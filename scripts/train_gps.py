#!/usr/bin/env python3
"""Train GP baselines for every curve in one packed HDF5 dataset."""
import os, sys, time
from pathlib import Path
import numpy as np
import argparse
import h5py
import pandas as pd
from sklearn.gaussian_process.kernels import RBF  # RationalQuadratic if you want

# -------------------------------------------------------------------
# project imports
# -------------------------------------------------------------------
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from utils import setup_logger, load_packed_hdf5, compute_metrics_for_fit
from train import fit_gp  # ChangePointKernel, fit_wong_with_restarts if you re-enable them
from train import barrier_from_sigma_samples
from wong import sigma as sigma_true


def main():
    ap = argparse.ArgumentParser()

    # --- training inputs/outputs ---
    ap.add_argument("--data", required=True, help="Input packed HDF5 dataset (.h5)")
    ap.add_argument("--out_h5", required=True, help="Output HDF5 file for all GP fit results")
    ap.add_argument("--log_dir", default=None, help="Directory for train_gps log files")

    # --- GP fit controls ---
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--n_restarts", type=int, default=20)          # GP optimizer restarts
    ap.add_argument("--n_samples", type=int, default=500)         # posterior draws saved to HDF5
    ap.add_argument("--random_state", type=int, default=None)
    ap.add_argument(
        "--target_space",
        choices=("linear", "log"),
        default="linear",
        help="Fit the GP to Sigma directly or to log(Sigma), transforming predictions back to Sigma space.",
    )

    # --- metric controls (must match what plotting expects) ---
    ap.add_argument("--pad", type=float, default=2.0,
                    help="Metrics region = [min(E_data)-pad, max(E_data)+pad] on x_pred")
    ap.add_argument("--peak_window", type=float, default=6.0,
                    help="Search window around E0_true for barrier peak metrics")
    ap.add_argument("--barrier_smooth", type=int, default=5,
                    help="Moving-average width for barrier peak picking")
    ap.add_argument("--max_items", type=int, default=None,
                    help="Optional cap on number of fits to process (debugging)")

    args = ap.parse_args()

    logger = setup_logger(name="train_gp", dataset_path=args.data, tag="gp", log_dir=args.log_dir)
    logger.info(f"=== Starting GP training on HDF5 dataset: {args.data} ===")
    logger.info(f"GP target space: {args.target_space}")

    # Packed files concatenate variable-length curves.  ``load_packed_hdf5``
    # returns a lazy accessor so we only materialize one curve at a time.
    ds = load_packed_hdf5(args.data)
    N_total = int(ds["N"])
    N = N_total if args.max_items is None else min(N_total, int(args.max_items))
    logger.info(f"Loaded {N_total} datasets. Will process {N}.")

    # prepare outputs
    out_h5 = args.out_h5
    out_csv = out_h5 + ".csv"

    if os.path.exists(out_h5):
        logger.warning(f"Overwriting existing HDF5: {out_h5}")
        os.remove(out_h5)

    # ensure CSV dir exists (we write at the end)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    rows = []
    start = time.time()

    with h5py.File(out_h5, "w") as f_out:
        for i in range(N):
            try:
                E_i, S_i, Se_i, p_true = ds["get_i"](i)
                
                # Paper baseline: a single RBF kernel.  The commented
                # changepoint/RQ kernels below are retained for exploratory
                # comparisons, but are not part of the release pipeline.
                #popt, pcov, chi2= fit_wong_with_restarts(E_i, S_i, Se_i, n_restarts=5)
                #E0 = popt[0]
                k_right = 1.0*RBF(length_scale_bounds=(1.0, 10.0))
                #k_left = 1.0*RBF(length_scale_bounds=(0.5, 10.0))

                res = fit_gp(
                    E_i, S_i, Se_i,
                    n_restarts=args.n_restarts, n_samples=args.n_samples, random_state=args.random_state,
                    kernel=k_right,
                    target_space=args.target_space,
                   # kernel=1.0*ChangePointKernel(
                   #     k1=k_right,
                   #     k2=k_left,
                   #     location=E0,
                   #     width=2.0,
                   #     location_bounds=(np.min(E_i), np.max(E_i)),
                   #     width_bounds=(1e-2, 50.0),)
                    
                  #  RationalQuadratic(
                   #         length_scale=90.0,
                   #         length_scale_bounds="fixed",
                   #         alpha=1.0,             # shape parameter
                   #         alpha_bounds=(1e-12, 1e2)
                   #         )
                ) 
           
                # One HDF5 group per fitted curve.  This mirrors the BNN merged
                # schema closely enough that the notebooks can compare methods
                # by swapping only the result directory.
                grp = f_out.create_group(f"{i}")

                grp.create_dataset("E_data", data=E_i)
                grp.create_dataset("Sigma_data", data=S_i)
                grp.create_dataset("Sigma_err", data=Se_i)

                grp.create_dataset("x_pred", data=res["x_pred"])
                grp.create_dataset("y_mean", data=res["y_mean"])
                grp.create_dataset("y_samples", data=res["y_samples"], compression="gzip")

                grp.attrs.update({
                    "gp_kernel_opt": res["gp_kernel_opt"],
                    "gp_log_marginal_likelihood": res["gp_log_marginal_likelihood"],
                    "n_restarts": res["n_restarts"],
                    "n_samples": res["n_samples"],
                    "target_space": res["target_space"],
                    "E0_true": float(p_true[0]),
                    "R0_true": float(p_true[1]),
                    "hw_true": float(p_true[2]),
                    "index": int(i),
                })

                # CSV metrics are redundant with the HDF5 predictions, but they
                # make the paper summary plots fast and lightweight to rerun.
                met = compute_metrics_for_fit(
                    x_pred=res["x_pred"],
                    y_mean=res["y_mean"],
                    y_samples=res["y_samples"],
                    E_data=E_i,
                    E0_true=float(p_true[0]),
                    R0_true=float(p_true[1]),
                    hw_true=float(p_true[2]),
                    sigma_true_fn=sigma_true,
                    barrier_from_sigma_samples_fn=barrier_from_sigma_samples,
                    pad=args.pad,
                    peak_window=args.peak_window,
                    barrier_smooth=args.barrier_smooth,
                )

                met.update({
                    "index": int(i),
                    "E0_true": float(p_true[0]),
                    "R0_true": float(p_true[1]),
                    "hw_true": float(p_true[2]),
                })
                rows.append(met)

            except Exception as e:
                logger.exception(f"[{i}] GP fit failed: {e}")
                continue

            if (i + 1) % args.log_every == 0 or (i + 1) == N:
                elapsed = (time.time() - start) / 60.0
                logger.info(f"Processed {i + 1}/{N} | Elapsed: {elapsed:.1f} min")

    # Write the CSV once at the end.  If an individual GP fit fails, it is
    # logged and omitted rather than aborting the full dataset.
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote metrics CSV with {len(df)} rows -> {out_csv}")

    logger.info(f"=== Finished {N} GP fits in {(time.time()-start)/60:.2f} min ===")
    logger.info(f"Wrote results HDF5 to {out_h5}")

    ds["file"].close()


if __name__ == "__main__":
    main()
