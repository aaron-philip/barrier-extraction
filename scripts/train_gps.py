#!/usr/bin/env python3
import os, sys, time
import numpy as np
import argparse
import h5py
from sklearn.gaussian_process.kernels import RationalQuadratic

# project imports
if "/mnt/research/ascsn/aaron-philip/barrier-extraction/src" not in sys.path:
    sys.path.append("/mnt/research/ascsn/aaron-philip/barrier-extraction/src")

from utils import setup_logger, load_packed_hdf5
from train import fit_gp   # your module where fit_gp now lives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input HDF5 dataset (.h5)")
    ap.add_argument("--out_h5", required=True, help="Output HDF5 file for all GP fit results")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--n_restarts", type=int, default=2)      # GP optimizer restarts
    ap.add_argument("--n_samples", type=int, default=500)     # posterior draws
    ap.add_argument("--random_state", type=int, default=None)     # posterior draws
    args = ap.parse_args()

    logger = setup_logger(name="train_gp", dataset_path=args.data, tag="gp")
    logger.info(f"=== Starting GP training on HDF5 dataset: {args.data} ===")

    # lazy input
    ds = load_packed_hdf5(args.data)
    N = ds["N"]
    logger.info(f"Loaded {N} datasets.")

    # prepare output
    out_path = args.out_h5
    if os.path.exists(out_path):
        logger.warning(f"Overwriting existing file: {out_path}")
        os.remove(out_path)

    start = time.time()
    with h5py.File(out_path, "w") as f_out:
        for i in range(N):
            try:
                E_i, S_i, Se_i, p_true = ds["get_i"](i)

                res = fit_gp(
                    E_i, S_i, Se_i,
                    n_restarts=args.n_restarts, n_samples=args.n_samples, random_state=args.random_state,
                    kernel=1.0*RationalQuadratic(
                length_scale=90.0,
                length_scale_bounds="fixed",
                alpha=1.0,             # shape parameter
                alpha_bounds=(1e-12, 1e2)
                ))

                grp = f_out.create_group(f"{i}")
                # raw data used for fit (handy for plotting)
                grp.create_dataset("E_data", data=E_i)
                grp.create_dataset("Sigma_data", data=S_i)
                grp.create_dataset("Sigma_err", data=Se_i)

                # predictions
                grp.create_dataset("x_pred", data=res["x_pred"])
                grp.create_dataset("y_mean", data=res["y_mean"])
                grp.create_dataset("y_samples", data=res["y_samples"], compression="gzip")

                # attrs / metadata
                grp.attrs.update({
                    "gp_kernel_opt": res["gp_kernel_opt"],
                    "gp_log_marginal_likelihood": res["gp_log_marginal_likelihood"],
                    "n_restarts": res["n_restarts"],
                    "n_samples": res["n_samples"],
                    "E0_true": float(p_true[0]),
                    "R0_true": float(p_true[1]),
                    "hw_true": float(p_true[2]),
                    "index": int(i),
                })

            except Exception as e:
                logger.exception(f"[{i}] GP fit failed: {e}")
                continue

            if (i + 1) % args.log_every == 0 or (i + 1) == N:
                elapsed = (time.time() - start) / 60.0
                logger.info(f"Processed {i + 1}/{N} | Elapsed: {elapsed:.1f} min")

    logger.info(f"=== Finished {N} GP fits in {(time.time()-start)/60:.2f} min ===")
    logger.info(f"Wrote results to {out_path}")
    ds["file"].close()


if __name__ == "__main__":
    main()