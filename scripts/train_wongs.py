#!/usr/bin/env python3
import os, sys, time
import numpy as np
import argparse
import h5py

# Import project modules
if "/mnt/research/ascsn/aaron-philip/barrier-extraction/src" not in sys.path:
    sys.path.append("/mnt/research/ascsn/aaron-philip/barrier-extraction/src")

from train import fit_wong
from utils import setup_logger, load_packed_hdf5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input HDF5 dataset (.h5)")
    ap.add_argument("--out_h5", required=True, help="Output HDF5 file for all fit results")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--n_restarts", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=500)
    args = ap.parse_args()

    logger = setup_logger(name="train_wong", dataset_path=args.data, tag="wong")
    logger.info(f"=== Starting Wong training on HDF5 dataset: {args.data} ===")

    # === Load packed data lazily ===
    ds = load_packed_hdf5(args.data)
    N = ds["N"]
    logger.info(f"Loaded {N} datasets from {args.data}")

    out_file = args.out_h5
    if os.path.exists(out_file):
        logger.warning(f"Overwriting existing file: {out_file}")
        os.remove(out_file)

    start_time = time.time()

    with h5py.File(out_file, "w") as f_out:
        for i in range(N):
            try:
                E_i, Sigma_i, Sigma_err_i, params_i = ds["get_i"](i)
                res = fit_wong(
                    E_i,
                    Sigma_i,
                    Sigma_err_i,
                    n_restarts=args.n_restarts,
                    n_samples=args.n_samples,
                )
                if res is None:
                    logger.warning(f"[{i}] Fit failed; skipping.")
                    continue

                grp = f_out.create_group(str(i))
                grp.create_dataset("E_data", data=E_i)
                grp.create_dataset("Sigma_data", data=Sigma_i)
                grp.create_dataset("Sigma_err", data=Sigma_err_i)
                grp.create_dataset("x_pred", data=res["x_pred"])
                grp.create_dataset("y_mean", data=res["y_mean"])
                grp.create_dataset("y_samples", data=res["y_samples"], compression="gzip")
          
                grp.attrs.update({
                    "E0_fit": res["best_popt"][0],
                    "R0_fit": res["best_popt"][1],
                    "hw_fit": res["best_popt"][2],
                    "chi2":   res["chi2"],
                    "E0_true": params_i[0],
                    "R0_true": params_i[1],
                    "hw_true": params_i[2],
                    "index":   i,
                })

            except Exception as e:
                logger.exception(f"[{i}] Fitting failed: {e}")
                continue

            if (i + 1) % args.log_every == 0 or (i + 1) == N:
                elapsed = time.time() - start_time
                logger.info(f"Processed {i + 1}/{N} fits | Elapsed: {elapsed/60:.1f} min")

    total_time = (time.time() - start_time) / 60
    logger.info(f"=== Finished all Wong fits in {total_time:.2f} min ===")
    logger.info(f"Results written to {out_file}")

    ds["file"].close()


if __name__ == "__main__":
    main()
