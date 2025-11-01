#!/usr/bin/env python3
import os, sys, time
import numpy as np
import pandas as pd
import argparse

# Import project modules
if "/mnt/research/ascsn/aaron-philip/barrier-extraction/src" not in sys.path:
    sys.path.append("/mnt/research/ascsn/aaron-philip/barrier-extraction/src")
from train import fit_wong
from utils import setup_logger, _load_packed_npz

# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--n_restarts", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=200)
    args = ap.parse_args()

    logger = setup_logger(name="train_wong", dataset_path=args.data, tag="wong")
    logger.info(f"=== Starting Wong training (packed format) on: {args.data} ===")

    # === Load packed data ===
    ds = _load_packed_npz(args.data)
    N = ds["N"]
    logger.info(f"Loaded {N} datasets (packed format).")

    # === Checkpoint recovery ===
    tmp_file = args.out_csv + ".tmp"
    start_index = 0
    results = []
    band_data = []  # collect large arrays separately

    if os.path.exists(tmp_file):
        logger.info(f"Checkpoint found: {tmp_file}. Resuming.")
        df_prev = pd.read_csv(tmp_file)
        results = df_prev.to_dict("records")
        start_index = len(results)
        logger.info(f"Resuming from dataset #{start_index + 1}")

    start_time = time.time()
    for i in range(start_index, N):
        try:
            E_i, Sigma_i, Sigma_err_i, params_i = ds["get_i"](i)

            res = fit_wong(
                E_i, Sigma_i, Sigma_err_i, params_i,
                n_restarts=args.n_restarts,
                n_samples=args.n_samples
            )

            # --- Separate band data (keep CSV small) ---
            band_data.append({
                "x_pred":        res["x_pred"],
                "y_mean":        res["y_mean"],
                "y_true":        res["y_true"],
                "D_mean":        res["D_mean"],
                "D_true":        res["D_true"],
                "sigma_bands":   res["sigma_bands"],
                "barrier_bands": res["barrier_bands"],
            })

            # Strip large arrays from metrics before CSV write
            for key in ("x_pred","y_mean","y_true","D_mean","D_true","sigma_bands","barrier_bands"):
                res.pop(key, None)

            results.append(res)

        except Exception as e:
            logger.exception(f"[{i}] Fitting failed: {e}")
            continue

        # Periodic logging
        if (i + 1) % args.log_every == 0 or (i + 1) == N:
            elapsed = time.time() - start_time
            logger.info(f"Processed {i + 1}/{N} fits | Elapsed: {elapsed/60:.1f} min")

        # Periodic checkpoint
        if (i + 1) % args.save_every == 0 or (i + 1) == N:
            pd.DataFrame(results).to_csv(tmp_file, index=False)
            logger.info(f"Checkpoint saved to {tmp_file} at fit {i + 1}")

    # === Final saves ===
    pd.DataFrame(results).to_csv(args.out_csv, index=False)
    logger.info(f"Completed {N} fits. Final results -> {args.out_csv}")

    bands_path = args.out_csv.replace(".csv", "_bands.npz")
    np.savez_compressed(bands_path, band_data=np.array(band_data, dtype=object))
    logger.info(f"Saved all uncertainty bands -> {bands_path}")

    if os.path.exists(tmp_file):
        os.remove(tmp_file)
        logger.info(f"Removed temporary checkpoint {tmp_file}")

    total_time = (time.time() - start_time) / 60
    logger.info(f"=== Finished all Wong fits in {total_time:.2f} min ===")

if __name__ == "__main__":
    main()
