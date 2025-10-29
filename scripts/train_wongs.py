#!/usr/bin/env python3
import os, sys, time, logging
import numpy as np
import pandas as pd
import argparse

# Import project modules
if "/mnt/research/ascsn/aaron-philip/barrier-extraction/src" not in sys.path:
    sys.path.append("/mnt/research/ascsn/aaron-philip/barrier-extraction/src")
from train import fit_wong
from utils import setup_logger   # updated import

# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_csv", required=True)
    
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)
    args = ap.parse_args()
   # args.data = "/mnt/home/philipaa/scratch/barrier-paper-datasets/cv0.00_spacing0.5.npz"

    logger = setup_logger(name="train_wong", dataset_path=args.data, tag="wong")
    logger.info(f"=== Starting Wong training on dataset: {args.data} ===")

    # === Load data ===
    npz = np.load(args.data, allow_pickle=True)
    E_all = npz["E"]
    Sigma_all = npz["Sigma"]
    Sigma_err_all = npz["Sigma_err"]
    params_all = npz["params"]

    N = len(E_all)
    logger.info(f"Loaded {N} datasets (variable grid lengths).")

    # === Checkpoint recovery ===
    tmp_file = args.out_csv + ".tmp"
    start_index = 0
    results = []

    if os.path.exists(tmp_file):
        logger.info(f"Checkpoint file found: {tmp_file}. Resuming previous progress.")
        df_prev = pd.read_csv(tmp_file)
        results = df_prev.to_dict("records")
        start_index = len(results)
        logger.info(f"Resuming from dataset #{start_index + 1}")

    # === Main loop ===
    start_time = time.time()
    for i in range(start_index, N):
        try:
            E_i = np.array(E_all[i], dtype=float)
            Sigma_i = np.array(Sigma_all[i], dtype=float)
            Sigma_err_i = np.array(Sigma_err_all[i], dtype=float)
            params_i = np.array(params_all[i], dtype=float)

            res = fit_wong(E_i, Sigma_i, Sigma_err_i, params_i)
            results.append(res)
        except Exception as e:
            logger.exception(f"[{i}] Fitting failed: {e}")
            continue

        # --- Periodic reporting ---
        if (i + 1) % args.log_every == 0 or (i + 1) == N:
            elapsed = time.time() - start_time
            logger.info(f"Processed {i + 1}/{N} fits | Elapsed: {elapsed/60:.1f} min")

        # --- Periodic checkpoint saving ---
        if (i + 1) % args.save_every == 0 or (i + 1) == N:
            df_partial = pd.DataFrame(results)
            df_partial.to_csv(tmp_file, index=False)
            logger.info(f"Checkpoint saved to {tmp_file} at fit {i + 1}")

    # === Finalize ===
    df_final = pd.DataFrame(results)
    df_final.to_csv(args.out_csv, index=False)
    logger.info(f"Completed {N} fits. Final results written to {args.out_csv}")

    if os.path.exists(tmp_file):
        os.remove(tmp_file)
        logger.info(f"Temporary checkpoint file {tmp_file} removed.")

    total_time = (time.time() - start_time) / 60
    logger.info(f"=== Finished all Wong fits in {total_time:.2f} min ===")


if __name__ == "__main__":
    main()
