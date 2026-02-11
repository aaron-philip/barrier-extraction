#!/usr/bin/env python3
import os, sys, time
import argparse
import h5py
import numpy as np
import jax

# project imports
SRC = "/mnt/research/ascsn/aaron-philip/barrier-extraction/src"
if SRC not in sys.path:
    sys.path.append(SRC)

from utils import setup_logger, load_packed_hdf5, compute_metrics_for_fit
from train import make_x_pred, barrier_from_sigma_samples
from wong import sigma as sigma_true

from bnn_utils import compute_Df  # assumes this returns sigma samples on E_pred grid

# autobnn import
if "../../probability/spinoffs/autobnn" not in sys.path:
    sys.path.insert(0, "../../probability/spinoffs/autobnn")
from autobnn import estimators


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--index", required=True, type=int,
                    help="Parameter-sample id (NOT raw dataset row). Actual row = index * stride.")
    ap.add_argument("--stride", type=int, default=10,
                    help="Replicates per parameter sample (e.g. 10). Use row = index*stride.")
    ap.add_argument("--out_h5", required=True,
                    help="FINAL merged output H5 name (we will write parts into out_h5 + '.parts/')")

    # BNN hyperparams
    ap.add_argument("--width", type=int, default=50)
    ap.add_argument("--num_chains", type=int, default=4)
    ap.add_argument("--num_draws", type=int, default=50)
    ap.add_argument("--num_warmup", type=int, default=1000)
    ap.add_argument("--random_state", type=int, default=0)

    # metrics knobs (match GP)
    ap.add_argument("--pad", type=float, default=2.0)
    ap.add_argument("--peak_window", type=float, default=6.0)
    ap.add_argument("--barrier_smooth", type=int, default=5)

    args = ap.parse_args()

    logger = setup_logger(name="train_bnn", dataset_path=args.data, tag="bnn")
    logger.info(f"Starting BNN training for param index {args.index} (stride={args.stride}) in {args.data}")

    ds = load_packed_hdf5(args.data)
    N = int(ds["N"])

    raw_idx = int(args.index) * int(args.stride)
    if raw_idx >= N:
        logger.error(f"Raw index {raw_idx} out of range (N={N}). index={args.index}, stride={args.stride}")
        ds["file"].close()
        return

    # Load one curve
    E_train, Sigma_train, Sigma_err_train, params_true = ds["get_i"](raw_idx)

    # --- store ORIGINAL sigma-space data for GP-compatible plotting ---
    Sigma_train_raw = np.asarray(Sigma_train, float)
    Sigma_err_raw   = np.asarray(Sigma_err_train, float)

    # --- train in log-space (your current approach) ---
    Sigma_train_log = np.log(np.maximum(Sigma_train_raw, 1e-30))

    # prediction grid (same helper you used)
    E_pred = make_x_pred(E_train)

    # normalization
    x_scale = float(np.std(E_train)) if np.std(E_train) > 0 else 1.0
    x_shift = float(np.mean(E_train))
    y_scale = float(np.std(Sigma_train_log)) if np.std(Sigma_train_log) > 0 else 1.0
    y_shift = float(np.mean(Sigma_train_log))

    x_norm = ((E_train - x_shift) / x_scale)[..., None]
    y_norm = ((Sigma_train_log - y_shift) / y_scale)[..., None]
    scales = (x_scale, x_shift, y_scale, y_shift)

    # -----------------------------
    # build + train MCMC BNN
    # -----------------------------
    seed = jax.random.PRNGKey(args.random_state)
    kernel = "sum_of_changepoints"

    bnn_mcmc = estimators.AutoBnnMCMCEstimator(
        kernel,
        likelihood_model="normal_likelihood_lognormal_noise",
        seed=seed,
        width=args.width,
        num_chains=args.num_chains,
        num_draws=args.num_draws,
        num_warmup=args.num_warmup,
        distributed=1,
    )

    logger.info("Training MCMC BNN...")
    t0 = time.time()
    bnn_mcmc = bnn_mcmc.fit(x_norm, y_norm)
    logger.info(f"MCMC training done in {time.time() - t0:.2f} s")

    # -----------------------------
    # posterior predictive sigma samples on E_pred grid
    # -----------------------------
    logger.info("Computing posterior predictive sigma samples...")
    # We assume compute_Df returns sigma samples in *sigma space* on E_pred.
    # If it returns log-sigma samples instead, we’ll exponentiate below.
    y_samples = compute_Df(bnn_mcmc, E_pred, scales)

    y_samples = np.asarray(y_samples)
    if y_samples.ndim == 1:
        y_samples = y_samples[None, :]  # (1, M)

    # Heuristic: if outputs look like log-sigma (can be negative), exponentiate
    # (This keeps plotting script happy, which expects sigma > 0)
    if np.nanmedian(y_samples) < 0:
        y_samples = np.exp(y_samples)

    y_mean = np.mean(y_samples, axis=0)

    # -----------------------------
    # compute metrics (same function as GP workflow)
    # -----------------------------
    met = compute_metrics_for_fit(
        x_pred=E_pred,
        y_mean=y_mean,
        y_samples=y_samples,
        E_data=E_train,
        E0_true=float(params_true[0]),
        R0_true=float(params_true[1]),
        hw_true=float(params_true[2]),
        sigma_true_fn=sigma_true,
        barrier_from_sigma_samples_fn=barrier_from_sigma_samples,
        pad=args.pad,
        peak_window=args.peak_window,
        barrier_smooth=args.barrier_smooth,
    )
    met.update({
        "index": int(args.index),           # IMPORTANT: matches group name later
        "E0_true": float(params_true[0]),
        "R0_true": float(params_true[1]),
        "hw_true": float(params_true[2]),
    })

    # -----------------------------
    # write PART outputs (safe for arrays)
    # -----------------------------
    out_base = args.out_h5
    parts_dir = out_base + ".parts"
    os.makedirs(parts_dir, exist_ok=True)

    part_h5  = os.path.join(parts_dir, f"{int(args.index):05d}.h5")
    part_csv = os.path.join(parts_dir, f"{int(args.index):05d}.csv")

    # Write part H5 with GP-compatible schema
    with h5py.File(part_h5, "w") as f:
        grp = f.create_group(str(int(args.index)))

        grp.create_dataset("E_data", data=np.asarray(E_train, float))
        grp.create_dataset("Sigma_data", data=np.asarray(Sigma_train_raw, float))
        grp.create_dataset("Sigma_err", data=np.asarray(Sigma_err_raw, float))

        grp.create_dataset("x_pred", data=np.asarray(E_pred, float))
        grp.create_dataset("y_mean", data=np.asarray(y_mean, float))
        grp.create_dataset("y_samples", data=np.asarray(y_samples, float), compression="gzip")

        grp.attrs.update({
            "width": int(args.width),
            "num_chains": int(args.num_chains),
            "num_draws": int(args.num_draws),
            "num_warmup": int(args.num_warmup),
            "params_true_E0": float(params_true[0]),
            "params_true_R0": float(params_true[1]),
            "params_true_hw": float(params_true[2]),
            "raw_dataset_index": int(raw_idx),
            "stride": int(args.stride),
            "E0_true": float(params_true[0]),
            "R0_true": float(params_true[1]),
            "hw_true": float(params_true[2]),
        })

    # Write one-row part CSV
    import pandas as pd
    pd.DataFrame([met]).to_csv(part_csv, index=False)

    logger.info(f"Finished param index {args.index}.")
    logger.info(f"  wrote part H5  -> {part_h5}")
    logger.info(f"  wrote part CSV -> {part_csv}")

    ds["file"].close()


if __name__ == "__main__":
    main()
