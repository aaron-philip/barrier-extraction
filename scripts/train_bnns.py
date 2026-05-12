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

from bnn_utils import compute_Df_log  

# autobnn import
if "/mnt/home/philipaa/research/probability/spinoffs/autobnn" not in sys.path:
    sys.path.insert(0, "/mnt/home/philipaa/research/probability/spinoffs/autobnn")
import autobnn
from autobnn import estimators, training_util


def _build_d3_penalty_kwargs(args, e_grid, x_scale, x_shift, y_scale, y_shift):
    """Build third-derivative penalty kwargs for AutoBNN."""
    if args.d3_lambda <= 0.0:
        return {}
    if np.asarray(e_grid).size < 5:
        raise ValueError("Third-derivative penalty requires at least 5 grid points.")

    x_penalty = ((np.asarray(e_grid, dtype=float) - x_shift) / x_scale)[..., None]
    penalty_lambda = float(args.d3_lambda)
    if args.d3_lambda_space == "physical":
        # The AutoBNN penalty is computed in normalized x-space. Rescale so the
        # CLI lambda corresponds to derivatives with respect to the original
        # energy axis instead.
        penalty_lambda /= float(x_scale) ** 6

    return {
        "penalty_x_grid": x_penalty,
        "penalty_lambda": penalty_lambda,
        "penalty_use_sigma_space": args.d3_on_sigma,
        "penalty_y_scale": y_scale,
        "penalty_y_shift": y_shift,
    }


def _compute_d3_loss_diagnostics(bnn_est, x_train, y_train, d3_penalty_kwargs):
    """Compute base loss, penalty loss, and their ratio for posterior samples."""
    params_batch = bnn_est.params_

    base_loss_fn = jax.jit(
        jax.vmap(lambda p: -bnn_est.net_.log_prob(p, data=x_train, observations=y_train))
    )
    base_losses = np.asarray(base_loss_fn(params_batch))

    if d3_penalty_kwargs:
        penalty_loss_fn = jax.jit(
            jax.vmap(
                lambda p: training_util.third_derivative_penalty_on_grid(
                    net=bnn_est.net_,
                    params=p,
                    x_grid=d3_penalty_kwargs["penalty_x_grid"],
                    lam=d3_penalty_kwargs["penalty_lambda"],
                    use_sigma_space=d3_penalty_kwargs["penalty_use_sigma_space"],
                    y_scale=d3_penalty_kwargs["penalty_y_scale"],
                    y_shift=d3_penalty_kwargs["penalty_y_shift"],
                )
            )
        )
        d3_losses = np.asarray(penalty_loss_fn(params_batch))
    else:
        d3_losses = np.zeros_like(base_losses)

    denom = np.maximum(np.abs(base_losses), 1e-12)
    d3_relative = d3_losses / denom

    return {
        "base_loss_samples": base_losses,
        "d3_penalty_loss_samples": d3_losses,
        "d3_to_base_loss_ratio_samples": d3_relative,
        "base_loss_mean": float(np.mean(base_losses)),
        "d3_penalty_loss_mean": float(np.mean(d3_losses)),
        "d3_to_base_loss_ratio_mean": float(np.mean(d3_relative)),
        "d3_to_base_loss_ratio_median": float(np.median(d3_relative)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--index", required=True, type=int,
                    help="Parameter-sample id (NOT raw dataset row). Actual row = index * stride.")
    ap.add_argument("--sub_index", type=int, default=None,
                help="Optional repeat/run id (e.g. 1-10) for multiple trainings of the same curve.")
    ap.add_argument("--stride", type=int, default=10,
                    help="Replicates per parameter sample (e.g. 10). Use row = index*stride.")
    ap.add_argument("--out_h5", required=True,
                    help="FINAL merged output H5 name (we will write parts into out_h5 + '.parts/')")
    ap.add_argument("--extra", type=str, default=None)
    ap.add_argument("--likelihood_model", type=str, default="normal_likelihood_lognormal_noise",)
    ap.add_argument("--kernel", type=str, default="sum_of_changepoints", help="Kernel for BNN (default: sum_of_changepoints)")
   # ap.add_argument("--periods", type=float, nargs="+", default=(0.0,), help="Periods for sum-of-changepoints kernel (default: [0.0])")

    # BNN hyperparams
    ap.add_argument("--width", type=int, default=50)
    ap.add_argument("--num_chains", type=int, default=4)
    ap.add_argument("--num_draws", type=int, default=1000)
    ap.add_argument("--num_warmup", type=int, default=1000)
    ap.add_argument("--random_state", type=int, default=0)

    # metrics knobs (match GP)
    ap.add_argument("--pad", type=float, default=2.0)
    ap.add_argument("--peak_window", type=float, default=6.0)
    ap.add_argument("--barrier_smooth", type=int, default=5)

    # penalty controls
    ap.add_argument("--d3_lambda", type=float, default=0.0)
    ap.add_argument("--d3_on_sigma", action="store_true")
    ap.add_argument(
        "--d3_lambda_space",
        choices=("physical", "normalized"),
        default="physical",
        help=(
            "Interpret --d3_lambda in physical energy units or normalized "
            "training-x units."
        ),
    )

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
    d3_penalty_kwargs = _build_d3_penalty_kwargs(
        args=args,
        e_grid=E_pred,
        x_scale=x_scale,
        x_shift=x_shift,
        y_scale=y_scale,
        y_shift=y_shift,
    )

    # -----------------------------
    # build + train MCMC BNN
    # -----------------------------
    if args.sub_index is None:
        seed_int = int(args.random_state)
    else:
        seed_int = int(args.random_state) + 10_000 * int(args.index) + int(args.sub_index)

    seed = jax.random.PRNGKey(seed_int)

    if d3_penalty_kwargs:
        logger.info(
            "Using d3 penalty: lambda=%s (%s x-space), sigma_space=%s",
            args.d3_lambda,
            args.d3_lambda_space,
            args.d3_on_sigma,
        )

    bnn_mcmc = estimators.AutoBnnMCMCEstimator(
        args.kernel,
        likelihood_model=args.likelihood_model,
        seed=seed,
        periods=(E_train.max() - E_train.min(),),
        width=args.width,
        num_chains=args.num_chains,
        num_draws=args.num_draws,
        num_warmup=args.num_warmup,
        distributed=1,
        **d3_penalty_kwargs,
    )

    logger.info("Training MCMC BNN...")
    
    t0 = time.time()
    if args.likelihood_model == "normal_likelihood_known_noise":
        std_log = Sigma_err_raw / np.maximum(Sigma_train_raw, 1e-30)     # shape (N,)
        std_log_norm = (std_log / y_scale)[..., None]               # shape (N,1)
        obs_train = np.concatenate([y_norm, std_log_norm], axis=1)    # (N,2)
    else:
        obs_train = y_norm
    bnn_mcmc = bnn_mcmc.fit(x_norm, obs_train)
    logger.info(f"MCMC training done in {time.time() - t0:.2f} s")

    logger.info("Computing D3/base-loss diagnostics on posterior samples...")
    d3_loss_diag = _compute_d3_loss_diagnostics(
        bnn_est=bnn_mcmc,
        x_train=x_norm,
        y_train=obs_train,
        d3_penalty_kwargs=d3_penalty_kwargs,
    )
    logger.info(
        "Mean base loss %.6g, mean d3 penalty %.6g, mean relative ratio %.6g",
        d3_loss_diag["base_loss_mean"],
        d3_loss_diag["d3_penalty_loss_mean"],
        d3_loss_diag["d3_to_base_loss_ratio_mean"],
    )

    # -----------------------------
    # posterior predictive sigma samples on E_pred grid
    # -----------------------------
    logger.info("Computing posterior predictive sigma samples...")
    y_samples, d2_samples = compute_Df_log(bnn_mcmc, E_pred, scales)

    y_samples = np.asarray(y_samples)
    if y_samples.ndim == 1:
        y_samples = y_samples[None, :]  # (1, M)
    y_mean = np.mean(y_samples, axis=0)
    # -----------------------------
    # compute metrics (only useful for Wong curves)
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
        d2_samples=d2_samples,
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
    if args.extra is not None:
        out_base += f"_{args.extra}"
    parts_dir = out_base + ".parts"
    os.makedirs(parts_dir, exist_ok=True)

    if args.sub_index is None:
        part_h5  = os.path.join(parts_dir, f"{int(args.index):05d}.h5")
        part_csv = os.path.join(parts_dir, f"{int(args.index):05d}.csv")
        group_name = str(int(args.index))
    else:
        part_h5  = os.path.join(parts_dir, f"{int(args.index):05d}_run{int(args.sub_index):02d}.h5")
        part_csv = os.path.join(parts_dir, f"{int(args.index):05d}_run{int(args.sub_index):02d}.csv")
        group_name = f"{int(args.index):05d}_run{int(args.sub_index):02d}"

    # Write part H5 with GP-compatible schema
    with h5py.File(part_h5, "w") as f:
        grp = f.create_group(group_name)

        grp.create_dataset("E_data", data=np.asarray(E_train, float))
        grp.create_dataset("Sigma_data", data=np.asarray(Sigma_train_raw, float))
        grp.create_dataset("Sigma_err", data=np.asarray(Sigma_err_raw, float))

        grp.create_dataset("x_pred", data=np.asarray(E_pred, float))
        grp.create_dataset("y_mean", data=np.asarray(y_mean, float))
        grp.create_dataset("y_samples", data=np.asarray(y_samples, float), compression="gzip")
        grp.create_dataset("d2_samples", data=np.asarray(d2_samples, float), compression="gzip")
        grp.create_dataset(
            "base_loss_samples",
            data=np.asarray(d3_loss_diag["base_loss_samples"], float),
            compression="gzip",
        )
        grp.create_dataset(
            "d3_penalty_loss_samples",
            data=np.asarray(d3_loss_diag["d3_penalty_loss_samples"], float),
            compression="gzip",
        )
        grp.create_dataset(
            "d3_to_base_loss_ratio_samples",
            data=np.asarray(d3_loss_diag["d3_to_base_loss_ratio_samples"], float),
            compression="gzip",
        )

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
            "d3_lambda": float(args.d3_lambda),
            "d3_on_sigma": bool(args.d3_on_sigma),
            "d3_lambda_space": args.d3_lambda_space,
            "d3_penalty_active": bool(d3_penalty_kwargs),
            "base_loss_mean": d3_loss_diag["base_loss_mean"],
            "d3_penalty_loss_mean": d3_loss_diag["d3_penalty_loss_mean"],
            "d3_to_base_loss_ratio_mean": d3_loss_diag["d3_to_base_loss_ratio_mean"],
            "d3_to_base_loss_ratio_median": d3_loss_diag["d3_to_base_loss_ratio_median"],
            "x_scale": float(x_scale),
            "x_shift": float(x_shift),
            "y_scale": float(y_scale),
            "y_shift": float(y_shift),
        })
        if args.sub_index is not None:
            grp.attrs["sub_index"] = int(args.sub_index)
            grp.attrs["seed_int"] = int(seed_int)

    # Write one-row part CSV
    import pandas as pd
    pd.DataFrame([met]).to_csv(part_csv, index=False)

    logger.info(f"Finished param index {args.index}.")
    logger.info(f"  wrote part H5  -> {part_h5}")
    logger.info(f"  wrote part CSV -> {part_csv}")

    ds["file"].close()

    print("BNN Diagnostics", bnn_mcmc.diagnostics_)


if __name__ == "__main__":
    main()
