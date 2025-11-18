#!/usr/bin/env python3
import os, sys, time
import argparse
import h5py
import numpy as np
import jax

# project imports
if "/mnt/research/ascsn/aaron-philip/barrier-extraction/src" not in sys.path:
    sys.path.append("/mnt/research/ascsn/aaron-philip/barrier-extraction/src")

from utils import setup_logger, load_packed_hdf5
from train import  make_x_pred
from bnn_utils import compute_Df
from wong import sigma


if "../../probability/spinoffs/autobnn" not in sys.path:
    sys.path.insert(0, "../../probability/spinoffs/autobnn")
    
from autobnn import estimators

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--index", required=True, type=int)
    ap.add_argument("--out_h5", required=True)
    ap.add_argument("--width", type=int, default=50)
    ap.add_argument("--num_chains", type=int, default=4)
    ap.add_argument("--num_draws", type=int, default=50)
    ap.add_argument("--num_warmup", type=int, default=1000)
    ap.add_argument("--random_state", type=int, default=0)
    args = ap.parse_args()

    logger = setup_logger(name="train_bnn", dataset_path=args.data, tag="bnn")
    logger.info(f"Starting BNN training for sample index {args.index} in {args.data}")

    # -----------------------------
    # load dataset + sample
    # -----------------------------
    ds = load_packed_hdf5(args.data)
    N = ds["N"]

    if args.index >= N:
        logger.error(f"Index {args.index} out of range (N={N})")
        return

    E_train, Sigma_train, Sigma_err_train, params_true = ds["get_i"](args.index)

    # transform
    Sigma_train = np.log(Sigma_train)
    E_pred = make_x_pred(E_train)

    # normalization
    x_scale = np.std(E_train)
    x_shift = np.mean(E_train)
    y_scale = np.std(Sigma_train)
    y_shift = np.mean(Sigma_train)

    x_norm = ((E_train - x_shift)/x_scale)[..., None]
    y_norm = ((Sigma_train - y_shift)/y_scale)[..., None]

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
    logger.info(f"MCMC training done in {time.time()-t0:.2f} s")

    # -----------------------------
    # derivatives
    # -----------------------------
    logger.info("Computing derivatives...")
    d2_mcmc = compute_Df(bnn_mcmc, E_pred, scales)
    d2_mcmc_mean = d2_mcmc.mean(axis=0) if d2_mcmc.ndim > 1 else d2_mcmc

    # true distribution
    d2_true = sigma(E_pred, *params_true)

    # -----------------------------
    # append to output HDF5 file
    # -----------------------------
    out_path = args.out_h5
    mode = "a" if os.path.exists(out_path) else "w"

    with h5py.File(out_path, mode) as f:
        if str(args.index) in f:
            logger.info(f"Sample {args.index} already exists → skipping.")
            return

        grp = f.create_group(str(args.index))

        grp.create_dataset("E_data", data=E_train)
        grp.create_dataset("Sigma_data", data=Sigma_train)
        grp.create_dataset("Sigma_err", data=Sigma_err_train)
        grp.create_dataset("x_pred", data=E_pred)
        grp.create_dataset("d2_mcmc", data=d2_mcmc, compression="gzip")
        grp.create_dataset("d2_mcmc_mean", data=d2_mcmc_mean)
        grp.create_dataset("d2_true", data=d2_true)

        grp.attrs.update({
            "width": args.width,
            "num_chains": args.num_chains,
            "num_draws": args.num_draws,
            "num_warmup": args.num_warmup,
            "params_true_E0": float(params_true[0]),
            "params_true_R0": float(params_true[1]),
            "params_true_hw": float(params_true[2]),
        })

    logger.info(f"Finished sample {args.index}. Written to {out_path}.")


if __name__ == "__main__":
    main()

