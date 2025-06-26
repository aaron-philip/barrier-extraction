import argparse
import numpy as np
import pandas as pd
import os
import sys 
if "../../" not in sys.path:
    sys.path.append("../../")

if "../../../probability/spinoffs/autobnn" not in sys.path:
    sys.path.insert(0, "../../../probability/spinoffs/autobnn")

import jax
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys


import autobnn
from autobnn import estimators
from autobnn import kernels
from src.bnn_utils import compute_Df, save_bnn_params

def main(args):
    
    E_train     = np.load(args.e_train)
    sigma_train = np.load(args.sigma_train)
    print(f"Loaded E_train from {args.e_train}, sigma_train from {args.sigma_train}")
    
    # Save and Load 
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        base = args.out_dir
    else:
        base = os.getcwd()
    print(base)

    #  Normalize I/O
    x_scale = np.std(E_train)
    x_shift = np.mean(E_train)
    y_scale = np.std(sigma_train)
    y_shift = np.mean(sigma_train)

    x_norm = ((E_train - x_shift) / x_scale)[..., None]
    y_norm = ((sigma_train-y_shift) / y_scale)[..., None]


    # Build & train the AutoBnnMapEstimator
    st = time.time()
    seed = jax.random.PRNGKey(0)
    seed, est_seed = jax.random.split(seed)
    rbf = kernels.ExponentiatedQuadraticBNN()
    bnn_est = estimators.AutoBnnMCMCEstimator(
        args.model,
        likelihood_model    = args.likelihood,
        seed                = est_seed,
        distributed         = args.distributed,
        num_chains          = args.num_chains, #4,
        num_draws           = args.num_draws, #50,
        num_warmup          = args.num_warmup
    )
    bnn_est = bnn_est.fit(x_norm, y_norm)
    print("Training time: ", time.time() - st)

    # Save BNN params
    params_f = os.path.join(base, f"bnn_params_{args.model}_{args.likelihood}.pkl")
    save_bnn_params(bnn_est.params_, params_f)
    print(f"Saved BNN params to {params_f}")

    print("Taking Derivatives Now")
    st = time.time()
    x_pred = np.arange(E_train[0]-5,E_train[-1]+5, 0.2)
    scales = (x_scale, x_shift, y_scale, y_shift)
    all_second_derivs = compute_Df(bnn_est, x_pred, scales)
    print("All Derivatives Taken in: ", time.time()-st)
    print("Derivative Shape After Pruning: ", all_second_derivs.shape)
    derivs = np.array(all_second_derivs)
    derivs_fname = os.path.join(base, f"second_derivs_{args.model}_{args.likelihood}.npy")
    np.save(derivs_fname, derivs)
    print(f"Saved second‐derivatives to {derivs_fname}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train one AutoBnn MCMC model")
    p.add_argument("--model",
                   choices=["sum_of_safe_shallow", "sum_of_changepoints","sum_of_products", "sum_of_stumps",
                             "sum_of_stumps_and_products", "changepoint_of_safe_products", 
                             "mlp_depth2", "mlp_depth3", "mlp_depth4", "mlp_depth5"],
                   required=True)
    p.add_argument("--likelihood",
                   choices=["normal_likelihood_logistic_noise",
                             "normal_likelihood_lognormal_noise"],
                   required=True)
    p.add_argument("--num_warmup", type=int, default=1000,
                   help="Number of warmup (burn-in) iterations")
    p.add_argument("--num_draws", type=int, default=50,
                   help="Number of  post-burn-in iterations per chain")
    p.add_argument("--num_chains", type=int, default=4,
                   help="Number of MCMC chains to run")
    p.add_argument("--distributed", type=int, default=1,
                   help="[0,1,2]=[sequential, vectorized, parallel]")
    p.add_argument("--e_train",       type=str, default=None,
                   help="Path to a .npy file with your E_train array")
    p.add_argument("--sigma_train",   type=str, default=None,
                   help="Path to a .npy file with your sigma_train array")
    p.add_argument("--out_dir",       type=str, default=None,
                   help="Base directory to save params, arrays & .csv")
    args = p.parse_args()
    main(args)