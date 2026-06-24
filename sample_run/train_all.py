import argparse
import numpy as np
import pandas as pd
import os
import sys 
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if "../../probability/spinoffs/autobnn" not in sys.path:
    sys.path.append("../../probability/spinoffs/autobnn")
from scipy.optimize import curve_fit
from src.wong import sigma
import jax
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys


import autobnn
from autobnn import estimators, training_util
from autobnn import kernels, likelihoods

from src.io import load_experimental_data, get_data_arrays
from src.bnn_utils import compute_Df, save_bnn_params, load_bnn_params
from src.utils import barrier_dist

dirname = PROJECT_ROOT / "data" / "experimental" / "16O208Pb_exp"
expfiles = ['expMort.dat','expDas.dat']

def main(args):
     #### SYSTEM ####
    exp = load_experimental_data(dirname, expfiles)
    E_data, Sigma_data, Sigma_err = get_data_arrays(exp[0])
    


    # Fit Wong formula to experimental data
    initial_guess = [70.0, 11.0, 4]  # e.g. E0≈10MeV, R0≈7fm, ħω≈4MeV
    popt, pcov = curve_fit(
        sigma,
        E_data,
        Sigma_data,
        p0=initial_guess,
        sigma=Sigma_err,
        absolute_sigma=True,
        bounds=([0.0, 8.0, 0.0], [np.inf, 15.0, np.inf])
    )
    E0_opt, R0_opt, hbarw_opt = popt
    perr = np.sqrt(np.diag(pcov))

    print(f"Best-fit parameters:")
    print(f"  E0    = {E0_opt:.4f} ± {perr[0]:.4f} MeV")
    print(f"  R0    = {R0_opt:.4f} ± {perr[1]:.4f} fm")
    print(f"  hbarw = {hbarw_opt:.4f} ± {perr[2]:.4f} MeV")

    # default training grid
    N = 1000
    E_fine = np.linspace(E_data.min(), E_data.max(), N)
    sigma_mean  = sigma(E_fine, E0_opt, R0_opt, hbarw_opt)

    if args.e_train is not None and args.sigma_train is not None:
        E_train     = np.load(args.e_train)
        sigma_train = np.load(args.sigma_train)
        print(f"Loaded E_train from {args.e_train}, sigma_train from {args.sigma_train}")
    else:
        interval = 10
        start = 0
        stop  = np.argmin(np.abs(E_fine - 90.0))
        E_train     = E_fine[start:stop:interval]
        sigma_train = sigma_mean[start:stop:interval]
        print("Using default E_train and sigma_train")
    
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
#    save_bnn_params(bnn_est.params_, f"bnn_params_{args.model}_{args.likelihood}.pkl")

    #save_bnn_params(bnn_est.params_, 'bnn_params.pkl')

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


    # Compute and select true distribution function at desired points
    true_d = barrier_dist(E_fine,sigma(E_fine,E0_opt,R0_opt,hbarw_opt),acc=4)
    idx    = np.searchsorted(E_fine, x_pred)
    d2_true = true_d[idx]

    true_peak_loc, true_peak_height, true_fwhm = compute_peak_metrics(x_pred, d2_true)

    #    all_second_derivs has shape (n_draws, n_x)
    metrics = [compute_peak_metrics(x_pred, d2) 
            for d2 in np.array(all_second_derivs)]  # convert to numpy

    df_samples = pd.DataFrame(metrics, columns=['peak_loc', 'peak_height', 'fwhm'])
    summary = pd.DataFrame({
        'true':  [true_peak_loc, true_peak_height, true_fwhm],
        'mean':  df_samples.mean().values,
        'std':   df_samples.std().values,
    }, index=['peak_loc', 'peak_height', 'fwhm'])

    samples_fname = os.path.join(base, f"samples_{args.model}_{args.likelihood}.csv")
    summary_fname = os.path.join(base, f"summary_{args.model}_{args.likelihood}.csv")
    df_samples.to_csv(samples_fname, index=False)
    summary.to_csv(summary_fname)
    print(f"Saved sample metrics to {samples_fname}")
    print(f"Saved summary metrics to {summary_fname}")

    print(summary)

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
