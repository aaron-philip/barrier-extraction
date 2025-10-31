import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import curve_fit
import pandas as pd
import sys
import scipy
if "../src" not in sys.path:
    sys.path.append("../src")  

from bnn_utils import compute_Df
from utils import compute_all_metrics, barrier_dist
from wong import sigma

def fit_wong(
    E, Sigma, Sigma_err, true_params,
    levels=(0.5, 0.68, 0.9, 0.95),
    n_restarts=5, n_samples=200, verbose=False
):
    """
    Fit Wong parameters to noisy data with multiple restarts, then
    draw posterior samples for sigma(E), compute metrics using compute_all_metrics(),
    and precompute uncertainty bands at each level.
    """
    bounds = ([40.0, 6.0, 1.0], [100.0, 16.0, 10.0])
    best_popt, best_pcov, best_chi2 = None, None, np.inf

    # --- Try multiple restarts ---
    for trial in range(n_restarts):
        p0 = [np.random.uniform(50, 90),
              np.random.uniform(8, 14),
              np.random.uniform(2, 8)]
        try:
            popt, pcov = curve_fit(
                sigma, E, Sigma, p0=p0,
                sigma=Sigma_err, absolute_sigma=True,
                bounds=bounds, maxfev=10_000
            )
            resid = (Sigma - sigma(E, *popt)) / np.maximum(Sigma_err, 1e-12)
            chi2 = np.sum(resid**2)
            if chi2 < best_chi2:
                best_popt, best_pcov, best_chi2 = popt, pcov, chi2
        except Exception as e:
            if verbose:
                print(f"[Restart {trial}] failed: {e}")

    if best_popt is None:
        return {
            "E0_fit": np.nan, "R0_fit": np.nan, "hw_fit": np.nan,
            "chi2": np.nan, "n_restarts": n_restarts, "n_samples": n_samples
        }

    # --- Mean prediction ---
    x_pred = np.arange(max(0, E.min() - 5), E.max() + 5, 0.2)
    y_mean = sigma(x_pred, *best_popt)

    # --- Draw posterior parameter samples ---
    samples = np.random.multivariate_normal(best_popt, best_pcov, n_samples)
    y_samples = np.array([sigma(x_pred, *p) for p in samples])

    # --- True curve ---
    y_true = sigma(x_pred, *true_params) if true_params is not None else np.nan * y_mean

    # --- Compute metrics using existing function (unchanged) ---
    metrics = compute_all_metrics(x_pred, y_true, y_mean, y_samples, levels=levels)

    # --- Compute uncertainty bands (percentiles) for σ and D ---
    def percentile_bands(samples, levels):
        bands = {}
        for alpha in levels:
            lower = np.percentile(samples, (1 - alpha) / 2 * 100, axis=0)
            upper = np.percentile(samples, (1 + alpha) / 2 * 100, axis=0)
            bands[f"lower_{int(alpha*100)}"] = lower
            bands[f"upper_{int(alpha*100)}"] = upper
        return bands

    sigma_bands = percentile_bands(y_samples, levels)

    # Barrier distributions from samples
    D_samples = np.array([barrier_dist(x_pred, y) for y in y_samples])
    D_mean = barrier_dist(x_pred, y_mean)
    D_true = barrier_dist(x_pred, y_true)
    barrier_bands = percentile_bands(D_samples, levels)

    # --- Attach metadata and bands ---
    metrics.update({
        "E0_fit": best_popt[0],
        "R0_fit": best_popt[1],
        "hw_fit": best_popt[2],
        "chi2": best_chi2,
        "n_restarts": n_restarts,
        "n_samples": n_samples,
        "x_pred": x_pred,
        "y_mean": y_mean,
        "y_true": y_true,
        "D_mean": D_mean,
        "D_true": D_true,
        "sigma_bands": sigma_bands,
        "barrier_bands": barrier_bands
    })

    return metrics


def fit_gp(E, Sigma, Sigma_err, true_params, levels=(0.5, 0.68, 0.9, 0.95)):
    """Fit GP and compute full metrics + empirical coverage bands."""
    range_x = E.max() - E.min()
    y_var = np.var(Sigma)
    kernel = ConstantKernel(y_var, (1e-6, 1e6)) * RBF(0.5 * range_x, (1e-6, 1e6)) \
             + WhiteKernel(noise_level=np.mean(Sigma_err**2), noise_level_bounds=(1e-10, 1e6))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=2)
    gp.fit(E.reshape(-1,1), Sigma)

    x_pred = np.arange(max(0, E.min()-5), E.max()+5, 0.2).reshape(-1,1)
    mean, std = gp.predict(x_pred, return_std=True)
    mean, std = mean.ravel(), std.ravel()

    # === True function ===
    if true_params is not None and sigma is not None:
        y_true = sigma(x_pred.ravel(), *true_params)
    else:
        y_true = np.full_like(mean, np.nan)

    # === Metrics ===
    metrics = compute_all_metrics(x_pred.ravel(), mean, y_true)

    # === Empirical coverage ===
    coverages = {}
    for alpha in levels:
        z = scipy.stats.norm.ppf(0.5 + alpha/2.0)
        lower, upper = mean - z*std, mean + z*std
        inside = (y_true >= lower) & (y_true <= upper)
        coverages[f"emp_cov_{int(alpha*100)}"] = np.mean(inside)
    metrics.update(coverages)

    # === Hyperparameters ===
    learned = gp.kernel_
    metrics.update({
        "gp_signal_variance": float(learned.k1.k1.constant_value),
        "gp_length_scale": float(learned.k1.k2.length_scale),
        "gp_noise_level": float(learned.k2.noise_level)
    })
    return metrics

def fit_bnn(E, Sigma, true_params, est, levels=(0.5, 0.68, 0.9, 0.95)):
    """
    Given a trained AutoBNN estimator, compute mean, std, and empirical coverage.
    """
    x_pred = np.arange(max(0,E.min()-5), E.max()+5, 0.2)
    all_preds = compute_Df(est, x_pred)

    mean = np.mean(all_preds, axis=0)
    std = np.std(all_preds, axis=0)

    # === True curve ===
    if true_params is not None and sigma is not None:
        y_true = sigma(x_pred, *true_params)
    else:
        y_true = np.full_like(mean, np.nan)

    # === Standard metrics ===
    metrics = compute_all_metrics(x_pred, mean, y_true)

    # === Empirical coverage ===
    coverages = {}
    for alpha in levels:
        z = scipy.stats.norm.ppf(0.5 + alpha/2.0)
        lower, upper = mean - z*std, mean + z*std
        inside = (y_true >= lower) & (y_true <= upper)
        coverages[f"emp_cov_{int(alpha*100)}"] = np.mean(inside)
    metrics.update(coverages)

    return metrics



def train_gp(E_train, sigma_mu_train, delta_sigma_train, n_restarts=10, kernel=None):
    """
    Train a Gaussian Process on 1D data with heteroscedastic noise.

    Parameters
    ----------
    E_train : array-like, shape (N,)
        Training energies.
    sigma_mu_train : array-like, shape (N,)
        Mean cross-section values at E_train.
    delta_sigma_train : array-like, shape (N,)
        Variance of cross-section at E_train (i.e., noise variances).
    n_restarts : int, optional (default=10)
        Number of restarts for the optimizer.

    Returns
    -------
    gp : GaussianProcessRegressor
        The fitted GP model.
    """
    # 1) Sort E_train for length-scale estimation
    E_sorted = np.sort(np.asarray(E_train))
    dE = np.diff(E_sorted)
    ell0 = max(np.median(dE), 1e-6)
    lbound, ubound = ell0 / 10.0, ell0 * 10.0

    # 2) Compute signal variance from the training mean
    var_y0 = np.var(sigma_mu_train)

    if kernel is None:
        # 3) Build kernel: Constant * RBF  + WhiteKernel
        k_rbf   = 1 * RBF(length_scale=1.0, )#length_scale_bounds=(lbound, ubound)) # *ConstantKernel(constant_value=var_y0, constant_value_bounds=(var_y0/100, var_y0*100)) 
        k_noise = WhiteKernel(noise_level=1e-2 * var_y0, noise_level_bounds=(1e-8, var_y0))
        kernel  = k_rbf + k_noise

    # 4) Instantiate and fit GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=np.asarray(delta_sigma_train)**2,
        normalize_y=False,
        n_restarts_optimizer=n_restarts
    )
    gp.fit(np.asarray(E_train).reshape(-1, 1), np.asarray(sigma_mu_train))
    return gp


import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

def gp_mean_and_derivatives(gp, X_eval, return_std=False):
    """
    Given a trained sklearn GaussianProcessRegressor `gp` whose `kernel_` is of the form
        (ConstantKernel * RBF) + WhiteKernel,
    compute at each point x in X_eval:
      • the predictive mean μ(x),
      • the first derivative μ′(x) = dμ/dx,
      • the second derivative μ″(x) = d²μ/dx²,
      • (optionally) the predictive standard deviation δ(x).

    Assumptions:
      – The fitted kernel must be exactly (ConstantKernel * RBF) + WhiteKernel.
      – Input `X_eval` is one‐dimensional (shape=(n_eval,) or (n_eval, 1)).
      – The GP was fit on one‐dimensional inputs (shape=(n_samples, 1)).

    Parameters
    ----------
    gp : sklearn.gaussian_process.GaussianProcessRegressor
        A trained GP whose fitted `kernel_` is (ConstantKernel * RBF) + WhiteKernel.
    X_eval : array‐like, shape (n_eval,) or (n_eval, 1)
        Points at which to evaluate μ(x), μ′(x), μ″(x), (and δ(x) if requested).
    return_std : bool (default=False)
        If True, also return the predictive standard deviation δ(x) from gp.predict(..., return_std=True).

    Returns
    -------
    If return_std=False:
        mu   : ndarray, shape (n_eval,)
        dmu  : ndarray, shape (n_eval,)
        d2mu : ndarray, shape (n_eval,)
    If return_std=True:
        mu   : ndarray, shape (n_eval,)
        dmu  : ndarray, shape (n_eval,)
        d2mu : ndarray, shape (n_eval,)
        std  : ndarray, shape (n_eval,)
    """

    # 1) Ensure X_eval is a 2D column vector of shape (n_eval, 1)
    X = np.atleast_2d(X_eval).reshape(-1, 1)   # shape = (n_eval, 1)

    # 2) Retrieve training inputs and alpha weights
    X_train = gp.X_train_      # shape = (n_train, 1)
    alpha = gp.alpha_.ravel()  # shape = (n_train,)

    # 3) Extract the RBF‐Constant subkernel from kernel_.
    #    We expect: kernel_ = (ConstantKernel * RBF) + WhiteKernel.
    k_total = gp.kernel_

    # In scikit‐learn, a sum‐kernel has attributes `k1` and `k2`.
    # One of them should be the product (Constant * RBF), the other the WhiteKernel.
    if not hasattr(k_total, "k1") or not hasattr(k_total, "k2"):
        raise ValueError(
            "Expected kernel_ to be (ConstantKernel * RBF) + WhiteKernel"
        )

    # Identify which subkernel is the product and which is white‐noise
    if isinstance(k_total.k1, WhiteKernel) and hasattr(k_total.k2, "k1") and hasattr(k_total.k2, "k2"):
        # then k2 should be Constant * RBF
        white_kern = k_total.k1
        prod_kern  = k_total.k2
    elif isinstance(k_total.k2, WhiteKernel) and hasattr(k_total.k1, "k1") and hasattr(k_total.k1, "k2"):
        # then k1 should be Constant * RBF
        white_kern = k_total.k2
        prod_kern  = k_total.k1
    else:
        raise ValueError(
            "Expected one part of the sum to be WhiteKernel, "
            "the other to be ConstantKernel * RBF."
        )

    # Now extract C and ℓ from prod_kern = ConstantKernel * RBF
    if isinstance(prod_kern.k1, ConstantKernel) and isinstance(prod_kern.k2, RBF):
        C = float(prod_kern.k1.constant_value)
        length_scale = float(prod_kern.k2.length_scale)
    elif isinstance(prod_kern.k2, ConstantKernel) and isinstance(prod_kern.k1, RBF):
        C = float(prod_kern.k2.constant_value)
        length_scale = float(prod_kern.k1.length_scale)
    else:
        raise ValueError(
            "The product subkernel must be ConstantKernel * RBF."
        )

    # 4) Build pairwise differences Δ = x_eval - x_train
    diff = (X[:, 0][:, None] - X_train[:, 0][None, :])  # shape (n_eval, n_train)

    # 5) Compute the cross‐covariance K_rbf(X_eval, X_train)
    #    k(x, x_i) = C * exp( -½((x - x_i)^2 / ℓ^2) )
    K_x = C * np.exp(-0.5 * (diff ** 2) / (length_scale ** 2))  # shape (n_eval, n_train)

    # 6) Predictive mean: μ(x) = K_x · α
    mu = K_x.dot(alpha)  # shape (n_eval,)

    # 7) First derivative of k_rbf w.r.t. x:
    #    ∂/∂x k(x, x_i) = k(x, x_i) * ( −(x − x_i) / ℓ² )
    dK_dx = K_x * (-(diff) / (length_scale ** 2))  # shape (n_eval, n_train)
    dmu = dK_dx.dot(alpha)  # shape (n_eval,)

    # 8) Second derivative of k_rbf w.r.t. x:
    #    ∂²/∂x² k(x, x_i) = k(x, x_i) * [ ((x − x_i)² / ℓ⁴) − (1 / ℓ² ) ]
    d2K_dx2 = K_x * ((diff ** 2) / (length_scale ** 4) - (1.0 / (length_scale ** 2)))
    d2mu = d2K_dx2.dot(alpha)  # shape (n_eval,)

    if return_std:
        # 9) Also return the predictive standard deviation δ(x) from the full GP model
        mu_only, std = gp.predict(X, return_std=True)  # std has shape (n_eval,)
        return mu, dmu, d2mu, std
    else:
        return mu, dmu, d2mu
    


def gp_second_derivative_of_x_sigma(gp: GaussianProcessRegressor, X_train, y_train, X_test):
    """
    For trained GP modeling sigma(x), compute the GP of d²/dx² [ x * sigma(x) ].
    
    Returns:
        d2_mean: mean of second derivative at X_test
        d2_var: variance of second derivative at X_test
    """
    kernel = gp.kernel_
    K_xx = kernel(X_train) + gp.alpha * np.eye(len(X_train))
    K_xx_inv = np.linalg.inv(K_xx)
    
    N = X_train.shape[0]
    M = X_test.shape[0]
    
    # Only 1D case — since you are doing d²/dx²
    assert X_train.shape[1] == 1
    D = 1
    
    # Prepare outputs
    mu = np.zeros(M)
    mu_prime = np.zeros(M)
    mu_double_prime = np.zeros(M)
    
    var_prime = np.zeros(M)
    var_double_prime = np.zeros(M)
    cov_prime_double_prime = np.zeros(M)
    
    # Precompute kernel derivatives
    length_scale = kernel.k1.k2.length_scale
    sigma_f2 = kernel.k1.k1.constant_value
    noise_var = kernel.k2.noise_level
    
    for m in range(M):
        x_s = X_test[m, :].reshape(1, -1)  # (1,1)
        diff = X_train - x_s  # (N,1)
        
        # Kernel value
        sq_dist = (diff / length_scale) ** 2
        K_xs = sigma_f2 * np.exp(-0.5 * sq_dist).flatten()  # (N,)
        
        # First derivative of kernel wrt x
        dK_dx = K_xs * (diff.flatten() / length_scale**2)  # (N,)
        
        # Second derivative of kernel wrt x
        d2K_dx2 = K_xs * (1 / length_scale**2 - (diff.flatten()**2) / length_scale**4)  # (N,)
        
        # Mean predictions
        alpha = K_xx_inv @ y_train
        mu[m] = K_xs @ alpha
        mu_prime[m] = dK_dx @ alpha
        mu_double_prime[m] = d2K_dx2 @ alpha
        
        # Variance terms
        v1 = K_xx_inv @ dK_dx
        v2 = K_xx_inv @ d2K_dx2
        
        k_ss_prime = 1 / length_scale**2 * sigma_f2  # ∂²k(x,x)/∂x² = sigma_f² / l²
        k_ss_double_prime = sigma_f2 * (1 / length_scale**2)  # same
        k_ss_prime_prime = sigma_f2 * (1 / length_scale**2)  # same
        
        # Var[∂f/∂x]
        var_prime[m] = k_ss_prime - dK_dx @ v1
        
        # Var[∂²f/∂x²]
        var_double_prime[m] = 3*k_ss_double_prime - d2K_dx2 @ v2
        
        # Cov[∂f/∂x, ∂²f/∂x²]
        cov_prime_double_prime[m] = -dK_dx @ v2  # cross term
    
    # Now compute desired function:
    # d²/dx² (x * sigma(x)) = 2 * mu' + x * mu''
    x_vals = X_test.flatten()
    d2_mean = 2 * mu_prime - x_vals * mu_double_prime
    
    d2_var = (
        4 * var_prime
        + x_vals**2 * var_double_prime
        + 4 * x_vals * cov_prime_double_prime
    )
    
    return d2_mean, d2_var
