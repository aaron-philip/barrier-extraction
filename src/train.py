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

# --- Constants shared across helpers ---
WONG_BOUNDS = ([40.0, 6.0, 1.0], [100.0, 16.0, 10.0])  # (E0, R0, hw)
R0_FIXED = 11.0
HW_FIXED = 6.0

def estimate_E0_from_data(E, Sigma, eps=1e-12, smooth_win=5):
    """
    Estimate barrier energy E0 as the energy where log10(Sigma) rises most steeply.
    Steps:
      - sort by E
      - clip Sigma to avoid log issues
      - moving-average smooth in log10 space
      - pick E at max positive gradient
    """
    E = np.asarray(E, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    order = np.argsort(E)
    E_sorted = E[order]
    S_sorted = Sigma[order]

    S_clipped = np.clip(S_sorted, eps, None)
    y = S_clipped

    w = max(3, int(smooth_win) | 1)  # odd window
    if 1 < w < y.size:
        kernel = np.ones(w, dtype=float) / w
        y_smooth = np.convolve(y, kernel, mode="same")
    else:
        y_smooth = y

    dy_dE = np.gradient(y_smooth, E_sorted)
    idx = int(np.argmax(dy_dE))
    return float(E_sorted[idx])

def estimate_E0_from_data(E, Sigma, eps=1e-12, smooth_win=5):
    """
    Estimate barrier energy E0 as the energy where log10(Sigma) rises most steeply.
    Steps:
      - sort by E
      - clip Sigma to avoid log issues
      - moving-average smooth in log10 space
      - pick E at max positive gradient
    """
    E = np.asarray(E, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    order = np.argsort(E)
    E_sorted = E[order]
    S_sorted = Sigma[order]

    S_clipped = np.clip(S_sorted, eps, None)
    y = np.log10(S_clipped)

    w = max(3, int(smooth_win) | 1)  # odd window
    if 1 < w < y.size:
        kernel = np.ones(w, dtype=float) / w
        y_smooth = np.convolve(y, kernel, mode="same")
    else:
        y_smooth = y

    dy_dE = np.gradient(y_smooth, E_sorted)
    idx = int(np.argmax(dy_dE))
    return float(E_sorted[idx])

def init_E0(E, Sigma, bounds=WONG_BOUNDS, verbose=False):
    """Return a data-driven E0, clamped to bounds; fallback to median(E) if needed."""
    try:
        e0 = estimate_E0_from_data(E, Sigma)
    except Exception as e:
        if verbose:
            print(f"[E0 init] fallback due to error: {e}")
        e0 = float(np.median(np.asarray(E, dtype=float)))
    lo, hi = bounds[0][0], bounds[1][0]
    return float(np.clip(e0, lo, hi))


def run_wong_trial(E, Sigma, Sigma_err, p0, bounds=WONG_BOUNDS, maxfev=10_000):
    """
    Try a single non-linear least squares fit. Returns (popt, pcov, chi2).
    Raises any curve_fit error to the caller.
    """
    popt, pcov = curve_fit(
        sigma, E, Sigma, p0=p0,
        sigma=Sigma_err, absolute_sigma=True,
        bounds=bounds, maxfev=maxfev
    )
    resid = (Sigma - sigma(E, *popt)) / np.maximum(Sigma_err, 1e-12)
    chi2 = float(np.sum(resid**2))
    return popt, pcov, chi2

def fit_wong_with_restarts(E, Sigma, Sigma_err, E0_center, n_restarts=5, jitter=2.0,
                           bounds=WONG_BOUNDS, verbose=False):
    """
    Multiple restarts around a data-driven E0. R0 and hw fixed to midpoints.
    Returns (best_popt, best_pcov, best_chi2) or (None, None, inf) if all fail.
    """
    best_popt, best_pcov, best_chi2 = None, None, np.inf
    lo, hi = bounds[0][0], bounds[1][0]

    for trial in range(int(n_restarts)):
        E0_init = float(np.clip(E0_center + np.random.uniform(-jitter, jitter), lo, hi))
        p0 = [E0_init, R0_FIXED, HW_FIXED]

        try:
            popt, pcov, chi2 = run_wong_trial(E, Sigma, Sigma_err, p0, bounds=bounds)
            if chi2 < best_chi2:
                best_popt, best_pcov, best_chi2 = popt, pcov, chi2
        except Exception as e:
            if verbose:
                print(f"[Restart {trial}] failed: {e}")

    return best_popt, best_pcov, best_chi2

def make_x_pred(E, pad_low=5.0, pad_high=5.0, step=0.2):
    """Dense energy grid for plotting/metrics."""
    Emin = float(np.min(E))
    Emax = float(np.max(E))
    start = max(0.0, Emin - pad_low)
    stop = Emax + pad_high
    return np.arange(start, stop, step)

def draw_sigma_samples(x_pred, mean_params, cov_params, n_samples):
    """
    Sample parameters ~ N(mean_params, cov_params) and evaluate sigma on x_pred.
    Returns (y_mean, samples_matrix) where samples_matrix has shape [n_samples, len(x_pred)].
    """
    # y_mean from MAP/ML fit
    y_mean = sigma(x_pred, *mean_params)

    # Robust sampling even if covariance is near-singular
    try:
        samples = np.random.multivariate_normal(mean_params, cov_params, int(n_samples))
    except np.linalg.LinAlgError:
        # Fallback: diagonalize with non-negative variances
        diag = np.clip(np.diag(cov_params), 1e-12, None)
        cov_fallback = np.diag(diag)
        samples = np.random.multivariate_normal(mean_params, cov_fallback, int(n_samples))

    y_samples = np.array([sigma(x_pred, *p) for p in samples])
    return y_mean, y_samples

def percentile_bands(samples, levels):
    """
    samples: [n_samples, n_points]
    levels: iterable of alphas (e.g., 0.5, 0.68, 0.9, 0.95)
    Returns dict: lower_XX, upper_XX -> arrays of length n_points
    """
    bands = {}
    for alpha in levels:
        lo = np.percentile(samples, (1 - alpha) / 2 * 100, axis=0)
        hi = np.percentile(samples, (1 + alpha) / 2 * 100, axis=0)
        bands[f"lower_{int(alpha * 100)}"] = lo
        bands[f"upper_{int(alpha * 100)}"] = hi
    return bands

def barrier_from_sigma_samples(x_pred, y_mean, y_samples, y_true=None):
    """
    Compute D(E) from sigma(E) for mean, samples, and (optionally) true.
    Returns (D_mean, D_samples, D_true or nan-array).
    """
    D_mean = barrier_dist(x_pred, y_mean)
    D_samples = np.array([barrier_dist(x_pred, ys) for ys in y_samples])

    if y_true is None or not np.all(np.isfinite(y_true)):
        D_true = np.full_like(D_mean, np.nan)
    else:
        D_true = barrier_dist(x_pred, y_true)

    return D_mean, D_samples, D_true

def fit_wong(
    E, Sigma, Sigma_err, true_params,
    levels=(0.5, 0.68, 0.9, 0.95),
    n_restarts=5, n_samples=200, verbose=False
):
    """
    Fit Wong parameters to noisy data with multiple restarts (data-driven E0),
    then draw posterior parameter samples, compute metrics & uncertainty bands.
    """
    # 1) Data-driven E0
    E0_data = init_E0(E, Sigma, bounds=WONG_BOUNDS, verbose=verbose)

    # 2) Multi-restart fit (R0, hw fixed to midpoints; jitter E0)
    best_popt, best_pcov, best_chi2 = fit_wong_with_restarts(
        E=np.asarray(E, dtype=float),
        Sigma=np.asarray(Sigma, dtype=float),
        Sigma_err=np.asarray(Sigma_err, dtype=float),
        E0_center=E0_data,
        n_restarts=n_restarts,
        jitter=2.0,
        bounds=WONG_BOUNDS,
        verbose=verbose,
    )

    if best_popt is None:
        return {
            "E0_fit": np.nan, "R0_fit": np.nan, "hw_fit": np.nan,
            "chi2": np.nan, "n_restarts": n_restarts, "n_samples": n_samples
        }

    # 3) Prediction grid
    x_pred = make_x_pred(E)

    # 4) Mean σ and σ-samples from parameter posterior
    y_mean, y_samples = draw_sigma_samples(x_pred, best_popt, best_pcov, n_samples)

    # 5) True σ if available
    if true_params is not None:
        try:
            y_true = sigma(x_pred, *true_params)
        except Exception:
            y_true = np.full_like(y_mean, np.nan)
    else:
        y_true = np.full_like(y_mean, np.nan)

    # 6) Metrics (existing function)
    metrics = compute_all_metrics(x_pred, y_true, y_mean, y_samples, levels=levels)

    # 7) Bands in σ-space
    sigma_bands = percentile_bands(y_samples, levels)

    # 8) Barrier distributions + bands
    D_mean, D_samples, D_true = barrier_from_sigma_samples(x_pred, y_mean, y_samples, y_true)
    barrier_bands = percentile_bands(D_samples, levels)

    # 9) Package
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




def fit_gp(
    E, Sigma, Sigma_err, true_params,
    levels=(0.5, 0.68, 0.9, 0.95),
    n_restarts=2, n_samples=200, verbose=False
):
    """
    Gaussian Process regression on (E, Sigma) using the provided kernel recipe.
    Draw posterior samples for sigma(E), compute metrics using compute_all_metrics(),
    and precompute uncertainty bands at each level—analogous to fit_wong.
    Assumes sklearn GP + kernels are already imported.
    """
    # --- Defensive casts & shapes ---
    E = np.asarray(E, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float).reshape(-1)
    Sigma_err = np.asarray(Sigma_err, dtype=float).reshape(-1)

    # --- Basic scales for kernel construction ---
    # y_var for ConstantKernel scale; range_x for RBF length-scale prior
    dE = np.diff(E)
    ell0 = min(np.abs(dE))
    lbound, ubound = ell0 , 10.0

    # --- Kernel & GP (your exact recipe) ---
    kernel = (
        RBF(length_scale=1.0, length_scale_bounds=(lbound, ubound))
       # + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10, 1e6))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=Sigma_err**2,                   # all noise is modeled via WhiteKernel
        n_restarts_optimizer=int(n_restarts),
        normalize_y=False,           # keep physical scale (change if you prefer)
        random_state=None
    )

    # --- Fit ---
    gp.fit(E.reshape(-1, 1), Sigma)
    # You can access gp.kernel_ (optimized) and gp.log_marginal_likelihood_value_

    # --- Prediction grid (same style as fit_wong) ---
    x_pred = np.arange(max(0.0, E.min() - 5.0), E.max() + 5.0, 0.2)
    Xp = x_pred.reshape(-1, 1)

    # --- Posterior mean and draws ---
    # (We prefer sampling to match the "y_samples" workflow in fit_wong.)
    y_mean, y_std = gp.predict(Xp, return_std=True)
    # Posterior function draws (shape: [n_points, n_samples]) -> transpose to [n_samples, n_points]
    y_samples = gp.sample_y(Xp, n_samples=n_samples, random_state=None).T

    # --- True curve (keep consistency with fit_wong interface) ---
    if true_params is not None:
        try:
            y_true = sigma(x_pred, *true_params)
        except Exception:
            # if sigma(true_params) is not available, fall back to NaNs
            y_true = np.full_like(y_mean, np.nan)
    else:
        y_true = np.full_like(y_mean, np.nan)

    # --- Metrics in σ-space (uses your existing helper) ---
    metrics = compute_all_metrics(x_pred, y_true, y_mean, y_samples, levels=levels)

    # --- Percentile bands helper (same as fit_wong) ---
    def percentile_bands(samples, levels):
        bands = {}
        for alpha in levels:
            lo = np.percentile(samples, (1 - alpha) / 2 * 100, axis=0)
            hi = np.percentile(samples, (1 + alpha) / 2 * 100, axis=0)
            bands[f"lower_{int(alpha * 100)}"] = lo
            bands[f"upper_{int(alpha * 100)}"] = hi
        return bands

    sigma_bands = percentile_bands(y_samples, levels)

    # --- Barrier distributions ---
    D_mean  = barrier_dist(x_pred, y_mean)
    D_true  = barrier_dist(x_pred, y_true) if np.all(np.isfinite(y_true)) else np.full_like(D_mean, np.nan)
    D_samples = np.array([barrier_dist(x_pred, ys) for ys in y_samples])
    barrier_bands = percentile_bands(D_samples, levels)

    # --- Attach GP metadata & bands (mirroring fit_wong returns) ---
    metrics.update({
        # No param triplet here; include GP details instead
        "gp_kernel_opt": str(gp.kernel_),
        "gp_log_marginal_likelihood": float(getattr(gp, "log_marginal_likelihood_value_", np.nan)),
        "n_restarts": int(n_restarts),
        "n_samples": int(n_samples),

        # Curves for downstream plotting/inspection
        "x_pred": x_pred,
        "y_mean": y_mean,
        "y_true": y_true,
        "D_mean": D_mean,
        "D_true": D_true,

        # Uncertainty bands (σ-space and barrier-space)
        "sigma_bands": sigma_bands,
        "barrier_bands": barrier_bands,
    })

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
