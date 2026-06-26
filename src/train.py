import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import curve_fit
import pandas as pd
import sys
if "../src" not in sys.path:
    sys.path.append("../src")  

from utils import compute_all_metrics, barrier_dist
from wong import sigma

# --- Constants shared across helpers ---
WONG_BOUNDS = ([40.0, 6.0, 1.0], [100.0, 16.0, 10.0])  # (E0, R0, hw)
R0_FIXED = 11.0
HW_FIXED = 6.0


def estimate_E0_from_data(E, Sigma, frac_threshold=0.2):
    """
    Crude estimate of the barrier energy E0:
    find the first energy where the curvature of σ(E)
    (second derivative of E*σ) exceeds a fraction of its maximum.

    Args:
      E : array of energies
      Sigma : array of cross sections
      frac_threshold : fraction of max curvature to count as 'significant'
    """
    E = np.asarray(E, float)
    S = np.asarray(Sigma, float)

    # Ensure monotonic energy grid
    order = np.argsort(E)
    E, S = E[order], S[order]

    y = E * S
    dydE = np.gradient(y, E)
    d2ydE2 = np.gradient(dydE, E)

    # Normalize curvature
    k = np.abs(d2ydE2)
    k /= (np.max(k) + 1e-12)

    # Find the first index where curvature exceeds threshold
    candidates = np.where(k > frac_threshold)[0]
    if len(candidates) == 0:
        # fallback to global maximum
        idx = np.nanargmax(k)
    else:
        idx = candidates[0]

    return float(E[idx])


def init_E0(E, Sigma):
    E0 = estimate_E0_from_data(E, Sigma)
    R0, hw = 11.0, 5.0
    return [E0, R0, hw]


def run_wong_trial(E, Sigma, Sigma_err, p0):
    try:
        popt, pcov = curve_fit(
            sigma,
            E,
            Sigma,
            p0=p0,
            sigma=Sigma_err,
            absolute_sigma=True,
            bounds=([40.0, 6.0, 1.0], [100.0, 16.0, 10.0]),
            maxfev=10000,
        )
        res = (Sigma - sigma(E, *popt)) / np.maximum(Sigma_err, 1e-8)
        chi2 = np.sum(res**2)
        return popt, pcov, chi2
    except Exception:
        return None


def fit_wong_with_restarts(E, Sigma, Sigma_err, n_restarts=5):
    best = None
    p0_init = init_E0(E, Sigma)
    result = run_wong_trial(E, Sigma, Sigma_err, p0_init)
    if result:
        best = result

    for _ in range(n_restarts - 1):
        p0 = [
            np.random.uniform(50, 90),
            np.random.uniform(8, 14),
            np.random.uniform(2, 8),
        ]
        result = run_wong_trial(E, Sigma, Sigma_err, p0)
        if result and (best is None or result[2] < best[2]):
            best = result
    return best


def make_x_pred(E):
    return np.arange(max(0, E.min() - 5), E.max() + 5, 0.2)


def draw_sigma_samples(x_pred, best_popt, best_pcov, n_samples=200):
    samples = np.random.multivariate_normal(best_popt, best_pcov, size=n_samples)
    return np.array([sigma(x_pred, *p) for p in samples])


def percentile_bands(samples, levels=(0.68, 0.95)):
    out = {}
    for lev in levels:
        lo = np.percentile(samples, (1 - lev) / 2 * 100, axis=0)
        hi = np.percentile(samples, (1 + lev) / 2 * 100, axis=0)
        out[f"lower_{int(lev*100)}"] = lo
        out[f"upper_{int(lev*100)}"] = hi
    return out

def barrier_from_sigma_samples(x_pred, y_samples):
    """Compute second derivative of E*sigma for each sample."""
    D = []
    for y in y_samples:
        D_i = barrier_dist(x_pred, y)
        D.append(D_i)
    return np.array(D)


def fit_wong(E, Sigma, Sigma_err, n_restarts=5, n_samples=200):
    best = fit_wong_with_restarts(E, Sigma, Sigma_err, n_restarts)
    if best is None:
        return None

    best_popt, best_pcov, chi2 = best
    x_pred = make_x_pred(E)
    y_mean = sigma(x_pred, *best_popt)
    y_samples = draw_sigma_samples(x_pred, best_popt, best_pcov, n_samples)

    return {
        "x_pred": x_pred,
        "y_mean": y_mean,
        "y_samples": y_samples,
        "best_popt": best_popt,
        "chi2": chi2,
    }





def fit_gp(
    E, Sigma, Sigma_err,
    n_restarts=10, n_samples=200, random_state=None, kernel=None,
    target_space="linear",
):
    """
    GP regression on (E, Sigma) with simple RBF kernel and heteroscedastic noise.

    target_space="linear" fits Sigma with alpha=Sigma_err^2.
    target_space="log" fits log(Sigma) with delta-method alpha=(Sigma_err/Sigma)^2,
    then transforms predictions back to Sigma space.

    Returns arrays for HDF5 writing: x_pred, y_mean, y_samples, and barrier-space summaries.
    """
    # --- defensives & shapes ---
    E = np.asarray(E, float).reshape(-1)
    Sigma = np.asarray(Sigma, float).reshape(-1)
    Sigma_err = np.asarray(Sigma_err, float).reshape(-1)

    # ensure sorted by energy (GP prefers monotonic X for some diagnostics)
    order = np.argsort(E)
    E, Sigma, Sigma_err = E[order], Sigma[order], Sigma_err[order]

    target_space = str(target_space).lower()
    if target_space not in {"linear", "log"}:
        raise ValueError("target_space must be 'linear' or 'log'")

    # --- kernel and GP ---
    # length scale bounds based on grid spacing; robust to irregular grids
    if len(E) > 1:
        dE = np.diff(E)
        ell_min = max(1e-3, np.min(np.abs(dE)))
        ell_max = max(5.0, 10.0 * np.median(np.abs(dE)))
    else:
        ell_min, ell_max = 1e-3, 10.0
    if kernel is None:
        kernel = 1.0*RBF(length_scale=1.0, length_scale_bounds=(ell_min, ell_max))
    

    if target_space == "log":
        if np.any(Sigma <= 0):
            raise ValueError("Log-space GP requires strictly positive Sigma values.")
        y_train = np.log(Sigma)
        alpha = np.maximum(Sigma_err / Sigma, 1e-12) ** 2
    else:
        y_train = Sigma
        alpha = np.maximum(Sigma_err, 1e-12) ** 2

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,   # heteroscedastic obs noise in the fitted target space
        n_restarts_optimizer=int(n_restarts),
        normalize_y=False,
        random_state=random_state,
    )

    # --- fit ---
    gp.fit(E.reshape(-1, 1), y_train)
    #kernel_opt = gp.kernel_.k2.length_scale
    #lml = float(getattr(gp, "log_marginal_likelihood_value_", np.nan))

    # --- prediction grid (match Wong plotting style) ---
    x_pred = np.arange(max(0.0, E.min() - 5.0), E.max() + 5.0, 0.2).astype(float)
    Xp = x_pred.reshape(-1, 1)

    # posterior mean and samples
    y_mean, y_std = gp.predict(Xp, return_std=True)
    y_samples = gp.sample_y(Xp, n_samples=n_samples, random_state=random_state).T  # (S, M)

    if target_space == "log":
        y_mean = np.exp(y_mean + 0.5 * y_std**2)
        y_samples = np.exp(y_samples)

    return {
        "x_pred": x_pred,
        "y_mean": y_mean,
        "y_samples": y_samples,      # (n_samples, len(x_pred))
        # metadata
        "gp_kernel_opt": np.nan, #kernel_opt,
        "gp_log_marginal_likelihood": np.nan,#lml,
        "n_restarts": int(n_restarts),
        "n_samples": int(n_samples),
        "target_space": target_space,
    }



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

from sklearn.gaussian_process.kernels import Kernel

class ChangePointKernel(Kernel):
    """
    Smooth changepoint kernel:
        K = S*K1 + T*K2
    where S_ij = s(x_i)s(x_j), T_ij = (1-s(x_i))(1-s(x_j)),
    s(x) = sigmoid((x - c)/w).

    This version supports sklearn's requirement that `kernel.theta` is SETTABLE.
    We store c,w as positive parameters represented in theta-space as log(c), log(w).
    """

    def __init__(self, k1, k2, location=50.0, width=5.0,
                 location_bounds=(1e-3, 1e3), width_bounds=(1e-3, 1e3)):
        self.k1 = k1
        self.k2 = k2
        self.location = float(location)
        self.width = float(width)
        self.location_bounds = location_bounds
        self.width_bounds = width_bounds

    # ---- required by sklearn optimizer / LML ----
    @property
    def theta(self):
        return np.concatenate([
            self.k1.theta,
            self.k2.theta,
            np.log([self.location, self.width]),
        ])

    @theta.setter
    def theta(self, theta):
        """Set hyperparameters from sklearn's theta vector (log-space)."""
        theta = np.asarray(theta, float)
        n1 = self.k1.theta.size
        n2 = self.k2.theta.size

        # update sub-kernels by delegating their theta setters
        self.k1.theta = theta[:n1]
        self.k2.theta = theta[n1:n1+n2]

        # last two are log(location), log(width)
        self.location = float(np.exp(theta[n1+n2]))
        self.width    = float(np.exp(theta[n1+n2+1]))

    @property
    def bounds(self):
        b1 = self.k1.bounds
        b2 = self.k2.bounds
        b_loc = np.log(np.asarray(self.location_bounds, float))[None, :]
        b_w   = np.log(np.asarray(self.width_bounds, float))[None, :]
        return np.vstack([b1, b2, b_loc, b_w])

    # ---- kernel guts ----
    def _sigmoid(self, X):
        z = (X - self.location) / np.maximum(self.width, 1e-12)
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if X.shape[1] != 1:
            raise ValueError("ChangePointKernel expects 1D inputs of shape (N,1).")

        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)

        sX = self._sigmoid(X)  # (N,1)
        sY = self._sigmoid(Y)  # (M,1)

        S = sX @ sY.T
        T = (1.0 - sX) @ (1.0 - sY).T

        if not eval_gradient:
            K1 = self.k1(X, Y, eval_gradient=False)
            K2 = self.k2(X, Y, eval_gradient=False)
            return S * K1 + T * K2

        if Y is not X:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        K1, G1 = self.k1(X, None, eval_gradient=True)
        K2, G2 = self.k2(X, None, eval_gradient=True)
        K = S * K1 + T * K2

        s = sX
        ds_dz = s * (1.0 - s)

        dloc = (-1.0 / np.maximum(self.width, 1e-12)) * ds_dz
        dwid = (-(X - self.location) / np.maximum(self.width, 1e-12)**2) * ds_dz

        dS_dloc = dloc @ s.T + s @ dloc.T
        dS_dwid = dwid @ s.T + s @ dwid.T

        u = 1.0 - s
        du_dloc = -dloc
        du_dwid = -dwid

        dT_dloc = du_dloc @ u.T + u @ du_dloc.T
        dT_dwid = du_dwid @ u.T + u @ du_dwid.T

        dK_dloc = dS_dloc * K1 + dT_dloc * K2
        dK_dwid = dS_dwid * K1 + dT_dwid * K2

        # theta uses log-params for positive vars: dK/dlog(p) = p * dK/dp
        dK_dlogloc = self.location * dK_dloc
        dK_dlogwid = self.width    * dK_dwid

        G = np.concatenate(
            [
                S[:, :, None] * G1,
                T[:, :, None] * G2,
                dK_dlogloc[:, :, None],
                dK_dlogwid[:, :, None],
            ],
            axis=2,
        )
        return K, G

    def diag(self, X):
        X = np.atleast_2d(X)
        s = self._sigmoid(X).reshape(-1)
        return (s**2) * self.k1.diag(X) + ((1.0 - s)**2) * self.k2.diag(X)

    def is_stationary(self):
        return False



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
