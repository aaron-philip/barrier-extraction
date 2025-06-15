import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

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
