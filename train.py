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
        k_rbf   = ConstantKernel(constant_value=var_y0, constant_value_bounds=(var_y0/100, var_y0*100)) \
                * RBF(length_scale=ell0, length_scale_bounds=(lbound, ubound))
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