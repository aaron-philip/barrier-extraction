import jax
import jax.numpy as jnp
import numpy as np
import pickle
import sys
if "/mnt/research/ascsn/aaron-philip/barrier-extraction/probability/spinoffs/autobnn" not in sys.path:
    sys.path.insert(0, "/mnt/research/ascsn/aaron-philip/barrier-extraction/probability/spinoffs/autobnn")

import autobnn
from autobnn import estimators, training_util
from autobnn import kernels, likelihoods


def compute_Df(
    bnn_est,
    x_pred: np.ndarray,
    scales: tuple,
    num_prune: int = 100
) -> np.ndarray:
    """
    Compute second derivatives of x*f(x) via JAX for each MCMC sample.
    Returns array of shape (num_prune, len(x_pred)).
    """
    x_scale, x_shift, y_scale, y_shift = scales
    X = x_pred
    # define g and its second derivative
    def f_sample(params, x):
        x_n = (x - x_shift)/x_scale
        x_b = jnp.atleast_2d([[x_n]])
        pred = bnn_est.net_.apply(params, x_b)
        return pred.squeeze()*y_scale + y_shift
    def g(params, x):
        return x * f_sample(params, x)
    grad2 = jax.grad(jax.grad(g, argnums=1), argnums=1)
    # vectorize
    all_d2 = jax.vmap(lambda p: jax.vmap(lambda x: grad2(p, x))(X))(bnn_est.params_)
    all_d2 = np.array(all_d2)
    # random prune
    key = jax.random.PRNGKey(0)
    idx = jax.random.choice(key, all_d2.shape[0], (num_prune,), replace=False)
    return all_d2[idx]

def compute_Df_log( bnn_est, x_pred: np.ndarray, scales: tuple, num_prune: int = 100 ) -> np.ndarray: 
    """ 
    Compute second derivatives of x*f(x) for model trained to log(f(x)) via JAX for each MCMC sample.
    Returns array of shape (num_prune, len(x_pred)). 
    """ 
    x_scale, x_shift, y_scale, y_shift = scales
    X = x_pred 
    # define g and its second derivative 
    def f_sample(params, x): 
        x_n = (x - x_shift)/x_scale 
        x_b = jnp.atleast_2d([[x_n]]) 
        pred = bnn_est.net_.apply(params, x_b) 
        return jnp.exp(pred.squeeze()*y_scale + y_shift) 
    def g(params, x): 
        return x * f_sample(params, x) 
    grad2 = jax.grad(jax.grad(g, argnums=1), argnums=1) 
    # vectorize 
    all_d2 = jax.vmap(lambda p: jax.vmap(lambda x: grad2(p, x))(X))(bnn_est.params_) 
    all_d2 = np.array(all_d2) 
    # random prune 
    key = jax.random.PRNGKey(0) 
    idx = jax.random.choice(key, all_d2.shape[0], (num_prune,), replace=False) 
    return all_d2[idx]


def save_bnn_params(params: dict, path: str) -> None:
    """
    Save BNN parameters PyTree to disk.
    """
    params = jax.device_get(params)
    with open(path, 'wb') as f:
        pickle.dump(params, f)


def load_bnn_params(path: str, bnn_est) -> estimators.AutoBnnMCMCEstimator:
    """
    Load params into an existing estimator.
    """
    with open(path, 'rb') as f:
        params = pickle.load(f)
    bnn_est.params_ = jax.tree_map(lambda x: jnp.array(x), params)
    return bnn_est

"""To load back in
    bnn_est = estimators.AutoBnnMCMCEstimator(
        rbf,#'sum_of_products',              
        likelihood_model='normal_likelihood_logistic_noise',
        seed=est_seed,
        distributed=1,
        num_chains=4, # num_chains and num_draws must be same as original trained BNN
        num_draws=50,
        num_warmup=1)
    bnn_est.fit(x_norm, y_norm)
    bnn_est = load_bnn_params("bnn_params.pkl", bnn_est)
"""