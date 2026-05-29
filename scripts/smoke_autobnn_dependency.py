#!/usr/bin/env python3
"""Check that the installed AutoBNN fork exposes the API used here."""

import inspect

import autobnn
import jax
import jax.numpy as jnp
from autobnn import estimators, likelihoods, models, training_util


def main():
    estimator_sig = inspect.signature(estimators.AutoBnnMCMCEstimator.__init__)
    required_args = (
        "num_warmup",
        "distributed",
        "penalty_x_grid",
        "penalty_lambda",
        "penalty_use_sigma_space",
        "penalty_y_scale",
        "penalty_y_shift",
    )
    missing = [name for name in required_args if name not in estimator_sig.parameters]
    if missing:
        raise RuntimeError(
            "Installed AutoBNN is missing barrier fork args: " + ", ".join(missing)
        )

    if "normal_likelihood_known_noise" not in likelihoods.NAME_TO_LIKELIHOOD_MODEL:
        raise RuntimeError("Installed AutoBNN is missing normal_likelihood_known_noise")

    if not hasattr(training_util, "third_derivative_penalty_on_grid"):
        raise RuntimeError(
            "Installed AutoBNN is missing third_derivative_penalty_on_grid"
        )

    # This mirrors the Bayeux/MCMC model-construction path. It catches
    # incompatible PyPI oryx wheels that pass simple import checks but fail
    # when `sum_of_changepoints` is trained.
    x_train = jnp.linspace(-1.0, 1.0, 8)[..., None]
    y_train = jnp.sin(x_train)
    likelihood_model = likelihoods.NAME_TO_LIKELIHOOD_MODEL[
        "normal_likelihood_lognormal_noise"
    ]()
    net = models.make_model(
        "sum_of_changepoints",
        likelihood_model=likelihood_model,
        time_series_xs=x_train,
    )
    training_util._make_bayeux_model(
        net, jax.random.PRNGKey(0), x_train, y_train, num_particles=2
    )

    version = getattr(autobnn, "__version__", "unknown")
    print(f"AutoBNN dependency check passed: autobnn {version}")


if __name__ == "__main__":
    main()
