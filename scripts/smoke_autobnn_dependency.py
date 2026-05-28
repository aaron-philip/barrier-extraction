#!/usr/bin/env python3
"""Check that the installed AutoBNN fork exposes the API used here."""

import inspect

import autobnn
from autobnn import estimators, likelihoods, training_util


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

    version = getattr(autobnn, "__version__", "unknown")
    print(f"AutoBNN dependency check passed: autobnn {version}")


if __name__ == "__main__":
    main()
