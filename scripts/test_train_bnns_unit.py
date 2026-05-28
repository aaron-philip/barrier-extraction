#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


SCRIPT_PATH = pathlib.Path(__file__).with_name("train_bnns.py")


def _load_train_bnns_module():
    utils_stub = types.ModuleType("utils")
    utils_stub.setup_logger = lambda *args, **kwargs: None
    utils_stub.load_packed_hdf5 = lambda *args, **kwargs: None
    utils_stub.compute_metrics_for_fit = lambda *args, **kwargs: None

    train_stub = types.ModuleType("train")
    train_stub.make_x_pred = lambda x: x
    train_stub.barrier_from_sigma_samples = lambda *args, **kwargs: None

    wong_stub = types.ModuleType("wong")
    wong_stub.sigma = lambda *args, **kwargs: None

    bnn_utils_stub = types.ModuleType("bnn_utils")
    bnn_utils_stub.compute_Df_log = lambda *args, **kwargs: None

    autobnn_stub = types.ModuleType("autobnn")
    autobnn_stub.estimators = types.SimpleNamespace()
    autobnn_training_util_stub = types.ModuleType("training_util")
    autobnn_training_util_stub.third_derivative_penalty_on_grid = (
        lambda *args, **kwargs: 0.0
    )
    autobnn_stub.training_util = autobnn_training_util_stub

    original_modules = {}
    stubbed_modules = {
        "utils": utils_stub,
        "train": train_stub,
        "wong": wong_stub,
        "bnn_utils": bnn_utils_stub,
        "autobnn": autobnn_stub,
        "autobnn.training_util": autobnn_training_util_stub,
    }

    try:
        for name, module in stubbed_modules.items():
            original_modules[name] = sys.modules.get(name)
            sys.modules[name] = module

        spec = importlib.util.spec_from_file_location("train_bnns_under_test", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class BuildD3PenaltyKwargsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_bnns = _load_train_bnns_module()

    def test_returns_empty_when_penalty_disabled(self):
        args = types.SimpleNamespace(
            d3_lambda=0.0,
            d3_lambda_space="physical",
            d3_on_sigma=False,
        )

        out = self.train_bnns._build_d3_penalty_kwargs(
            args=args,
            e_grid=np.linspace(1.0, 2.0, 6),
            x_scale=2.0,
            x_shift=0.5,
            y_scale=3.0,
            y_shift=-1.0,
        )

        self.assertEqual(out, {})

    def test_physical_lambda_is_rescaled_from_original_energy_units(self):
        args = types.SimpleNamespace(
            d3_lambda=64.0,
            d3_lambda_space="physical",
            d3_on_sigma=True,
        )

        out = self.train_bnns._build_d3_penalty_kwargs(
            args=args,
            e_grid=np.linspace(10.0, 20.0, 6),
            x_scale=2.0,
            x_shift=12.0,
            y_scale=5.0,
            y_shift=7.0,
        )

        np.testing.assert_allclose(
            out["penalty_x_grid"].squeeze(),
            np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
        )
        self.assertEqual(out["penalty_lambda"], 1.0)
        self.assertTrue(out["penalty_use_sigma_space"])
        self.assertEqual(out["penalty_y_scale"], 5.0)
        self.assertEqual(out["penalty_y_shift"], 7.0)

    def test_normalized_lambda_is_passed_through_unchanged(self):
        args = types.SimpleNamespace(
            d3_lambda=3.5,
            d3_lambda_space="normalized",
            d3_on_sigma=False,
        )

        out = self.train_bnns._build_d3_penalty_kwargs(
            args=args,
            e_grid=np.linspace(-2.0, 2.0, 5),
            x_scale=4.0,
            x_shift=0.0,
            y_scale=1.5,
            y_shift=-0.25,
        )

        self.assertEqual(out["penalty_lambda"], 3.5)
        self.assertFalse(out["penalty_use_sigma_space"])

    def test_requires_five_grid_points_for_third_derivative_stencil(self):
        args = types.SimpleNamespace(
            d3_lambda=1.0,
            d3_lambda_space="physical",
            d3_on_sigma=False,
        )

        with self.assertRaisesRegex(ValueError, "at least 5 grid points"):
            self.train_bnns._build_d3_penalty_kwargs(
                args=args,
                e_grid=np.linspace(0.0, 1.0, 4),
                x_scale=1.0,
                x_shift=0.0,
                y_scale=1.0,
                y_shift=0.0,
            )


if __name__ == "__main__":
    unittest.main()
