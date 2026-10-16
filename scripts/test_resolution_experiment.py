"""Scientific and pairing checks for the resolution experiment."""
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from resolution_experiment import (
    DEFAULT_DATA, ROOT, RESOLUTIONS, LEVELS, prepare, score, tag, wong_barrier,
)
from utils import paper_three_point_barrier
from wong import sigma


class ResolutionExperimentTest(unittest.TestCase):
    def test_subsamples_preserve_measurements_parameters_and_endpoints(self):
        with h5py.File(ROOT / 'data/packed/res_0.25MeV_noise0.01.h5') as source:
            for resolution in RESOLUTIONS:
                with h5py.File(DEFAULT_DATA / (tag(resolution) + '.h5')) as f:
                    self.assertEqual(len(np.unique(f['params'][:], axis=0)), 10)
                    for i, row in enumerate(f['source_rows'][:]):
                        a, b = source['idx_ptr'][row:row+2]
                        c, d = f['idx_ptr'][i:i+2]
                        for key in ('E_cat', 'Sigma_cat', 'Sigma_err_cat', 'Sigma_true_cat'):
                            np.testing.assert_array_equal(f[key][c:d], source[key][a:b:int(resolution/0.25)])
                        np.testing.assert_allclose(np.diff(f['E_cat'][c:d]), resolution)
                        np.testing.assert_array_equal(f['E_cat'][[c,d-1]], source['E_cat'][[a,b-1]])

    def test_analytic_barrier_matches_fine_three_point(self):
        params = (65., 11., 4.)
        e = np.linspace(60, 70, 10001)
        x, d, _ = paper_three_point_barrier(e, sigma(e, *params), np.zeros_like(e))
        np.testing.assert_allclose(d, wong_barrier(x, params), atol=0.003, rtol=2e-5)

    def test_absolute_coverage_error_and_raw_mae(self):
        x = np.linspace(60, 70, 11)
        params = (65., 11., 4.)
        truth = wong_barrier(x, params)
        lo = np.tile(truth - 2, (len(LEVELS), 1))
        hi = np.tile(truth + 2, (len(LEVELS), 1))
        result = score(x, truth + 1, lo, hi, params, (61, 69))
        self.assertAlmostEqual(result['barrier_MAE'], 1.)
        self.assertAlmostEqual(result['coverage_MAE'], np.mean(1-LEVELS))
        self.assertEqual(result['n_eval'], 9)
        lo[0, 3] = np.nan
        with self.assertRaises(ValueError):
            score(x, truth, lo, hi, params, (61, 69))

    def test_existing_datasets_cannot_be_replaced_by_new_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = ROOT / 'data/packed/res_0.25MeV_noise0.01.h5'
            prepare(source, directory)
            prepare(source, directory)
            with self.assertRaises(ValueError):
                prepare(source, directory, seed=43)


if __name__ == '__main__':
    unittest.main()
