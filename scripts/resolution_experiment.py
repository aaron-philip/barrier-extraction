#!/usr/bin/env python3
"""Prepare and score the paired paper resolution experiment (no interpolation)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utils import paper_three_point_barrier
from train import barrier_from_sigma_samples

RESOLUTIONS = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
LEVELS = np.array([0.25, 0.5, 0.68, 0.9, 0.95])
DEFAULT_DATA = ROOT / 'data/packed/resolution_experiment'
DEFAULT_RESULTS = ROOT / 'data/results/resolution_experiment_soc_d3lam1_gp_log'


def tag(resolution):
    return f'res_{resolution:.2f}MeV'


def prepare(source, out_dir, seed=42):
    """Keep identical measured values across resolutions; record full provenance."""
    source, out_dir = Path(source), Path(out_dir)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with h5py.File(source) as f:
        for key, expected in [('cv', 0.0), ('mean_spacing', 0.25), ('noise_ratio', 0.01)]:
            if not np.isclose(f.attrs[key], expected):
                raise ValueError(f'Unexpected source {key}: {f.attrs[key]}')
        n_noise = int(f.attrs['n_noise'])
        n_curves = int(f.attrs['n_curves'])
        ids = np.sort(np.random.default_rng(seed).choice(n_curves, 10, replace=False))
        rows = ids * n_noise
        params = f['params'][rows]
        if len(np.unique(params, axis=0)) != 10:
            raise ValueError('Expected ten distinct parameter samples')
        ptr = f['idx_ptr'][:]
        fields = ('E_cat', 'Sigma_cat', 'Sigma_err_cat', 'Sigma_true_cat')
        curves = [{k: f[k][ptr[r]:ptr[r+1]] for k in fields} for r in rows]
    for c in curves:
        if len(c['E_cat']) != 121 or not np.allclose(np.diff(c['E_cat']), 0.25):
            raise ValueError('Expected 121 points on a regular 30 MeV interval')
    manifest = dict(source=str(source.resolve()), source_sha256=digest, selection_seed=seed,
                    source_parameter_indices=ids.tolist(), source_rows=rows.tolist(),
                    noise_realization=0, resolutions=list(RESOLUTIONS),
                    evaluation_inset_MeV=3.0, coverage_levels=LEVELS.tolist())
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / 'manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        # Location is descriptive; content hashes define identity across checkouts.
        if {k: v for k, v in existing.items() if k != 'source'} != {
                k: v for k, v in manifest.items() if k != 'source'}:
            raise ValueError('Existing dataset manifest differs; choose a fresh output directory')
    for resolution in RESOLUTIONS:
        step = int(round(resolution / 0.25))
        arrays = {k: np.concatenate([c[k][::step] for c in curves]) for k in fields}
        lengths = [len(c['E_cat'][::step]) for c in curves]
        arrays.update(idx_ptr=np.r_[0, np.cumsum(lengths)], params=params,
                      source_parameter_indices=ids, source_rows=rows)
        path = out_dir / (tag(resolution) + '.h5')
        if path.exists():
            with h5py.File(path) as f:
                if any(k not in f or not np.array_equal(f[k][:], v) for k, v in arrays.items()):
                    raise ValueError(f'Existing dataset differs: {path}')
                if f.attrs.get('source_sha256') != digest:
                    raise ValueError(f'Existing dataset provenance differs: {path}')
        else:
            with h5py.File(path, 'x') as f:
                for k, v in arrays.items():
                    f.create_dataset(k, data=v)
                f.attrs.update(noise_ratio=0.01, cv=0.0, mean_spacing=resolution,
                               n_curves=10, n_noise=1, source_sha256=digest,
                               selection_seed=seed, noise_realization=0,
                               source=str(source.resolve()))
        print(f'{path}: 10 curves, {lengths[0]} points each; source rows {rows.tolist()}')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


def wong_barrier(x, params):
    """Analytic d2(E*sigma)/dE2, including sigma's fm^2 -> mb factor of 10."""
    e0, r0, hw = params
    z = 2 * np.pi * (np.asarray(x) - e0) / hw
    return 20 * np.pi**2 * r0**2 / hw * expit(z) * expit(-z)


def score(x, center, lower, upper, params, bounds):
    mask = (x >= bounds[0] - 1e-8) & (x <= bounds[1] + 1e-8)
    if not np.any(mask):
        raise ValueError('Empty scoring interval')
    truth = wong_barrier(x[mask], params)
    if not all(np.all(np.isfinite(a)) for a in (center[mask], lower[:, mask], upper[:, mask])):
        raise ValueError('Nonfinite predictions: refusing to silently drop a fit')
    coverage = np.mean((lower[:, mask] <= truth) & (truth <= upper[:, mask]), axis=1)
    return dict(barrier_MAE=float(np.mean(np.abs(center[mask] - truth))),
                coverage_MAE=float(np.mean(np.abs(coverage - LEVELS))),
                n_eval=int(mask.sum()),
                **{f'coverage_{int(100*l)}': float(c) for l, c in zip(LEVELS, coverage)})


def summarize(data_dir=DEFAULT_DATA, result_dir=DEFAULT_RESULTS):
    """Require all 210 method/curve cases; never average incomplete results."""
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    records = []
    for resolution in RESOLUTIONS:
        name = tag(resolution)
        with h5py.File(data_dir / (name + '.h5')) as data, \
             h5py.File(result_dir / (name + '_bnn.h5')) as bnn, \
             h5py.File(result_dir / (name + '_gp.h5')) as gp:
            expected = {str(i) for i in range(10)}
            if set(bnn) != expected or set(gp) != expected:
                raise ValueError(f'{name}: expected exactly ten BNN and ten GP fits')
            ptr = data['idx_ptr'][:]
            for i in range(10):
                sl = slice(ptr[i], ptr[i+1])
                e, s, se = [data[k][sl] for k in ('E_cat', 'Sigma_cat', 'Sigma_err_cat')]
                params = data['params'][i]
                bounds = (e[0] + 3.0, e[-1] - 3.0)
                common = dict(resolution=resolution, index=i,
                              source_parameter_index=int(data['source_parameter_indices'][i]),
                              source_row=int(data['source_rows'][i]), n_train=len(e))
                x, center, err = paper_three_point_barrier(e, s, se, dE=1)
                z = norm.ppf((1 + LEVELS) / 2)[:, None]
                records.append(dict(common, method='3-point', **score(
                    x, center, center - z*err, center + z*err, params, bounds)))
                for method, file in [('BNN', bnn), ('GP', gp)]:
                    g = file[str(i)]
                    for key, actual in [('E_data', e), ('Sigma_data', s), ('Sigma_err', se)]:
                        np.testing.assert_array_equal(g[key][:], actual)
                    np.testing.assert_allclose([g.attrs[k] for k in ('E0_true', 'R0_true', 'hw_true')], params)
                    x = g['x_pred'][:]
                    if method == 'BNN':
                        samples = g['d2_samples'][:]
                        center = samples.mean(axis=0)
                    else:
                        samples = barrier_from_sigma_samples(x, g['y_samples'][:])
                        center = barrier_from_sigma_samples(x, g['y_mean'][:][None, :])[0]
                    lower = np.quantile(samples, (1-LEVELS)/2, axis=0)
                    upper = np.quantile(samples, (1+LEVELS)/2, axis=0)
                    records.append(dict(common, method=method, **score(x, center, lower, upper, params, bounds)))
    frame = pd.DataFrame(records)
    summary = frame.groupby(['resolution', 'method'], as_index=False).agg(
        barrier_MAE=('barrier_MAE', 'mean'), coverage_MAE=('coverage_MAE', 'mean'),
        n_curves=('index', 'count'), n_train=('n_train', 'first'))
    if len(frame) != 210 or not (summary.n_curves == 10).all():
        raise ValueError('Incomplete experiment')
    frame.to_csv(result_dir / 'per_curve_metrics.csv', index=False)
    summary.to_csv(result_dir / 'summary.csv', index=False)
    return frame, summary


def plot_summary(summary, plot_dir):
    """The notebook and dependent batch job share the same two figures."""
    import matplotlib.pyplot as plt
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {'BNN': 'green', 'GP': 'blue', '3-point': 'black'}
    plt.rcParams.update({'font.family': 'serif', 'font.size': 13})
    figures = []
    for metric, ylabel, filename in [
        ('barrier_MAE', 'Mean barrier MAE (mb/MeV)', 'barrier_mae'),
        ('coverage_MAE', 'Mean absolute coverage deviation', 'coverage_mae'),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
        for method, color in colors.items():
            rows = summary[summary.method == method].sort_values('resolution')
            ax.plot(rows.resolution, rows[metric], 'o-', color=color, label=method)
        ax.set(xlabel='Measurement resolution (MeV)', ylabel=ylabel)
        ax.set_xticks(sorted(summary.resolution.unique()))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        for extension in ('png', 'pdf'):
            fig.savefig(plot_dir / f'{filename}.{extension}', dpi=300)
        figures.append(fig)
    return figures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'summarize'])
    ap.add_argument('--source', type=Path, default=ROOT / 'data/packed/res_0.25MeV_noise0.01.h5')
    ap.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--result-dir', type=Path, default=DEFAULT_RESULTS)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.action == 'prepare':
        prepare(args.source, args.data_dir, args.seed)
    else:
        _, summary = summarize(args.data_dir, args.result_dir)
        print(summary.to_string(index=False))
        plot_summary(summary, args.result_dir / 'plots')


if __name__ == '__main__':
    main()
