"""Cached interior scoring for PaperFigures, using saved posterior curves."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from train import barrier_from_sigma_samples
from utils import compute_metrics_for_fit, peak_metrics_anchored
from wong import sigma


def resolve_prediction_file(csv_path, indices, full_h5_root=None):
    """Prefer the repo HDF5; use an explicit full-results root for slim copies."""
    csv_path = Path(csv_path)
    local = Path(str(csv_path).removesuffix('.csv'))
    candidates = [local]
    if full_h5_root is not None:
        candidates.append(Path(full_h5_root).expanduser() / csv_path.parent.name / local.name)
    required = {str(int(i)) for i in indices}
    failures = []
    for candidate in candidates:
        if not candidate.is_file():
            failures.append(f'{candidate}: file absent')
            continue
        with h5py.File(candidate, 'r') as f:
            missing = required - set(f.keys())
        if not missing:
            return candidate
        failures.append(f'{candidate}: missing {len(missing)} requested fits '
                        f'(examples: {sorted(missing, key=int)[:5]})')
    raise ValueError('Interior rescoring requires saved predictions for every paired fit. '
                     'Set SETTINGS["full_h5_root"] to the full d3lambda1_rerun directory. '
                     'No fits were dropped.\n' + '\n'.join(failures))


def interior_metrics(csv_path, is_bnn, inset_mev=0.0, indices=None, full_h5_root=None):
    """Recompute raw metrics in [min(E_data)+inset, max(E_data)-inset].

    Normalize in the plotting loader, exactly once, after this function returns.
    Original CSVs/HDF5s remain untouched. Only requested paired fits are read.
    """
    csv_path = Path(csv_path)
    inset_mev = float(inset_mev)
    if not np.isfinite(inset_mev) or inset_mev < 0:
        raise ValueError('inset_mev must be finite and nonnegative')
    source = pd.read_csv(csv_path)
    source['index'] = pd.to_numeric(source['index'], errors='raise').astype(int)
    if source['index'].duplicated().any():
        raise ValueError(f'Duplicate fit indices in {csv_path}')
    if indices is not None:
        requested = set(map(int, indices))
        missing = requested - set(source['index'])
        if missing:
            raise ValueError(f'{csv_path}: missing paired indices {sorted(missing)}')
        source = source[source['index'].isin(requested)].copy()
    if source.empty:
        raise ValueError(f'No fits selected in {csv_path}')
    h5_path = resolve_prediction_file(csv_path, source['index'], full_h5_root)
    # Include source and calculation fingerprints so caches cannot silently
    # survive changes in predictions, CSV metadata, pairing, or scoring code.
    dependencies = [csv_path, h5_path, Path(__file__),
                    Path(__file__).with_name('utils.py'), Path(__file__).with_name('train.py'),
                    Path(__file__).with_name('wong.py')]
    signature = dict(inset_mev=inset_mev, is_bnn=bool(is_bnn),
                     indices=source['index'].tolist(),
                     files=[(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                            for p in dependencies])
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    cache_dir = csv_path.parent / 'interior_metric_cache'
    cache = cache_dir / f'{csv_path.name}.inset{inset_mev:g}.{digest}.csv'
    if cache.exists():
        return pd.read_csv(cache)

    print(f'Rescoring {len(source)} fits: {h5_path}, inset={inset_mev:g} MeV', flush=True)
    rows = []
    with h5py.File(h5_path, 'r') as f:
        for _, row in source.iterrows():
            index = int(row['index'])
            if str(index) not in f:
                raise ValueError(f'{h5_path}: no saved curves for fit {index}; full HDF5 is required')
            g = f[str(index)]
            xp = np.asarray(g['x_pred'][:], float)
            e = np.asarray(g['E_data'][:], float)
            bounds = (float(e.min()+inset_mev), float(e.max()-inset_mev))
            mask = (xp >= bounds[0]) & (xp <= bounds[1])
            if mask.sum() < 10:
                raise ValueError(f'{h5_path}, fit {index}: fewer than ten interior prediction points')
            params = tuple(float(row[k]) for k in ('E0_true', 'R0_true', 'hw_true'))
            np.testing.assert_allclose(
                [g.attrs[k] for k in ('E0_true', 'R0_true', 'hw_true')], params,
                err_msg=f'CSV/HDF5 parameter mismatch: {h5_path}, fit {index}')
            samples = np.asarray(g['y_samples'][:], float)
            mean = np.asarray(g['y_mean'][:], float)
            d2 = np.asarray(g['d2_samples'][:], float) if is_bnn and 'd2_samples' in g else None
            for arr in (samples[:, mask], mean[mask], d2[:, mask] if d2 is not None else xp[mask]):
                if not np.isfinite(arr).all():
                    raise ValueError(f'{h5_path}, fit {index}: nonfinite interior predictions')
            metrics = compute_metrics_for_fit(
                xp, mean, samples, e, *params,
                sigma_true_fn=sigma,
                barrier_from_sigma_samples_fn=barrier_from_sigma_samples,
                d2_samples=d2, pad=-inset_mev,
            )
            x = xp[mask]
            barriers = d2[:, mask] if d2 is not None else barrier_from_sigma_samples(x, samples[:, mask])
            truth = barrier_from_sigma_samples(x, sigma(x, *params)[None, :])[0]
            median = np.median(barriers, axis=0)
            error = median - truth
            loc, ht, width = peak_metrics_anchored(x, median, params[0], window=6., smooth=5)
            metrics.update(
                barrier_MAE_median=float(np.mean(np.abs(error))),
                barrier_RMSE_median=float(np.sqrt(np.mean(error**2))),
                barrier_peak_loc_pred_median=loc,
                barrier_peak_ht_pred_median=ht,
                barrier_fwhm_pred_median=width,
                barrier_peak_loc_resid_median=loc-metrics['barrier_peak_loc_true'],
                barrier_peak_ht_resid_median=ht-metrics['barrier_peak_ht_true'],
                barrier_fwhm_resid_median=width-metrics['barrier_fwhm_true'],
                score_inset_mev=inset_mev, score_n_points=int(mask.sum()),
                score_grid_min=float(x[0]), score_grid_max=float(x[-1]),
            )
            if not np.isfinite([metrics['barrier_MAE'],metrics['barrier_MAE_median'],
                                *[v for k,v in metrics.items() if '_emp_cov_' in k]]).all():
                raise ValueError(f'{h5_path}, fit {index}: nonfinite recomputed metrics')
            updated = row.to_dict()
            updated.update(metrics)
            rows.append(updated)
    result = pd.DataFrame(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache, index=False)
    return result
