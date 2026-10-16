"""Rescore saved resolution fits using PaperFigures' median and normalization."""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.stats import norm
from resolution_experiment import (
    DEFAULT_DATA, DEFAULT_RESULTS, RESOLUTIONS, tag, wong_barrier,
    paper_three_point_barrier, barrier_from_sigma_samples,
)
from wong import sigma

# PaperFigures selects every saved nominal level below 100%, including 0%.
PAPER_LEVELS = np.array([0., .25, .5, .68, .9, .95])
PAPER_MAE_LABEL = r'$(\frac{\hbar \omega_0 }{R_0 ^2}) \|\delta D_f\|_1$ (mb/MeV)'
PAPER_RMSE_LABEL = r'$(\frac{\hbar \omega_0 }{R_0 ^2}) \|\delta D_f\|_2$ (mb/MeV)'


def nearest_stencil_offset(energy, target_mev):
    energy = np.asarray(energy, dtype=float)
    spacing = np.diff(energy)
    if not np.isfinite(target_mev) or target_mev <= 0:
        raise ValueError('Target interval must be finite and positive')
    if len(spacing) < 2 or spacing[0] <= 0 or not np.allclose(spacing, spacing[0]):
        raise ValueError('Interval selection requires a regular increasing grid')
    offsets = np.arange(1, (len(energy) - 1) // 2 + 1)
    distances = np.abs(offsets * spacing[0] - target_mev)
    offset = int(offsets[np.flatnonzero(np.isclose(
        distances, distances.min(), rtol=1e-10, atol=1e-10))[0]])
    return offset, float(offset * spacing[0])


def metric_record(x, center, truth, lower, upper, params):
    if not all(np.isfinite(a).all() for a in (x, center, truth, lower, upper)):
        raise ValueError('Nonfinite prediction; refusing to drop a curve')
    raw = float(np.mean(np.abs(center - truth)))
    rmse = float(np.sqrt(np.mean((center - truth)**2)))
    scale = float(params[1]**2 / params[2])
    coverage = np.mean((lower <= truth) & (truth <= upper), axis=1)
    return dict(barrier_MAE_raw=raw, normalization_R0sq_over_hw=scale,
                barrier_MAE=raw / scale, barrier_RMSE_raw=rmse,
                barrier_RMSE=rmse / scale, n_eval=len(x),
                eval_min=float(x.min()), eval_max=float(x.max()),
                coverage_MAE_per_curve=float(np.mean(np.abs(coverage-PAPER_LEVELS))),
                **{f'coverage_{int(l*100)}': float(c) for l,c in zip(PAPER_LEVELS,coverage)})


def score_saved_fits(data_dir=DEFAULT_DATA, result_dir=DEFAULT_RESULTS, interval='interior', inset_mev=0.0):
    """interior: measured range with optional inset, matching PaperFigures.

    common: legacy 3-MeV interior; paper: original padded CSV bounds.

    GP/truth differentiation is performed AFTER masking, exactly as in the
    PaperFigures median-metric helper. Three-point truth is analytic: applying
    the same coarse stencil to the truth would hide its discretization bias.
    """
    if interval not in ('interior', 'common', 'paper'):
        raise ValueError('interval must be interior, common, or paper')
    if not np.isfinite(inset_mev) or inset_mev < 0:
        raise ValueError('inset_mev must be finite and nonnegative')
    inset = 3.0 if interval == 'common' else float(inset_mev)
    data_dir, result_dir = Path(data_dir), Path(result_dir)
    records=[]
    for resolution in RESOLUTIONS:
        name=tag(resolution)
        with h5py.File(data_dir/(name+'.h5')) as data:
            for method in ('BNN','GP'):
                base=result_dir/(name+'_'+method.lower()+'.h5')
                bounds_df=pd.read_csv(str(base)+'.csv').set_index('index')
                with h5py.File(base) as f:
                    if set(f) != {str(i) for i in range(10)}:
                        raise ValueError(f'Incomplete fits: {base}')
                    for i in range(10):
                        g=f[str(i)]; a,b=data['idx_ptr'][i:i+2]
                        e=data['E_cat'][a:b]; params=data['params'][i]
                        for key, packed in [('E_data','E_cat'),('Sigma_data','Sigma_cat'),('Sigma_err','Sigma_err_cat')]:
                            np.testing.assert_array_equal(g[key][:],data[packed][a:b])
                        np.testing.assert_allclose([g.attrs[k] for k in ('E0_true','R0_true','hw_true')],params)
                        bounds=(e[0]+inset,e[-1]-inset) if interval!='paper' else (
                            float(bounds_df.loc[i,'x_min']),float(bounds_df.loc[i,'x_max']))
                        xp=g['x_pred'][:]; mask=(xp>=bounds[0])&(xp<=bounds[1]); x=xp[mask]
                        if len(x)<5: raise ValueError('Insufficient evaluation points')
                        samples=g['d2_samples'][:,mask] if method=='BNN' else barrier_from_sigma_samples(x,g['y_samples'][:,mask])
                        truth=barrier_from_sigma_samples(x,sigma(x,*params)[None,:])[0]
                        lo=np.quantile(samples,(1-PAPER_LEVELS)/2,axis=0)
                        hi=np.quantile(samples,(1+PAPER_LEVELS)/2,axis=0)
                        records.append(dict(resolution=resolution,index=i,method=method,
                            source_row=int(data['source_rows'][i]), target_interval_MeV=0.,
                            requested_min=bounds[0],requested_max=bounds[1],
                            center='median',truth_convention='paper_numerical',interval_mode=interval,
                            **metric_record(x,np.median(samples,axis=0),truth,lo,hi,params)))
            for i in range(10):
                a,b=data['idx_ptr'][i:i+2]
                e,s,se=[data[k][a:b] for k in ('E_cat','Sigma_cat','Sigma_err_cat')]
                params=data['params'][i]
                for target in (0.,2.,3.):
                    offset,actual=(1,float(e[1]-e[0])) if target==0 else nearest_stencil_offset(e,target)
                    x,center,err=paper_three_point_barrier(e,s,se,dE=offset)
                    bounds=(e[0]+inset,e[-1]-inset) if interval!='paper' else (e[0]-2.,e[-1]+2.)
                    mask=(x>=bounds[0]-1e-8)&(x<=bounds[1]+1e-8)
                    x,center,err=x[mask],center[mask],err[mask]
                    z=norm.ppf((1+PAPER_LEVELS)/2)[:,None]
                    records.append(dict(resolution=resolution,index=i,method='3-point',
                        source_row=int(data['source_rows'][i]),target_interval_MeV=target,
                        actual_interval_MeV=actual,stencil_offset=offset,
                        requested_min=bounds[0],requested_max=bounds[1],
                        center='three_point',truth_convention='analytic',interval_mode=interval,
                        **metric_record(x,center,wong_barrier(x,params),center-z*err,center+z*err,params)))
    frame=pd.DataFrame(records)
    models=frame[frame.method!='3-point']
    comparison=pd.concat([frame[frame.method=='3-point'],
                         *[models.assign(target_interval_MeV=t) for t in (0.,2.,3.)]],ignore_index=True)
    covcols=[f'coverage_{int(l*100)}' for l in PAPER_LEVELS]
    keys=['target_interval_MeV','resolution','method']
    summary=comparison.groupby(keys,as_index=False)[['barrier_MAE','barrier_MAE_raw','barrier_RMSE','barrier_RMSE_raw',
                               'coverage_MAE_per_curve',*covcols]].mean()
    # Match PaperFigures: average empirical coverage across curves FIRST.
    summary['coverage_MAE']=np.abs(summary[covcols].to_numpy()-PAPER_LEVELS).mean(axis=1)
    counts=comparison.groupby(keys).size()
    if len(summary)!=63 or not (counts==10).all(): raise ValueError('Incomplete experiment')
    summary['n_curves']=10
    output=result_dir/('paper_metrics_interior_inset'+f'{inset:g}MeV' if interval=='interior' else 'paper_metrics_'+interval)
    output.mkdir(parents=True,exist_ok=True)
    comparison.to_csv(output/'per_curve_metrics.csv',index=False)
    summary.to_csv(output/'summary.csv',index=False)
    frame[frame.method=='3-point'].to_csv(output/'stencil_audit.csv',index=False)
    return comparison,summary
