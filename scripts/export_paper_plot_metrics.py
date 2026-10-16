#!/usr/bin/env python3
"""Export compact interior paper-plot tables from locally available predictions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from paper_metrics import interior_metrics
from score_resolution_paper import score_saved_fits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inset', type=float, default=0.)
    ap.add_argument('--full-h5-root', type=Path,
                    default=Path.home()/'scratch/barrier-paper-results/d3lambda1_rerun')
    args = ap.parse_args()
    out = ROOT / 'data/paper_plot_metrics' / f'interior_inset{args.inset:g}MeV'
    out.mkdir(parents=True, exist_ok=True)
    notebook = json.loads((ROOT/'notebooks/PaperFigures.ipynb').read_text())
    names = {}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                if key in ('GP_CSVS', 'BNN_CSVS'):
                    names[key] = ast.literal_eval(node.value)
    result_root = ROOT/'data/results/d3lambda1_rerun'
    manifest = dict(schema=1, inset_mev=args.inset, normalized=False,
                    description='Raw per-curve metrics; plotting applies hw/R0^2 normalization.',
                    files={}, code_sha256={str(p.relative_to(ROOT)):digest(p) for p in
                    [ROOT/'src/paper_metrics.py',ROOT/'src/train.py',ROOT/'src/utils.py',
                     ROOT/'scripts/score_resolution_paper.py']})
    for gp_name, bnn_name in zip(names['GP_CSVS'],names['BNN_CSVS']):
        bnn_path=result_root/'bnns_with_df'/bnn_name
        bnn_ids=pd.read_csv(bnn_path)['index'].astype(int).tolist()
        for family,name,is_bnn in [('bnns_with_df',bnn_name,True),('gps',gp_name,False),('gps_log',gp_name,False)]:
            source=result_root/family/name
            available=set(pd.read_csv(source)['index'].astype(int))
            ids=[i for i in bnn_ids if i in available] if is_bnn else [10*i for i in bnn_ids if 10*i in available]
            df=interior_metrics(source,is_bnn,args.inset,ids,args.full_h5_root)
            # Retain plot inputs and energy-range audit fields, not posterior arrays.
            columns=[c for c in df if c in ('index','E0_true','R0_true','hw_true','x_min','x_max')
                     or c.startswith(('barrier_','score_'))]
            target=out/family/name;target.parent.mkdir(exist_ok=True)
            df[columns].to_csv(target,index=False)
            key=str(target.relative_to(out))
            manifest['files'][key]=dict(rows=len(df),source_csv_sha256=digest(source),sha256=digest(target))
            print('Exported',key,len(df),'curves',flush=True)
    _, summary=score_saved_fits(interval='interior',inset_mev=args.inset)
    summary.to_csv(out/'resolution_summary.csv',index=False)
    records=ROOT/'data/results/resolution_experiment_soc_d3lam1_gp_log'/f'paper_metrics_interior_inset{args.inset:g}MeV/per_curve_metrics.csv'
    pd.read_csv(records).to_csv(out/'resolution_per_curve.csv',index=False)
    for name in ('resolution_summary.csv','resolution_per_curve.csv'):
        manifest['files'][name]=dict(sha256=digest(out/name))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Export complete:',out,flush=True)


if __name__=='__main__':
    main()
