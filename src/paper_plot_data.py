"""Read small, integrity-checked exports without opening prediction HDF5s."""
import hashlib
import json
from pathlib import Path
import pandas as pd


def _read_table(root, inset_mev, relative_name, source_csv=None):
    directory = Path(root) / f'interior_inset{float(inset_mev):g}MeV'
    manifest_path = directory / 'manifest.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 1 or manifest.get('inset_mev') != float(inset_mev):
        raise ValueError(f'Unexpected plot-data manifest: {manifest_path}')
    entry = manifest['files'].get(relative_name)
    if entry is None:
        return None
    target = directory / relative_name
    if hashlib.sha256(target.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError(f'Plot table checksum mismatch: {target}; regenerate the export')
    if source_csv is not None and Path(source_csv).exists():
        if hashlib.sha256(Path(source_csv).read_bytes()).hexdigest() != entry['source_csv_sha256']:
            raise ValueError(f'Plot export differs from source CSV: {source_csv}; regenerate the export')
    return pd.read_csv(target)


def load_compact_metrics(csv_path, root, inset_mev=0., indices=None):
    csv_path = Path(csv_path)
    df = _read_table(root, inset_mev, csv_path.parent.name+'/'+csv_path.name, csv_path)
    if df is None:
        return None
    if df['index'].duplicated().any():
        raise ValueError(f'Duplicate curve indices in export for {csv_path}')
    if indices is not None:
        requested = set(map(int, indices))
        if requested - set(df['index'].astype(int)):
            raise ValueError(f'Export lacks requested curve indices: {csv_path}')
        df = df[df['index'].isin(requested)].copy()
    return df


def load_resolution_summary(root, inset_mev=0.):
    return _read_table(root, inset_mev, 'resolution_summary.csv')


def load_resolution_records(root, inset_mev=0.):
    return _read_table(root, inset_mev, 'resolution_per_curve.csv')
