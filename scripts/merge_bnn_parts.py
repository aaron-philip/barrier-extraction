#!/usr/bin/env python3
import os, sys, argparse, glob
import h5py
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_h5", required=True, help="Final merged HDF5 path (same as used in train_bnns.py)")
    args = ap.parse_args()

    out_h5 = args.out_h5
    out_csv = out_h5 + ".csv"
    parts_dir = out_h5 + ".parts"

    if not os.path.isdir(parts_dir):
        raise FileNotFoundError(f"Parts dir not found: {parts_dir}")

    part_h5s = sorted(glob.glob(os.path.join(parts_dir, "*.h5")))
    part_csvs = sorted(glob.glob(os.path.join(parts_dir, "*.csv")))

    if len(part_h5s) == 0:
        raise RuntimeError(f"No part H5 files found in {parts_dir}")

    # Merge H5 groups
    if os.path.exists(out_h5):
        os.remove(out_h5)

    with h5py.File(out_h5, "w") as fout:
        for ph in part_h5s:
            with h5py.File(ph, "r") as fin:
                for k in fin.keys():
                    fin.copy(k, fout, name=k)

    # Merge CSV rows
    rows = []
    for pc in part_csvs:
        try:
            rows.append(pd.read_csv(pc))
        except Exception:
            pass

    df = pd.concat(rows, ignore_index=True) if len(rows) else pd.DataFrame()
    df.to_csv(out_csv, index=False)

    print(f"Merged {len(part_h5s)} part H5s -> {out_h5}")
    print(f"Merged {len(part_csvs)} part CSVs -> {out_csv}")

if __name__ == "__main__":
    main()
