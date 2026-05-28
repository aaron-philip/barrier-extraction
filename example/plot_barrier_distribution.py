#!/usr/bin/env python3
"""Plot extracted barrier distributions from a merged BNN HDF5 file."""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _bands(samples, levels=(0.68, 0.95)):
    out = {}
    for level in levels:
        out[level] = (
            np.nanpercentile(samples, 100.0 * (1.0 - level) / 2.0, axis=0),
            np.nanpercentile(samples, 100.0 * (1.0 + level) / 2.0, axis=0),
        )
    return out


def _select_groups(h5_file, group, max_groups):
    names = sorted(h5_file.keys())
    if group is not None:
        if group not in h5_file:
            raise KeyError(f"Group {group!r} not found. Available groups: {names}")
        return [group]
    return names[:max_groups]


def plot_barrier(h5_path, out_path, group=None, max_groups=4):
    with h5py.File(h5_path, "r") as f:
        group_names = _select_groups(f, group, max_groups)

        fig, axes = plt.subplots(
            len(group_names),
            1,
            figsize=(6.5, 3.2 * len(group_names)),
            squeeze=False,
            constrained_layout=True,
        )

        for ax, group_name in zip(axes[:, 0], group_names):
            grp = f[group_name]
            x = grp["x_pred"][:]
            d_samples = grp["d2_samples"][:]
            d_mean = np.nanmean(d_samples, axis=0)
            bands = _bands(d_samples)

            lo95, hi95 = bands[0.95]
            lo68, hi68 = bands[0.68]
            ax.fill_between(x, lo95, hi95, color="#8fb6d9", alpha=0.25, lw=0)
            ax.fill_between(x, lo68, hi68, color="#2f6fa3", alpha=0.30, lw=0)
            ax.plot(x, d_mean, color="#153f66", lw=2.0)
            ax.set_title(f"Fit {group_name}")
            ax.set_xlabel(r"$E_{\rm cm}$ [MeV]")
            ax.set_ylabel(r"$D_f(E)$")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--out", default="example/output/barrier_distribution.png")
    parser.add_argument("--group", default=None)
    parser.add_argument("--max_groups", type=int, default=4)
    args = parser.parse_args()
    plot_barrier(args.h5, args.out, group=args.group, max_groups=args.max_groups)


if __name__ == "__main__":
    main()
