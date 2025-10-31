#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import scipy.stats
import findiff as fd
import sys

if "/mnt/home/philipaa/research/barrier-extraction/src" not in sys.path:
    sys.path.append("/mnt/home/philipaa/research/barrier-extraction/src")  

from utils import barrier_dist
from wong import sigma

# -------------------------------------------------------------------
# Main plotting utility
# -------------------------------------------------------------------
def plot_wong_sanity(data_path, bands_path, csv_path, n_samples=5, levels=(0.68, 0.9, 0.95)):
    """
    Plot fitted Wong cross sections and barrier distributions with uncertainty bands.

    Parameters
    ----------
    data_path : str
        Path to the original .npz dataset used for fitting (contains E, Sigma, Sigma_err).
    bands_path : str
        Path to the _bands.npz file saved by train_wongs.py.
    csv_path : str
        Path to the .csv file of Wong fit metrics.
    n_samples : int
        Number of random examples to visualize.
    levels : tuple of float
        Confidence levels for uncertainty bands.
    """
    df = pd.read_csv(csv_path)
    bands = np.load(bands_path, allow_pickle=True)["band_data"]
    npz = np.load(data_path, allow_pickle=True)

    E_all = npz["E"]
    Sigma_all = npz["Sigma"]
    Sigma_err_all = npz["Sigma_err"]

    total = len(bands)
    idxs = np.random.choice(total, size=min(n_samples, total), replace=False)

    fig, axes = plt.subplots(len(idxs), 2, figsize=(12, 3.5 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[np.newaxis, :]

    for row, i in enumerate(idxs):
        entry = bands[i]
        E = entry["x_pred"]
        y_mean, y_true = entry["y_mean"], entry["y_true"]
        D_mean, D_true = entry["D_mean"], entry["D_true"]
        sigma_bands = entry["sigma_bands"]
        barrier_bands = entry["barrier_bands"]
        print(y_mean, y_true, D_mean, D_true, sigma_bands, barrier_bands)

        # Experimental (fitting) data
        E_data = np.array(E_all[i], dtype=float)
        Sigma_data = np.array(Sigma_all[i], dtype=float)
        Sigma_err = np.array(Sigma_err_all[i], dtype=float)

        # --- σ(E) plot ---
        ax1 = axes[row, 0]
        ax1.errorbar(E_data, Sigma_data, yerr=Sigma_err, fmt="o", color="black",
                     alpha=0.6, markersize=4, label="Data")
        ax1.plot(E, y_true, "k--", lw=1.5, label="True σ(E)")
        ax1.plot(E, y_mean, "r-", lw=1.5, label="Mean σ(E)")
        for alpha in levels:
            lo = sigma_bands[f"lower_{int(alpha*100)}"]
            hi = sigma_bands[f"upper_{int(alpha*100)}"]
            ax1.fill_between(E, lo, hi, alpha=0.15, label=f"{int(alpha*100)}% band")
        ax1.set_xlabel("E [MeV]")
        ax1.set_ylabel("σ(E)")
        ax1.set_title(f"Cross Section Fit #{i}")
        ax1.legend(frameon=False, fontsize=8)

        # --- D(E) plot ---
        ax2 = axes[row, 1]
        ax2.plot(E, D_true, "k", lw=1.5, label="True D(E)")
        ax2.plot(E, D_mean, "r", lw=1.5, label="Mean D(E)", linestyle="--")
        for alpha in levels:
            lo = barrier_bands[f"lower_{int(alpha*100)}"]
            hi = barrier_bands[f"upper_{int(alpha*100)}"]
            ax2.fill_between(E, lo, hi, alpha=0.15, label=f"{int(alpha*100)}% band")
        ax2.set_xlabel("E [MeV]")
        ax2.set_ylabel("D(E)")
        ax2.set_title(f"Barrier Distribution #{i}")
        ax2.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig("wong_fit_sanity.png", dpi=300)
    print("Saved plot → wong_fit_sanity.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    #ap.add_argument("--data", required=True, help="Path to .npz dataset used for training")
    #ap.add_argument("--fits", required=True, help="Path to .csv file of Wong fit results")
    #ap.add_argument("--bands", required=True, help="Path to _bands.npz file of Wong fit results")
    ap.add_argument("--n_samples", type=int, default=5)
    args = ap.parse_args()

    args.data = "/mnt/home/philipaa/scratch/barrier-paper-datasets/cv0.00_spacing0.5.npz"
    args.fits = "/mnt/home/philipaa/scratch/barrier-paper-results/cv0.00_spacing0.5_wong.csv"
    args.bands = "/mnt/home/philipaa/scratch/barrier-paper-results/cv0.00_spacing0.5_wong_bands.npz"
    plot_wong_sanity(args.data, args.bands, args.fits, n_samples=args.n_samples)
