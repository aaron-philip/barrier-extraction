#!/usr/bin/env python3
import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys

if "/mnt/home/philipaa/research/barrier-extraction/src" not in sys.path:
    sys.path.append("/mnt/home/philipaa/research/barrier-extraction/src")

from train import  init_E0, run_wong_trial, fit_wong_with_restarts, make_x_pred, draw_sigma_samples,percentile_bands,barrier_from_sigma_samples,fit_wong
from utils import load_packed_hdf5 # updated import
from wong import sigma


# -------------------------------------------------------------------
# Plot σ(E) and D(E) in a 2-panel row
# -------------------------------------------------------------------
def plot_row(ax_sigma, ax_barrier, x, y_samples, y_mean,
             E_data, Sigma_data, Sigma_err_data,
             D_samples, D_mean, D_true,
             E0_fit, E0_true, fit_index):
    """
    Draws the two-panel plot for one fit.
    """

    # ================= σ(E) ====================
    for y in y_samples:
        ax_sigma.plot(x, y, color="gray", alpha=0.20, lw=0.7)

    ax_sigma.plot(x, y_mean, "r--", lw=2, label="Mean σ(E)")

    ax_sigma.errorbar(
        E_data, Sigma_data, yerr=Sigma_err_data,
        fmt="o", color="black", markersize=4, alpha=0.8,
        label="Training data"
    )

    ax_sigma.set_xlabel("Energy [MeV]")
    ax_sigma.set_ylabel("σ(E)")
    ax_sigma.set_title(f"Fit #{fit_index}: σ(E)\nE0_fit={E0_fit:.2f}, E0_true={E0_true:.2f}")
    ax_sigma.legend(frameon=False, fontsize=8)

    # ================= D(E) ====================
    for D in D_samples:
        ax_barrier.plot(x, D, color="gray", alpha=0.20, lw=0.7)

    ax_barrier.plot(x, D_mean, "r--", lw=2, label="Mean D(E)")
    ax_barrier.plot(x, D_true, "b--", lw=2, label="True D(E)")

    # 68% band
    bands = percentile_bands(D_samples, levels=(0.68,))
    D_lo = bands["lower_68"]
    D_hi = bands["upper_68"]
    ax_barrier.fill_between(x, D_lo, D_hi, color="red", alpha=0.15, label="68% band")

    ax_barrier.set_xlabel("Energy [MeV]")
    ax_barrier.set_ylabel("D(E)")
    ax_barrier.set_title(f"Fit #{fit_index}: Barrier Dist.")
    ax_barrier.set_ylim(-1000, 2500)
    ax_barrier.legend(frameon=False, fontsize=8)


# -------------------------------------------------------------------
# Main function: randomly sample indices and plot them
# -------------------------------------------------------------------
def plot_multiple_fits(h5_path, n_samples):
    """
    Randomly pick n_samples fits from the HDF5 file and plot σ(E) & D(E).
    """

    with h5py.File(h5_path, "r") as f:
        all_keys = sorted([int(k) for k in f.keys()])

    total = len(all_keys)
    n_samples = min(n_samples, total)

    idxs = np.random.choice(all_keys, size=n_samples, replace=False)
    print(f"Selected indices: {idxs}")

    # Prepare figure grid
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    # Loop through sampled indices
    with h5py.File(h5_path, "r") as f:
        for row, fit_index in enumerate(idxs):
            grp = f[f"{fit_index}"]

            # Load arrays
            x = grp["x_pred"][:]
            y_mean = grp["y_mean"][:]
            y_samples = grp["y_samples"][:]

            E_data = grp["E_data"][:]
            Sigma_data = grp["Sigma_data"][:]
            Sigma_err = grp["Sigma_err"][:]

            E0_true = grp.attrs["E0_true"]
            R0_true = grp.attrs["R0_true"]
            hw_true = grp.attrs["hw_true"]
            E0_fit = grp.attrs.get("E0_fit", np.nan)

            # Compute D(E)
            D_samples = barrier_from_sigma_samples(x, y_samples)
            D_mean = barrier_from_sigma_samples(x, y_mean.reshape(1, -1))[0]

            sigma_true = sigma(x, E0=E0_true, R0=R0_true, hbarw=hw_true)
            D_true = barrier_from_sigma_samples(x, sigma_true.reshape(1, -1))[0]

            # Plot on this row
            ax1, ax2 = axes[row]
            plot_row(ax1, ax2,
                     x, y_samples, y_mean,
                     E_data, Sigma_data, Sigma_err,
                     D_samples, D_mean, D_true,
                     E0_fit, E0_true,
                     fit_index)

    plt.tight_layout()
    plt.savefig(f"{os.path.basename(h5_path)}.png", dpi=300)
    print(f"\nSaved → {os.path.basename(h5_path)}.png")
    plt.show()


# -------------------------------------------------------------------
# Command-line entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=str, required=True, help="Path to GP .h5 results")
    ap.add_argument("--n_samples", type=int, default=5, help="Number of random fits to plot")
    args = ap.parse_args()

    plot_multiple_fits(args.h5, args.n_samples)


