#!/usr/bin/env python3
import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys

if "/mnt/home/philipaa/research/barrier-extraction/src" not in sys.path:
    sys.path.append("/mnt/home/philipaa/research/barrier-extraction/src")

from train import percentile_bands, barrier_from_sigma_samples
from wong import sigma

# NEW: CSV reading
import pandas as pd


# -------------------------------------------------------------------
# Draw a "cross" from (loc, ht, fwhm) using symmetric endpoints
# (we don't have left/right half-max crossings in the CSV)
# -------------------------------------------------------------------
def draw_cross_from_metrics(ax, loc, ht, fwhm, color):
    if np.isfinite(loc) and np.isfinite(ht):
        ax.axvline(loc, color=color, lw=1.2, alpha=0.9)
        ax.plot([loc], [ht], marker="x", ms=8, mew=2, color=color)

        if np.isfinite(fwhm) and fwhm > 0:
            left = loc - 0.5 * fwhm
            right = loc + 0.5 * fwhm
            ax.hlines(0.5 * ht, left, right, colors=color, lw=2.0, alpha=0.9)


# -------------------------------------------------------------------
# Plot σ(E) and D(E) in a 2-panel row (cross now comes from CSV)
# -------------------------------------------------------------------
def plot_row(ax_sigma, ax_barrier, x, y_samples, y_mean,
             E_data, Sigma_data, Sigma_err_data,
             D_samples, D_mean, D_true,
             E0_fit, E0_true, fit_index,
             # NEW: cross metrics (from CSV)
             t_loc, t_ht, t_fwhm,
             p_loc, p_ht, p_fwhm):
    """
    Draws the two-panel plot for one fit.
    Cross overlay on D(E) uses metrics loaded from CSV.
    """

    # ================= σ(E) ====================
    
    # 68% band for sigma samples
    sbands = percentile_bands(y_samples, levels=(0.68,))
    s_lo = sbands["lower_68"]
    s_hi = sbands["upper_68"]
    ax_sigma.fill_between(x, s_lo, s_hi, color="red", alpha=0.15, label="68% band")

    # mean
    ax_sigma.plot(x, y_mean, "r--", lw=2, label="Mean σ(E)")

    # data
    ax_sigma.errorbar(
        E_data, Sigma_data, yerr=Sigma_err_data,
        fmt="o", color="black", markersize=4, alpha=0.8,
        label="Training data"
    )

    ax_sigma.set_xlabel("Energy [MeV]")
    ax_sigma.set_ylabel("σ(E)")
    ax_sigma.set_title(f"Fit #{fit_index}: σ(E)\nE0_fit={E0_fit:.2f}, E0_true={E0_true:.2f}")

    # more stable limits (avoid weird behavior if Sigma_data has tiny range)
    y_min = np.nanmin(np.concatenate([Sigma_data, s_lo, y_mean]))
    y_max = np.nanmax(np.concatenate([Sigma_data, s_hi, y_mean]))
    pad = 0.05 * (y_max - y_min + 1e-12)
    ax_sigma.set_ylim(y_min - pad, y_max + pad)

    ax_sigma.legend(frameon=False, fontsize=8)

    # ================= D(E) ====================
   # for D in D_samples:
   #     ax_barrier.plot(x, D, color="gray", alpha=0.20, lw=0.7)

    ax_barrier.plot(x, D_true, "b-", lw=2, label="True D(E)")
    ax_barrier.plot(x, D_mean, "r--", lw=2, label="Mean D(E)")

    # 68% band
    bands = percentile_bands(D_samples, levels=(0.68,))
    D_lo = bands["lower_68"]
    D_hi = bands["upper_68"]
    ax_barrier.fill_between(x, D_lo, D_hi, color="red", alpha=0.15, label="68% band")

    # NEW: overlay crosses from CSV metrics
    draw_cross_from_metrics(ax_barrier, t_loc, t_ht, t_fwhm, "b")
    draw_cross_from_metrics(ax_barrier, p_loc, p_ht, p_fwhm, "r")

    ax_barrier.set_xlabel("Energy [MeV]")
    ax_barrier.set_ylabel("D(E)")
    def fmt(x):
        return f"{x:.2f}" if np.isfinite(x) else "nan"

    title = (
        f"Fit #{fit_index}: "
        f"[{fmt(t_loc)}, {fmt(t_ht)}, {fmt(t_fwhm)}]; "
        f"[{fmt(p_loc)}, {fmt(p_ht)}, {fmt(p_fwhm)}]"
        )

    ax_barrier.set_title(title)
    #ax_barrier.set_title(f"Fit #{fit_index}: Barrier Dist. (cross from CSV)")
    ax_barrier.set_ylim(np.min(D_true)-200, np.max(D_true) + 1000)
    ax_barrier.legend(frameon=False, fontsize=8)


# -------------------------------------------------------------------
# Pick 5 indices from CSV and plot those fits from HDF5
# -------------------------------------------------------------------
def plot_random_5_from_csv(h5_path, n_samples=5, seed=None):
    rng = np.random.default_rng(seed)
    csv_path = h5_path + ".csv"
    df = pd.read_csv(csv_path)

    # Filtering outlier fits
    cutoff = 1000
    print(f"Before MAE filter (<= {cutoff}): {len(df)} fits")
    if "sigma_MAE" in df.columns:
        df = df[pd.to_numeric(df["barrier_MAE"], errors="coerce") <= cutoff]
    print(f"After MAE filter (<= {cutoff}): {len(df)} fits")

    if "index" not in df.columns:
        raise ValueError("CSV must contain an 'index' column matching HDF5 group keys.")

    # sample rows from CSV
    n = min(int(n_samples), len(df))
    pick = df.sample(n=n, random_state=None if seed is None else int(seed))

    idxs = pick["index"].astype(int).to_numpy()
    print(f"Selected indices from CSV: {idxs}")

    # Prepare figure grid
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    # Open HDF5 once
    with h5py.File(h5_path, "r") as f:
        for row, (fit_index, row_metrics) in enumerate(zip(idxs, pick.to_dict(orient="records"))):
            grp = f[str(int(fit_index))]

            # Load arrays
            x = grp["x_pred"][:]
            y_mean = grp["y_mean"][:]
            y_samples = grp["y_samples"][:]

            E_data = grp["E_data"][:]
            Sigma_data = grp["Sigma_data"][:]
            Sigma_err = grp["Sigma_err"][:]

            E0_true = float(grp.attrs["E0_true"])
            R0_true = float(grp.attrs["R0_true"])
            hw_true = float(grp.attrs["hw_true"])
            E0_fit = float(grp.attrs.get("E0_fit", np.nan))

            # Compute D(E) curves (for plotting)
            D_samples = barrier_from_sigma_samples(x, y_samples)
            D_mean = barrier_from_sigma_samples(x, y_mean.reshape(1, -1))[0]

            sigma_true = sigma(x, E0=E0_true, R0=R0_true, hbarw=hw_true)
            D_true = barrier_from_sigma_samples(x, sigma_true.reshape(1, -1))[0]

            # Pull cross metrics from CSV (NaNs ok)
            t_loc  = float(row_metrics.get("barrier_peak_loc_true", np.nan))
            t_ht   = float(row_metrics.get("barrier_peak_ht_true", np.nan))
            t_fwhm = float(row_metrics.get("barrier_fwhm_true", np.nan))

            p_loc  = float(row_metrics.get("barrier_peak_loc_pred", np.nan))
            p_ht   = float(row_metrics.get("barrier_peak_ht_pred", np.nan))
            p_fwhm = float(row_metrics.get("barrier_fwhm_pred", np.nan))

            # Plot
            ax1, ax2 = axes[row]
            plot_row(
                ax1, ax2,
                x, y_samples, y_mean,
                E_data, Sigma_data, Sigma_err,
                D_samples, D_mean, D_true,
                E0_fit, E0_true,
                int(fit_index),
                t_loc, t_ht, t_fwhm,
                p_loc, p_ht, p_fwhm,
            )

    plt.tight_layout()
    out_png = f"../plots/{os.path.basename(h5_path)}_csv_cross.png"
    plt.savefig(out_png, dpi=300)
    print(f"\nSaved → {out_png}")
    plt.show()


# -------------------------------------------------------------------
# Command-line entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=str, required=True, help="Path to .h5 results")
   # ap.add_argument("--csv", type=str, required=True, help="Path to metrics CSV produced by summary script")
    ap.add_argument("--n_samples", type=int, default=5, help="Number of random fits to plot (default 5)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible sampling")
    args = ap.parse_args()

    plot_random_5_from_csv(args.h5, n_samples=args.n_samples, seed=args.seed)

