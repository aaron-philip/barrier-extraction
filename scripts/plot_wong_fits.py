#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import scipy.stats
import findiff as fd
import sys

if "../src" not in sys.path:
    sys.path.append("../src")  

from utils import barrier_dist
from wong import sigma

# -------------------------------------------------------------------
# Main plotting utility
# -------------------------------------------------------------------
def plot_wong_sanity(data_path, csv_path, n_samples=5, levels=(0.68, 0.9, 0.95), seed=42):
    np.random.seed(seed)
    npz = np.load(data_path, allow_pickle=True)
    df = pd.read_csv(csv_path)

    E_all = npz["E"]
    Sigma_all = npz["Sigma"]
    Sigma_err_all = npz["Sigma_err"]
    params_all = npz["params"]

    total = len(E_all)
    idxs = np.random.choice(total, size=min(n_samples, total), replace=False)

    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, i in enumerate(idxs):
        E = np.array(E_all[i], dtype=float)
        Sigma_true = np.array(Sigma_all[i], dtype=float)
        Sigma_err = np.array(Sigma_err_all[i], dtype=float)
        true_params = params_all[i]


        fit = df.iloc[i]
        E0_fit, R0_fit, hw_fit = fit["E0_fit"], fit["R0_fit"], fit["hw_fit"]
        print(true_params)
        print(E0_fit, R0_fit, hw_fit)

        # ---- Predict curves ----
        x_pred = np.arange(max(0, E.min() - 5), E.max() + 5, 0.2)
        y_pred = sigma(x_pred, E0_fit, R0_fit, hw_fit)
        y_true = sigma(x_pred, *true_params)

        # ---- Approximate uncertainty ----
        eps = 1e-5
        popt = np.array([E0_fit, R0_fit, hw_fit])
        J = np.zeros((len(x_pred), len(popt)))
        pcov = np.eye(3) * 0.1  # dummy variance if unavailable
        for j in range(len(popt)):
            dp = np.zeros_like(popt)
            dp[j] = eps
            J[:, j] = (sigma(x_pred, *(popt + dp)) - sigma(x_pred, *(popt - dp))) / (2 * eps)
        pred_var = np.einsum("ij,jk,ik->i", J, pcov, J)
        pred_std = np.sqrt(np.maximum(pred_var, 0))

        # ---- Barrier distributions ----
        D_pred = barrier_dist(x_pred, y_pred)
        D_true = barrier_dist(x_pred, y_true)
        D_up = barrier_dist(x_pred, y_pred + pred_std)
        D_down = barrier_dist(x_pred, y_pred - pred_std)
        D_std = 0.5 * np.abs(D_up - D_down)

        # ---- Plot σ(E) ----
        ax1 = axes[row, 0]
        ax1.errorbar(E, Sigma_true, yerr=Sigma_err, fmt="o", color="black", alpha=0.6, label="Data")
        ax1.plot(x_pred, y_true, "k--", lw=1.5, label="True σ(E)")
        ax1.plot(x_pred, y_pred, "r-", lw=1.5, label="Fit σ(E)")
        for alpha in levels:
            z = scipy.stats.norm.ppf(0.5 + alpha / 2.0)
            ax1.fill_between(x_pred, y_pred - z*pred_std, y_pred + z*pred_std,
                             alpha=0.15, label=f"{int(alpha*100)}% CI")
        ax1.set_xlabel("E [MeV]")
        ax1.set_ylabel("σ(E) [arb.]")
        ax1.set_title(f"Cross Section Fit #{i}")
        ax1.legend(frameon=False)

        # ---- Plot D(E) ----
        ax2 = axes[row, 1]
        ax2.plot(x_pred, D_true, "k--", lw=1.5, label="True D(E)")
        ax2.plot(x_pred, D_pred, "r-", lw=1.5, label="Fit D(E)")
        for alpha in levels:
            z = scipy.stats.norm.ppf(0.5 + alpha / 2.0)
            ax2.fill_between(x_pred, D_pred - z*D_std, D_pred + z*D_std,
                             alpha=0.15, label=f"{int(alpha*100)}% CI")
        ax2.set_xlabel("E [MeV]")
        ax2.set_ylabel("D(E) [arb.]")
        ax2.set_title(f"Barrier Distribution #{i}")
        ax2.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("wong_fit_sanity.png", dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to .npz dataset used for training")
    ap.add_argument("--fits", required=True, help="Path to .csv file of Wong fit results")
    ap.add_argument("--n_samples", type=int, default=5)
    args = ap.parse_args()

    plot_wong_sanity(args.data, args.fits, n_samples=args.n_samples)

