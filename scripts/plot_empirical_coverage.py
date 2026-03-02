#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def _finite(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def _mae_rmse_of_resid(resid):
    r = _finite(resid)
    if r.size == 0:
        return np.nan, np.nan
    return float(np.mean(np.abs(r))), float(np.sqrt(np.mean(r**2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="GP metrics CSV (e.g. out.h5.csv)")
   # ap.add_argument("--out", default=None, help="Optional output image path (png/pdf)")
    ap.add_argument("--which", choices=["barrier", "sigma", "both"], default="barrier",
                    help="Which empirical coverage columns to plot")
    ap.add_argument("--bins", type=int, default=40, help="Histogram bins")
    ap.add_argument("--normalize", action="store_true",
                    help="Normalize sigma MAE/RMSE by (hw_true * R0_true^2) using CSV columns hw_true, R0_true.")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    
    # Filtering outlier fits
    cutoff = 20000
    print(f"Before MAE filter (<= {cutoff}): {len(df)} fits")
    if "barrier_RMSE" in df.columns:
        df = df[pd.to_numeric(df["barrier_RMSE"], errors="coerce") <= cutoff]
    else: 
        print("No changes made")
    print(f"After MAE filter (<= {cutoff}): {len(df)} fits")



    # Optional normalization scale for sigma metrics: s = hw_true * R0_true^2
    if args.normalize:
        if ("hw_true" not in df.columns) or ("R0_true" not in df.columns):
            raise ValueError("Need hw_true and R0_true columns in CSV for --normalize")

        s = pd.to_numeric(df["hw_true"], errors="coerce") * (pd.to_numeric(df["R0_true"], errors="coerce") ** 2)
        s = s.to_numpy(dtype=float)
        s[~np.isfinite(s)] = np.nan
        s[s == 0] = np.nan

        # Create normalized sigma metrics (barrier left untouched)
        if "sigma_MAE" in df.columns:
            df["sigma_MAE_norm_hwR02"] = pd.to_numeric(df["sigma_MAE"], errors="coerce").to_numpy(dtype=float) / s
        if "sigma_RMSE" in df.columns:
            df["sigma_RMSE_norm_hwR02"] = pd.to_numeric(df["sigma_RMSE"], errors="coerce").to_numpy(dtype=float) / s
        
        # Create normalized sigma metrics (barrier left untouched)
        if "barrier_MAE" in df.columns:
            df["barrier_MAE_norm_hwR02"] = pd.to_numeric(df["barrier_MAE"], errors="coerce").to_numpy(dtype=float) / s
        if "barrier_RMSE" in df.columns:
            df["barrier_RMSE_norm_hwR02"] = pd.to_numeric(df["barrier_RMSE"], errors="coerce").to_numpy(dtype=float) / s

    # ---------------------------
    # Panel 1: empirical coverage
    # ---------------------------
    def collect_cov(prefix: str):
        cols = [c for c in df.columns if c.startswith(prefix)]
        out = []
        for c in cols:
            try:
                pct = int(c.split("_")[-1])
                level = pct / 100.0
            except Exception:
                continue
            vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
            vals = _finite(vals)
            if vals.size:
                out.append((level, float(np.mean(vals))))
        out.sort(key=lambda t: t[0])
        return out

    cov_sigma = collect_cov("sigma_emp_cov_")
    cov_bar   = collect_cov("barrier_emp_cov_")

    # ---------------------------
    # Panels 2 & 3: histograms
    # ---------------------------
    if args.normalize:
        sigma_rmse = _finite(pd.to_numeric(df.get("sigma_RMSE_norm_hwR02"), errors="coerce").to_numpy())
        sigma_mae  = _finite(pd.to_numeric(df.get("sigma_MAE_norm_hwR02"), errors="coerce").to_numpy())
        barrier_rmse = _finite(pd.to_numeric(df.get("barrier_RMSE_norm_hwR02"), errors="coerce").to_numpy())
        barrier_mae  = _finite(pd.to_numeric(df.get("barrier_MAE_norm_hwR02"), errors="coerce").to_numpy())
    else:
        sigma_rmse = _finite(pd.to_numeric(df.get("sigma_RMSE"), errors="coerce").to_numpy())
        sigma_mae  = _finite(pd.to_numeric(df.get("sigma_MAE"), errors="coerce").to_numpy())
        barrier_rmse = _finite(pd.to_numeric(df.get("barrier_RMSE"), errors="coerce").to_numpy())
        barrier_mae  = _finite(pd.to_numeric(df.get("barrier_MAE"), errors="coerce").to_numpy())

    # choose which error distribution to histogram: barrier by default
    if args.which == "sigma":
        rmse_vals, mae_vals = sigma_rmse, sigma_mae
        if args.normalize:
            rmse_label, mae_label = "σ RMSE / (hw·R0²) (per fit)", "σ MAE / (hw·R0²) (per fit)"
        else:
            rmse_label, mae_label = "σ RMSE (per fit)", "σ MAE (per fit)"
    elif args.which == "both":
        # concatenate for histograms (still per-fit values, mixed)
        rmse_vals = np.concatenate([sigma_rmse, barrier_rmse]) if (sigma_rmse.size and barrier_rmse.size) else (sigma_rmse if sigma_rmse.size else barrier_rmse)
        mae_vals  = np.concatenate([sigma_mae,  barrier_mae ]) if (sigma_mae.size  and barrier_mae.size ) else (sigma_mae if sigma_mae.size else barrier_mae)
        rmse_label, mae_label = "RMSE (σ + D mixed)", "MAE (σ + D mixed)"
    else:
        rmse_vals, mae_vals = barrier_rmse, barrier_mae
        if args.normalize:
            rmse_label, mae_label = "Barrier RMSE / (hw·R0²) (per fit)", "Barrier MAE / (hw·R0²) (per fit)"
        else:
            rmse_label, mae_label = "Barrier RMSE (per fit)", "Barrier MAE (per fit)"


    # ---------------------------
    # Barrier peak residual summaries (for titles)
    # ---------------------------
    loc_mae, loc_rmse = _mae_rmse_of_resid(pd.to_numeric(df.get("barrier_peak_loc_resid"), errors="coerce").to_numpy())
    ht_mae,  ht_rmse  = _mae_rmse_of_resid(pd.to_numeric(df.get("barrier_peak_ht_resid"), errors="coerce").to_numpy())
    fw_mae,  fw_rmse  = _mae_rmse_of_resid(pd.to_numeric(df.get("barrier_fwhm_resid"), errors="coerce").to_numpy())

    # ---------------------------
    # Make 3-panel figure
    # ---------------------------
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 15))

    # Panel 1: coverage
    ax = axes[0]

    def plot_cov(series, label):
        if not series:
            return
        levels = [t[0] for t in series]
        means  = [t[1] for t in series]
        ax.plot(levels, means, marker="o", label=label)

    if args.which in ("sigma", "both"):
        plot_cov(cov_sigma, "σ empirical coverage")
    if args.which in ("barrier", "both"):
        plot_cov(cov_bar, "Barrier empirical coverage")

    # ideal y=x for available levels
    # --- true diagonal reference: y = x / 100 ---
    x_ref = np.linspace(0.0, 1.0, 4)
    y_ref = x_ref
    ax.plot(x_ref, y_ref, linestyle="--", linewidth=1.5, label="ideal (y = x)")
    ax.set_xlabel("Nominal credibility level")
    ax.set_ylabel("Average empirical coverage")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 1.0)

    ax.set_title("Empirical coverage (avg over fits)")
    ax.legend(frameon=False, fontsize=8)

    # Panel 2: RMSE histogram
    ax = axes[1]
    if rmse_vals.size:
        ax.hist(rmse_vals, bins=args.bins)
    ax.set_xlabel(rmse_label)
    ax.set_ylabel("Count")
    ax.set_title(
        "Full Curve RMSE Distribution\n"
        f"Wong Barrier RMSE: Location={loc_rmse:.3g} MeV,\n Height={ht_rmse:.3g} MeV, FWHM={fw_rmse:.3g} MeV"
    )

    # Panel 3: MAE histogram
    ax = axes[2]
    if mae_vals.size:
        ax.hist(mae_vals, bins=args.bins)
    ax.set_xlabel(mae_label)
    ax.set_ylabel("Count")
    ax.set_title(
        "Full Curve MAE Distribution\n"
        f"Wong Barrier MAE: Location={loc_mae:.3g} MeV,\n Height={ht_mae:.3g} MeV, FWHM={fw_mae:.3g} MeV"
    )

    plt.tight_layout()

    prefix = "norm_" if args.normalize else ""
    out_pdf = f"/mnt/home/philipaa/research/barrier-extraction/plots/{prefix}EC_1.0_20_{os.path.basename(args.csv)}.pdf"
    plt.savefig(out_pdf, dpi=300)
    print(f"Saved -> {out_pdf}")
    

if __name__ == "__main__":
    main()
