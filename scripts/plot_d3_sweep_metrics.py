#!/usr/bin/env python3
"""Plot d3 penalty sweep coverage and calibration metrics."""

import argparse
import colorsys
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "d3_sweep_single_curve"
PENALTY_ORDER = ("1", "1e1", "1e2")
PENALTY_LABELS = {
    "1": r"$\lambda_{D3}=1$",
    "1e1": r"$\lambda_{D3}=10$",
    "1e2": r"$\lambda_{D3}=100$",
}
PENALTY_BASE_COLORS = {
    "1": "tab:green",
    "1e1": "tab:blue",
    "1e2": "purple",
}
FILENAME_RE = re.compile(
    r"(?P<dataset>.+)_curve0_bnn_soc_d3lam_(?P<penalty>1e2|1e1|1)\.h5\.csv$"
)


def setup_style():
    plt.style.use(["science", "notebook", "ieee", "grid"])
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.labelsize": 24,
            "axes.titlesize": 30,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 14,
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.2,
        }
    )


def get_shades(base_color, n, lightness=(0.32, 0.76)):
    r, g, b = to_rgb(base_color)
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    return [
        colorsys.hls_to_rgb(h, float(lightness_value), s)
        for lightness_value in np.linspace(lightness[0], lightness[1], n)
    ]


def dataset_sort_key(dataset):
    if dataset.startswith("res_"):
        match = re.match(r"res_(?P<res>[0-9.]+)MeV_noise(?P<noise>[0-9.]+)", dataset)
        if match:
            return (
                0,
                float(match.group("noise")),
                float(match.group("res")),
                dataset,
            )
    if dataset.startswith("cv"):
        match = re.match(r"cv(?P<cv>[0-9.]+)_spacing(?P<spacing>[0-9.]+)", dataset)
        if match:
            return (
                1,
                float(match.group("spacing")),
                float(match.group("cv")),
                dataset,
            )
    return (2, dataset)


def discover_csvs(results_dir):
    records = []
    for path in sorted(results_dir.glob("*.h5.csv")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        records.append(
            {
                "path": path,
                "dataset": match.group("dataset"),
                "penalty": match.group("penalty"),
            }
        )
    if not records:
        raise FileNotFoundError(f"No d3 sweep CSVs found in {results_dir}")
    return records


def coverage_columns(df, prefix="barrier_emp_cov_"):
    cols = []
    for col in df.columns:
        if not col.startswith(prefix):
            continue
        try:
            nominal = int(col.rsplit("_", 1)[-1])
        except ValueError:
            continue
        cols.append((nominal, col))
    return sorted(cols)


def load_summary(record):
    df = pd.read_csv(record["path"])
    numeric_cols = [
        col
        for col in df.columns
        if any(token in col for token in ("MAE", "RMSE", "cov", "resid", "true", "pred"))
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cov_cols = coverage_columns(df)
    if not cov_cols:
        raise ValueError(f"No barrier empirical coverage columns in {record['path']}")

    levels = np.array([nominal for nominal, _ in cov_cols], dtype=float)
    empirical = np.array([df[col].mean() * 100.0 for _, col in cov_cols], dtype=float)
    coverage_abs_error = np.abs(empirical - levels)
    barrier_mae = float(df["barrier_MAE"].mean())

    return {
        **record,
        "levels": levels,
        "empirical": empirical,
        "coverage_abs_error": coverage_abs_error,
        "coverage_mae": float(np.nanmean(coverage_abs_error)),
        "barrier_mae": barrier_mae,
    }


def build_color_map(summaries):
    datasets = sorted({s["dataset"] for s in summaries}, key=dataset_sort_key)
    colors = {}
    for penalty in PENALTY_ORDER:
        shades = get_shades(PENALTY_BASE_COLORS[penalty], len(datasets))
        for dataset, shade in zip(datasets, shades):
            colors[(penalty, dataset)] = shade
    return colors


def plot_coverage(summaries, colors, out_path, penalty=None):
    filtered = [s for s in summaries if penalty is None or s["penalty"] == penalty]
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    ax.plot([0, 100], [0, 100], "k--", alpha=0.55, linewidth=1.5, label="Ideal")

    for summary in sorted(filtered, key=lambda s: (PENALTY_ORDER.index(s["penalty"]), dataset_sort_key(s["dataset"]))):
        label = None
        ax.plot(
            summary["levels"],
            summary["empirical"],
            color=colors[(summary["penalty"], summary["dataset"])],
            alpha=0.78,
            marker="o",
            markersize=3.8,
            linewidth=1.35,
            label=label,
        )

    handles = []
    if penalty is None:
        for penalty_value in PENALTY_ORDER:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=PENALTY_BASE_COLORS[penalty_value],
                    lw=2.5,
                    label=PENALTY_LABELS[penalty_value],
                )
            )
    else:
        handles.append(
            Line2D(
                [0],
                [0],
                color=PENALTY_BASE_COLORS[penalty],
                lw=2.5,
                label=PENALTY_LABELS[penalty],
            )
        )
    handles.insert(0, Line2D([0], [0], color="k", linestyle="--", lw=1.5, label="Ideal"))

    ax.set_xlabel("Nominal Credibility (%)")
    ax.set_ylabel("Empirical Coverage (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    title_suffix = "All D3 Penalties" if penalty is None else PENALTY_LABELS[penalty]
    ax.set_title(rf"$D_f$ Empirical Coverage: {title_suffix}", pad=15)
    ax.legend(handles=handles, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(summaries, colors, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    zero_floor = 0.05
    markers = {
        0: "o",
        25: "s",
        50: "^",
        68: "D",
        90: "P",
        95: "X",
        100: "v",
    }

    for summary in sorted(summaries, key=lambda s: (PENALTY_ORDER.index(s["penalty"]), dataset_sort_key(s["dataset"]))):
        color = colors[(summary["penalty"], summary["dataset"])]
        for level, error in zip(summary["levels"], summary["coverage_abs_error"]):
            plot_error = max(float(error), zero_floor)
            ax.scatter(
                summary["barrier_mae"],
                plot_error,
                color=color,
                marker=markers.get(int(level), "o"),
                s=48,
                alpha=0.78,
                edgecolors="none",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(zero_floor, color="0.45", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"Barrier MAE")
    ax.set_ylabel("Coverage Absolute Error (%)")
    ax.set_title(r"$D_f$ MAE vs. Coverage Error", pad=15)

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PENALTY_BASE_COLORS[penalty],
            markersize=8,
            label=PENALTY_LABELS[penalty],
        )
        for penalty in PENALTY_ORDER
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=markers.get(level, "o"),
            color="0.25",
            linestyle="none",
            markersize=7,
            label=f"{level}%",
        )
        for level in (0, 25, 50, 68, 90, 95, 100)
    ]
    first_legend = ax.legend(handles=color_handles, frameon=False, loc="upper left")
    ax.add_artist(first_legend)
    ax.legend(
        handles=marker_handles,
        frameon=False,
        loc="lower right",
        title="Nominal",
        fontsize=10,
        title_fontsize=11,
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mean_scatter(summaries, colors, out_path):
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for summary in sorted(summaries, key=lambda s: (PENALTY_ORDER.index(s["penalty"]), dataset_sort_key(s["dataset"]))):
        ax.scatter(
            summary["barrier_mae"],
            summary["coverage_mae"],
            color=colors[(summary["penalty"], summary["dataset"])],
            s=56,
            alpha=0.82,
            edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Barrier MAE")
    ax.set_ylabel("Coverage MAE (%)")
    ax.set_title(r"$D_f$ MAE vs. Mean Coverage MAE", pad=15)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PENALTY_BASE_COLORS[penalty],
            markersize=8,
            label=PENALTY_LABELS[penalty],
        )
        for penalty in PENALTY_ORDER
    ]
    ax.legend(handles=handles, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    setup_style()
    out_dir = args.out_dir or (args.results_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = discover_csvs(args.results_dir)
    summaries = [load_summary(record) for record in records]
    colors = build_color_map(summaries)

    plot_coverage(summaries, colors, out_dir / "d3_barrier_empirical_coverage_all.png")
    for penalty in PENALTY_ORDER:
        plot_coverage(
            summaries,
            colors,
            out_dir / f"d3_barrier_empirical_coverage_d3lam_{penalty}.png",
            penalty=penalty,
        )

    plot_scatter(
        summaries,
        colors,
        out_dir / "d3_barrier_mae_vs_coverage_error_by_interval.png",
    )
    plot_mean_scatter(
        summaries,
        colors,
        out_dir / "d3_barrier_mae_vs_mean_coverage_mae.png",
    )

    print(f"Loaded {len(summaries)} CSV files from {args.results_dir}")
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
