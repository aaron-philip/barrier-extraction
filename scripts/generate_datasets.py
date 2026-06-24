#!/usr/bin/env python3
"""This script generated all 16 Wong datasets each with 1000 parameter samples x 10 noise samples. 
The datasets are saved in the OUTDIR directory as .npz files."""
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))
from generator import generate_datasets

# Base output directory
OUTDIR = "/mnt/home/philipaa/scratch/barrier-paper-datasets"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# Experiment 1: Resolution vs. 2 noise ratios
# ============================================================
# These correspond to mean spacings of 0.25, 0.5, 1.0, and 2.0 MeV
mean_spacings = [0.25, 0.5, 1.0, 2.0]
for spacing in mean_spacings:
    for noise_ratio in [0.02, 0.01]:
        tag = f"res_{spacing:.2f}MeV_noise{noise_ratio}.npz"
        out_file = os.path.join(OUTDIR, tag)
        generate_datasets(
            n_curves=100,
            n_noise=10,
            noise_ratio=noise_ratio,
            cv=0.0,               # perfectly regular grid
            mean_spacing=spacing,
            out_file=out_file,
            seed=42
        )

# ============================================================
# Experiment 2: Irregularities with mean spacing 0.5 MeV
# ============================================================
cv_values = [0.0, 0.25, 0.5, 0.75]
mean_spacing = 0.5
for cv in cv_values:
    tag = f"cv{cv:.2f}_spacing0.5.npz"
    out_file = os.path.join(OUTDIR, tag)
    generate_datasets(
        n_curves=100,
        n_noise=10,
        noise_ratio=0.01,
        cv=cv,
        mean_spacing=mean_spacing,
        out_file=out_file,
        seed=0
    )

# ============================================================
# Experiment 3: Irregularities with mean spacing 1.0 MeV
# ============================================================
cv_values = [0.0, 0.25, 0.5, 0.75]
mean_spacing = 1.0
for cv in cv_values:
    tag = f"cv{cv:.2f}_spacing1.0.npz"
    out_file = os.path.join(OUTDIR, tag)
    generate_datasets(
        n_curves=100,
        n_noise=10,
        noise_ratio=0.01,
        cv=cv,
        mean_spacing=mean_spacing,
        out_file=out_file,
        seed=0
    )
