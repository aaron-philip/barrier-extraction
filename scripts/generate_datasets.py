#!/usr/bin/env python3
import os
import sys
if "/mnt/research/ascsn/aaron-philip/barrier-extraction/src" not in sys.path:
    sys.path.append("/mnt/research/ascsn/aaron-philip/barrier-extraction/src")  
from generator import generate_datasets

# Base output directory
OUTDIR = "/mnt/home/philipaa/scratch/wong-paper-datasets"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# Experiment 1: Resolution vs 2 noise ratios
# -----------------------------
factors = [5, 10, 20, 40]
for f in factors:
    # Equivalent n_points as before
    n_points = int((100 - 50) / 0.2 / f)  
    for noise_ratio in [0.1, 0.01]:
        tag = f"res{f}_noise{noise_ratio}.npz"
        out_file = os.path.join(OUTDIR, tag)
        generate_datasets(
            n_curves=100,        
            n_noise=100,         
            noise_ratio=noise_ratio,
            dev_points=0.0,      
            n_points=n_points,
            out_file=out_file,
            seed=42
        )

# -----------------------------
# Experiment 2: Irregularities with mean spacing 0.5 MeV
# -----------------------------
cv_values = [0.0, 0.25, 0.5, 0.75]
mean_spacing = 0.5
E_min, E_max = 50, 80   # match your E_fine[:stop_idx]
for cv in cv_values:
    n_points = int((E_max - E_min) / mean_spacing)
    tag = f"cv{cv:.2f}_spacing0.5.npz"
    out_file = os.path.join(OUTDIR, tag)
    generate_datasets(
        n_curves=100,
        n_noise=100,
        noise_ratio=0.01,   
        dev_points=cv,      
        n_points=n_points,
        out_file=out_file,
        seed=0
    )

# -----------------------------
# Experiment 3: Irregularities with mean spacing 1.0 MeV
# -----------------------------
cv_values = [0.0, 0.25, 0.5, 0.75]
mean_spacing = 1.0
E_min, E_max = 50, 80
for cv in cv_values:
    n_points = int((E_max - E_min) / mean_spacing)
    tag = f"cv{cv:.2f}_spacing1.0.npz"
    out_file = os.path.join(OUTDIR, tag)
    generate_datasets(
        n_curves=100,
        n_noise=100,
        noise_ratio=0.01,
        dev_points=cv,
        n_points=n_points,
        out_file=out_file,
        seed=0
    )
