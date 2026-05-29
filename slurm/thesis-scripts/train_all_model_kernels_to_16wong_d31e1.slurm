#!/bin/bash --login
#SBATCH --job-name=bnn_nb_models_d3lam1e1
#SBATCH --output=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_nb_models_d3lam1e1_%A_%a.out
#SBATCH --error=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_nb_models_d3lam1e1_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=frib-nodes
#SBATCH --array=0-15

# Train the four AutoBNN model types used by notebooks/BNNDataAnalysis.ipynb
# on the four datasets in its final two comparison figures, with d3 lambda=1e1.
#
# Array mapping:
#   model_id   = task_id / 4   in [0..3]
#   dataset_id = task_id % 4   in [0..3]
#
# Submit:
#   sbatch slurm/thesis-scripts/train_all_model_kernels_to_16wong_d31e1.slurm
#
# Useful overrides:
#   sbatch --export=ALL,NUM_DRAWS=100,RESULT_DIR=/path/to/results \
#     slurm/thesis-scripts/train_all_model_kernels_to_16wong_d31e1.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

DATA_DIR="${DATA_DIR:-/mnt/home/philipaa/scratch/barrier-paper-datasets}"
RESULT_DIR="${RESULT_DIR:-/mnt/home/philipaa/scratch/barrier-paper-results/bnn_data_analysis_models_d3lam1e1}"
LOG_DIR="${LOG_DIR:-/mnt/home/philipaa/scratch/barrier-paper-logs}"
ENV_PATH="${ENV_PATH:-$HOME/envs/autobnn-test-fork}"
DATAFILE="${DATAFILE:-bnn_data_analysis_4curves.h5}"
STRIDE="${STRIDE:-1}"
RANDOM_STATE="${RANDOM_STATE:-0}"
NUM_DRAWS="${NUM_DRAWS:-50}"
NUM_CHAINS="${NUM_CHAINS:-4}"
NUM_WARMUP="${NUM_WARMUP:-1000}"
D3_LAMBDA_SPACE="${D3_LAMBDA_SPACE:-physical}"

mkdir -p "$DATA_DIR" "$RESULT_DIR" "$LOG_DIR"

if [ -f "$DATAFILE" ]; then
  DATA_PATH="$DATAFILE"
else
  DATA_PATH="$DATA_DIR/$DATAFILE"
fi

source ~/.bashrc
conda deactivate
conda activate "$ENV_PATH"

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

if [ ! -f "$DATA_PATH" ]; then
  LOCKDIR="${DATA_PATH}.lock"
  until mkdir "$LOCKDIR" 2>/dev/null; do
    echo "Waiting for dataset writer lock: $LOCKDIR"
    sleep 10
  done
  trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

  if [ ! -f "$DATA_PATH" ]; then
    echo "Creating BNNDataAnalysis four-curve H5: $DATA_PATH"
    PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" python - "$DATA_PATH" "$REPO_DIR/data" <<'PY'
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import curve_fit

out_file = Path(sys.argv[1])
data_dir = Path(sys.argv[2])
exp_dir = data_dir.parents[1] / "barrier-distribution-sample" / "16O208Pb_exp"

def sigma(E, E0, R0, hbarw):
    return 10.0 * (0.5 * R0**2 * hbarw / E) * np.log(
        1.0 + np.exp(2.0 * np.pi * (E - E0) / hbarw)
    )

exp_frames = []
for file_name in ("expMort.dat", "expDas.dat"):
    arr = np.loadtxt(exp_dir / file_name)
    if arr[0, 0] > arr[-1, 0]:
        arr = arr[::-1]
    exp_frames.append(arr)

E_data = exp_frames[0][:, 0].astype(float)
Sigma_data = exp_frames[0][:, 1].astype(float)
Sigma_err = exp_frames[0][:, 2].astype(float)
popt, _ = curve_fit(
    sigma,
    E_data,
    Sigma_data,
    p0=[70.0, 11.0, 4.0],
    sigma=Sigma_err,
    absolute_sigma=True,
    bounds=([0.0, 8.0, 0.0], [np.inf, 15.0, np.inf]),
)

datasets = [
    ("E_20_1e-1_log", "E_tr_20.npy", "y_tr_noisy_20_1e-1.npy", "var_tr_20_1e-1.npy"),
    ("E_20_1e-2_log", "E_tr_20.npy", "y_tr_noisy_20_1e-2.npy", "var_tr_20_1e-2.npy"),
    ("E_cv_05_log", "E_tr_cv_0.50.npy", "y_tr_noisy_cv_0.50.npy", "var_tr_cv_0.50.npy"),
    ("E_cv_05_1MEV_log", "E_tr_cv_0.50_1MEV.npy", "y_tr_noisy_cv_0.50_1MEV.npy", "var_tr_cv_0.50_1MEV.npy"),
]

E_chunks = []
S_chunks = []
Se_chunks = []
Strue_chunks = []
params = []
names = []

for name, e_file, sigma_file, var_file in datasets:
    E = np.load(data_dir / e_file).astype(float).ravel()
    S = np.load(data_dir / sigma_file).astype(float).ravel()
    var = np.load(data_dir / var_file).astype(float).ravel()

    n = min(E.size, S.size, var.size)
    E, S, var = E[:n], S[:n], var[:n]
    good = np.isfinite(E) & np.isfinite(S) & np.isfinite(var) & (S > 0.0)
    E, S, var = E[good], S[good], var[good]
    order = np.argsort(E)
    E, S, var = E[order], S[order], var[order]

    E_chunks.append(E)
    S_chunks.append(S)
    Se_chunks.append(np.sqrt(np.maximum(var, 0.0)))
    Strue_chunks.append(sigma(E, *popt))
    params.append(popt)
    names.append(name)

lengths = np.array([len(x) for x in E_chunks], dtype=np.int64)
idx_ptr = np.zeros(len(lengths) + 1, dtype=np.int64)
idx_ptr[1:] = np.cumsum(lengths)

out_file.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(out_file, "w") as f:
    f.create_dataset("E_cat", data=np.concatenate(E_chunks), compression="gzip")
    f.create_dataset("Sigma_cat", data=np.concatenate(S_chunks), compression="gzip")
    f.create_dataset("Sigma_err_cat", data=np.concatenate(Se_chunks), compression="gzip")
    f.create_dataset("Sigma_true_cat", data=np.concatenate(Strue_chunks), compression="gzip")
    f.create_dataset("idx_ptr", data=idx_ptr)
    f.create_dataset("params", data=np.asarray(params, dtype=float))
    f.create_dataset("curve_names", data=np.asarray(names, dtype=h5py.string_dtype()))
    f.attrs.update(
        noise_ratio=-1.0,
        cv=-1.0,
        mean_spacing=-1.0,
        n_curves=len(names),
        n_noise=1,
        source="notebooks/BNNDataAnalysis.ipynb",
    )

print(f"Wrote {len(names)} curves to {out_file}")
PY
  fi

  rmdir "$LOCKDIR" 2>/dev/null || true
  trap - EXIT
fi

if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: Dataset not found: $DATA_PATH"
  exit 1
fi

DATASET_NAMES=(
  "E_20_1e-1_log"
  "E_20_1e-2_log"
  "E_cv_05_log"
  "E_cv_05_1MEV_log"
)

MODELS=(
  "changepoint_of_safe_products"
  "sum_of_changepoints"
  "sum_of_products"
  "sum_of_stumps"
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
MODEL_ID=$(( TASK_ID / 4 ))
DATASET_ID=$(( TASK_ID % 4 ))

if [ "$MODEL_ID" -ge "${#MODELS[@]}" ]; then
  echo "ERROR: MODEL_ID=$MODEL_ID out of range for MODELS."
  exit 1
fi
if [ "$DATASET_ID" -ge "${#DATASET_NAMES[@]}" ]; then
  echo "ERROR: DATASET_ID=$DATASET_ID out of range for DATASET_NAMES."
  exit 1
fi

MODEL="${MODELS[$MODEL_ID]}"
DATASET_NAME="${DATASET_NAMES[$DATASET_ID]}"
H5_BASENAME="$(basename "$DATA_PATH" .h5)"
OUTFILE="$RESULT_DIR/${H5_BASENAME}_${DATASET_NAME}_${MODEL}_bnn_d3lam1e1.h5"

echo "Node:            $(hostname)"
echo "Repo:            $REPO_DIR"
echo "Dataset H5:      $DATA_PATH"
echo "Dataset ID:      $DATASET_ID ($DATASET_NAME)"
echo "Model ID:        $MODEL_ID ($MODEL)"
echo "Stride:          $STRIDE"
echo "Random state:    $RANDOM_STATE"
echo "Num chains:      $NUM_CHAINS"
echo "Num draws:       $NUM_DRAWS"
echo "Num warmup:      $NUM_WARMUP"
echo "d3 lambda:       10.0"
echo "d3 lambda space: $D3_LAMBDA_SPACE"
echo "Output:          $OUTFILE"

python "$REPO_DIR/scripts/train_bnns.py" \
  --data "$DATA_PATH" \
  --index "$DATASET_ID" \
  --stride "$STRIDE" \
  --random_state "$RANDOM_STATE" \
  --out_h5 "$OUTFILE" \
  --num_draws "$NUM_DRAWS" \
  --num_chains "$NUM_CHAINS" \
  --width 50 \
  --num_warmup "$NUM_WARMUP" \
  --likelihood_model "normal_likelihood_lognormal_noise" \
  --kernel "$MODEL" \
  --d3_lambda 10.0 \
  --d3_lambda_space "$D3_LAMBDA_SPACE"

python "$REPO_DIR/scripts/merge_bnn_parts.py" --out_h5 "$OUTFILE"

echo "Done dataset_id=$DATASET_ID model_id=$MODEL_ID -> $OUTFILE"
