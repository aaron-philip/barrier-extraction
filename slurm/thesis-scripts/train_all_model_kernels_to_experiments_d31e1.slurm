#!/bin/bash --login
#SBATCH --job-name=bnn_exp7_models_d3lam1e1
#SBATCH --output=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_exp7_models_d3lam1e1_%A_%a.out
#SBATCH --error=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_exp7_models_d3lam1e1_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=frib-nodes
#SBATCH --array=0-34

# Train one BNN for every notebook experimental curve and model type with
# third-derivative penalty lambda = 1e1.
#
# Array mapping:
#   model_id = task_id / 7   in [0..4]
#   curve_id = task_id % 7   in [0..6]
#
# Submit:
#   sbatch slurm/thesis-scripts/train_all_model_kernels_to_experiments_d31e1.slurm
#
# Optional overrides:
#   sbatch --export=ALL,DATAFILE=/path/to/experiments7.h5,ENV_PATH=~/envs/autobnn-test-fork \
#     slurm/thesis-scripts/train_all_model_kernels_to_experiments_d31e1.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/home/philipaa/research/barrier-extraction}"
export REPO_DIR

DATA_DIR="${DATA_DIR:-/mnt/home/philipaa/scratch/barrier-paper-datasets}"
RESULT_DIR="${RESULT_DIR:-/mnt/home/philipaa/scratch/barrier-paper-results/experiments7_models_d3lam1e1}"
LOG_DIR="${LOG_DIR:-/mnt/home/philipaa/scratch/barrier-paper-logs}"
ENV_PATH="${ENV_PATH:-$HOME/envs/autobnn-test-fork}"

DATAFILE="${DATAFILE:-experiments7.h5}"
STRIDE="${STRIDE:-1}"
RANDOM_STATE="${RANDOM_STATE:-0}"
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

if [ ! -f "$DATA_PATH" ] && [ "$(basename "$DATA_PATH")" = "experiments7.h5" ]; then
  LOCKDIR="${DATA_PATH}.lock"
  until mkdir "$LOCKDIR" 2>/dev/null; do
    echo "Waiting for dataset writer lock: $LOCKDIR"
    sleep 10
  done
  trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

  if [ ! -f "$DATA_PATH" ]; then
    echo "Creating seven-curve experimental H5: $DATA_PATH"
    python - "$DATA_PATH" "$REPO_DIR/data" <<'PY'
import sys
from pathlib import Path

import h5py
import numpy as np

out_file = Path(sys.argv[1])
data_dir = Path(sys.argv[2])

datasets = [
    ("Aljuwaire1984", "Aljuwairetal.1984"),
    ("Dasgupta2007", "Dasguptaetal.2007"),
    ("Jiang2010", "Jiangetal.2010"),
    ("Montagnoli2012", "Montagnolietal.2012"),
    ("Morton1999", "Mortonetal.1999"),
    ("Stefanini2009", "Stefaninietal.2009"),
    ("Trotta2001", "Trottaetal.2001"),
]

e_chunks = []
s_chunks = []
se_chunks = []
params = []
names = []

for short_name, stem in datasets:
    e = np.load(data_dir / f"E_{stem}.npy").astype(float).ravel()
    s = np.load(data_dir / f"sigma_{stem}.npy").astype(float).ravel()
    se = np.load(data_dir / f"error_{stem}.npy").astype(float).ravel()

    n = min(e.size, s.size, se.size)
    e, s, se = e[:n], s[:n], se[:n]
    good = np.isfinite(e) & np.isfinite(s)
    e, s, se = e[good], s[good], se[good]
    se = np.nan_to_num(se, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(e)

    e_chunks.append(e[order])
    s_chunks.append(s[order])
    se_chunks.append(se[order])
    params.append([0.0, 0.0, 0.0])
    names.append(short_name)

lengths = np.array([len(x) for x in e_chunks], dtype=np.int64)
idx_ptr = np.zeros(len(lengths) + 1, dtype=np.int64)
idx_ptr[1:] = np.cumsum(lengths)

out_file.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(out_file, "w") as f:
    f.create_dataset("E_cat", data=np.concatenate(e_chunks), compression="gzip")
    f.create_dataset("Sigma_cat", data=np.concatenate(s_chunks), compression="gzip")
    f.create_dataset("Sigma_err_cat", data=np.concatenate(se_chunks), compression="gzip")
    f.create_dataset("Sigma_true_cat", data=np.concatenate(s_chunks), compression="gzip")
    f.create_dataset("idx_ptr", data=idx_ptr)
    f.create_dataset("params", data=np.asarray(params, dtype=float))
    f.create_dataset("curve_names", data=np.asarray(names, dtype=h5py.string_dtype()))
    f.attrs.update(noise_ratio=0.0, cv=0.0, mean_spacing=-1.0, n_curves=len(names), n_noise=1)

print(f"Wrote {len(names)} curves and {int(lengths.sum())} points to {out_file}")
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
  "Aljuwaire1984"
  "Dasgupta2007"
  "Jiang2010"
  "Montagnoli2012"
  "Morton1999"
  "Stefanini2009"
  "Trotta2001"
)

MODELS=(
  "changepoint_of_safe_products"
  "sum_of_changepoints"
  "sum_of_products"
  "sum_of_stumps_and_products"
  "sum_of_stumps"
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
MODEL_ID=$(( TASK_ID / 7 ))
CURVE_ID=$(( TASK_ID % 7 ))

if [ "$MODEL_ID" -ge "${#MODELS[@]}" ]; then
  echo "ERROR: MODEL_ID=$MODEL_ID out of range for MODELS."
  exit 1
fi
if [ "$CURVE_ID" -ge "${#DATASET_NAMES[@]}" ]; then
  echo "ERROR: CURVE_ID=$CURVE_ID out of range for DATASET_NAMES."
  exit 1
fi

MODEL="${MODELS[$MODEL_ID]}"
CURVE_NAME="${DATASET_NAMES[$CURVE_ID]}"
H5_BASENAME="$(basename "$DATA_PATH" .h5)"
OUTFILE="$RESULT_DIR/${H5_BASENAME}_${CURVE_NAME}_${MODEL}_bnn_d3lam1e1.h5"

echo "Node:            $(hostname)"
echo "Repo:            $REPO_DIR"
echo "Dataset H5:      $DATA_PATH"
echo "Curve ID:        $CURVE_ID ($CURVE_NAME)"
echo "Model ID:        $MODEL_ID ($MODEL)"
echo "Stride:          $STRIDE"
echo "Random state:    $RANDOM_STATE"
echo "d3 lambda:       10.0"
echo "d3 lambda space: $D3_LAMBDA_SPACE"
echo "Output:          $OUTFILE"

python "$REPO_DIR/scripts/train_bnns.py" \
  --data "$DATA_PATH" \
  --index "$CURVE_ID" \
  --stride "$STRIDE" \
  --random_state "$RANDOM_STATE" \
  --out_h5 "$OUTFILE" \
  --num_draws 50 \
  --num_chains 4 \
  --width 50 \
  --num_warmup 1000 \
  --likelihood_model "normal_likelihood_lognormal_noise" \
  --kernel "$MODEL" \
  --d3_lambda 10.0 \
  --d3_lambda_space "$D3_LAMBDA_SPACE"

python "$REPO_DIR/scripts/merge_bnn_parts.py" --out_h5 "$OUTFILE"

echo "Done curve_id=$CURVE_ID model_id=$MODEL_ID -> $OUTFILE"
