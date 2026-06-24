#!/bin/bash --login
#SBATCH --job-name=bnn_train_reps
#SBATCH --output=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_rep_%A_%a.out
#SBATCH --error=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_rep_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=frib-nodes
#SBATCH --array=0-9   # sub_index = 1..10

# Repeated fits to the SAME curve (same param index), varying sub_index.
# Submit like:
#   sbatch --export=ALL,DATAFILE=res_2.00MeV_noise0.01.h5,PARAM_INDEX=72 \
#     slurm/paper-scripts/misc-retrain_bnns.slurm
#
# Or with full path:
#   sbatch --export=ALL,DATAFILE=/mnt/home/philipaa/scratch/barrier-paper-datasets/res_2.00MeV_noise0.01.h5,PARAM_INDEX=72 \
#     slurm/paper-scripts/misc-retrain_bnns.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/home/philipaa/research/barrier-extraction}"
export REPO_DIR

# === Directories ===
DATA_DIR=/mnt/home/philipaa/scratch/barrier-paper-datasets
RESULT_DIR=/mnt/home/philipaa/scratch/barrier-paper-results
LOG_DIR=/mnt/home/philipaa/scratch/barrier-paper-logs
mkdir -p "$RESULT_DIR" "$LOG_DIR"

# --- Required inputs via --export ---
if [ -z "${DATAFILE:-}" ]; then
  echo "ERROR: Must pass DATAFILE (basename like foo.h5 or full path)."
  exit 1
fi

if [ -z "${PARAM_INDEX:-}" ]; then
  echo "ERROR: Must pass PARAM_INDEX (e.g. 72)."
  exit 1
fi

# Optional: stride + random state
STRIDE="${STRIDE:-10}"
RANDOM_STATE="${RANDOM_STATE:-0}"

# Resolve DATAFILE against DATA_DIR if it's not an existing file path.
if [ -f "$DATAFILE" ]; then
  DATA_PATH="$DATAFILE"
else
  DATA_PATH="$DATA_DIR/$DATAFILE"
fi

if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: Dataset not found: $DATA_PATH"
  exit 1
fi

BASENAME=$(basename "$DATA_PATH" .h5)

# Make OUTFILE unique per PARAM_INDEX so repeats don't mix with other indices unless you want them to.
OUTFILE="$RESULT_DIR/${BASENAME}_bnn_soc_idx${PARAM_INDEX}.h5"

# --- Environment ---
source ~/.bashrc
conda deactivate
conda activate ~/envs/autobnn-test-fork

# Force JAX to CPU and prevent any CUDA init attempt
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

SUB_INDEX="$SLURM_ARRAY_TASK_ID"

echo "Node:        $(hostname)"
echo "Dataset:     $DATA_PATH"
echo "Output:      $OUTFILE"
echo "Param index: $PARAM_INDEX"
echo "Sub index:   $SUB_INDEX"
echo "Stride:      $STRIDE"
echo "Rand state:  $RANDOM_STATE"

python "$REPO_DIR/scripts/train_bnns.py" \
  --data "$DATA_PATH" \
  --index "$PARAM_INDEX" \
  --sub_index "$SUB_INDEX" \
  --stride "$STRIDE" \
  --random_state "$RANDOM_STATE" \
  --out_h5 "$OUTFILE"

echo "Done param index $PARAM_INDEX sub_index $SUB_INDEX"
