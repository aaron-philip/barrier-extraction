#!/bin/bash --login
#SBATCH --job-name=bnn_train_8x10
#SBATCH --output=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_rep_%A_%a.out
#SBATCH --error=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_rep_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=frib-nodes
#SBATCH --array=0-39   # 4 datasets * 10 reps  => task_id = 0..39

# Trains 10 models per curve for 4 curves in ONE dataset H5.
# Each SLURM task picks:
#   curve_id = task_id // 10   in [0..3]
#   rep_id   = task_id % 10    in [0..9]  (passed as --sub_index)
#
# Submit like:
#   sbatch --export=ALL,DATAFILE=experiments.h5 experiment_train_bnn.slurm
# or:
#   sbatch --export=ALL,DATAFILE=/full/path/experiments.h5 experiment_train_bnn.slurm
set -euo pipefail

# === Directories ===
DATA_DIR=/mnt/home/philipaa/scratch/barrier-paper-datasets
RESULT_DIR=/mnt/home/philipaa/scratch/barrier-paper-results/experiments8
LOG_DIR=/mnt/home/philipaa/scratch/barrier-paper-logs
mkdir -p "$RESULT_DIR" "$LOG_DIR"

# --- Required input via --export ---
if [ -z "${DATAFILE:-}" ]; then
  echo "ERROR: Must pass DATAFILE (basename like experiments.h5 or full path)."
  exit 1
fi

# Optional knobs
STRIDE="${STRIDE:-1}"           # for experiments.h5: each curve is one entry => stride=1
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

H5_BASENAME="$(basename "$DATA_PATH" .h5)"

# ============================================================
# EDIT THIS SECTION ONLY: give names for each of the 8 curves
# (used in the final OUTFILE name)
# ============================================================
DATASET_NAMES=(
  "16O208PbMorton1999"
  "40Ca40CaMontagnoli2012"
  "48Ca40CaJiang2010"
  "48Ca48CaStefanini2009"
)
# ============================================================

# --- Map array task -> (curve_id, rep_id) ---
TASK_ID="${SLURM_ARRAY_TASK_ID}"
CURVE_ID=$(( TASK_ID / 10 ))   # 0..3
REP_ID=$(( TASK_ID % 10 ))     # 0..9

if [ "$CURVE_ID" -ge "${#DATASET_NAMES[@]}" ]; then
  echo "ERROR: CURVE_ID=$CURVE_ID out of range for DATASET_NAMES (len=${#DATASET_NAMES[@]})."
  exit 1
fi

CURVE_NAME="${DATASET_NAMES[$CURVE_ID]}"

# Name final result based on the curve name (and also include source h5 basename)
OUTFILE="$RESULT_DIR/${H5_BASENAME}_${CURVE_NAME}_bnn_soc.h5"

# --- Environment ---
source ~/.bashrc
conda deactivate
conda activate ~/envs/autobnn-gpu

# Force JAX to CPU and prevent any CUDA init attempt
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

echo "Node:        $(hostname)"
echo "Dataset H5:  $DATA_PATH"
echo "Curve ID:    $CURVE_ID  (name=$CURVE_NAME)"
echo "Rep ID:      $REP_ID"
echo "Stride:      $STRIDE"
echo "Rand state:  $RANDOM_STATE"
echo "Output:      $OUTFILE"

python /mnt/research/ascsn/aaron-philip/barrier-extraction/scripts/train_bnns.py \
  --data "$DATA_PATH" \
  --index "$CURVE_ID" \
  --sub_index "$REP_ID" \
  --stride "$STRIDE" \
  --random_state "$RANDOM_STATE" \
  --out_h5 "$OUTFILE" \
  --num_draws 1000 \
  --num_chains 8 \
  --width 50  \
  --num_warmup 1000 \
  --extra "8x1000_lognormal_soc" \
  --likelihood_model "normal_likelihood_lognormal_noise" \
  --kernel "sum_of_changepoints"

echo "Done curve_id=$CURVE_ID rep_id=$REP_ID -> $OUTFILE"