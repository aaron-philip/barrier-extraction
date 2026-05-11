#!/bin/bash --login
#SBATCH --job-name=bnn_train_d3
#SBATCH --output=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_d3_%A_%a.out
#SBATCH --error=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_d3_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=frib-nodes
#SBATCH --array=0-159   # 4 lambdas * 4 datasets * 10 reps

# Trains 10 models per curve for 4 curves in ONE dataset H5, repeated across
# multiple third-derivative penalty strengths.
#
# Array index mapping:
#   lambda_id = task_id // 40   in [0..3]
#   within_lambda = task_id % 40
#   curve_id  = within_lambda // 10   in [0..3]
#   rep_id    = within_lambda % 10    in [0..9]
#
# Submit like:
#   sbatch --export=ALL,DATAFILE=experiments.h5 experiment_train_bnn_d3_sweep.slurm
# or:
#   sbatch --export=ALL,DATAFILE=/full/path/experiments.h5 experiment_train_bnn_d3_sweep.slurm
set -euo pipefail

# === Directories ===
DATA_DIR=/mnt/home/philipaa/scratch/barrier-paper-datasets
RESULT_DIR=/mnt/home/philipaa/scratch/barrier-paper-results/experiments8_d3_sweep
LOG_DIR=/mnt/home/philipaa/scratch/barrier-paper-logs
mkdir -p "$RESULT_DIR" "$LOG_DIR"

# --- Required input via --export ---
if [ -z "${DATAFILE:-}" ]; then
  echo "ERROR: Must pass DATAFILE (basename like experiments.h5 or full path)."
  exit 1
fi

# Optional knobs
STRIDE="${STRIDE:-1}"
RANDOM_STATE="${RANDOM_STATE:-0}"
D3_LAMBDA_SPACE="${D3_LAMBDA_SPACE:-physical}"

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
# Four experimental curves in the H5, matching experiment_train_bnn.slurm
# ============================================================
DATASET_NAMES=(
  "16O208PbMorton1999"
  "40Ca40CaMontagnoli2012"
  "48Ca40CaJiang2010"
  "48Ca48CaStefanini2009"
)

# Third-derivative penalty strengths to sweep over.
# Keep the tag strings filesystem-safe and easy to read.
D3_LAMBDAS=(
  "0.0"
  "1e-6"
  "1e-4"
  "1e-2"
)
D3_TAGS=(
  "0"
  "1e-6"
  "1e-4"
  "1e-2"
)
# ============================================================

TASK_ID="${SLURM_ARRAY_TASK_ID}"
LAMBDA_ID=$(( TASK_ID / 40 ))
WITHIN_LAMBDA=$(( TASK_ID % 40 ))
CURVE_ID=$(( WITHIN_LAMBDA / 10 ))
REP_ID=$(( WITHIN_LAMBDA % 10 ))

if [ "$LAMBDA_ID" -ge "${#D3_LAMBDAS[@]}" ]; then
  echo "ERROR: LAMBDA_ID=$LAMBDA_ID out of range for D3_LAMBDAS (len=${#D3_LAMBDAS[@]})."
  exit 1
fi

if [ "$CURVE_ID" -ge "${#DATASET_NAMES[@]}" ]; then
  echo "ERROR: CURVE_ID=$CURVE_ID out of range for DATASET_NAMES (len=${#DATASET_NAMES[@]})."
  exit 1
fi

CURVE_NAME="${DATASET_NAMES[$CURVE_ID]}"
D3_LAMBDA="${D3_LAMBDAS[$LAMBDA_ID]}"
D3_TAG="${D3_TAGS[$LAMBDA_ID]}"

OUTFILE="$RESULT_DIR/${H5_BASENAME}_${CURVE_NAME}_bnn_soc_d3lam_${D3_TAG}.h5"

# --- Environment ---
source ~/.bashrc
conda deactivate
conda activate ~/envs/autobnn-gpu

# Force JAX to CPU and prevent any CUDA init attempt
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

echo "Node:            $(hostname)"
echo "Dataset H5:      $DATA_PATH"
echo "Lambda ID:       $LAMBDA_ID"
echo "d3 lambda:       $D3_LAMBDA"
echo "d3 lambda space: $D3_LAMBDA_SPACE"
echo "Curve ID:        $CURVE_ID  (name=$CURVE_NAME)"
echo "Rep ID:          $REP_ID"
echo "Stride:          $STRIDE"
echo "Rand state:      $RANDOM_STATE"
echo "Output:          $OUTFILE"

python /mnt/research/ascsn/aaron-philip/barrier-extraction/scripts/train_bnns.py \
  --data "$DATA_PATH" \
  --index "$CURVE_ID" \
  --sub_index "$REP_ID" \
  --stride "$STRIDE" \
  --random_state "$RANDOM_STATE" \
  --out_h5 "$OUTFILE" \
  --num_draws 1000 \
  --num_chains 8 \
  --width 50 \
  --num_warmup 1000 \
  --extra "8x1000_lognormal_soc_d3lam_${D3_TAG}" \
  --likelihood_model "normal_likelihood_lognormal_noise" \
  --kernel "sum_of_changepoints" \
  --d3_lambda "$D3_LAMBDA" \
  --d3_lambda_space "$D3_LAMBDA_SPACE"

echo "Done lambda_id=$LAMBDA_ID curve_id=$CURVE_ID rep_id=$REP_ID -> $OUTFILE"
