#!/bin/bash --login
#SBATCH --job-name=bnn_train_d3
#SBATCH --output=logs/bnn_d3_%A_%a.out
#SBATCH --error=logs/bnn_d3_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=frib-nodes
#SBATCH --array=0-47   # 16 datasets * 3 lambdas

# Trains exactly one sum_of_changepoints BNN on one curve from each of the
# 16 paper datasets, repeated only across third-derivative penalty strengths.
#
# Array index mapping:
#   lambda_id  = task_id / 16   in [0..2]
#   dataset_id = task_id % 16   in [0..15]
#
# This script intentionally has no repeat/sub_index dimension.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_DIR

DATA_DIR="${DATA_DIR:-$REPO_DIR/data/packed}"
RESULT_DIR="${RESULT_DIR:-$REPO_DIR/data/results/d3_sweep_single_curve}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
ENV_PATH="${ENV_PATH:-$HOME/envs/autobnn-test-fork}"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

# Curve-selection knobs. train_bnns.py maps raw row = CURVE_INDEX * STRIDE.
# The default selects the first parameter sample / first curve in each H5.
CURVE_INDEX="${CURVE_INDEX:-0}"
STRIDE="${STRIDE:-10}"
RANDOM_STATE="${RANDOM_STATE:-0}"
D3_LAMBDA_SPACE="${D3_LAMBDA_SPACE:-physical}"

DATASETS=(
  cv0.00_spacing0.5.h5
  cv0.25_spacing0.5.h5
  cv0.50_spacing0.5.h5
  cv0.75_spacing0.5.h5
  cv0.00_spacing1.0.h5
  cv0.25_spacing1.0.h5
  cv0.50_spacing1.0.h5
  cv0.75_spacing1.0.h5
  res_0.25MeV_noise0.01.h5
  res_0.50MeV_noise0.01.h5
  res_1.00MeV_noise0.01.h5
  res_2.00MeV_noise0.01.h5
  res_0.25MeV_noise0.02.h5
  res_0.50MeV_noise0.02.h5
  res_1.00MeV_noise0.02.h5
  res_2.00MeV_noise0.02.h5
)

D3_LAMBDAS=(
  "1.0"
  "1e1"
  "1e2"
)
D3_TAGS=(
  "1"
  "1e1"
  "1e2"
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
LAMBDA_ID=$(( TASK_ID / 16 ))
DATASET_ID=$(( TASK_ID % 16 ))

if [ "$LAMBDA_ID" -ge "${#D3_LAMBDAS[@]}" ]; then
  echo "ERROR: LAMBDA_ID=$LAMBDA_ID out of range for D3_LAMBDAS (len=${#D3_LAMBDAS[@]})."
  exit 1
fi

if [ "$DATASET_ID" -ge "${#DATASETS[@]}" ]; then
  echo "ERROR: DATASET_ID=$DATASET_ID out of range for DATASETS (len=${#DATASETS[@]})."
  exit 1
fi

DATAFILE="${DATASETS[$DATASET_ID]}"
DATA_PATH="$DATA_DIR/$DATAFILE"
if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: Dataset not found: $DATA_PATH"
  exit 1
fi

D3_LAMBDA="${D3_LAMBDAS[$LAMBDA_ID]}"
D3_TAG="${D3_TAGS[$LAMBDA_ID]}"
BASENAME="$(basename "$DATA_PATH" .h5)"
OUTFILE="$RESULT_DIR/${BASENAME}_curve${CURVE_INDEX}_bnn_soc_d3lam_${D3_TAG}.h5"

source ~/.bashrc
conda deactivate
conda activate "$ENV_PATH"

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

echo "Node:            $(hostname)"
echo "Repo:            $REPO_DIR"
echo "Dataset ID:      $DATASET_ID"
echo "Dataset H5:      $DATA_PATH"
echo "Curve index:     $CURVE_INDEX"
echo "Stride:          $STRIDE"
echo "Lambda ID:       $LAMBDA_ID"
echo "d3 lambda:       $D3_LAMBDA"
echo "d3 lambda space: $D3_LAMBDA_SPACE"
echo "Rand state:      $RANDOM_STATE"
echo "Output:          $OUTFILE"

python "$REPO_DIR/scripts/train_bnns.py" \
  --data "$DATA_PATH" \
  --index "$CURVE_INDEX" \
  --stride "$STRIDE" \
  --random_state "$RANDOM_STATE" \
  --out_h5 "$OUTFILE" \
  --num_draws 50 \
  --num_chains 4 \
  --width 50 \
  --num_warmup 1000 \
  --likelihood_model "normal_likelihood_lognormal_noise" \
  --kernel "sum_of_changepoints" \
  --d3_lambda "$D3_LAMBDA" \
  --d3_lambda_space "$D3_LAMBDA_SPACE"

python "$REPO_DIR/scripts/merge_bnn_parts.py" --out_h5 "$OUTFILE"

echo "Done lambda_id=$LAMBDA_ID dataset_id=$DATASET_ID curve_index=$CURVE_INDEX -> $OUTFILE"
