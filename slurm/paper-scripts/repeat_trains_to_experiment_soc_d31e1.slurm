#!/bin/bash --login
#SBATCH --job-name=bnn_exp_reps_soc_d3
#SBATCH --output=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_exp_reps_soc_d3_%A_%a.out
#SBATCH --error=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_exp_reps_soc_d3_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=frib-nodes
#SBATCH --array=0-39   # 4 experimental datasets * 10 repeats

# Train one sum_of_changepoints BNN 10 times on each of the 4 experimental
# datasets, using a fixed third-derivative penalty lambda = 1e1.
#
# Array mapping:
#   curve_id = task_id / 10   in [0..3]
#   rep_id   = task_id % 10   in [0..9]  (passed as --sub_index)
#
# Submit:
#   sbatch slurm/paper-scripts/repeat_trains_to_experiment_soc_d31e1.slurm
#
# Optional overrides:
#   sbatch --export=ALL,DATAFILE=/path/to/experiments.h5,ENV_PATH=~/envs/autobnn-test-fork \
#     slurm/paper-scripts/repeat_trains_to_experiment_soc_d31e1.slurm

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

DATA_DIR="${DATA_DIR:-/mnt/home/philipaa/scratch/barrier-paper-datasets}"
RESULT_DIR="${RESULT_DIR:-/mnt/home/philipaa/scratch/barrier-paper-results/experimental_reps_soc_d3lam1e1}"
LOG_DIR="${LOG_DIR:-/mnt/home/philipaa/scratch/barrier-paper-logs}"
ENV_PATH="${ENV_PATH:-$HOME/envs/autobnn-test-fork}"

DATAFILE="${DATAFILE:-experiments.h5}"
STRIDE="${STRIDE:-1}"
RANDOM_STATE="${RANDOM_STATE:-0}"
NUM_DRAWS="${NUM_DRAWS:-100}"
NUM_CHAINS="${NUM_CHAINS:-4}"
NUM_WARMUP="${NUM_WARMUP:-1000}"
D3_LAMBDA_SPACE="${D3_LAMBDA_SPACE:-physical}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

if [ -f "$DATAFILE" ]; then
  DATA_PATH="$DATAFILE"
else
  DATA_PATH="$DATA_DIR/$DATAFILE"
fi

if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: Dataset not found: $DATA_PATH"
  exit 1
fi

DATASET_NAMES=(
  "16O208PbMorton1999"
  "40Ca40CaMontagnoli2012"
  "48Ca40CaJiang2010"
  "48Ca48CaStefanini2009"
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
CURVE_ID=$(( TASK_ID / 10 ))
REP_ID=$(( TASK_ID % 10 ))

if [ "$CURVE_ID" -ge "${#DATASET_NAMES[@]}" ]; then
  echo "ERROR: CURVE_ID=$CURVE_ID out of range for DATASET_NAMES."
  exit 1
fi

CURVE_NAME="${DATASET_NAMES[$CURVE_ID]}"
H5_BASENAME="$(basename "$DATA_PATH" .h5)"
OUTFILE="$RESULT_DIR/${H5_BASENAME}_${CURVE_NAME}_sum_of_changepoints_d3lam1e1_4x100_reps.h5"

source ~/.bashrc
conda deactivate
conda activate "$ENV_PATH"

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

echo "Node:            $(hostname)"
echo "Repo:            $REPO_DIR"
echo "Dataset H5:      $DATA_PATH"
echo "Curve ID:        $CURVE_ID ($CURVE_NAME)"
echo "Repeat ID:       $REP_ID"
echo "Stride:          $STRIDE"
echo "Random state:    $RANDOM_STATE"
echo "Num draws:       $NUM_DRAWS"
echo "Num chains:      $NUM_CHAINS"
echo "Num warmup:      $NUM_WARMUP"
echo "Kernel:          sum_of_changepoints"
echo "d3 lambda:       10.0"
echo "d3 lambda space: $D3_LAMBDA_SPACE"
echo "Output:          $OUTFILE"

python "$REPO_DIR/scripts/train_bnns.py" \
  --data "$DATA_PATH" \
  --index "$CURVE_ID" \
  --sub_index "$REP_ID" \
  --stride "$STRIDE" \
  --random_state "$RANDOM_STATE" \
  --out_h5 "$OUTFILE" \
  --num_draws "$NUM_DRAWS" \
  --num_chains "$NUM_CHAINS" \
  --width 50 \
  --num_warmup "$NUM_WARMUP" \
  --likelihood_model "normal_likelihood_lognormal_noise" \
  --kernel "sum_of_changepoints" \
  --d3_lambda 10.0 \
  --d3_lambda_space "$D3_LAMBDA_SPACE"

echo "Done curve_id=$CURVE_ID rep_id=$REP_ID -> $OUTFILE"
