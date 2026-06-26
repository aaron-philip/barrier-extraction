#!/bin/bash --login
#SBATCH --job-name=bnn_train_d3lam1
#SBATCH --output=logs/bnn_d3lam1_%A_%a.out
#SBATCH --error=logs/bnn_d3lam1_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=frib-nodes
#SBATCH --array=0-99   # one per parameter sample


# Train one parameter sample from a Wong toy dataset with the
# sum_of_changepoints BNN and third-derivative penalty lambda = 1.0.
#
# Usual full-paper run:
#   sbatch slurm/paper-scripts/1-run_soc_wong_submitter-1.0d3.slurm
#
# To run the full pipeline from a login shell:
#   bash slurm/paper-scripts/1-submit_all_soc_trains_to_wong-1.0d3.sh
#
# To run one dataset manually, pass DATAFILE as either a basename from
# DATA_DIR or an absolute path:
#   sbatch --array=0-99 --export=ALL,DATAFILE=cv0.25_spacing1.0.h5 \
#     slurm/paper-scripts/1-train_all_soc_to_wong-1.0d3.slurm
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_DIR

# === Directories ===
DATA_DIR="$REPO_DIR/data/packed"
RESULT_ROOT=$REPO_DIR/data/results/d3lambda1_rerun
RESULT_DIR="$RESULT_ROOT/bnns_with_df"
LOG_DIR=$REPO_DIR/logs
mkdir -p "$RESULT_DIR" "$LOG_DIR"

# --- dataset argument ---
# You pass DATAFILE as either basename (recommended) or absolute path.
if [ -z "${DATAFILE:-}" ]; then
  echo "ERROR: Must pass DATAFILE (basename like foo.h5 or full path)."
  exit 1
fi

# Resolve DATAFILE against DATA_DIR if it is a basename.  This lets users submit
# from anywhere without editing the Slurm file.
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
OUTFILE="$RESULT_DIR/${BASENAME}_bnn_soc_d3lam1.h5"

submit_sbatch () {
  local output status job_id
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e

  if [[ $status -ne 0 ]]; then
    printf '%s\n' "$output" >&2
    return "$status"
  fi

  job_id="$(printf '%s\n' "$output" | sed -n 's/^\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -z "$job_id" ]]; then
    echo "ERROR: Could not parse Slurm job id from sbatch output." >&2
    printf '%s\n' "$output" >&2
    return 1
  fi

  printf '%s\n' "$job_id"
}

# If submitted without --array, turn this file into a convenience wrapper:
# submit the 100-task array and a dependent serial merge job.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Submitting array job for dataset: $DATA_PATH"
  echo "Output will be: $OUTFILE"

  ARRAY_JOB_ID=$(submit_sbatch sbatch --parsable --array=0-99 --export=ALL,DATAFILE="$DATA_PATH" "$0")
  echo "Submitted array job: $ARRAY_JOB_ID"

  echo "Submitting merge job dependent on array completion..."
  MERGE_JOB_ID=$(submit_sbatch sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --job-name=bnn_merge_d3lam1 \
    --output="$LOG_DIR/bnn_merge_d3lam1_${BASENAME}_%j.out" \
    --error="$LOG_DIR/bnn_merge_d3lam1_${BASENAME}_%j.err" \
    --time=00:20:00 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --partition=frib-nodes \
    --export=ALL,OUTFILE="$OUTFILE" \
    --wrap="source ~/.bashrc; conda deactivate; conda activate ~/envs/autobnn-test-fork; python \"$REPO_DIR/scripts/merge_bnn_parts.py\" --out_h5 \"$OUTFILE\""
  )
  echo "Submitted merge job: $MERGE_JOB_ID (afterok:$ARRAY_JOB_ID)"
  exit 0
fi

# Array task body.  Each task trains one parameter sample and writes one part
# file; no two tasks touch the same HDF5 file.
source ~/.bashrc
conda deactivate
conda activate ~/envs/autobnn-test-fork

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

echo "Node: $(hostname)"
echo "Dataset: $DATA_PATH"
echo "Output:  $OUTFILE"
echo "Param index: $SLURM_ARRAY_TASK_ID"
echo "d3 lambda: 1.0"
echo "d3 lambda space: physical"
echo "num draws: 50"

python "$REPO_DIR/scripts/train_bnns.py" \
  --data "$DATA_PATH" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --stride 10 \
  --out_h5 "$OUTFILE" \
  --num_draws 50 \
  --d3_lambda 1.0 \
  --d3_lambda_space physical

echo "Done param index $SLURM_ARRAY_TASK_ID"
