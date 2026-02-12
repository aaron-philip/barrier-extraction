#!/bin/bash --login
#SBATCH --job-name=bnn_train
#SBATCH --output=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_%A_%a.out
#SBATCH --error=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=frib-nodes
#SBATCH --array=0-99   # one per parameter sample


# do not sbatch this directly; use submit_all_bnns.slurm.
set -euo pipefail

# === Directories ===
DATA_DIR=/mnt/home/philipaa/scratch/barrier-paper-datasets
RESULT_DIR=/mnt/home/philipaa/scratch/barrier-paper-results
LOG_DIR=/mnt/home/philipaa/scratch/barrier-paper-logs
mkdir -p "$RESULT_DIR" "$LOG_DIR"

# --- dataset argument ---
# You pass DATAFILE as either basename (recommended) or absolute path.
if [ -z "${DATAFILE:-}" ]; then
  echo "ERROR: Must pass DATAFILE (basename like foo.h5 or full path)."
  exit 1
fi

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
OUTFILE="$RESULT_DIR/${BASENAME}_bnn_soc.h5"

# ------------------------------------------------------------
# If this script is run as a normal sbatch submission (not an
# array task), auto-submit the array and then the merge job.
# ------------------------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Submitting array job for dataset: $DATA_PATH"
  echo "Output will be: $OUTFILE"

  # Submit the array job (this same script), capture job id
  ARRAY_JOB_ID=$(sbatch --parsable --array=0-99 --export=ALL,DATAFILE="$DATA_PATH" "$0")
  echo "Submitted array job: $ARRAY_JOB_ID"

  # Submit merge job dependent on successful completion of the array
  echo "Submitting merge job dependent on array completion..."
  MERGE_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --job-name=bnn_merge \
    --output="$LOG_DIR/bnn_merge_${BASENAME}_%j.out" \
    --error="$LOG_DIR/bnn_merge_${BASENAME}_%j.err" \
    --time=00:20:00 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    --partition=frib-nodes \
    --export=ALL,OUTFILE="$OUTFILE" \
    --wrap="source ~/.bashrc; conda deactivate; conda activate ~/envs/autobnn-reproduce2; python /mnt/research/ascsn/aaron-philip/barrier-extraction/scripts/merge_bnn_parts.py --out_h5 \"$OUTFILE\""
  )
  echo "Submitted merge job: $MERGE_JOB_ID (afterok:$ARRAY_JOB_ID)"
  exit 0
fi

# ------------------------------------------------------------
# Array task body (runs when SLURM_ARRAY_TASK_ID is set)
# ------------------------------------------------------------
source ~/.bashrc
conda deactivate
conda activate ~/envs/autobnn-reproduce2

# Force JAX to CPU and prevent any CUDA init attempt
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

echo "Node: $(hostname)"
echo "Dataset: $DATA_PATH"
echo "Output:  $OUTFILE"
echo "Param index: $SLURM_ARRAY_TASK_ID"

python /mnt/research/ascsn/aaron-philip/barrier-extraction/scripts/train_bnns.py \
  --data "$DATA_PATH" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --stride 10 \
  --out_h5 "$OUTFILE"

echo "Done param index $SLURM_ARRAY_TASK_ID"

