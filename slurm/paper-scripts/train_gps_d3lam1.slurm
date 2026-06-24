#!/bin/bash --login
#SBATCH --job-name=gp_train_d3lam1
#SBATCH --output=/mnt/home/philipaa/scratch/barrier-paper-logs/gp_d3lam1_%A_%a.out
#SBATCH --error=/mnt/home/philipaa/scratch/barrier-paper-logs/gp_d3lam1_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=standard

# Companion GP rerun into the same results root as the d3-lambda-1 BNN rerun.
# No derivative penalty is applied to GP training; this copy only redirects paths.

REPO_DIR="${REPO_DIR:-/mnt/home/philipaa/research/barrier-extraction}"
export REPO_DIR
DATA_DIR=/mnt/home/philipaa/scratch/barrier-paper-datasets
RESULT_DIR=/mnt/home/philipaa/scratch/barrier-paper-results/d3lambda1_rerun/gps
LOG_DIR=/mnt/home/philipaa/scratch/barrier-paper-logs
mkdir -p "$RESULT_DIR" "$LOG_DIR"

FILES=("$DATA_DIR"/*.h5)
NUM_FILES=${#FILES[@]}

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Submitting as array of $NUM_FILES jobs..."
    sbatch --array=0-$(($NUM_FILES - 1)) "$0"
    exit 0
fi

DATAFILE=${FILES[$SLURM_ARRAY_TASK_ID]}
BASENAME=$(basename "$DATAFILE" .h5)
OUTH5="$RESULT_DIR/${BASENAME}_gp_rbf.h5"

source ~/.bashrc
conda deactivate
conda activate ~/envs/autobnn-reproduce2

echo "Activated environment:"
which python
python -c "import sklearn; print('scikit-learn version:', sklearn.__version__)"

echo "Running GP on dataset: $DATAFILE"
python "$REPO_DIR/scripts/train_gps.py" \
  --data "$DATAFILE" \
  --out_h5 "$OUTH5" \
  --log_every 50
