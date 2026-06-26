#!/bin/bash --login
#SBATCH --job-name=wong_train
#SBATCH --output=logs/wong_%A_%a.out
#SBATCH --error=logs/wong_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=standard

# === Directories ===
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_DIR
DATA_DIR="$REPO_DIR/data/packed"
RESULT_DIR=$REPO_DIR/data/results
LOG_DIR=$REPO_DIR/logs
mkdir -p $RESULT_DIR $LOG_DIR

# === Compute array length dynamically ===
FILES=($DATA_DIR/*.h5)
NUM_FILES=${#FILES[@]}

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Submitting as array of $NUM_FILES jobs..."
    sbatch --array=0-$(($NUM_FILES - 1)) "$0"
    exit 0
fi

# === Dataset selection ===
DATAFILE=${FILES[$SLURM_ARRAY_TASK_ID]}
BASENAME=$(basename "$DATAFILE")
OUTH5=$RESULT_DIR/${BASENAME}_wong.h5

# === Activate environment right before Python ===
source ~/.bashrc
conda deactivate
conda activate ~/envs/autobnn-test-fork

echo "Activated environment:"
which python
python -c "import sklearn; print('scikit-learn version:', sklearn.__version__)"

# === Run training ===
echo "Running on dataset: $DATAFILE"
python "$REPO_DIR/scripts/train_wongs.py" \
  --data "$DATAFILE" \
  --out_h5 "$OUTH5" \
  --log_every 50
