#!/bin/bash --login
#SBATCH --job-name=gp_train
#SBATCH --output=/mnt/home/philipaa/scratch/barrier-paper-logs/gp_%A_%a.out
#SBATCH --error=/mnt/home/philipaa/scratch/barrier-paper-logs/gp_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=standard

# === Directories ===
DATA_DIR=/mnt/home/philipaa/scratch/barrier-paper-datasets
RESULT_DIR=/mnt/home/philipaa/scratch/barrier-paper-results
LOG_DIR=/mnt/home/philipaa/scratch/barrier-paper-logs
mkdir -p "$RESULT_DIR" "$LOG_DIR"

# === Compute array length dynamically ===
FILES=("$DATA_DIR"/*.h5)
NUM_FILES=${#FILES[@]}

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Submitting as array of $NUM_FILES jobs..."
    sbatch --array=0-$(($NUM_FILES - 1)) "$0"
    exit 0
fi

# === Dataset selection ===
DATAFILE=${FILES[$SLURM_ARRAY_TASK_ID]}
BASENAME=$(basename "$DATAFILE" .h5)
OUTH5="$RESULT_DIR/${BASENAME}_gp.h5"

# === Activate environment right before Python ===
source ~/.bashrc
conda deactivate
conda activate ~/envs/autobnn-reproduce2

echo "Activated environment:"
which python
python -c "import sklearn; print('scikit-learn version:', sklearn.__version__)"

# === Run training ===
echo "Running GP on dataset: $DATAFILE"
python /mnt/research/ascsn/aaron-philip/barrier-extraction/scripts/train_gps.py \
  --data "$DATAFILE" \
  --out_h5 "$OUTH5" \
  --log_every 50 

