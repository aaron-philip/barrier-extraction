#!/bin/bash --login
#SBATCH --job-name=gp_train_d3lam1
#SBATCH --output=logs/gp_d3lam1_%A_%a.out
#SBATCH --error=logs/gp_d3lam1_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=standard

# Companion GP baseline for the same Wong toy datasets used by the 1.0d3 BNN
# paper pipeline. No derivative penalty is applied to GP training; the 1.0d3
# tag means these GP outputs live beside the lambda = 1.0 BNN result family.
#
# Run after, before, or alongside the BNN pipeline with:
#   sbatch slurm/paper-scripts/1-train_gps_companion-1.0d3.slurm
#
# This auto-submits an array over all H5 files in DATA_DIR and writes:
#   $REPO_DIR/data/results/d3lambda1_rerun/gps      (linear, default)
#   $REPO_DIR/data/results/d3lambda1_rerun/gps_log  (log-space target)
#
# To train in log space:
#   GP_TARGET_SPACE=log sbatch slurm/paper-scripts/1-train_gps_companion-1.0d3.slurm

find_repo_dir() {
    local candidates=()
    if [[ -n "${REPO_DIR:-}" ]]; then
        candidates+=("$REPO_DIR")
    fi
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        candidates+=("$SLURM_SUBMIT_DIR" "$SLURM_SUBMIT_DIR/barrier-extraction")
    fi
    candidates+=("$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)")

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate/scripts/train_gps.py" && -d "$candidate/src" && -d "$candidate/data" ]]; then
            cd "$candidate" && pwd
            return 0
        fi
    done

    echo "Could not locate barrier-extraction repo. Set REPO_DIR explicitly." >&2
    return 1
}

REPO_DIR="$(find_repo_dir)"
export REPO_DIR
GP_TARGET_SPACE="${GP_TARGET_SPACE:-linear}"
export GP_TARGET_SPACE

if [[ "$GP_TARGET_SPACE" != "linear" && "$GP_TARGET_SPACE" != "log" ]]; then
    echo "GP_TARGET_SPACE must be 'linear' or 'log' (got '$GP_TARGET_SPACE')" >&2
    exit 2
fi

DATA_DIR="$REPO_DIR/data/packed"
SCRATCH_DIR="${SCRATCH_DIR:-/mnt/home/philipaa/scratch}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_DIR/data/results}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
if [[ "$GP_TARGET_SPACE" == "log" ]]; then
    RESULT_DIR=$RESULT_ROOT/d3lambda1_rerun/gps_log
else
    RESULT_DIR=$RESULT_ROOT/d3lambda1_rerun/gps
fi
mkdir -p "$RESULT_DIR" "$LOG_DIR"

FILES=("$DATA_DIR"/*.h5)
NUM_FILES=${#FILES[@]}

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Submitting as array of $NUM_FILES jobs..."
    sbatch --export=ALL,GP_TARGET_SPACE,SCRATCH_DIR,RESULT_ROOT,LOG_DIR,REPO_DIR --array=0-$(($NUM_FILES - 1)) "$REPO_DIR/slurm/paper-scripts/1-train_gps_companion-1.0d3.slurm"
    exit 0
fi

DATAFILE=${FILES[$SLURM_ARRAY_TASK_ID]}
BASENAME=$(basename "$DATAFILE" .h5)
OUTH5="$RESULT_DIR/${BASENAME}_gp_rbf.h5"

source ~/.bashrc
conda deactivate
conda activate ~/envs/autobnn-test-fork

echo "Activated environment:"
which python
python -c "import sklearn; print('scikit-learn version:', sklearn.__version__)"

echo "Running GP on dataset: $DATAFILE"
echo "Target space: $GP_TARGET_SPACE"
python "$REPO_DIR/scripts/train_gps.py" \
  --data "$DATAFILE" \
  --out_h5 "$OUTH5" \
  --log_dir "$LOG_DIR" \
  --target_space "$GP_TARGET_SPACE" \
  --log_every 50
