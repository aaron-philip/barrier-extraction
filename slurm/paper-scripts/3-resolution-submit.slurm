#!/bin/bash --login
#SBATCH --job-name=resolution_submit
#SBATCH --output=logs/resolution_submit_%j.out
#SBATCH --error=logs/resolution_submit_%j.err
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=frib-nodes
# From repository root: sbatch slurm/paper-scripts/3-resolution-submit.slurm
# Or: bash slurm/paper-scripts/3-resolution-submit.slurm
# Export REPO_DIR explicitly when submitting from outside the repository.
set -euo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO_DIR="$(cd "$REPO_DIR" && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data/packed/resolution_experiment}"
RESULT_DIR="${RESULT_DIR:-$REPO_DIR/data/results/resolution_experiment_soc_d3lam1_gp_log}"
ENV_PATH="${ENV_PATH:-$HOME/envs/autobnn-test-fork}"
export REPO_DIR DATA_DIR RESULT_DIR ENV_PATH
cd "$REPO_DIR"
source "$REPO_DIR/slurm/paper-scripts/3-resolution-env.sh"
"$PYTHON" scripts/smoke_autobnn_dependency.py
"$PYTHON" scripts/resolution_experiment.py prepare --data-dir "$DATA_DIR"
mkdir -p "$REPO_DIR/logs" "$RESULT_DIR"
# Atomic guard prevents duplicate arrays and accidental overwrites on resubmission.
if ! mkdir "$RESULT_DIR/.submission_lock"; then
  echo "Existing submission in $RESULT_DIR. Inspect jobs before using a fresh RESULT_DIR." >&2
  exit 1
fi
"$PYTHON" -m pip freeze > "$RESULT_DIR/environment.freeze.txt"
git rev-parse HEAD > "$RESULT_DIR/repository_commit.txt"
sha256sum scripts/train_bnns.py scripts/train_gps.py scripts/resolution_experiment.py \
  slurm/paper-scripts/3-resolution-* > "$RESULT_DIR/code.sha256"
WORKER="$REPO_DIR/slurm/paper-scripts/3-resolution-worker.slurm"
BNN_JOB=$(sbatch --parsable --array=0-69 --job-name=resolution_bnn --export=ALL,MODE=bnn "$WORKER")
BNN_JOB=${BNN_JOB%%;*}
printf 'BNN_JOB=%s\n' "$BNN_JOB" > "$RESULT_DIR/jobs.env"
GP_JOB=$(sbatch --parsable --array=0-6 --job-name=resolution_gp --export=ALL,MODE=gp "$WORKER")
GP_JOB=${GP_JOB%%;*}
printf 'GP_JOB=%s\n' "$GP_JOB" >> "$RESULT_DIR/jobs.env"
FINAL_JOB=$(sbatch --parsable --dependency="afterok:$BNN_JOB:$GP_JOB" \
  --job-name=resolution_summary --time=00:30:00 --mem=8G \
  --export=ALL,MODE=finalize "$WORKER")
FINAL_JOB=${FINAL_JOB%%;*}
printf 'FINAL_JOB=%s\n' "$FINAL_JOB" >> "$RESULT_DIR/jobs.env"
cat "$RESULT_DIR/jobs.env"
