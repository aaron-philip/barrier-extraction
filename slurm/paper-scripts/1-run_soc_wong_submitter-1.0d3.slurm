#!/bin/bash --login
#SBATCH --job-name=d3lam1_submitter
#SBATCH --output=logs/d3lam1_submitter_%j.out
#SBATCH --error=logs/d3lam1_submitter_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=frib-nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

# Submit this file once to run the full 1.0d3 toy-data BNN pipeline:
#   sbatch slurm/paper-scripts/1-run_soc_wong_submitter-1.0d3.slurm
#
# The job stays alive as a lightweight submitter/monitor. It calls
# 1-submit_all_soc_trains_to_wong-1.0d3.sh, which submits each BNN array and
# its merge/coverage/metric follow-up jobs in sequence.

mkdir -p "$REPO_DIR/logs"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_DIR

echo "Submitter job: ${SLURM_JOB_ID:-unknown}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"

cd "$REPO_DIR"
bash "$REPO_DIR/slurm/paper-scripts/1-submit_all_soc_trains_to_wong-1.0d3.sh"

echo "Finished at: $(date)"
