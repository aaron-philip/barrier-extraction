#!/bin/bash --login
#SBATCH --job-name=d3lam1e2_submitter
#SBATCH --output=/mnt/home/philipaa/scratch/barrier-paper-logs/d3lam1e2_submitter_%j.out
#SBATCH --error=/mnt/home/philipaa/scratch/barrier-paper-logs/d3lam1e2_submitter_%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=frib-nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

mkdir -p /mnt/home/philipaa/scratch/barrier-paper-logs
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Submitter job: ${SLURM_JOB_ID:-unknown}"
echo "Running on: $(hostname)"
echo "Started at: $(date)"

cd "$REPO_DIR"
bash "$REPO_DIR/slurm/paper-scripts/submit_all_bnns_d3lam1e2.sh"

echo "Finished at: $(date)"
