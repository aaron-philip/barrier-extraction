#!/bin/bash --login
#SBATCH --job-name=bnn_train
#SBATCH --output=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_%A_%a.out
#SBATCH --error=/mnt/home/philipaa/scratch/barrier-paper-logs/bnn_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=frib-nodes
#SBATCH --array=0-999        # ← always 1000 samples per dataset

# Expect dataset variables
if [ -z "$DATAFILE" ] || [ -z "$OUTFILE" ]; then
    echo "ERROR: DATAFILE or OUTFILE not provided."
    exit 1
fi

# Activate environment
source ~/.bashrc
conda deactivate
conda activate ~/envs/autobnn-reproduce2
export JAX_PLATFORM=cpu

echo "Node: $(hostname)"
echo "Dataset: $DATAFILE"
echo "Output:  $OUTFILE"
echo "Sample index: $SLURM_ARRAY_TASK_ID"

python /mnt/research/ascsn/aaron-philip/barrier-extraction/scripts/train_bnns.py \
    --data "$DATAFILE" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --out_h5 "$OUTFILE"

echo "Done sample $SLURM_ARRAY_TASK_ID"
