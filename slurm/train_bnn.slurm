#!/bin/bash --login
#SBATCH --job-name=train_bnn_${MODEL}_${LIKELIHOOD}
#SBATCH --time=2:00:00
#SBATCH --partition=frib-nodes
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --export=ALL,MODEL,LIKELIHOOD

# load your env
source ~/envs/choose_autobnn.sh

echo "Running on node $(hostname)"
echo "Model:      $MODEL"
echo "Likelihood: $LIKELIHOOD"

python train_all.py \
    --model      $MODEL \
    --likelihood $LIKELIHOOD \
    --num_warmup 1000

