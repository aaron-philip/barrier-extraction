#!/bin/bash --login
#SBATCH --time=2:00:00
#SBATCH --partition=frib-nodes
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --export=ALL,MODEL,LIKELIHOOD,BASEDIR

echo "Running on node: $(hostname)"
echo "Model:      ${MODEL}"
echo "Likelihood: ${LIKELIHOOD}"
echo "Base dir:   ${BASEDIR}"

# make per-run results dir
mkdir -p "${RESULTS_DIR}"

# load your env
source ~/envs/choose_autobnn.sh

# call the correct script
python train_all.py \
    --model   "$MODEL" \
    --likelihood "$LIKELIHOOD" \
    --num_warmup 1000 \
    --out_dir   "$BASEDIR"

#--e_train    "${ETRAIN_FILE:-}" \
#--sigma_train "${SIGMA_FILE:-}" \


