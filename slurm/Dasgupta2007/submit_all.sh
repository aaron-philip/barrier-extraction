#!/bin/bash
models=(sum_of_changepoints sum_of_products \
        sum_of_stumps sum_of_stumps_and_products changepoint_of_safe_products)

likelihoods=(normal_likelihood_lognormal_noise)

# no spaces around =
BASEDIR=/mnt/home/philipaa/scratch/barrier-extraction/Dasgupta2007

mkdir -p "${BASEDIR}/logs"

for model in "${models[@]}"; do
  for lik in "${likelihoods[@]}"; do
    sbatch \
      --job-name=train_bnn_${model}_${lik} \
      --output=${BASEDIR}/logs/train_bnn_${model}_${lik}.out \
      --error=${BASEDIR}/logs/train_bnn_${model}_${lik}.err \
      --export=MODEL=${model},LIKELIHOOD=${lik},BASEDIR=${BASEDIR} \
      train_all.slurm
  done
done
