#!/bin/bash
# submit_all.sh

models=(sum_of_safe_shallow sum_of_changepoints sum_of_products sum_of_stumps sum_of_stumps_and_products changepoint_of_safe_products mlp_depth2 mlp_depth3 mlp_depth4 mlp_depth5)
likelihoods=(normal_likelihood_logistic_noise normal_likelihood_lognormal_noise)

mkdir -p logs
for model in "${models[@]}"; do
  for lik in "${likelihoods[@]}"; do
    sbatch \
      --job-name=train_bnn_${model}_${lik} \
      --output=logs/train_bnn_${model}_${lik}.out \
      --error=logs/train_bnn_${model}_${lik}.err \
      --export=MODEL=${model},LIKELIHOOD=${lik} \
      train_bnn.slurm
  done
done
