#!/bin/bash --login
#SBATCH --job-name=resolution_fit
#SBATCH --output=logs/resolution_fit_%A_%a.out
#SBATCH --error=logs/resolution_fit_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=frib-nodes
set -euo pipefail
: "${REPO_DIR:?Submit via 3-resolution-submit.slurm}"
: "${DATA_DIR:?}"
: "${RESULT_DIR:?}"
: "${MODE:?}"
ENV_PATH="${ENV_PATH:-$HOME/envs/autobnn-test-fork}"
source "$REPO_DIR/slurm/paper-scripts/3-resolution-env.sh"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
cd "$REPO_DIR"
TAGS=(res_0.25MeV res_0.50MeV res_1.00MeV res_1.50MeV res_2.00MeV res_2.50MeV res_3.00MeV)
if [[ "$MODE" == finalize ]]; then
  for TAG in "${TAGS[@]}"; do
    "$PYTHON" scripts/merge_bnn_parts.py --out_h5 "$RESULT_DIR/${TAG}_bnn.h5"
  done
  "$PYTHON" scripts/resolution_experiment.py summarize --data-dir "$DATA_DIR" --result-dir "$RESULT_DIR"
  exit 0
fi
TASK=${SLURM_ARRAY_TASK_ID:?}
if [[ "$MODE" == bnn ]]; then
  RES_ID=$(( TASK / 10 ))
  INDEX=$(( TASK % 10 ))
elif [[ "$MODE" == gp ]]; then
  RES_ID=$TASK
  INDEX=all
else
  echo "Unknown MODE=$MODE" >&2
  exit 1
fi
[[ "$RES_ID" -ge 0 && "$RES_ID" -lt 7 ]]
TAG=${TAGS[$RES_ID]}
DATA_PATH="$DATA_DIR/$TAG.h5"
OUTFILE="$RESULT_DIR/${TAG}_${MODE}.h5"
echo "mode=$MODE task=$TASK resolution_id=$RES_ID index=$INDEX dataset=$DATA_PATH output=$OUTFILE seed=0 host=$(hostname)"
if [[ "$MODE" == bnn ]]; then
  echo 'BNN: sum_of_changepoints, log Sigma, width=50, chains=4, warmup=1000, draws=100, d3_lambda=1 physical, penalty on log Sigma'
  "$PYTHON" scripts/train_bnns.py --data "$DATA_PATH" --index "$INDEX" --stride 1 \
    --out_h5 "$OUTFILE" --random_state 0 --kernel sum_of_changepoints \
    --likelihood_model normal_likelihood_lognormal_noise --width 50 \
    --num_chains 4 --num_warmup 1000 --num_draws 100 \
    --d3_lambda 1.0 --d3_lambda_space physical
else
  echo 'GP: log Sigma, RBF, restarts=20, samples=500; ten curves'
  "$PYTHON" scripts/train_gps.py --data "$DATA_PATH" --out_h5 "$OUTFILE" \
    --target_space log --random_state 0 --n_restarts 20 --n_samples 500 --log_every 1
  # train_gps catches individual failures: turn incomplete output into job failure.
  "$PYTHON" - "$OUTFILE" <<'PY'
import sys
import h5py
with h5py.File(sys.argv[1]) as f:
    assert set(f) == {str(i) for i in range(10)}, 'Missing GP fits'
    for g in f.values():
        assert 'y_samples' in g and 'hw_true' in g.attrs, 'Incomplete GP group'
PY
fi
