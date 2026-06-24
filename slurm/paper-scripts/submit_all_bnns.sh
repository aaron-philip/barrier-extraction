#!/bin/bash --login
set -euo pipefail

DATA_DIR=/mnt/home/philipaa/scratch/barrier-paper-datasets
RESULT_DIR=/mnt/home/philipaa/scratch/barrier-paper-results/bnns_with_df
LOG_DIR=/mnt/home/philipaa/scratch/barrier-paper-logs

REPO_DIR="${REPO_DIR:-/mnt/home/philipaa/research/barrier-extraction}"
export REPO_DIR
TRAIN_SLURM="$REPO_DIR/slurm/paper-scripts/train_bnns.slurm"
MERGE_PY="$REPO_DIR/scripts/merge_bnn_parts.py"
PLOT_EC="$REPO_DIR/scripts/plot_empirical_coverage.py"
PLOT_MET="$REPO_DIR/scripts/plot_with_metrics.py"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

DATASETS=(
  cv0.00_spacing0.5.h5
  cv0.25_spacing0.5.h5
  cv0.50_spacing0.5.h5
  cv0.75_spacing0.5.h5
  cv0.00_spacing1.0.h5
  cv0.25_spacing1.0.h5
  cv0.50_spacing1.0.h5
  cv0.75_spacing1.0.h5
  res_0.25MeV_noise0.01.h5
  res_0.50MeV_noise0.01.h5
  res_1.00MeV_noise0.01.h5
  res_2.00MeV_noise0.01.h5
  res_0.25MeV_noise0.02.h5
  res_0.50MeV_noise0.02.h5
  res_1.00MeV_noise0.02.h5
  res_2.00MeV_noise0.02.h5
)

# throttle within a dataset (array task concurrency)
ARRAY_SPEC="0-99"

wait_for_job () {
  local jobid="$1"
  echo "Waiting for job $jobid to finish..."

  # Prefer sacct if available (works even after job leaves queue)
  if command -v sacct >/dev/null 2>&1; then
    while true; do
      # state can be COMPLETED, FAILED, CANCELLED, TIMEOUT, etc.
      state="$(sacct -j "$jobid" --noheader --format=State | head -n 1 | awk '{print $1}')"
      if [[ -z "${state:-}" ]]; then
        # sacct can lag; fall back to squeue briefly
        if squeue -j "$jobid" -h >/dev/null 2>&1 && squeue -j "$jobid" -h | grep -q .; then
          sleep 30
          continue
        fi
        # still nothing; try again
        sleep 30
        continue
      fi

      # normalize state like COMPLETED+ to COMPLETED
      state="${state%%+*}"

      case "$state" in
        COMPLETED)
          echo "Job $jobid completed."
          return 0
          ;;
        FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
          echo "ERROR: Job $jobid ended with state: $state"
          return 1
          ;;
        *)
          # PENDING, RUNNING, CONFIGURING, COMPLETING, etc.
          sleep 30
          ;;
      esac
    done
  else
    # Fallback: only knows while job is in queue
    while squeue -j "$jobid" -h | grep -q .; do
      sleep 30
    done
    echo "Job $jobid left queue (no sacct available to verify final state)."
    return 0
  fi
}

for FILE in "${DATASETS[@]}"; do
  DATA_PATH="$DATA_DIR/$FILE"
  if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: missing dataset: $DATA_PATH"
    exit 1
  fi

  BASENAME="${FILE%.h5}"
  OUTH5="$RESULT_DIR/${BASENAME}_bnn_soc.h5"
  OUTCSV="${OUTH5}.csv"

  echo "=============================="
  echo "Submitting pipeline for: $BASENAME"
  echo "=============================="

  # 1) training array
  TRAIN_JOB_ID=$(sbatch --parsable \
    --job-name="bnn_${BASENAME}" \
    --array="$ARRAY_SPEC" \
    --export=ALL,DATAFILE="$DATA_PATH" \
    "$TRAIN_SLURM")
  echo "  train: $TRAIN_JOB_ID"

  # 2) merge (after training)
  MERGE_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:$TRAIN_JOB_ID \
    --job-name="merge_${BASENAME}" \
    --output="$LOG_DIR/merge_${BASENAME}_%j.out" \
    --error="$LOG_DIR/merge_${BASENAME}_%j.err" \
    --time=00:20:00 --ntasks=1 --cpus-per-task=1 --mem=4G \
    --partition=frib-nodes \
    --wrap="source ~/.bashrc; conda deactivate; conda activate ~/envs/autobnn-reproduce2; python \"$MERGE_PY\" --out_h5 \"$OUTH5\"")
  echo "  merge: $MERGE_JOB_ID"

  # 3a) empirical coverage plot (UNNORMALIZED) after merge
  EC_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:$MERGE_JOB_ID \
    --job-name="ec_${BASENAME}" \
    --output="$LOG_DIR/ec_${BASENAME}_%j.out" \
    --error="$LOG_DIR/ec_${BASENAME}_%j.err" \
    --time=00:10:00 --ntasks=1 --cpus-per-task=1 --mem=2G \
    --partition=frib-nodes \
    --wrap="source ~/.bashrc; conda deactivate; conda activate ~/envs/autobnn-reproduce2; python \"$PLOT_EC\" --csv \"$OUTCSV\" --which barrier")
  echo "  ec:        $EC_JOB_ID"

  # 3a.2) empirical coverage plot (NORMALIZED) after merge
  ECN_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:$MERGE_JOB_ID \
    --job-name="ecn_${BASENAME}" \
    --output="$LOG_DIR/ecn_${BASENAME}_%j.out" \
    --error="$LOG_DIR/ecn_${BASENAME}_%j.err" \
    --time=00:10:00 --ntasks=1 --cpus-per-task=1 --mem=2G \
    --partition=frib-nodes \
    --wrap="source ~/.bashrc; conda deactivate; conda activate ~/envs/autobnn-reproduce2; python \"$PLOT_EC\" --csv \"$OUTCSV\" --which barrier --normalize")
  echo "  ec (norm): $ECN_JOB_ID"

  # 3b) plot_with_metrics (after merge)
  MET_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:$MERGE_JOB_ID \
    --job-name="met_${BASENAME}" \
    --output="$LOG_DIR/met_${BASENAME}_%j.out" \
    --error="$LOG_DIR/met_${BASENAME}_%j.err" \
    --time=00:10:00 --ntasks=1 --cpus-per-task=1 --mem=2G \
    --partition=frib-nodes \
    --wrap="source ~/.bashrc; conda deactivate; conda activate ~/envs/autobnn-reproduce2; python \"$PLOT_MET\" --h5 \"$OUTH5\"")
  echo "  met:       $MET_JOB_ID"

  # ---- SEQUENTIAL GATE ----
  # Wait until the "last thing" is done. MET depends on MERGE; EC/ECN also depend on MERGE.
  # So we wait for ALL three plot jobs (or just wait for MET + ECN, your call).
  wait_for_job "$EC_JOB_ID"
  wait_for_job "$ECN_JOB_ID"
  wait_for_job "$MET_JOB_ID"

  echo "Pipeline finished for: $BASENAME"
done

echo "Done (sequential): all dataset pipelines completed."
