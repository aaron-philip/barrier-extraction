#!/bin/bash --login
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_DIR

DATA_DIR="$REPO_DIR/data/packed"
RESULT_ROOT=$REPO_DIR/data/results/d3lambda1e2_rerun
RESULT_DIR="$RESULT_ROOT/bnns_with_df"
LOG_DIR=$REPO_DIR/logs

TRAIN_SLURM="$REPO_DIR/slurm/paper-scripts/misc-train_bnns_d3lam1e2.slurm"
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

ARRAY_SPEC="0-99"

wait_for_job () {
  local jobid="$1"
  echo "Waiting for job $jobid to finish..."

  if command -v sacct >/dev/null 2>&1; then
    while true; do
      state="$(sacct -j "$jobid" --noheader --format=State | head -n 1 | awk '{print $1}')"
      if [[ -z "${state:-}" ]]; then
        if squeue -j "$jobid" -h >/dev/null 2>&1 && squeue -j "$jobid" -h | grep -q .; then
          sleep 30
          continue
        fi
        sleep 30
        continue
      fi

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
          sleep 30
          ;;
      esac
    done
  else
    while squeue -j "$jobid" -h | grep -q .; do
      sleep 30
    done
    echo "Job $jobid left queue (no sacct available to verify final state)."
    return 0
  fi
}

dataset_has_active_pipeline () {
  local basename="$1"
  local queue
  queue="$(squeue -u "$USER" -h -o '%j' 2>/dev/null || true)"
  [[ "$queue" == *"bnn_d3lam1e2_${basename}"* ]] && return 0
  [[ "$queue" == *"merge_d3lam1e2_${basename}"* ]] && return 0
  [[ "$queue" == *"ec_d3lam1e2_${basename}"* ]] && return 0
  [[ "$queue" == *"ecn_d3lam1e2_${basename}"* ]] && return 0
  [[ "$queue" == *"met_d3lam1e2_${basename}"* ]] && return 0
  return 1
}

submit_with_retry () {
  local kind="$1"
  shift
  local output job_id
  while true; do
    set +e
    output="$("$@" 2>&1)"
    local status=$?
    set -e
    if [[ $status -eq 0 ]]; then
      job_id="$(printf '%s\n' "$output" | sed -n 's/^\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
      if [[ -n "$job_id" ]]; then
        printf '%s\n' "$job_id"
        return 0
      fi

      echo "ERROR: Could not parse Slurm job id while queuing $kind." >&2
      printf '%s\n' "$output" >&2
      return 1
    fi

    if printf '%s\n' "$output" | grep -q "QOSMaxSubmitJobPerUserLimit"; then
      echo "Submit limit hit while queuing $kind. Sleeping 120s and retrying..." >&2
      sleep 120
      continue
    fi

    printf '%s\n' "$output" >&2
    return "$status"
  done
}

for FILE in "${DATASETS[@]}"; do
  DATA_PATH="$DATA_DIR/$FILE"
  if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: missing dataset: $DATA_PATH"
    exit 1
  fi

  BASENAME="${FILE%.h5}"
  OUTH5="$RESULT_DIR/${BASENAME}_bnn_soc_d3lam1e2.h5"
  OUTCSV="${OUTH5}.csv"

  # Skip datasets that already made it all the way through merge.
  if [[ -f "$OUTH5" && -f "$OUTCSV" ]]; then
    echo "Skipping already-merged dataset: $BASENAME"
    continue
  fi

  if dataset_has_active_pipeline "$BASENAME"; then
    echo "Skipping already-active dataset: $BASENAME"
    continue
  fi

  echo "=============================="
  echo "Submitting d3lam1e2 pipeline for: $BASENAME"
  echo "=============================="

  TRAIN_JOB_ID=$(submit_with_retry "train $BASENAME" sbatch --parsable \
    --job-name="bnn_d3lam1e2_${BASENAME}" \
    --array="$ARRAY_SPEC" \
    --export=ALL,DATAFILE="$DATA_PATH" \
    "$TRAIN_SLURM")
  echo "  train: $TRAIN_JOB_ID"

  MERGE_JOB_ID=$(submit_with_retry "merge $BASENAME" sbatch --parsable \
    --dependency=afterok:$TRAIN_JOB_ID \
    --job-name="merge_d3lam1e2_${BASENAME}" \
    --output="$LOG_DIR/merge_d3lam1e2_${BASENAME}_%j.out" \
    --error="$LOG_DIR/merge_d3lam1e2_${BASENAME}_%j.err" \
    --time=06:00:00 --ntasks=1 --cpus-per-task=1 --mem=4G \
    --partition=frib-nodes \
    --wrap="source ~/.bashrc; conda deactivate; conda activate ~/envs/autobnn-reproduce2; python \"$MERGE_PY\" --out_h5 \"$OUTH5\"")
  echo "  merge: $MERGE_JOB_ID"

  EC_JOB_ID=$(submit_with_retry "ec $BASENAME" sbatch --parsable \
    --dependency=afterok:$MERGE_JOB_ID \
    --job-name="ec_d3lam1e2_${BASENAME}" \
    --output="$LOG_DIR/ec_d3lam1e2_${BASENAME}_%j.out" \
    --error="$LOG_DIR/ec_d3lam1e2_${BASENAME}_%j.err" \
    --time=06:00:00 --ntasks=1 --cpus-per-task=1 --mem=2G \
    --partition=frib-nodes \
    --wrap="source ~/.bashrc; conda deactivate; conda activate ~/envs/autobnn-reproduce2; python \"$PLOT_EC\" --csv \"$OUTCSV\" --which barrier")
  echo "  ec:        $EC_JOB_ID"

  ECN_JOB_ID=$(submit_with_retry "ecn $BASENAME" sbatch --parsable \
    --dependency=afterok:$MERGE_JOB_ID \
    --job-name="ecn_d3lam1e2_${BASENAME}" \
    --output="$LOG_DIR/ecn_d3lam1e2_${BASENAME}_%j.out" \
    --error="$LOG_DIR/ecn_d3lam1e2_${BASENAME}_%j.err" \
    --time=06:00:00 --ntasks=1 --cpus-per-task=1 --mem=2G \
    --partition=frib-nodes \
    --wrap="source ~/.bashrc; conda deactivate; conda activate ~/envs/autobnn-reproduce2; python \"$PLOT_EC\" --csv \"$OUTCSV\" --which barrier --normalize")
  echo "  ec (norm): $ECN_JOB_ID"

  MET_JOB_ID=$(submit_with_retry "met $BASENAME" sbatch --parsable \
    --dependency=afterok:$MERGE_JOB_ID \
    --job-name="met_d3lam1e2_${BASENAME}" \
    --output="$LOG_DIR/met_d3lam1e2_${BASENAME}_%j.out" \
    --error="$LOG_DIR/met_d3lam1e2_${BASENAME}_%j.err" \
    --time=06:00:00 --ntasks=1 --cpus-per-task=1 --mem=2G \
    --partition=frib-nodes \
    --wrap="source ~/.bashrc; conda deactivate; conda activate ~/envs/autobnn-reproduce2; python \"$PLOT_MET\" --h5 \"$OUTH5\"")
  echo "  met:       $MET_JOB_ID"

  wait_for_job "$EC_JOB_ID"
  wait_for_job "$ECN_JOB_ID"
  wait_for_job "$MET_JOB_ID"

  echo "Pipeline finished for: $BASENAME"
done

echo "Done (sequential): all d3lam1e2 dataset pipelines completed."
