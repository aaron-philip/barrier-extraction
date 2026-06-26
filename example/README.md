# Example BNN Workflow

`example_points.csv` shows the intended minimal input data shape:

```text
E_cm,Sigma
```

Run all commands from the repository root. This example builds a packed HDF5
dataset, trains a third-derivative-penalized SOC AutoBNN, merges the part file,
and plots the extracted barrier distribution.

## Single Train

1. Build the HDF5 dataset:

```bash
mkdir -p example/output
cd example/output
python ../../scripts/experimental_h5.py
cd ../..
```

2. Train one paper-style SOC BNN:

```bash
python scripts/train_bnns.py \
  --data example/output/experiments.h5 \
  --index 0 \
  --stride 1 \
  --out_h5 example/output/example_bnn.h5 \
  --kernel sum_of_changepoints \
  --likelihood_model normal_likelihood_lognormal_noise \
  --width 50 \
  --num_chains 4 \
  --num_draws 50 \
  --num_warmup 1000 \
  --d3_lambda 1.0 \
  --d3_lambda_space physical
```

3. Merge the part output:

```bash
python scripts/merge_bnn_parts.py \
  --out_h5 example/output/example_bnn.h5
```

4. Plot the extracted barrier distribution with posterior bands:

```bash
python example/plot_barrier_distribution.py \
  --h5 example/output/example_bnn.h5 \
  --out example/output/example_barrier.png
```

## Multiple Trains

BNN posterior fits can vary from run to run. Repeating a calculation is often
the easiest way to separate a robust barrier feature from a poor fit.

```bash
for run in 0 1 2 3; do
  python scripts/train_bnns.py \
    --data example/output/experiments.h5 \
    --index 0 \
    --sub_index "$run" \
    --stride 1 \
    --out_h5 example/output/example_bnn_repeats.h5 \
    --kernel sum_of_changepoints \
    --likelihood_model normal_likelihood_lognormal_noise \
    --width 50 \
    --num_chains 4 \
    --num_draws 50 \
    --num_warmup 1000 \
    --d3_lambda 1.0 \
    --d3_lambda_space physical
done

python scripts/merge_bnn_parts.py \
  --out_h5 example/output/example_bnn_repeats.h5

python example/plot_barrier_distribution.py \
  --h5 example/output/example_bnn_repeats.h5 \
  --out example/output/example_barrier_repeats.png \
  --max_groups 4
```

## If The First Fit Looks Bad

The first things to try are:

- Swap the SOC kernel for the changepoint-of-safe-products kernel:
  `--kernel changepoint_of_safe_products`.
- Change the penalty strength, for example `--d3_lambda 0.1`, `1.0`, or `10.0`.
- Retrain a few times and compare the merged posterior barrier plots.

## Paper-Scale Runs

The full calculations use the same training entry point with Slurm arrays. The
main paper-style toy-data run is:

```bash
sbatch slurm/paper-scripts/1-run_soc_wong_submitter-1.0d3.slurm
```

For one packed dataset, follow the comments in:

```text
slurm/paper-scripts/1-train_all_soc_to_wong-1.0d3.slurm
```
