# Example BNN Workflow

`example_points.csv` shows the intended minimal input data shape:

```text
E_cm,Sigma
```

The commands below use the existing scripts in `scripts/` to reproduce the BNN
training workflow: build the packed HDF5 dataset, train, merge part files, and
plot. Run all commands from the repository root.

## Single Train

1. Build the HDF5 dataset:

```bash
mkdir -p example/output
cd example/output
python ../../scripts/experimental_h5.py
cd ../..
```

2. Train one penalized BNN:

```bash
python scripts/train_bnns.py \
  --data example/output/experiments.h5 \
  --index 0 \
  --stride 1 \
  --out_h5 example/output/example_bnn.h5 \
  --kernel sum_of_changepoints \
  --likelihood_model normal_likelihood_lognormal_noise \
  --width 20 \
  --num_chains 2 \
  --num_draws 20 \
  --num_warmup 20 \
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

1. Build the HDF5 dataset as in the single-train workflow.

2. Run several repeated trainings:

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
    --width 20 \
    --num_chains 2 \
    --num_draws 20 \
    --num_warmup 20 \
    --d3_lambda 1.0 \
    --d3_lambda_space physical
done
```

3. Merge the repeated trainings:

```bash
python scripts/merge_bnn_parts.py \
  --out_h5 example/output/example_bnn_repeats.h5
```

4. Plot the extracted barrier distributions with posterior bands:

```bash
python example/plot_barrier_distribution.py \
  --h5 example/output/example_bnn_repeats.h5 \
  --out example/output/example_barrier_repeats.png \
  --max_groups 4
```

## HPC / SLURM Runs

For a single paper-style dataset, use the SLURM script in
`slurm/barrier-paper`. Submit it from the repository root and pass the dataset
through `DATAFILE`:

```bash
sbatch --export=ALL,DATAFILE=/path/to/dataset.h5 \
  slurm/barrier-paper/train_bnns_d3lam1.slurm
```

For example:

```bash
sbatch --export=ALL,DATAFILE=/mnt/home/philipaa/scratch/barrier-paper-datasets/res_0.50MeV_noise0.01.h5 \
  slurm/barrier-paper/train_bnns_d3lam1.slurm
```

That script submits the BNN training array and then submits the merge job with
the correct dependency. It is configured for the paper dataset layout and writes
outputs under the result paths set inside `slurm/barrier-paper/train_bnns_d3lam1.slurm`.

To submit the full hardcoded d3-lambda-1 paper batch, run:

```bash
bash slurm/barrier-paper/submit_all_bnns_d3lam1.sh
```

That submitter loops over the datasets listed in the script, submits the BNN
array jobs, merges each dataset after successful training, and queues the
post-merge plotting/coverage jobs.
