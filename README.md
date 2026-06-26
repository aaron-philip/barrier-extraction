# barrier-extraction

This repository contains code, notebooks, Slurm workflows, examples, and saved
results for extracting heavy-ion fusion barrier distributions from fusion
cross-section data. It compares Gaussian-process reconstructions with AutoBNN
compositional Bayesian neural-network reconstructions [3], including the
third-derivative regularization used in the paper and thesis.

Contact: philipaa@mit.edu

## Quick Start

Create the conda environment used by the Slurm scripts:

```bash
mkdir ~/envs
conda create -p ~/envs/autobnn-test-fork python=3.10 -y
conda activate ~/envs/autobnn-test-fork
python -m pip install -r requirements.txt
python scripts/smoke_autobnn_dependency.py
```

If you want to try out the AutoBNN regression tool for your own fusion data, start with `example/`. Just swap in your fusion data and follow the prompts to train an AutoBNN model and extract a barrier distribution.

Training too slow? GPU users can add the CUDA 12 JAX extras after the baseline install:

```bash
python -m pip install -r requirements-cuda12.txt
```

## Repository Map

`example/` is the fastest way to try the workflow on your own data. 

`notebooks/` reproduces the paper and thesis plots. The notebooks can be run
directly to explore the Gaussian-process and AutoBNN Bayesian inference
approaches investigated in the paper [1] and thesis [2].

`scripts/` contains the reusable command-line training, merging, metric, and
plotting entry points.

`slurm/` contains the heavy batch calculations used to reproduce the results in
`data/`. The paper-scale scripts live in `slurm/paper-scripts/`; they call the
commands in `scripts/` and write the HDF5/CSV outputs used by the notebooks.

`data/` contains packed datasets and saved result files used by the notebooks.

## Running On Your HPC System

The Slurm files are meant to be edited lightly for local clusters. The usual
changes are:

- Update `#SBATCH --partition`, `--time`, `--mem`, and array sizes.
- Update the conda activation line if your environment path differs from
  `autobnn-test-fork`.
- Update `REPO_DIR`, `DATA_DIR`, `RESULT_DIR`, or scratch paths if your cluster
  separates code, datasets, and long-lived outputs.
- Decide whether to run CPU-only, as in several paper scripts, or enable GPU
  execution and install `requirements-cuda12.txt`.

For the main toy-data paper pipeline, see:

```bash
sbatch slurm/paper-scripts/1-run_soc_wong_submitter-1.0d3.slurm
```

For one dataset, see the comments in:

```text
slurm/paper-scripts/1-train_all_soc_to_wong-1.0d3.slurm
```

## Dependency Notes

This installs `autobnn` from the `aaron-philip/autobnn-barrier` source snapshot pinned in
`requirements.txt`. We slightly modified the original autobnn code (`tensorflow/probability/spinoffs/autobnn`) to expose certain MCMC parameters, add priors in the form of derivative penalties, and potentially incorporate quoted exp. uncertainties. Visit `aaron-philip/autobnn-barrier` to learn more!

`oryx` is also installed from a compatible upstream Git commit. The plain PyPI
`oryx==0.2.9` wheel can fail during `sum_of_changepoints` initialization with:

```text
TypeError: abstract_eval_fun() missing 1 required keyword-only argument: 'debug_info'
```

## Citations

[1] Paper citation: TODO, manuscript forthcoming.

[2] Thesis citation: TODO, manuscript forthcoming.

[3] Urs Köster, "AutoBNN: Probabilistic time series forecasting with
compositional Bayesian neural networks," Google Research Blog, March 28, 2024.
https://research.google/blog/autobnn-probabilistic-time-series-forecasting-with-compositional-bayesian-neural-networks/
