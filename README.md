# barrier-extraction

## Quick Start

Create the conda environment used by the Slurm scripts:

```bash
conda create -n autobnn-test-fork python=3.10 -y
conda activate autobnn-test-fork
python -m pip install -r requirements.txt
python scripts/smoke_autobnn_dependency.py
```

If you want to run something immediately, start with `example/`. It contains a
small CSV input and copy-paste commands for a complete BNN fit, merge, and
barrier plot.

GPU users can add the CUDA 12 JAX extras after the baseline install:

```bash
python -m pip install -r requirements-cuda12.txt
```

## Repository Map

`example/` is the fastest way to try the workflow on a tiny dataset.

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

This installs `autobnn` from the `aaron-philip/autobnn-barrier` fork pinned in
`requirements.txt`. While that fork is private, installation requires SSH access
to the repository. Once the fork is public, the dependency can be switched to an
HTTPS GitHub URL.

`oryx` is also installed from a compatible upstream Git commit. The plain PyPI
`oryx==0.2.9` wheel can fail during `sum_of_changepoints` initialization with:

```text
TypeError: abstract_eval_fun() missing 1 required keyword-only argument: 'debug_info'
```

## Citations

[1] Paper citation: TODO, arXiv forthcoming.

[2] Thesis citation: TODO, arXiv forthcoming.
