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

## Paired resolution experiment

`notebooks/ResolutionExperiment.ipynb` compares barrier MAE and empirical-coverage
error for BNN (green), log-space RBF GP (blue), and adjacent three-point estimates
(black). Ten parameter samples share the same noise realization across seven
subsampled resolutions from 0.25 to 3 MeV. The dataset manifest records the source
hash and selection; existing data with different provenance are rejected.

From the repository root:

```bash
sbatch slurm/paper-scripts/3-resolution-submit.slurm
```

This prepares/verifies `data/packed/resolution_experiment`, submits 70 BNN fits and
70 GP fits (seven GP tasks), then merges and scores all fits with an `afterok`
dependency. Results, job IDs (`jobs.env`), per-curve scores, averages, and PNG/PDF
plots go under `data/results/resolution_experiment_soc_d3lam1_gp_log/`.
The notebook regenerates scores and plots without training.

MAE is unnormalized, in mb/MeV, evaluated against analytic Wong truth at native
prediction points without interpolating the three-point estimate. All methods
use the same fixed interior energy interval (3 MeV removed at each end).
Coverage error averages absolute deviations from 25%, 50%, 68%, 90%, and 95%
nominal coverage within each curve, then across curves. Missing or nonfinite
fits cause scoring to fail rather than silently reducing the sample count.

The launch scripts use the explicit `ENV_PATH/bin/python` interpreter and prefer
its site-packages to user-site copies, clearing inherited module PYTHONPATHs.
They run the AutoBNN API/model-initialization smoke check before submitting.
On this cluster, remaining dependencies are supplied by the Python 3.10 user
site; a clean installation should use `requirements.txt`. Override `ENV_PATH`
for another installation and `REPO_DIR` when submitting outside the repository.
Use a fresh `RESULT_DIR` for another run; the submission lock prevents duplicates.
Do not remove a submission lock without first checking `jobs.env` and SLURM.

Lightweight scientific checks after dataset preparation:

```bash
~/envs/autobnn-test-fork/bin/python scripts/test_resolution_experiment.py
for f in slurm/paper-scripts/3-resolution-*.slurm; do bash -n "$f"; done
```

The resolution notebook now defaults to corrected paper-style scores: pointwise
posterior median barriers and per-curve MAE divided by R0^2/hbaromega. Its coverage
summary measures the deviation of the average coverage curve at the same levels
as PaperFigures (including 0%, excluding 100%). This differs from the original
batch summary's raw mean-based MAE and per-curve coverage error.
`scripts/score_resolution_paper.py` rescores saved HDF5 curves, writing separate
`paper_metrics_common/` or `paper_metrics_paper/` outputs. The common mode keeps
the 3 MeV interior cut for all methods; paper mode uses the original model CSV
bounds and cannot give three-point estimates the same extrapolation support.
The notebook's interval table records the actual approximately 2/3 MeV stencil
spacings; exact target intervals are unavailable on some coarser grids.

### Interior-only paper violin and coverage plots

In `notebooks/PaperFigures.ipynb`, the plotting `SETTINGS` now enables
`interior_metrics=True` and `inset_mev=0.0`. Violin and empirical-coverage metrics
are recomputed from saved posterior curves within the measured energy range.
Set `inset_mev=3.0` to remove another 3 MeV from each end, or set
`interior_metrics=False` to use the original saved scores. Re-run the settings
cell and the desired violin/coverage cells after changing either setting.
Normalization and median-based violin predictions retain the paper convention.

When no compact export exists for the chosen inset, the first run rescores the
paired GP/BNN curves; subsequent calls reuse separate
`interior_metric_cache/` CSVs beside the source result files. Cache keys include
input and scoring-code fingerprints. Original result CSVs and HDF5s are preserved.
Figures are saved under `SAVE_DIR/interior_inset0MeV/gps/` or `gps_log/` (with the
inset reflected in the directory name). Experimental stencils select integer steps with average intervals closest to
2 or 3 MeV for each reaction.

Interior rescoring needs full posterior HDF5s, not only the published metric
CSVs. Some repository HDF5s contain only a plotted example. `PaperFigures` now
checks that every requested fit exists and, when needed, uses
`SETTINGS['full_h5_root']` (default:
`~/scratch/barrier-paper-results/d3lambda1_rerun`). That directory must contain
`gps/`, `gps_log/`, and `bnns_with_df/`. The chosen HDF5 participates in the cache
fingerprint, and its truth parameters are checked against the CSV. Missing fits
are never silently dropped or substituted with the old padded metrics.

`ResolutionExperiment.ipynb` now also defaults to the measured energy range:
`INTERVAL_MODE='interior'`, `INSET_MEV=0.0`, matching PaperFigures' current inset.
Its refreshed scores and figures use `paper_metrics_interior_inset0MeV/`;
prior padded and 3 MeV-inset outputs remain separate. The resolution-plot x-axis
label is delta E (MeV).

### GitHub-sized plot inputs

`PaperFigures.ipynb` includes a vertically stacked two-panel resolution figure
after the violin plots: (a) normalized RMSE (L2) and (b) Mean Coverage Deviation.
It shows BNN, GP, and the approximately 3 MeV three-point estimate (solid black
triangles), with one legend in the upper panel. Exports are
`resolution_comparison.png` and `.pdf`; the x-label is delta E (MeV).

The default interior violin, EC, and resolution plots now load compact checked
CSVs from `data/paper_plot_metrics/interior_inset0MeV/`. They do not need full
posterior HDF5s or scratch access. Existing experimental/example figures continue
to use their existing inputs. The standalone resolution notebook uses the same
compact exports. See `data/paper_plot_metrics/README.md` for the column conventions
and the explicit regeneration command. Other inset settings still require
rescoring saved predictions if no matching export is present.

Include the compact export directory, the seven small subsampled input HDF5s,
notebooks, helper/export scripts, and SLURM scripts when committing this experiment.
The approximately 85 MB of resolution posterior results and local interior metric
caches are ignored; they can be regenerated using the documented SLURM workflow.

For a fresh checkout, run either plotting notebook directly using the committed
compact exports. Notebook outputs are cleared to avoid embedding duplicate images.
Before SLURM submission, run `mkdir -p logs` from the repository root. Dataset
preparation checks source hashes and values independently of checkout location.

For interactive plotting, install `seaborn` and `jupyterlab` in the environment
(`python -m pip install seaborn jupyterlab`). PaperFigures uses SciencePlots
with LaTeX by default; without a TeX installation, set
`plt.rcParams["text.usetex"] = False` after the first cell. Both plotting
notebooks were checked end to end with this Matplotlib mathtext setting using
only versioned inputs. Generated files in `plots/` are intentionally ignored.
