# barrier-extraction

## Installation

After cloning this repository, install the pinned CPU dependencies with one command (optionally in a conda environment, which is what we do):

```bash
python -m pip install -r requirements.txt
```

This is the recommended default for new users because it avoids machine-specific
CUDA wheel resolution while still reproducing the AutoBNN/JAX/Oryx stack used by
the sample runs.

This installs `autobnn` from the `aaron-philip/autobnn-barrier` fork pinned to:

```text
d9bc1bdd7b0fd3197eea8af7c22f473ce85c0eb5
```

It also installs `oryx` from the upstream Git commit that is compatible with
`jax==0.5.3`. The plain PyPI `oryx==0.2.9` wheel can fail during
`sum_of_changepoints` initialization with:

```text
TypeError: abstract_eval_fun() missing 1 required keyword-only argument: 'debug_info'
```

While `autobnn-barrier` is private, this command requires SSH access to that
repository. Once the fork is public, this dependency can be switched to the
HTTPS GitHub URL without local source checkouts or path edits.

GPU users can add the CUDA 12 JAX extras after the baseline install:

```bash
python -m pip install -r requirements-cuda12.txt
```

The code has previously been run on Nvidia V100 GPUs with NVIDIA-SMI 560.35.05,
driver 560.35.05, and CUDA 12.6.

You can check that the correct forked API is available with:

```bash
python scripts/smoke_autobnn_dependency.py
```

Conda users can instead create an environment with:

```bash
conda env create -f environment.yml
conda activate barrier-extraction
```
