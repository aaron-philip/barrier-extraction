# barrier-extraction

## Installation

> **Note:**  
> All code is tested on Nvidia V100 GPUs with  
> - NVIDIA-SMI 560.35.05  
> - Driver Version: 560.35.05  
> - CUDA Version: 12.6

After cloning this repository, install the pinned dependencies with one command:

```bash
python -m pip install -r requirements.txt
```

This installs `autobnn` from the `aaron-philip/autobnn-barrier` fork pinned to:

```text
d9bc1bdd7b0fd3197eea8af7c22f473ce85c0eb5
```

While `autobnn-barrier` is private, this command requires SSH access to that
repository. Once the fork is public, this dependency can be switched to the
HTTPS GitHub URL without local source checkouts or path edits.

You can check that the correct forked API is available with:

```bash
python scripts/smoke_autobnn_dependency.py
```

Conda users can instead create an environment with:

```bash
conda env create -f environment.yml
conda activate barrier-extraction
```
