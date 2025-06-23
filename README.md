# barrier-extraction

## Installation

> **Note:**  
> All code is tested on Nvidia V100 GPUs with  
> - NVIDIA-SMI 560.35.05  
> - Driver Version: 560.35.05  
> - CUDA Version: 12.6  

```bash
# 1. Create and activate a Python 3.10 conda environment
conda create -p ~/envs/autobnn-env python=3.10
conda activate ~/envs/autobnn-env

# 2. Install AutoBNN (v0.0.2.dev0)
pip install autobnn==0.0.2.dev0

# 3. Install JAX with CUDA-12 support
pip install "jax[cuda12]==0.5.3"

# 4. Install Bayeux for MCMC
pip install bayeux-ml==0.1.15
