# barrier-extraction


Note: 
All codes run on Nvidia V100s with 
NVIDIA-SMI 560.35.05              Driver Version: 560.35.05      CUDA Version: 12.6 

Install instructions for these specs: 
conda create -p ~/envs/autobnn-env python=3.10
conda activate ~/envs/autobnn-env

pip install autobnn (I used version 0.0.2.dev0)
pip install "jax[cuda12]==0.5.3"
pip install bayeux-ml==0.1.15

Your system may require other tweaks
