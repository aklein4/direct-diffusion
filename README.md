# Direct Diffusion

A diffusion model that replaces the VAE from standard implementations with a lossless reshaping and linear projection.


# TPU VM Setup Instructions

1. Create VM with version: tpu-ubuntu2204-base

2. git clone https://github.com/aklein4/direct-diffusion.git

3. cd ~/direct-diffusion && . setup_vm.sh <HF_TOKEN> <WANDB_TOKEN>
