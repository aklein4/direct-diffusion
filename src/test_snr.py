import torch

from diffusers import DDIMScheduler

from utils.diffusion_utils import compute_min_snr


config = DDIMScheduler.load_config('runwayml/stable-diffusion-v1-5', subfolder='scheduler')
config['rescale_betas_zero_snr'] = True
scheduler = DDIMScheduler.from_config(config)
scheduler.set_timesteps(scheduler.config.num_train_timesteps)

t = torch.arange(0, scheduler.config.num_train_timesteps).long()
print(compute_min_snr(scheduler, t, 5.0))