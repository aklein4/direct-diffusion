import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.constants as constants


def encode_prompts(prompts, text_encoder, tokenizer):

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(text_encoder.device)

    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        input_ids,
        attention_mask=attention_mask
    )[0]

    return prompt_embeds


def encode_images(images):
    
    latents = F.unfold(
        images,
        kernel_size=constants.PATCH_SIZE,
        stride=constants.PATCH_SIZE
    )

    latents = latents.view(
       images.shape[0],
       constants.LATENT_DEPTH,
       constants.LATENT_SIZE,
       constants.LATENT_SIZE
    )

    return latents


def decode_latents(latents):
  
    latents = latents.view(
        latents.shape[0],
        constants.LATENT_DEPTH,
        constants.LATENT_SIZE**2
    )

    images = F.fold(
        latents,
        (constants.IMAGE_SIZE, constants.IMAGE_SIZE),
        kernel_size=constants.PATCH_SIZE,
        stride=constants.PATCH_SIZE
    )

    return images


def compute_min_snr(scheduler, timesteps, gamma, epsilon):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2

    # apply min
    min_snr = torch.clip(snr, min=epsilon, max=gamma)

    return min_snr


def add_more_noise(
    scheduler,
    samples: torch.Tensor,
    new_noise: torch.Tensor,
    timesteps: torch.IntTensor,
    new_timesteps: torch.IntTensor,
) -> torch.Tensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    # Move the scheduler.alphas_cumprod to device to avoid redundant CPU to GPU data movement
    # for the subsequent add_noise calls
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=samples.device)
    alphas_cumprod = scheduler.alphas_cumprod.to(dtype=samples.dtype)
    timesteps = timesteps.to(samples.device)
    new_timesteps = new_timesteps.to(samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    new_sqrt_alpha_prod = alphas_cumprod[new_timesteps] ** 0.5
    new_sqrt_alpha_prod = new_sqrt_alpha_prod.flatten()

    while len(sqrt_alpha_prod.shape) < len(samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        new_sqrt_alpha_prod = new_sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    new_sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[new_timesteps]) ** 0.5
    new_sqrt_one_minus_alpha_prod = new_sqrt_one_minus_alpha_prod.flatten()
    
    while len(sqrt_one_minus_alpha_prod.shape) < len(samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        new_sqrt_one_minus_alpha_prod = new_sqrt_one_minus_alpha_prod.unsqueeze(-1)

    signal_ratio = new_sqrt_alpha_prod / sqrt_alpha_prod
    new_samples = (
            signal_ratio * samples + 
            new_sqrt_one_minus_alpha_prod * (1 - (signal_ratio * sqrt_one_minus_alpha_prod)) * new_noise
    )

    return new_samples


def step_to(
    scheduler,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    prev_timestep
):
    assert prev_timestep <= timestep

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep]
    while len(alpha_prod_t.shape) < len(sample.shape):
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        alpha_prod_t_prev = alpha_prod_t_prev.unsqueeze(-1)

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - 0.0**2) ** (0.5) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    return prev_sample


def insert_noise(
    scheduler,
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.IntTensor,
) -> torch.Tensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    # Move the scheduler.alphas_cumprod to device to avoid redundant CPU to GPU data movement
    # for the subsequent add_noise calls
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=original_samples.device)
    alphas_cumprod = scheduler.alphas_cumprod.to(dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples