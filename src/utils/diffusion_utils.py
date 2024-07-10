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