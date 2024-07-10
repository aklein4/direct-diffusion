import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.xla_base_trainer import XLABaseTrainer
from utils.data_utils import DotDict
from  utils.training_utils import masked_mse_loss
from utils.diffusion_utils import (
    encode_prompts,
    encode_images, decode_latents,
    compute_min_snr
)
import utils.constants as constants


class XLADirectTrainer(XLABaseTrainer):

    def train_step(
        self,
        model,
        scheduler,
        tokenizer,
        text_encoder,
        prompts,
        x,
        mask,
        uncond_embeds
    ):

        # encode prompts
        with torch.no_grad():
            prompt_embeds = encode_prompts(
                prompts,
                text_encoder,
                tokenizer
            )

        # replace with uncond_embeds
        uncond_coin = torch.rand(
            x.shape[0],
            device=constants.XLA_DEVICE()
        ) < self.uncond_prob
        embeds = torch.where(
            uncond_coin.unsqueeze(-1).unsqueeze(-1),
            uncond_embeds,
            prompt_embeds
        )

        # diffusion process
        noise = torch.randn_like(x)
        t = torch.randint(
            0, scheduler.config.num_train_timesteps,
            [len(x)],
            device=constants.XLA_DEVICE(),
            dtype=torch.long
        )

        # input peturbation
        peturbation = torch.randn_like(x)
        peturbed_noise = noise + self.ip_gamma * peturbation

        # add noise
        noisy = scheduler.add_noise(x, peturbed_noise, t)

        # get the model output
        model_pred = model(
            encode_images(noisy),
            t,
            embeds,
        ).sample
        model_pred = decode_latents(model_pred)

        weight = compute_min_snr(scheduler, t, self.snr_gamma, self.snr_epsilon)
        loss = masked_mse_loss(
            model_pred,
            x,
            mask,
            weight=weight,
            mask_epsilon=self.mask_epsilon
        )
        uncond_loss = masked_mse_loss(
            model_pred,
            x,
            torch.logical_and(mask, uncond_coin),
            weight=weight,
            mask_epsilon=self.mask_epsilon
        )
        cond_loss = masked_mse_loss(
            model_pred,
            x,
            torch.logical_and(mask, ~uncond_coin),
            weight=weight,
            mask_epsilon=self.mask_epsilon
        )

        return DotDict(
            loss=loss,
            uncond_loss=uncond_loss,
            cond_loss=cond_loss
        )
