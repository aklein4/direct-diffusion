import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.xla_base_trainer import XLABaseTrainer
from utils.data_utils import DotDict
from  utils.training_utils import masked_mse_loss
from utils.diffusion_utils import (
    encode_prompts,
    encode_images, decode_latents,
    compute_min_snr,
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

        x0 = x
        t = torch.randint(
            0, scheduler.config.num_train_timesteps,
            [len(x)],
            device=constants.XLA_DEVICE(),
            dtype=torch.long
        )

        if self.il_max is not None:

            x_il = scheduler.add_noise(x, torch.randn_like(x), t)
            pred_il = model(
                encode_images(x_il),
                t,
                prompt_embeds,
            ).sample.detach()
            pred_il = decode_latents(pred_il)

            scale_il = torch.rand(
                x.shape[0],
                device=constants.XLA_DEVICE()
            ) * self.il_max
            while len(scale_il.shape) < len(pred_il.shape):
                scale_il = scale_il.unsqueeze(-1)
            
            x0 = x0 + scale_il * (pred_il - x0)

        # diffusion process
        noise = torch.randn_like(x) + self.ip_gamma * torch.randn_like(x)
        noisy = scheduler.add_noise(x0, noise, t)

        # get the model output
        model_pred = model(
            encode_images(noisy),
            t,
            embeds,
        ).sample
        model_pred = decode_latents(model_pred)

        weight = self.snr_scale * compute_min_snr(scheduler, t, self.snr_gamma, self.snr_epsilon)
        loss, loss_denom = masked_mse_loss(
            model_pred,
            x,
            mask,
            weight=weight,
        )
        uncond_loss, uncond_loss_denom = masked_mse_loss(
            model_pred,
            x,
            torch.logical_and(mask, uncond_coin),
            weight=weight,
        )
        cond_loss, cond_loss_denom = masked_mse_loss(
            model_pred,
            x,
            torch.logical_and(mask, ~uncond_coin),
            weight=weight,
        )

        loss_out = loss / (loss_denom + self.mask_epsilon)

        return loss_out, DotDict(
            loss=(loss, loss_denom),
            uncond_loss=(uncond_loss, uncond_loss_denom),
            cond_loss=(cond_loss, cond_loss_denom),
        )
