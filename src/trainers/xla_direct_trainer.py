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
    add_more_noise, step_to, insert_noise
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
        noisy = scheduler.add_noise(x, noise, t)

        if self.il_lambda is not None:
            with torch.no_grad():
                t_il = torch.clamp(t + self.il_step, max=scheduler.config.num_train_timesteps - 1)

                noise_il = torch.randn_like(x)
                noisy_il = add_more_noise(scheduler, noisy, noise_il, t, t_il)

                il_pred = model(
                    torch.cat([encode_images(noisy_il), encode_images(noisy_il)], dim=0),
                    torch.cat([t_il, t_il], dim=0),
                    torch.cat([uncond_embeds.expand(t.shape[0], -1, -1), prompt_embeds], dim=0),
                ).sample
                uncond_pred, cond_pred = il_pred.chunk(2, dim=0)
                il_pred = (cond_pred + self.il_guidance * (cond_pred - uncond_pred))
                il_pred = decode_latents(il_pred)

                il_sample = step_to(scheduler, il_pred, t_il, noisy_il, t)
                il_diff = il_sample - noisy

                il_scale = torch.zeros_like(t).float()
                il_scale.exponential_(self.il_lambda).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                noisy = noisy + il_scale * il_diff

        # input peturbation
        peturbation = self.ip_gamma * torch.randn_like(x)
        noisy = insert_noise(scheduler, noisy, peturbation, t)

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
