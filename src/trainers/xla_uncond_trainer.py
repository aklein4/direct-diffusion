from typing import Optional

import torch

import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast, syncfree

import os
import numpy as np

from trainers.xla_base_trainer import XLABaseTrainer
import utils.constants as constants
from utils.data_utils import DotDict
from utils.logging_utils import LogSection, log_print, log_master_print
from utils.diffusion_utils import (
    insert_noise, compute_min_snr
)


class XLAUncondTrainer(XLABaseTrainer):

    def train(
        self,
        model,
        scheduler,
        loader
    ):

        # create ema weights
        ema = {k: v.clone().detach() for k, v in model.state_dict().items()}

        # init models
        model.requires_grad_(True)
        model.train()

        # init training objs
        optimizer = syncfree.AdamW(
            model.parameters(), lr=self.start_lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        lr_scheduler = self._get_lr_scheduler(optimizer)

        # loop
        curr_step = 0
        image_tracker = xm.RateTracker()
        step_tracker = xm.RateTracker()
        self.seen_images = 0
        while True:
            for x, labels in loader:

                # prepare x for accum
                n_x = x.shape[0]
                if n_x % self.mini_bs != 0:
                    log_print(f"Warning: sample size {n_x} not divisible by mini batch size {self.mini_bs}")
                if n_x * constants.NUM_XLA_DEVICES() != self.bs:
                    log_print(f"Warning: sample size {n_x} with {constants.NUM_XLA_DEVICES()} devices does not match batch size {self.bs}")
               
                x_split = torch.split(x, self.mini_bs, dim=0)

                # accumulate gradients
                results_accum = DotDict()
                for split_idx in range(len(x_split)):
                    
                    mini_x = x_split[split_idx]

                    # get results from train step
                    with autocast(constants.XLA_DEVICE()):
                        results = self.train_step(
                            model,
                            scheduler,
                            mini_x,
                        )

                        # scale results for accumulation
                        for k, v in results.items():
                            results[k] = v / (len(x_split) * constants.NUM_XLA_DEVICES())

                        # save results
                        with torch.no_grad():
                            for k, v in results.items():
                                if k not in results_accum:
                                    results_accum[k] = 0.0
                                results_accum[k] = results_accum[k] + v.detach()
                
                    results.loss.backward()
                    if len(x_split) > 1:
                        xm.mark_step()

                # perform a single optimizer step
                xm.optimizer_step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                
                # update the ema weights
                for k, v in model.state_dict().items():
                    if self.ema is not None:
                        ema[k] = self.ema * ema[k] + (1 - self.ema) * v.detach()
                    else:
                        ema[k] = v

                # update lr
                self.log.lr = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()

                # tracking
                image_tracker.add(self.bs)
                step_tracker.add(1)
                curr_step += 1
                self.log.steps_completed = curr_step

                def _post_step():

                    # log seen images
                    self.seen_images += self.bs
                    self.log.seen_images = self.seen_images

                    for k, v in results_accum.items():
                        r = xm.mesh_reduce(f"{k}_reduce", v, np.sum)
                        self.log[k] = r

                    # print update
                    msg = [
                        f"Step {curr_step}",
                        f"LR = {self.log.lr:.2e}",
                        f"Loss = {self.log.loss:.4f}",
                        f"{step_tracker.rate():.2f} steps/s",
                        f"{round(3600*image_tracker.rate()):_} img/h"
                    ]
                    log_master_print("{: >15}{: >20}{: >20}{: >20}{: >23}".format(*msg))
                
                    # save
                    self.log_step()
                    if curr_step % self.checkpoint_interval == 0:
                        try:
                            self.save_checkpoint(
                                {
                                    'ema': (ema, True)
                                },
                                curr_step
                            )
                        except:
                            log_master_print("Warning: checkpoint save failed!")
                
                # add closure
                xm.add_step_closure(_post_step)
        

    def train_step(
        self,
        model,
        scheduler,
        x
    ):
        bs = len(x)

        # diffusion process
        noise = torch.randn_like(x) + self.ip_gamma * torch.randn_like(x)
        t = torch.randint(
            0, scheduler.config.num_train_timesteps,
            [bs],
            device=constants.XLA_DEVICE(),
            dtype=torch.long
        )
        noisy = scheduler.add_noise(x, noise, t)

        # get the model output
        model_pred = model(noisy, t)

        weight = self.snr_scale * compute_min_snr(scheduler, t, self.snr_gamma, self.snr_epsilon)
        loss = (x.view(bs, -1) - model_pred.view(bs, -1)).pow(2).mean(-1)
        loss = (loss * weight).mean()

        return DotDict(
            loss=loss
        )
