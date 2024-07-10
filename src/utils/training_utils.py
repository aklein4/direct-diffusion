from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logging_utils import log_master_print


def masked_mse_loss(
    pred,
    target,
    mask,
    weight=None
):
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)
    mask = mask.float()

    loss = (pred - target).pow(2).mean(dim=-1)

    if weight is not None:
        log_master_print(loss.dtype, weight.dtype)
        loss = loss * weight
    
    mask = mask.float()
    loss = loss * mask

    loss = loss.sum() / (mask.sum() + 1e-6)

    return loss