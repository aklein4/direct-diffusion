from typing import List, Dict

import torch
import torch.nn.functional as F

try:
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    pass

import numpy as np
import PIL
import datasets
import requests

import utils.constants as constants
from utils.logging_utils import log_print


ZERO_IMG = np.zeros((constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3), dtype=np.uint8)


def load_image(url):
    try:
      image = PIL.Image.open(requests.get(url, stream=True, timeout=0.2).raw)
    except:
      return ZERO_IMG, True
    
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    w, h = image.size

    if w >= h:
        upper, lower = 0, h

        left = np.random.randint(w - h + 1)
        right = left + h

    else:
        left, right = 0, w

        upper = np.random.randint(h - w + 1)
        lower = upper + w

    image = image.crop((left, upper, right, lower))
    image = image.resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE))

    return np.asarray(image), False


def simple_collate_fn(data):

    images = []
    prompts = []
    valids = []

    for example in data:

        image, valid = load_image(example['url'])
        images.append(image)
        valids.append(valid)

        choice = np.random.randint(3)
        if choice == 0:
            prompts.append(example['caption'])
        elif choice == 1:
            prompts.append(example['caption_llava_short'])
        else:
            prompts.append(example['caption_llava'])

    tens = []
    for image in images:
        ten = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        ten = (ten * 2.0) - 1.0
        tens.append(ten)

    return prompts, torch.stack(tens), torch.tensor(valids)


def get_simple_loader(
    name: str,
    split: str,
    bs: int,
    mini_bs: int,
):

    # prepare batch sizes
    if constants.XLA_AVAILABLE:
        total_mini_bs = mini_bs * constants.NUM_XLA_DEVICES()
        if bs % total_mini_bs != 0:
            raise ValueError(f"Batch size {bs} not divisible by total mini batch size {total_mini_bs}")
        if total_mini_bs > bs:
            log_print(f"Warning: total mini batch size {total_mini_bs} larger than batch size {bs}")
        sample_size = mini_bs * (bs // total_mini_bs)
    else:
        sample_size = bs

    # get streaming dataset
    dataset = datasets.load_dataset(name, split)

    # wrap in loader with collator
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=sample_size,
        collate_fn=simple_collate_fn,
        drop_last=True,
        num_workers=8
    )

    if not constants.XLA_AVAILABLE:
        return loader

    # wrap with xla loader
    wrapper_type = pl.MpDeviceLoader if constants.NUM_XLA_DEVICES() > 1 else pl.ParallelLoader
    xm_loader = wrapper_type(loader, device=constants.XLA_DEVICE())

    return xm_loader
