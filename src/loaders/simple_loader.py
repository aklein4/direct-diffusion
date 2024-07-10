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
      image = PIL.Image.open(requests.get(url, stream=True, timeout=0.5).raw)
    except:
      return ZERO_IMG, False
    
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

    return np.asarray(image).copy(), True


def simple_collate_fn(data):

    images = []
    valids = []
    prompts = []
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
    bs: int
):

    # prepare batch sizes
    if constants.XLA_AVAILABLE:
        if bs % constants.NUM_XLA_DEVICES() != 0:
            raise ValueError(f"Batch size {bs} not divisible by number of devices {constants.NUM_XLA_DEVICES()}")
        sample_size = bs // constants.NUM_XLA_DEVICES()
    else:
        sample_size = bs

    # get streaming dataset
    dataset = datasets.load_dataset(name, split=split, streaming=True)

    # wrap in loader with collator
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=sample_size,
        collate_fn=simple_collate_fn,
        drop_last=True,
        num_workers=(8 if constants.XLA_AVAILABLE else 0),
    )

    if not constants.XLA_AVAILABLE:
        return loader

    # wrap with xla loader
    wrapper_type = pl.MpDeviceLoader if constants.NUM_XLA_DEVICES() > 1 else pl.ParallelLoader
    xm_loader = wrapper_type(loader, device=constants.XLA_DEVICE())

    return xm_loader
