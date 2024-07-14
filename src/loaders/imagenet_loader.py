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
from multiprocessing.pool import ThreadPool

import utils.constants as constants
from utils.logging_utils import log_print


def format_image(image):

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
    image = image.resize((128, 128))

    if np.random.rand() < 0.5:
        image = PIL.ImageOps.mirror(image)

    return np.asarray(image).copy()


def imagenet_collate_fn(data):

    images = [format_image(example['image']) for example in data]
    labels = [example['label'] for example in data]

    tens = []
    for image in images:
        ten = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        ten = (ten * 2.0) - 1.0
        tens.append(ten)

    return torch.stack(tens), torch.tensor(labels).long()


def get_imagenet_loader(
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
    dataset = datasets.load_dataset(
        name,
        split=split,
        streaming=True,
        trust_remote_code=True
    )

    # wrap in loader with collator
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=sample_size,
        collate_fn=imagenet_collate_fn,
        drop_last=True,
        num_workers=(min(8, dataset.n_shards) if constants.XLA_AVAILABLE else 0),
    )

    if not constants.XLA_AVAILABLE:
        return loader

    # wrap with xla loader
    wrapper_type = pl.MpDeviceLoader if constants.NUM_XLA_DEVICES() > 1 else pl.ParallelLoader
    xm_loader = wrapper_type(loader, device=constants.XLA_DEVICE())

    return xm_loader
