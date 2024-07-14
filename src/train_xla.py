import torch

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import os
import argparse
import huggingface_hub as hf

from diffusers import DDIMScheduler

from loaders.simple_loader import get_simple_loader
from models import CONFIG_DICT, MODEL_DICT
from trainers import TRAINER_DICT

import utils.constants as constants
from utils.config_utils import load_model_config, load_train_config
from utils.logging_utils import log_print


def _mp_fn(index, args):

    # setup
    torch.set_default_dtype(torch.float32)

    # debug infp
    log_print(
        f"Local Ordinal: {xm.get_local_ordinal()}, Local Master: {xm.is_master_ordinal(local=True)}, Master: {xm.is_master_ordinal(local=False)}, World Size: {xm.xrt_world_size()}"
    )

    log_print("Loading configs...")
    model_config = load_model_config(args.model_config)
    train_config = load_train_config(args.train_config)

    log_print("Loading model...")
    model_type = model_config["model_type"]
    model = MODEL_DICT[model_type](
        CONFIG_DICT[model_type](**model_config)
    )

    model = model.to(constants.XLA_DEVICE())
    
    if not args.debug:
        log_print("Syncing model...")

        # broadcast with bfloat16 for speed
        model = model.to(torch.bfloat16)
        xm.broadcast_master_param(model)
        model = model.to(torch.float32)

    log_print("Loading scheduler...")
    scheduler = DDIMScheduler(**model_config["scheduler"])
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)

    log_print("Loading data...")
    loader = get_simple_loader(
        args.dataset,
        "train",
        train_config["bs"],
    )

    log_print("Train!")
    trainer_type = train_config["trainer_type"]
    trainer = TRAINER_DICT[trainer_type](
        args.project,
        args.name,
        train_config,
        debug=args.debug
    )
    trainer.train(
        model,
        scheduler,
        loader
    )


if __name__ == '__main__':
  
    # setup PJRT runtime
    os.environ['PJRT_DEVICE'] = 'TPU'
    os.environ['XLA_NO_SPECIAL_SCALARS'] = '1'

    # handle arguments
    args = argparse.ArgumentParser()
    args.add_argument("--project", type=str, required=True)
    args.add_argument("--name", type=str, required=True)
    args.add_argument("--model_config", type=str, required=True)
    args.add_argument("--train_config", type=str, required=True)
    args.add_argument("--dataset", type=str, required=True)
    args.add_argument("--debug", action="store_true")
    args = args.parse_args()

    xmp.spawn(_mp_fn, args=(args,))
