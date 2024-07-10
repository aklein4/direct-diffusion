import torch

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import os
import argparse
import huggingface_hub as hf

from loaders.simple_loader import get_simple_loader
from models import get_components
from trainers import TRAINER_DICT

import utils.constants as constants
from utils.config_utils import load_train_config
from utils.logging_utils import log_print


# overwrite the default checkpointing function
from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint_fn
torch.utils.checkpoint.checkpoint = xla_checkpoint_fn


def _mp_fn(index, args):

    # setup
    torch.set_default_dtype(torch.float32)

    # debug infp
    log_print(
        f"Local Ordinal: {xm.get_local_ordinal()}, Local Master: {xm.is_master_ordinal(local=True)}, Master: {xm.is_master_ordinal(local=False)}, World Size: {xm.xrt_world_size()}"
    )

    log_print("Loading configs...")
    train_config = load_train_config(args.train_config)

    log_print("Loading model...")
    model, scheduler, tokenizer, text_encoder = get_components(args.model_url)

    model = model.to(constants.XLA_DEVICE())
    text_encoder = text_encoder.to(torch.bfloat16).to(constants.XLA_DEVICE())

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
        tokenizer,
        text_encoder,
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
    args.add_argument("--model_url", type=str, required=True)
    args.add_argument("--train_config", type=str, required=True)
    args.add_argument("--dataset", type=str, required=True)
    args.add_argument("--debug", action="store_true")
    args = args.parse_args()

    xmp.spawn(_mp_fn, args=(args,))
