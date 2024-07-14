import torch

import numpy as np

from models.dit import DiTConfig, DiT
from models.rsdit import RSDiTConfig, RSDiT

from utils.config_utils import load_model_config

DIT_CONFIG = "med-dit"
RSDIT_CONFIG = "med-rsdit"


def main():
    
    base_model = DiT(DiTConfig(**load_model_config(DIT_CONFIG)))
    model = RSDiT(RSDiTConfig(**load_model_config(RSDIT_CONFIG)))

    print(f"Base: {np.sum([p.numel() for p in base_model.parameters()]):_}")
    print(f"RS: {np.sum([p.numel() for p in model.parameters()]):_}")
    print(f"Diff: {
        np.sum([p.numel() for p in model.parameters()]) -
        np.sum([p.numel() for p in base_model.parameters()])
    :_}")
    

if __name__ == '__main__':
    main()