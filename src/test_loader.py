
from tqdm import tqdm

from loaders.simple_loader import get_simple_loader


loader = get_simple_loader(
    'aklein4/coyo-llava-hq',
    "validation",
    8,
)

with tqdm() as pbar:
    for prompts, x, mask in loader:
        pbar.update(mask.int().sum().item())
