
from tqdm import tqdm

from loaders.simple_loader import get_simple_loader
from loaders.imagenet_loader import get_imagenet_loader

loader = get_imagenet_loader(
    'ILSVRC/imagenet-1k',
    "train",
    8,
)

for x, labels in loader:
    print(labels)
    break
