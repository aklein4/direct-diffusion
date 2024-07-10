

from loaders.simple_loader import get_simple_loader


loader = get_simple_loader(
    'aklein4/coyo-llava-hq',
    "validation",
    4,
    2
)


for prompts, x, mask in loader:
    print(prompts)
    print(x)
    print(mask)
    break
