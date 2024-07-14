import torch

from models.dit import DiTConfig, DiT


def main():
    
    model = DiT(DiTConfig())

    x = torch.randn(4, 3, 128, 128)
    t = torch.randint(0, 1000, (4,)).long()
    y = model(x, t)

    print(y)
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    main()