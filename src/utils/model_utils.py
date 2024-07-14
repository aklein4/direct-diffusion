import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class RMSHeadNorm(nn.Module):

    def __init__(self, num_heads: int, head_dim: int, norm_eps):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim

        self.norm_eps = norm_eps
        self.mean_div = np.sqrt(self.head_dim)

        self.scales = nn.Parameter(torch.ones(1, self.num_heads, 1, self.head_dim))

    
    def forward(self, hidden_states: torch.Tensor):
        assert hidden_states.shape[1] == self.num_heads
        assert hidden_states.shape[-1] == self.head_dim

        inv_norms = torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        hidden_states = hidden_states * inv_norms

        out = hidden_states * self.scales
        return out.to(hidden_states.dtype)


class AdaLayerNorm(nn.Module):

    def __init__(self, hidden_size: int, cond_size: int, norm_eps: float, rank=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cond_size = cond_size
        self.rank = rank

        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps, elementwise_affine=False)

        if rank is None:
            self.bias = nn.Linear(cond_size, hidden_size, bias=True)
            self.gain = nn.Linear(cond_size, hidden_size, bias=True)
        else:
            self.bias = nn.Sequential(
                nn.Linear(cond_size, rank, bias=False),
                nn.Linear(rank, hidden_size, bias=True)
            )
            self.gain = nn.Sequential(
                nn.Linear(cond_size, rank, bias=False),
                nn.Linear(rank, hidden_size, bias=True)
            )

    
    def forward(self, hidden_states: torch.Tensor, cond_states: torch.Tensor):
        hidden_states = self.norm(hidden_states)
        
        shift = self.bias(cond_states)
        scale = self.gain(cond_states)
        
        return hidden_states * scale + shift
    

    def init_weights(self, std):

        if self.rank is None:

            self.bias.weight.data.zero_()
            self.bias.bias.data.zero_()

            self.gain.weight.data.zero_()
            self.gain.bias.data.fill_(1.0)

        else:

            self.bias[0].weight.data.normal_(mean=0.0, std=std)
            self.bias[1].weight.data.zero_()
            self.bias[1].bias.data.zero_()

            self.gain[0].weight.data.normal_(mean=0.0, std=std)
            self.gain[1].weight.data.zero_()
            self.gain[1].bias.data.fill_(1.0)


class AdaGate(nn.Module):

    def __init__(self, hidden_size: int, cond_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cond_size = cond_size
    
        self.gate = nn.Linear(cond_size, hidden_size, bias=True)


    def forward(self, hidden_states: torch.Tensor, cond_states: torch.Tensor):
        scale = self.gate(cond_states)

        return hidden_states * scale
    

    def init_weights(self, std):
        self.gate.weight.data.zero_()
        self.gate.bias.data.fill_(1.0)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -np.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
