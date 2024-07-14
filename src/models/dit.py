from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from transformers.activations import ACT2FN

from models.xla import XLAModel, XLAConfig
from utils.model_utils import (
    RMSHeadNorm,
    AdaLayerNorm, AdaGate,
    get_timestep_embedding, get_2d_sincos_pos_embed
)


class DiTConfig(XLAConfig):

    def __init__(
        self,
        hidden_size: int = 768,
        mlp_size: int = 1536,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        patch_size: int = 8,
        in_kernel_size: int = 3,
        out_kernel_size: int = 3,
        activation_fn: str = "gelu_pytorch_tanh",
        use_qkv_bias: bool = True,
        use_qk_norm: bool = True,
        initializer_range: float = 0.02,
        norm_eps: float = 1e-5,
        *args,
        **kwargs
    ):
        
        self.hidden_size = hidden_size
        self.mlp_size = mlp_size

        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads

        self.num_channels = num_channels
        self.patch_size = patch_size

        self.in_kernel_size = in_kernel_size
        self.out_kernel_size = out_kernel_size

        self.activation_fn = activation_fn
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm

        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        
        super().__init__(*args, **kwargs)


class Attention(nn.Module):

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.head_dim = self.hidden_size // self.num_heads
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads})."
            )
        
        self.use_qkv_bias = config.use_qkv_bias
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.use_qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.use_qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.use_qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.use_qk_norm = config.use_qk_norm
        self.q_norm = RMSHeadNorm(self.num_heads, self.head_dim, config.norm_eps) if self.use_qk_norm else None
        self.k_norm = RMSHeadNorm(self.num_heads, self.head_dim, config.norm_eps) if self.use_qk_norm else None


    def forward(
        self,
        hidden_states: torch.Tensor
    ):
        if isinstance(hidden_states, tuple):
            query_states, key_states, value_states = hidden_states
        else:
            query_states = key_states = value_states = hidden_states

        bsz, q_len, _ = query_states.shape

        # get tensors for attention
        query_states = self.q_proj(query_states)
        key_states = self.k_proj(key_states)
        value_states = self.v_proj(value_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        return self.o_proj(attn_output)
    

    def init_weights(self, std):
        self.q_proj.weight.data.normal_(mean=0.0, std=std)
        self.k_proj.weight.data.normal_(mean=0.0, std=std)
        self.v_proj.weight.data.normal_(mean=0.0, std=std)
        self.o_proj.weight.data.normal_(mean=0.0, std=std)

        if self.use_qkv_bias:
            self.q_proj.bias.data.zero_()
            self.k_proj.bias.data.zero_()
            self.v_proj.bias.data.zero_()


class GatedMLP(nn.Module):

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.mlp_size = config.mlp_size
        self.hidden_size = config.hidden_size

        self.gate_proj = nn.Linear(self.hidden_size, self.mlp_size, bias=False)
        self.value_proj = nn.Linear(self.hidden_size, self.mlp_size, bias=False)
        self.proj_down = nn.Linear(self.mlp_size, self.hidden_size, bias=False)

        self.act_fn = ACT2FN[config.activation_fn]

    
    def forward(
        self,
        hidden_states: torch.Tensor
    ):
        if isinstance(hidden_states, tuple):
            gate_states, value_states = hidden_states
        else:
            gate_states = value_states = hidden_states
        
        gate = self.gate_proj(gate_states)
        value = self.value_proj(value_states)

        intermediate = self.act_fn(gate) * value

        return self.proj_down(intermediate)


    def init_weights(self, std):
        self.gate_proj.weight.data.normal_(mean=0.0, std=std)
        self.value_proj.weight.data.normal_(mean=0.0, std=std)
        self.proj_down.weight.data.normal_(mean=0.0, std=std)


class DiTLayer(nn.Module):

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.attention = Attention(config)
        self.mlp = GatedMLP(config)

        self.norm1 = AdaLayerNorm(
            config.hidden_size, config.hidden_size,
            config.norm_eps, rank=None
        )
        self.norm2 = AdaLayerNorm(
            config.hidden_size, config.hidden_size,
            config.norm_eps, rank=None
        )

        self.gate1 = AdaGate(config.hidden_size, config.hidden_size)
        self.gate2 = AdaGate(config.hidden_size, config.hidden_size)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_emb: torch.Tensor
    ):
        
        attn_in = self.norm1(hidden_states, cond_emb)
        attn_out = self.attention(attn_in)
        hidden_states = hidden_states + self.gate1(attn_out, cond_emb)
        
        self.mlp_in = self.norm2(hidden_states, cond_emb)
        mlp_out = self.mlp(self.mlp_in)
        hidden_states = hidden_states + self.gate2(mlp_out, cond_emb)

        return hidden_states

    
    def init_weights(self, std):

        self.attention.init_weights(std)
        self.mlp.init_weights(std)

        self.norm1.init_weights(std)
        self.norm2.init_weights(std)

        self.gate1.init_weights(std)
        self.gate2.init_weights(std)


class DiT(XLAModel):

    layer_type = DiTLayer

    def __init__(self, config: DiTConfig):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.num_layers = config.num_layers

        # io
        self.proj_in = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.hidden_size,
            kernel_size=config.in_kernel_size * self.patch_size,
            stride=self.patch_size,
            bias=False,
            padding=(config.in_kernel_size * self.patch_size - 1) // 2
        )
        self.proj_out = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=self.num_channels * self.patch_size * self.patch_size,
            kernel_size=config.out_kernel_size,
            stride=1,
            bias=False,
            padding=(config.out_kernel_size - 1) // 2
        )

        # conditioning
        self.pos_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.layers = nn.ModuleList(
            [self.layer_type(config) for _ in range(self.num_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

        # Compute configuration
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    
    def init_weights(self):
        std = self.config.initializer_range

        # io
        self.proj_in.weight.data.normal_(mean=0.0, std=std)
        self.proj_out.weight.data.zero_()

        # conditioning
        self.pos_proj.weight.data.zero_()

        for layer in self.layers:
            layer.init_weights(std)


    def get_hidden_states(self, x, t):
        bs, c, h, w = x.shape
        
        assert h % self.patch_size == 0 and w % self.patch_size == 0, f"Image dimensions ({h}, {w}) must be divisible by patch size ({self.patch_size})."
        hp, wp = h // self.patch_size, w // self.patch_size

        hidden_states = self.proj_in(x) # [bs, d, h, w]
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous() # [bs, h, w//p, d//p]
        hidden_states = hidden_states.view(bs, hp*wp, self.hidden_size) # [bs, l, d]

        pos_emb = get_2d_sincos_pos_embed(self.hidden_size, (hp, wp), base_size=self.patch_size) # [l, d]
        pos_emb = torch.from_numpy(pos_emb).to(hidden_states.dtype).to(hidden_states.device) # [l, d]
        hidden_states = hidden_states + self.pos_proj(pos_emb).view(1, hp*wp, self.hidden_size) # [bs, l, d]

        return hidden_states
    

    def get_cond_emb(self, t):
        return get_timestep_embedding(t, self.hidden_size).unsqueeze(1)
    

    def get_output(self, hidden_states, x):
        bs, c, h, w = x.shape
        hp, wp = h // self.patch_size, w // self.patch_size

        out = hidden_states.view(bs, hp, wp, self.config.hidden_size) # [bs, hp, wp, d]
        out = out.permute(0, 3, 1, 2).contiguous() # [bs, d, hp, wp]
        out = self.proj_out(out) # [bs, c*p*p, hp, wp]
        out = out.view(bs, c*self.patch_size*self.patch_size, hp*wp) # [bs, c, l, p*p]
        out = F.fold(
            out,
            (h, w),
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size)
        ) # [bs, c, h*w]
        out = out.view(bs, c, h, w) # [bs, c, h, w]

        return out


    def forward(self, x, t):
        
        hidden_states = self.get_hidden_states(x, t)
        cond_emb = self.get_cond_emb(t)

        # run transformer
        for layer in self.layers:

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    cond_emb
                )

            else:
                hidden_states = layer(hidden_states, cond_emb)

        hidden_states = self.norm(hidden_states)
        out = self.get_output(hidden_states, x)

        return out
    