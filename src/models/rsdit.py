from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from transformers.activations import ACT2FN

from models.xla import XLAModel
from models.dit import (
    DiT, DiTConfig
)
from utils.model_utils import (
    RMSHeadNorm, AdaLayerNorm, AdaGate,
    get_2d_sincos_pos_embed
)


class RSDiTConfig(DiTConfig):

    model_type = "rsdit"

    def __init__(
        self,
        num_residuals: int = 4,
        selection_groups: int = 4,
        ada_rank: int = 64,
        *args,
        **kwargs
    ):
        
        self.num_residuals = num_residuals
        self.selection_groups = selection_groups

        self.ada_rank = ada_rank
        
        super().__init__(*args, **kwargs)


class SelectIn(nn.Module):

    def __init__(self, config: RSDiTConfig, num_out=1):
        super().__init__()
        self.config = config
        self.num_out = num_out

        self.hidden_size = config.hidden_size
        self.num_residuals = config.num_residuals
        self.selection_groups = config.selection_groups

        self.proj_in = nn.Conv1d(
            self.hidden_size,
            self.hidden_size*num_out,
            kernel_size=self.num_residuals,
            stride=self.num_residuals,
            groups=self.selection_groups,
            bias=False
        )

    
    def forward(self, hidden_states):
        bs, l, d, r = hidden_states.shape

        hidden_states = hidden_states.view(bs*l, d, r)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = hidden_states.view(bs, l, d * self.num_out)

        return hidden_states
    

    def init_weights(self, std):
        assert self.hidden_size % self.selection_groups == 0, f"Hidden size ({self.hidden_size}) must be divisible by selection groups ({self.selection_groups})."
        params_per = self.num_residuals * self.hidden_size / self.selection_groups

        self.proj_in.weight.data.normal_(
            mean=0.0,
            std=1/params_per
        )

class SelectOut(nn.Module):

    def __init__(self, config: RSDiTConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_residuals = config.num_residuals
        self.selection_groups = config.selection_groups

        self.proj_out = nn.Conv1d(
            self.hidden_size,
            self.num_residuals * self.hidden_size,
            kernel_size=1,
            stride=1,
            groups=self.selection_groups,
            bias=False
        )

    
    def forward(self, hidden_states):
        bs, l, d = hidden_states.shape
        r = self.num_residuals

        hidden_states = hidden_states.view(bs*l, d, 1) # [bs*l, d, 1]
        hidden_states = self.proj_out(hidden_states) # [bs*l, r*d, 1]
        
        hidden_states = hidden_states.view(bs, l, d, r) # [bs, l, d, r]

        return hidden_states


    def init_weights(self, std):
        params_per = self.hidden_size / self.selection_groups
        self.proj_out.weight.data.normal_(
            mean=0.0,
            std=1/params_per
        )


class CondIn(nn.Module):

    def __init__(self, config: RSDiTConfig, num_out: int):
        super().__init__()
        self.config = config
        self.num_out = num_out

        self.proj_in = SelectIn(config, num_out)
        self.norm = AdaLayerNorm(
            num_out * config.hidden_size, config.hidden_size,
            config.norm_eps, config.ada_rank
        )
    

    def forward(self, hidden_states, cond_emb):
        bs, l, d, r = hidden_states.shape
        
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.norm(
            hidden_states,
            cond_emb
        )

        return hidden_states
    

    def init_weights(self, std):
        self.proj_in.init_weights(std)
        self.norm.init_weights(std)


class CondOut(nn.Module):

    def __init__(self, config: RSDiTConfig):
        super().__init__()
        self.config = config

        self.proj_out = SelectOut(config)
        self.gate = AdaGate(
            config.hidden_size, config.hidden_size
        )
    

    def forward(self, hidden_states, cond_emb):
        hidden_states = self.gate(hidden_states, cond_emb)
        hidden_states = self.proj_out(hidden_states)

        return hidden_states
    

    def init_weights(self, std):
        self.proj_out.init_weights(std)
        self.gate.init_weights(std)


class RSAttention(nn.Module):

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
        self.qkv_proj = nn.Conv1d(
            3 * self.hidden_size,
            3 * self.hidden_size,
            kernel_size=1,
            stride=1,
            groups=3,
            bias=self.use_qkv_bias
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.use_qk_norm = config.use_qk_norm
        self.q_norm = RMSHeadNorm(self.num_heads, self.head_dim, config.norm_eps) if self.use_qk_norm else None
        self.k_norm = RMSHeadNorm(self.num_heads, self.head_dim, config.norm_eps) if self.use_qk_norm else None


    def forward(
        self,
        hidden_states: torch.Tensor
    ):
        bsz, q_len, _ = hidden_states.shape

        # get tensors for attention
        qkv_states = self.qkv_proj(hidden_states.view(bsz*q_len, 3*self.hidden_size, 1)).view(bsz, q_len, 3*self.hidden_size)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)

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
        self.qkv_proj.weight.data.normal_(mean=0.0, std=std)
        self.o_proj.weight.data.normal_(mean=0.0, std=std)

        if self.use_qkv_bias:
            self.qkv_proj.bias.data.zero_()


class RSGatedMLP(nn.Module):

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.mlp_size = config.mlp_size
        self.hidden_size = config.hidden_size

        self.up_proj = nn.Conv1d(
            2 * self.hidden_size,
            2 * self.mlp_size,
            kernel_size=1,
            stride=1,
            bias=False,
            groups=2
        )
        self.proj_down = nn.Linear(self.mlp_size, self.hidden_size, bias=False)

        self.act_fn = ACT2FN[config.activation_fn]

    
    def forward(
        self,
        hidden_states: torch.Tensor
    ):
        bs, l, _ = hidden_states.shape
        
        states = self.up_proj(hidden_states.view(bs*l, 2*self.hidden_size, 1)).view(bs, l, 2*self.mlp_size)
        gate, value = states.chunk(2, dim=-1)

        intermediate = self.act_fn(gate) * value

        return self.proj_down(intermediate)


    def init_weights(self, std):
        self.up_proj.weight.data.normal_(mean=0.0, std=std)
        self.proj_down.weight.data.normal_(mean=0.0, std=std)


class RSDiTLayer(nn.Module):

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.attention = RSAttention(config)
        self.mlp = RSGatedMLP(config)

        self.attn_norm = CondIn(config, 3)
        self.mlp_norm = CondIn(config, 2)
        
        self.attn_gate = CondOut(config)
        self.mlp_gate = CondOut(config)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_emb: torch.Tensor
    ):
        
        attn_in = self.attn_norm(hidden_states, cond_emb)
        attn_out = self.attention(attn_in)
        hidden_states = hidden_states + self.attn_gate(attn_out, cond_emb)
        
        mlp_in = self.mlp_norm(hidden_states, cond_emb)
        mlp_out = self.mlp(mlp_in)
        hidden_states = hidden_states + self.mlp_gate(mlp_out, cond_emb)

        return hidden_states

    
    def init_weights(self, std):

        self.attention.init_weights(std)
        self.mlp.init_weights(std)

        self.attn_norm.init_weights(std)
        self.mlp_norm.init_weights(std)

        self.attn_gate.init_weights(std)
        self.mlp_gate.init_weights(std)


class RSDiT(DiT):

    config_class = RSDiTConfig
    layer_type = RSDiTLayer

    def __init__(self, config: DiTConfig):
        XLAModel.__init__(self, config)

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
            padding=((config.in_kernel_size - 1) * self.patch_size) // 2
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

        # RS
        self.prep_x = SelectOut(config)
        self.prep_pos = SelectOut(config)
        self.post_select = SelectIn(config)

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

        # RS
        self.prep_x.init_weights(std)
        self.prep_pos.init_weights(std)
        self.post_select.init_weights(std)

        for layer in self.layers:
            layer.init_weights(std)


    def get_hidden_states(self, x, t):
        bs, c, h, w = x.shape
        
        assert h % self.patch_size == 0 and w % self.patch_size == 0, f"Image dimensions ({h}, {w}) must be divisible by patch size ({self.patch_size})."
        hp, wp = h // self.patch_size, w // self.patch_size

        hidden_states = self.proj_in(x) # [bs, d, h, w]
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous() # [bs, h, w//p, d//p]
        hidden_states = hidden_states.view(bs, hp*wp, self.hidden_size) # [bs, l, d]
        hidden_states = self.prep_x(hidden_states) # [bs, l, r, d]

        pos_emb = get_2d_sincos_pos_embed(self.hidden_size, (hp, wp), base_size=self.patch_size) # [l, d]
        pos_emb = torch.from_numpy(pos_emb).to(hidden_states.dtype).to(hidden_states.device) # [l, d]
        pos_emb = self.pos_proj(pos_emb).view(1, hp*wp, self.hidden_size) # [bs, l, d]
        pos_emb = self.prep_pos(pos_emb) # [bs, l, r, d]
        hidden_states = hidden_states + pos_emb

        return hidden_states
    

    def get_output(self, hidden_states, x):
        hidden_states = self.post_select(hidden_states) # [bs, l, d, r]
        return super().get_output(hidden_states, x)
