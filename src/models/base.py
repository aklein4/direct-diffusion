from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint_fn
except ImportError:
    pass

import functools

from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.activations import ACT2FN

from utils.logging_utils import log_print
import utils.constants as constants


class BaseConfig(PretrainedConfig):
    """
    Base configuration class for experiments.

    Args:
        vocab_size (`int`):
            Vocabulary size of the model. Defines the number of different tokens that
            can be represented by the `inputs_ids`.
        max_sequence_length (`int`):
            The maximum sequence length that this model might ever be used with.
        hidden_size (`int`):
            Number of hidden layers in the Transformer decoder.
        mlp_size (`int`):
            Dimension of the MLP representations.
        num_hidden_layers (`int`):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`):
            Number of attention heads for each attention layer in the Transformer encoder.
        use_qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not the model should use bias for qkv layers.
        dropout (`float`, *optional*, defaults to `None`):
            The dropout ratio for the output of each attention and ff component.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing
             all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the normalization layers.
        bos_token_id (int, *optional*, defaults to 0):
            The id of the `BOS` token in the vocabulary.
        eos_token_id (int, *optional*, defaults to 0):
            The id of the `EOS` token in the vocabulary.
        pad_token_id (int, *optional*, defaults to 0):
            The id of the `PAD` token in the vocabulary.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether or not to use gradient checkpointing for the model.
        gradient_checkpointing_layers (`int`, *optional*, defaults to 0):
            The number of layers to use gradient checkpointing for.
    """

    model_type = 'base'

    def __init__(
        self,
        vocab_size,
        max_sequence_length,
        hidden_size,
        mlp_size,
        num_hidden_layers,
        num_attention_heads,
        use_qkv_bias=True,
        dropout=None,
        hidden_act="silu",
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        gradient_checkpointing=False,
        gradient_checkpointing_layers=0,
        *args,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length

        self.hidden_size = hidden_size
        self.mlp_size = mlp_size

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        self.use_qkv_bias = use_qkv_bias
        self.dropout = dropout
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        # requires workaround
        tmp_gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_layers = gradient_checkpointing_layers

        # init with work arounds
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing = tmp_gradient_checkpointing


class BaseModel(PreTrainedModel):

    config_class = BaseConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    # from StableLM
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


    # converted from torch to torch xla
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        
        gradient_checkpointing_func = functools.partial(xla_checkpoint_fn, **gradient_checkpointing_kwargs)
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        
        log_print(f"Gradient checkpointing enabled for {self.__class__.__name__}: {self.gradient_checkpointing}")


class BaseAttention(nn.Module):

    def __init__(self, config: BaseConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.num_heads * self.head_dim, bias=config.use_qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ):
        bsz, q_len, _ = hidden_states.shape

        # get tensors for attention
        query_states, key_states, value_states = self.qkv_proj(hidden_states).chunk(3, dim=-1)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # update/apply cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if attention_mask is None and q_len > 1 else False

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output


class BaseMLP(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.mlp_size = config.mlp_size

        self.in_proj = nn.Linear(self.hidden_size, 2*self.mlp_size, bias=False)
        self.down_proj = nn.Linear(self.mlp_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]


    def forward(self, hidden_state):
        gate, h = self.in_proj(hidden_state).chunk(2, dim=-1)

        return self.down_proj(self.act_fn(gate) * h)


class BaseLayer(nn.Module):
    def __init__(self, config: BaseConfig, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.attn = BaseAttention(config, layer_idx)
        self.mlp = BaseMLP(config)

        self.attn_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_layernorm = None

        self.dropout = nn.Dropout(config.dropout) if config.dropout is not None else nn.Identity()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ):

        # Self Attention
        attn_out = self.attn(
            self.attn_layernorm(hidden_states),
            attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = hidden_states + self.dropout(attn_out)

        # GLU MLP
        mlp_out = self.mlp(self.mlp_layernorm(hidden_states))
        hidden_states = hidden_states + self.dropout(mlp_out)

        return hidden_states


class BaseTransformer(BaseModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # info
        self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length

        # weights
        self.vocab_embs = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embs = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.layers = nn.ModuleList(
            [BaseLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Compute configuration
        self.gradient_checkpointing_layers = config.gradient_checkpointing_layers
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def get_tokens(
        self,
        input_ids: torch.LongTensor
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    
    @torch.no_grad()
    def _get_mask(
        self,
        input_ids: torch.LongTensor,
        mask: Optional[torch.BoolTensor]=None,
        segment_ids: Optional[torch.LongTensor]=None,
        cached_mask=False
    ) -> torch.BoolTensor:
        batch_size, seq_length = input_ids.shape

        if cached_mask:
            return mask

        # error check
        if (mask is not None or segment_ids is not None) and self._attn_implementation.count('flash_attention_2'):
            raise ValueError("Custom attention mask and segmend_ids are not supported for Flash Attention!")

        # default eager causal mask
        if mask is None:
            mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)
        else:
            assert mask.dtype == torch.bool, f"Non-cached mask must be boolean, got {mask.dtype}"

        # must have batch dimension
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # apply segment ids
        if segment_ids is not None:
            assert segment_ids.shape == input_ids.shape, f"Segment ids ({segment_ids.shape}) must have same shape as input ids ({input_ids.shape})"

            segment_mask = segment_ids[:, None, :] != segment_ids[:, :, None]
            mask = mask | segment_mask

        # process for attn version
        if self._attn_implementation == 'eager':
            # eager uses attn bias
            # https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/stablelm/modeling_stablelm.py#L290
            mask = torch.masked_fill(torch.zeros_like(mask).float(), mask, float('-inf'))
        elif self._attn_implementation == 'sdpa':
            # sdpa uses True = NOT masked
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            mask = ~mask
        else:
            mask = None

        # final processing
        if mask is not None:

            # cannot broadcast to batch size
            if mask.shape[0] == 1:
                mask = mask.expand(batch_size, -1, -1)

            # head dim
            mask = mask.unsqueeze(1)

        return mask


    @torch.no_grad()
    def _get_position_ids(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor]=None
    ) -> torch.LongTensor:
        batch_size, seq_length = input_ids.shape
        
        # default
        # we use relative position ids so segment_ids can be ignored
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)

        # must have batch dimension
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        return position_ids


    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
        segment_ids: Optional[torch.LongTensor]=None,
        cached_mask=False,
        kv: Optional[Cache]=None,
        extra_states=None,
    ) -> DotDict:
        """ Forward pass of the LM

        Args:
            input_ids (torch.LongTensor): token input ids [bs, seq_length]
            position_ids (Optional[torch.LongTensor], optional): Position ids [bs|None, seq_length]. Defaults to None.
            attention_mask (Optional[torch.BoolTensor], optional): Attention mask [bs|None, seq_length, seq_length]. True = MASKED. Defaults to None.
            kv (Optional[Cache], optional): Key-Value cache. Defaults to None.
            
        Returns:
            DotDict:
                hidden_states [bs, seq_length, hidden_size]
                kv [Cache]
        """
        batch_size, seq_length = input_ids.shape

        # get inputs
        hidden_states = self._get_tokens(input_ids)
        attention_mask = self._get_mask(input_ids, attention_mask, segment_ids, cached_mask)
        position_ids = self._get_position_ids(input_ids, position_ids)

        # apply extras
        if extra_states is not None:
            hidden_states = hidden_states + extra_states

        # run transformer
        for layer_idx, decoder_layer in enumerate(self.layers):

            if self.gradient_checkpointing and self.training and layer_idx < self.gradient_checkpointing_layers:
                if kv is not None:
                    raise ValueError("Gradient checkpointing is not compatible with cache!")

                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    False,
                )[0]

            else:
                hidden_states = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=kv,
                    output_attentions=False,
                    use_cache=(kv is not None),
                )[0]

        return self.norm(hidden_states)


class BaseLmModel(BaseModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # transformer
        self.model = BaseTransformer(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # extras
        self.disable_segment_ids = config.disable_segment_ids

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
        segment_ids: Optional[torch.LongTensor]=None,
        kv: Optional[Cache]=None,
    ) -> DotDict:
        """ Forward pass of the LM

        Args:
            input_ids (torch.LongTensor): token input ids [bs, seq_length]
            position_ids (Optional[torch.LongTensor], optional): Position ids [bs|None, seq_length]. Defaults to None.
            attention_mask (Optional[torch.BoolTensor], optional): Attention mask [bs|None, seq_length, seq_length]. True = MASKED. Defaults to None.
            kv (Optional[Cache], optional): Key-Value cache. Defaults to None.
        
        Returns:
            DotDict:
                lm_logits: log-softmaxed token probs [bs, seq_length, vocab_size]
        """
        if self.disable_segment_ids:
            segment_ids = None

        # get lm predictions
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            kv=kv
        )

        lm_logits = self.lm_head(out)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits
