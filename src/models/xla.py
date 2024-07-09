import torch

try:
    from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint_fn
except ImportError:
    pass

import functools

from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

from utils.logging_utils import log_print


class XLAConfig(PretrainedConfig):

    model_type = 'xla'

    def __init__(
        self,
        gradient_checkpointing=False,
        *args,
        **kwargs,
    ):
        # requires workaround
        tmp_gradient_checkpointing = gradient_checkpointing

        # init with work arounds
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing = tmp_gradient_checkpointing


class XLAModel(PreTrainedModel):

    config_class = XLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    # converted from torch to torch xla
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        
        gradient_checkpointing_func = functools.partial(xla_checkpoint_fn, **gradient_checkpointing_kwargs)
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        
        log_print(f"Gradient checkpointing enabled for {self.__class__.__name__}: {self.gradient_checkpointing}")
