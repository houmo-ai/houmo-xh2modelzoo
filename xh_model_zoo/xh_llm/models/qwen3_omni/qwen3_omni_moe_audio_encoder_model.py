from copy import deepcopy
from typing import List, Optional, Union, cast

import torch
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.qwen3_moe import Qwen3MoeForCausalLM

from ..base_llm_model import LLMBaseModel
from ..builder import MODELS


@MODELS.register_module()
class XHQwen3OmniMoeAudioEncoderModel(LLMBaseModel):
    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type,
        allow_quant=True,
        export_cfg=None,
    ):
        super().__init__(
            hf_model,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            frontend_type=frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )

    def _set_dtype(self, dtype):
        if self._wrap_model is not None:
            self._wrap_model.to(dtype)

        if self._frontend_model is not None:
            self._frontend_model.to(dtype)

        if self._quanted_model is not None:
            self._quanted_model.to(dtype)

        self._dtype = dtype

    def init_wrap_model(self, hf_model=None):
        from ._audio_model import register_wrap_modules  # noqa F401

        register_wrap_modules()

        super().init_wrap_model(hf_model)
        hf_model = self.wrap_model

        hf_model = None

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        return data

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        return data

    def _forward(
        self,
        padded_feature,
        padded_mask_after_cnn,
        aftercnn_lens=None,
    ):

        hidden_states = self(
                padded_feature,
                padded_mask_after_cnn,
                aftercnn_lens=aftercnn_lens,
        )
        
        return BaseModelOutput(last_hidden_state=hidden_states)