from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

# import transformers_modules
from transformers import AutoModel
from types import MethodType
from ..base_model import BaseModel
from ..builder import MODELS, wrap_llm_model
from .minicpmo_base_model import XHMiniCPMOBaseModel
from ..base_llm_model import LLMBaseModel

@MODELS.register_module()
class XHMiniCPMOTTSModel(LLMBaseModel, XHMiniCPMOBaseModel):
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

    def get_hf_model(self, device_map="cpu", **kwargs):
        hf_model = super().get_hf_model(device_map=device_map, **kwargs)
        return hf_model

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()
        from ._tts_model_impl import register_wrap_cls as tts_register_wrap_cls  # noqa F401

        tts_register_wrap_cls(hf_model)
        tts_model = hf_model.tts
        super().init_wrap_model(tts_model)

        self.token_embedding = tts_model.model.get_input_embeddings()
        self.generation_config = tts_model.model.generation_config
        self.config = tts_model.model.config
        self.num_hidden_layers = tts_model.model.config.num_hidden_layers
        head_dim = tts_model.model.config.hidden_size // tts_model.model.config.num_attention_heads
        batch_size = self.wrap_cfg.batch_size
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [batch_size, tts_model.model.config.num_key_value_heads, self.cache_length, head_dim],
            )

    def _set_device(self, device: torch.device) -> None:
        super()._set_device(device)

    def prepare_inputs_for_graph(self, data: Dict[str, Union[torch.Tensor, Any]]) -> Any:
        inputs = self.prepare_inputs(data)
        return inputs

    def prepare_inputs(self, data: Dict[str, Union[torch.Tensor, Any]]) -> Any:
        inputs_embeds = data["inputs_embeds"]
        past_seq_length = data["past_seq_length"]
        current_input_length = data["current_input_length"]
        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches
        attention_mask = data["attention_mask"]
        return (inputs_embeds, past_seq_length, current_input_length, attention_mask, past_key_caches, past_value_caches)

    def _forward(
        self,
        inputs_embeds,
        past_seq_length,
        current_input_length,
        attention_mask,
        past_key_caches,
        past_value_caches,
    ) -> List[Tensor]:
        out = self(
            inputs_embeds,
            past_seq_length,
            current_input_length,
            attention_mask,
            past_key_caches,
            past_value_caches,
        )

        return out
