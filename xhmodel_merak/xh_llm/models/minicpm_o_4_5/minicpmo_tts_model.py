from typing import Any, Dict, List, Union

import torch
from torch import Tensor

from xhmodel_merak.xh_other_model.base_llm_model import LLMBaseModel

from ...builder import register_llm_model
from .minicpmo_base_model import XHMiniCPMOBaseModel


@register_llm_model("MiniCPMO45TTSModel", master=False)
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
        self._init_wrap_model_with_llm_registry(tts_model.model if hasattr(tts_model, "model") else tts_model)

        self.token_embedding = tts_model.model.get_input_embeddings()
        self.generation_config = getattr(tts_model.model, "generation_config", None)
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
        # Ensure attention_mask comes AFTER KV cache names in export_cfg.input_names
        # to match the wrap_model's forward signature:
        # (inputs_embeds, past_seq_length, current_input_length,
        #  past_key_cache_0..N, past_value_cache_0..N, attention_mask)
        if hasattr(self, "export_cfg") and self.export_cfg is not None:
            input_names = list(self.export_cfg.get("input_names", []))
            if "attention_mask" in input_names:
                input_names.remove("attention_mask")
                input_names.append("attention_mask")
                self.export_cfg["input_names"] = input_names
        del hf_model

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
        # Order must match the wrap_model's forward signature created by wrap_llm_model:
        # (inputs_embeds, past_seq_length, current_input_length, past_key_cache, past_value_cache, attention_mask)
        # KV caches come BEFORE attention_mask, and export_cfg.input_names must follow the same order.
        return (
            inputs_embeds,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
            attention_mask,
        )

    def _forward(
        self,
        inputs_embeds,
        past_seq_length,
        current_input_length,
        past_key_caches,
        past_value_caches,
        attention_mask,
    ) -> List[Tensor]:
        # Pass to the wrapped model / exported graph in the same positional order.
        # attention_mask is LAST — matching wrap_model's forward signature.
        out = self(
            inputs_embeds,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
            attention_mask,
        )
        return out
