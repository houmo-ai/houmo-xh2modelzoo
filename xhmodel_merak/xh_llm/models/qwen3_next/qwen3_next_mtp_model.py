from __future__ import annotations

from typing import cast

import torch

from xhquant.api import FrontendType, to_frontend_graph

from ...base_model import XHSubModel
from ...builder import register_llm_model
from ...kv_cache_mixin import EmptyKVCacheMixin
from ...types import LLMModelState, ModelMeta
from .xh_qwen3_next_config import XHQwen3NextMTPConfig


class _MTPDataProcessor:
    def __call__(self, dummy_inputs: dict) -> list:
        return list(dummy_inputs.values())

    def to(self, *args, **kwargs):
        return self


@register_llm_model("Qwen3NextMTP", master=False)
class XHQwen3NextMTPDraftModel(XHSubModel):
    HF_MODEL_CLS = None
    HF_AUTO_MODEL_CLS = None
    META_CLS = ModelMeta
    CONFIG_CLS = XHQwen3NextMTPConfig

    def __init__(self, config: XHQwen3NextMTPConfig):
        super().__init__(config)
        self.config = cast(XHQwen3NextMTPConfig, self.config)
        self.config.model_type = "Qwen3NextMTP"

    def to_wrap(self, hf_model=None):
        if self._state == LLMModelState.WRAP:
            return
        self._to_wrap(None)
        self._state = LLMModelState.WRAP

    def _to_wrap(self, hf_model):
        from ._mtp_model_impl import MTPModel

        self._wrap_model = MTPModel.from_pretrained(
            self.config.hf_model_dir,
            dtype=torch.float16,
            input_sequence_length=self.config.input_sequence_length,
            max_pe_length=self.config.max_pe_length,
            use_cache=self.config.use_cache,
            mtp_layer_index=self.config.mtp_layer_index,
            flash_attention=self.config.flash_attention,
        )

    def _to_fronted(self, wrap_model):
        return to_frontend_graph(
            wrap_model,
            FrontendType.TorchFX,
            list(self.get_dummy_inputs().values()),
        )

    def get_dummy_inputs(self) -> dict:
        cfg = self.config
        inputs = {
            "next_token_embedding": torch.randn(
                cfg.batch_size, cfg.input_sequence_length, cfg.hidden_size, dtype=torch.float16
            ),
            "post_norm_hidden": torch.randn(
                cfg.batch_size, cfg.input_sequence_length, cfg.hidden_size, dtype=torch.float16
            ),
            "past_seq_length": torch.tensor([0], dtype=torch.int64),
            "current_input_length": torch.tensor([cfg.input_sequence_length], dtype=torch.int64),
        }
        if cfg.use_cache:
            cache_shape = (
                cfg.batch_size,
                cfg.num_key_value_heads,
                cfg.context_max_length,
                cfg.head_dim,
            )
            inputs["past_key_cache"] = torch.zeros(cache_shape, dtype=torch.float16)
            inputs["past_value_cache"] = torch.zeros(cache_shape, dtype=torch.float16)
        return inputs

    def get_export_cfg(self) -> dict[str, list[str]]:
        inputs = [
            "next_token_embedding",
            "post_norm_hidden",
            "past_seq_length",
            "current_input_length",
        ]
        if self.config.use_cache:
            inputs.extend(["past_key_cache", "past_value_cache"])
        return {"input_names": inputs, "output_names": ["logits", "post_norm_out"]}

    def get_kvcache_mixin(self):
        return EmptyKVCacheMixin()

    def _get_data_preprocessor(self):
        return _MTPDataProcessor()

    def export_hmonnx(self, output_dir: str) -> ModelMeta:
        meta = self.get_export_metadata_cls()()
        meta.hmonnx = str(super()._export_hmonnx(output_dir))
        return meta
