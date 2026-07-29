"""
XHQwen3_5MTPDraftModel — MTP draft model wrapper for Qwen3.5 speculative decoding.

Follows XHSubModel pattern (same as Vision). The MTP model is built directly
from xhquant ops, so _to_wrap loads weights via MTPModel.from_pretrained()
and _to_fronted uses TorchExport frontend.
"""

from typing import cast

import torch

from xhquant.api import FrontendType, to_frontend_graph

from ...base_model import XHSubModel
from ...builder import register_llm_model
from ...kv_cache_mixin import EmptyKVCacheMixin
from ...types import LLMModelState, ModelMeta
from .xh_qwen3_5_config import XHQwen3_5_MTPConfig


class _MTPDataProcessor:
    def __call__(self, dummy_inputs: dict) -> list:
        return [v for v in dummy_inputs.values()]

    def to(self, *args, **kwargs):
        return self


@register_llm_model("Qwen3_5_MTP_Draft", master=False)
class XHQwen3_5MTPDraftModel(XHSubModel):
    HF_MODEL_CLS = None
    HF_AUTO_MODEL_CLS = None
    META_CLS = ModelMeta
    CONFIG_CLS = XHQwen3_5_MTPConfig

    def __init__(self, config: XHQwen3_5_MTPConfig):
        super().__init__(config)
        self.config = cast(XHQwen3_5_MTPConfig, self.config)
        if self.config.model_type is None:
            self.config.model_type = "Qwen3_5_MTP_Draft"

    def to_wrap(self, hf_model=None):
        if self._state == LLMModelState.WRAP:
            return
        from xhquant.api import get_xhquant_logger

        logger = get_xhquant_logger()
        logger.info(f"Converting model {type(self).__name__} to wrap mode...")
        self._to_wrap(None)
        self._state = LLMModelState.WRAP

    def _to_wrap(self, hf_model):
        from ._mtp_model_impl import MTPModel

        model = MTPModel.from_pretrained(
            self.config.hf_model_dir,
            dtype=torch.float16,
            input_sequence_length=self.config.input_sequence_length,
            max_pe_length=self.config.max_pe_length,
            use_cache=self.config.use_cache,
            flash_attention=self.config.flash_attention,
        )
        self._wrap_model = model

    def _to_fronted(self, wrap_model):
        dummy_inputs = self.get_dummy_inputs()
        dummy_args = list(dummy_inputs.values())
        return to_frontend_graph(wrap_model, FrontendType.TorchFX, dummy_args)

    def get_dummy_inputs(self) -> dict:
        bsz = self.config.batch_size
        seq_len = self.config.input_sequence_length
        hidden_size = self.config.hidden_size
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        context_max_length = self.config.context_max_length

        inputs = {
            "next_token_embedding": torch.randn(bsz, seq_len, hidden_size, dtype=torch.float16),
            "post_norm_hidden": torch.randn(bsz, seq_len, hidden_size, dtype=torch.float16),
            "past_seq_length": torch.tensor([0], dtype=torch.int64),
            "current_input_length": torch.tensor([seq_len], dtype=torch.int64),
        }
        if self.config.use_cache:
            inputs["past_key_cache"] = torch.zeros(bsz, num_kv_heads, context_max_length, head_dim, dtype=torch.float16)
            inputs["past_value_cache"] = torch.zeros(
                bsz, num_kv_heads, context_max_length, head_dim, dtype=torch.float16
            )
        return inputs

    def get_export_cfg(self) -> dict[str, list[str]]:
        input_names = [
            "next_token_embedding",
            "post_norm_hidden",
            "past_seq_length",
            "current_input_length",
        ]
        if self.config.use_cache:
            input_names.extend(["past_key_cache", "past_value_cache"])

        output_names = ["logits", "post_norm_out"]
        return dict(input_names=input_names, output_names=output_names)

    def get_kvcache_mixin(self):
        return EmptyKVCacheMixin()

    def _get_data_preprocessor(self) -> _MTPDataProcessor:
        return _MTPDataProcessor()

    def export_hmonnx(self, output_dir: str) -> ModelMeta:
        meta_info = self.get_export_metadata_cls()()
        exported_hmonnx_file = super()._export_hmonnx(output_dir)
        meta_info.hmonnx = str(exported_hmonnx_file)
        return meta_info
