"""
XHQwen3_5DFlashDraftModel — DFlash draft model wrapper for Qwen3.5 speculative decoding.

DFlash uses cross-attention between draft tokens and target model hidden states.
Each instance handles one mode (context or decode), producing a separate ONNX export.
"""

import json
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph

from ...base_model import XHSubModel
from ...builder import register_llm_model
from ...kv_cache_mixin import EmptyKVCacheMixin
from ...types import LLMModelState, ModelMeta
from .xh_qwen3_5_config import XHQwen3_5_DFlashConfig


def _build_dflash_export_adapter(
    core_model: nn.Module,
    *,
    mode: str,
    num_hidden_layers: int,
) -> nn.Module:
    if mode == "context":
        arg_names = [
            "target_hidden",
            "past_seq_length",
            "current_input_length",
            *[f"past_key_cache_{idx}" for idx in range(num_hidden_layers)],
            *[f"past_value_cache_{idx}" for idx in range(num_hidden_layers)],
        ]
        core_call = "self.core.forward_context"
    elif mode == "decode":
        arg_names = [
            "noise_embedding",
            "past_seq_length",
            "current_input_length",
            "attn_mask",
            *[f"past_key_cache_{idx}" for idx in range(num_hidden_layers)],
            *[f"past_value_cache_{idx}" for idx in range(num_hidden_layers)],
        ]
        core_call = "self.core.forward_decode"
    else:
        raise ValueError(f"Unsupported DFlash mode: {mode}")

    forward_src = (
        f"def forward(self, {', '.join(arg_names)}):\n"
        f"    return {core_call}({', '.join(arg_names)})\n"
    )
    namespace: dict[str, Any] = {}
    exec(forward_src, {}, namespace)
    forward_impl = namespace["forward"]

    class _DFlashExportAdapter(nn.Module):
        def __init__(self, core: nn.Module):
            super().__init__()
            self.core = core

    _DFlashExportAdapter.forward = forward_impl
    return _DFlashExportAdapter(core_model)


class _DFlashDataProcessor:
    def __call__(self, dummy_inputs: dict) -> list:
        return [v for v in dummy_inputs.values()]

    def to(self, *args, **kwargs):
        return self


@register_llm_model("Qwen3_5_DFlash_Draft", master=False)
class XHQwen3_5DFlashDraftModel(XHSubModel):
    HF_MODEL_CLS = None
    HF_AUTO_MODEL_CLS = None
    META_CLS = ModelMeta
    CONFIG_CLS = XHQwen3_5_DFlashConfig

    def __init__(self, config: XHQwen3_5_DFlashConfig):
        super().__init__(config)
        self.config = cast(XHQwen3_5_DFlashConfig, self.config)
        if self.config.model_type is None:
            self.config.model_type = "Qwen3_5_DFlash_Draft"

    def to_wrap(self, hf_model=None):
        if self._state == LLMModelState.WRAP:
            return
        from xhquant.api import get_xhquant_logger

        logger = get_xhquant_logger()
        logger.info(f"Converting model {type(self).__name__} to wrap mode...")
        self._to_wrap(None)
        self._state = LLMModelState.WRAP

    def _to_wrap(self, hf_model):
        from ._dflash_model_impl import DFlashModel

        core_model = DFlashModel.from_pretrained(
            self.config.dflash_model_dir,
            self.config.target_model_dir,
            mode=self.config.mode,
            dtype=torch.float16,
            input_sequence_length=self.config.input_sequence_length,
            max_pe_length=self.config.max_pe_length,
            max_sequence_length=self.config.max_sequence_length,
        )
        self._wrap_model = _build_dflash_export_adapter(
            core_model,
            mode=self.config.mode,
            num_hidden_layers=self.config.num_hidden_layers,
        )

    def _to_fronted(self, wrap_model):
        logger = get_xhquant_logger()
        dummy_inputs = self.get_dummy_inputs()
        dummy_args = list(dummy_inputs.values())
        logger.info("Using TorchFX frontend for DFlash draft graph export.")
        return to_frontend_graph(wrap_model, FrontendType.TorchFX, dummy_args)

    def get_dummy_inputs(self) -> dict:
        bsz = self.config.batch_size
        seq_len = self.config.input_sequence_length
        hidden_size = self.config.hidden_size
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        num_layers = self.config.num_hidden_layers
        cache_len = self.config.max_sequence_length

        cache_shape = (bsz, num_kv_heads, cache_len, head_dim)

        if self.config.mode == "context":
            num_target_layers = self._get_num_target_hidden_layers()
            inputs: dict[str, Any] = {
                "target_hidden": torch.randn(
                    bsz, seq_len, num_target_layers * hidden_size, dtype=torch.float16
                ),
                "past_seq_length": torch.tensor([0], dtype=torch.int64),
                "current_input_length": torch.tensor([seq_len], dtype=torch.int64),
            }
        else:
            inputs = {
                "noise_embedding": torch.randn(
                    bsz, seq_len, hidden_size, dtype=torch.float16
                ),
                "past_seq_length": torch.tensor([0], dtype=torch.int64),
                "current_input_length": torch.tensor([seq_len], dtype=torch.int64),
                "attn_mask": torch.zeros(bsz, cache_len, dtype=torch.float16),
            }

        for idx in range(num_layers):
            inputs[f"past_key_cache_{idx}"] = torch.zeros(cache_shape, dtype=torch.float16)
        for idx in range(num_layers):
            inputs[f"past_value_cache_{idx}"] = torch.zeros(cache_shape, dtype=torch.float16)

        return inputs

    def _get_num_target_hidden_layers(self) -> int:
        """Return the number of target hidden states consumed by DFlash.

        DFlash does not concatenate all target model layers.  Its projection
        consumes only the layers listed by ``dflash_config.target_layer_ids``.
        Keep the export dummy input width aligned with the loaded draft model
        so context/context_decode ONNX signatures are [B, S, len(ids) * H].
        """
        wrap_model = getattr(self, "_wrap_model", None)
        target_layer_ids = getattr(wrap_model, "target_layer_ids", None)
        if target_layer_ids is None and hasattr(wrap_model, "core"):
            target_layer_ids = getattr(wrap_model.core, "target_layer_ids", None)
        if target_layer_ids:
            return len(target_layer_ids)

        config_path = Path(self.config.dflash_model_dir) / "config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                configured_ids = json.load(f).get("dflash_config", {}).get("target_layer_ids")
            if configured_ids:
                return len(configured_ids)

        return self.config.num_target_layers

    def get_export_cfg(self) -> dict[str, list[str]]:
        num_layers = self.config.num_hidden_layers

        if self.config.mode == "context":
            input_names = ["target_hidden", "past_seq_length", "current_input_length"]
            output_names = [
                f"present_key_cache_{idx}" for idx in range(num_layers)
            ] + [
                f"present_value_cache_{idx}" for idx in range(num_layers)
            ]
        else:
            input_names = [
                "noise_embedding",
                "past_seq_length",
                "current_input_length",
                "attn_mask",
            ]
            output_names = ["logits"]

        input_names.extend(f"past_key_cache_{idx}" for idx in range(num_layers))
        input_names.extend(f"past_value_cache_{idx}" for idx in range(num_layers))

        return dict(input_names=input_names, output_names=output_names)

    def get_kvcache_mixin(self):
        return EmptyKVCacheMixin()

    def _get_data_preprocessor(self) -> _DFlashDataProcessor:
        return _DFlashDataProcessor()

    def export_hmonnx(self, output_dir: str) -> ModelMeta:
        meta_info = self.get_export_metadata_cls()()
        exported_hmonnx_file = super()._export_hmonnx(output_dir)
        meta_info.hmonnx = str(exported_hmonnx_file)
        return meta_info
