# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast

import torch

from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph

from ...base_model import XHSubModel
from ...builder import register_llm_model
from ...kv_cache_mixin import EmptyKVCacheMixin
from ...types import LLMModelState, ModelMeta
from .dflash_draft import (
    HunyuanOCRDFlashCheckpoint,
    HunyuanOCRDFlashModel,
    build_hunyuan_ocr_dflash_export_adapter,
    dflash_graph_io_contract,
    load_hunyuan_ocr_dflash_checkpoint,
)
from .xh_hunyuan_ocr_config import XHHunYuanOCRDFlashConfig


class _DFlashDataProcessor:
    def __call__(self, dummy_inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        return list(dummy_inputs.values())

    def to(self, *args: Any, **kwargs: Any) -> _DFlashDataProcessor:
        del args, kwargs
        return self


@register_llm_model("HunYuanOCR_DFlash_Draft", master=False)
class XHHunYuanOCRDFlashModel(XHSubModel):
    """Merak owner for HunyuanOCR DFlash context, context-decode and decode graphs."""

    HF_MODEL_CLS = None
    HF_AUTO_MODEL_CLS = None
    META_CLS = ModelMeta
    CONFIG_CLS = XHHunYuanOCRDFlashConfig

    def __init__(self, config: XHHunYuanOCRDFlashConfig) -> None:
        super().__init__(config)
        self.config = cast(XHHunYuanOCRDFlashConfig, self.config)
        self._checkpoint: HunyuanOCRDFlashCheckpoint | None = None

    def to_wrap(self, hf_model=None) -> None:
        del hf_model
        if self._state == LLMModelState.WRAP:
            return
        if self._state != LLMModelState.NONE:
            raise RuntimeError(f"Invalid state transition: {self._state} -> {LLMModelState.WRAP}")
        self._to_wrap(None)
        self._state = LLMModelState.WRAP

    def _to_wrap(self, hf_model) -> None:
        del hf_model
        checkpoint = load_hunyuan_ocr_dflash_checkpoint(
            self.config.dflash_model_dir,
            self.config.target_model_dir,
        )
        self._checkpoint = checkpoint
        core = HunyuanOCRDFlashModel.from_checkpoint(
            checkpoint,
            max_sequence_length=self.config.context_max_length,
            input_sequence_length=self.config.input_sequence_length,
        ).eval()
        self._wrap_model = build_hunyuan_ocr_dflash_export_adapter(
            core,
            mode=self.config.mode,
            num_hidden_layers=checkpoint.config.num_hidden_layers,
        )

    def _to_fronted(self, wrap_model):
        dummy_args = list(self.get_dummy_inputs().values())
        get_xhquant_logger().info("Using TorchFX frontend for HunyuanOCR DFlash export.")
        return to_frontend_graph(wrap_model, FrontendType.TorchFX, dummy_args)

    def _draft_config(self):
        if self._checkpoint is None:
            self._checkpoint = load_hunyuan_ocr_dflash_checkpoint(
                self.config.dflash_model_dir,
                self.config.target_model_dir,
            )
        return self._checkpoint.config

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        config = self._draft_config()
        sequence_length = self.config.input_sequence_length
        cache_shape = (
            self.config.batch_size,
            config.num_key_value_heads,
            self.config.context_max_length,
            config.head_dim,
        )
        if self.config.mode in ("context", "context_decode"):
            inputs: dict[str, torch.Tensor] = {
                "target_hidden": torch.zeros(
                    self.config.batch_size,
                    sequence_length,
                    len(config.target_layer_ids) * config.hidden_size,
                    dtype=torch.float16,
                ),
                "past_seq_length": torch.tensor([0], dtype=torch.int64),
                "current_input_length": torch.tensor([sequence_length], dtype=torch.int64),
            }
        else:
            inputs = {
                "noise_embedding": torch.zeros(
                    self.config.batch_size,
                    sequence_length,
                    config.hidden_size,
                    dtype=torch.float16,
                ),
                "past_seq_length": torch.tensor([0], dtype=torch.int64),
                "current_input_length": torch.tensor([sequence_length], dtype=torch.int64),
                "attn_mask": torch.zeros(
                    self.config.batch_size,
                    self.config.context_max_length,
                    dtype=torch.float16,
                ),
            }
        for name in self.get_export_cfg()["cache_input_names"]:
            inputs[name] = torch.zeros(cache_shape, dtype=torch.float16)
        return inputs

    def get_export_cfg(self) -> dict[str, list[str]]:
        return dflash_graph_io_contract(mode=self.config.mode, num_hidden_layers=5)

    def get_kvcache_mixin(self):
        return EmptyKVCacheMixin()

    def _get_data_preprocessor(self) -> _DFlashDataProcessor:
        return _DFlashDataProcessor()

    def export_hmonnx(self, output_dir: str) -> ModelMeta:
        metadata = self.get_export_metadata_cls()()
        metadata.hmonnx = str(self._export_hmonnx(output_dir))
        return metadata


__all__ = ["XHHunYuanOCRDFlashModel"]