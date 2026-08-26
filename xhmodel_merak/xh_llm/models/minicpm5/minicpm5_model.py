"""Registered MiniCPM5-MoE model and remote-code loading hooks."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, PreTrainedModel

from ...builder import register_llm_model
from ...text_llm_model import TextLLMModel, TextLLMModelConfig
from .minicpm5_hmonnx_inference import XHMiniCPM5HMONNXModel


class XHMiniCPM5ModelConfig(TextLLMModelConfig):
    """Merak export configuration for MiniCPM5."""


@register_llm_model("MiniCPM5MoEForCausalLM")
class XHMiniCPM5Model(TextLLMModel):
    HF_MODEL_CLS = PreTrainedModel
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HMONNXINFERENCE_CLS = XHMiniCPM5HMONNXModel
    CONFIG_CLS = XHMiniCPM5ModelConfig
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.minicpm5.workflow:MiniCPM5Workflow"

    @classmethod
    def _load_hf_model(cls, hf_model_dir: str, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return super()._load_hf_model(hf_model_dir, **kwargs)

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir: str | Path, **kwargs) -> Any:
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("dtype", cls.HF_MODEL_DTYPE)
        config = AutoConfig.from_pretrained(str(hf_model_dir), trust_remote_code=True)
        with init_empty_weights():
            hf_model = cls.HF_AUTO_MODEL_CLS.from_config(config, **kwargs)
            if hf_model.can_generate():
                try:
                    hf_model.generation_config = GenerationConfig.from_pretrained(str(hf_model_dir))
                except OSError:
                    pass
        return hf_model

    @classmethod
    def _get_hf_model_for_compatible(cls, hf_model_dir=None):
        return cls.get_empty_hf_model(hf_model_dir, trust_remote_code=True)

    def init_wrap_model(self, hf_model: nn.Module) -> Any:
        from ._model import register_wrap_modules

        register_wrap_modules(hf_model)
        return super().init_wrap_model(hf_model)

    def _wraped_post(self, hf_model: nn.Module):
        super()._wraped_post(hf_model)
        language_model = self._get_language_model(self._wrap_model)
        if self.use_cache:
            num_decoder_layers = int(language_model.config.num_hidden_layers)
            max_layers = self.config.get_max_decode_layers()
            if max_layers > 0:
                num_decoder_layers = min(num_decoder_layers, max_layers)
            self.kvcache_config.num_layers = num_decoder_layers
            self.kvcache_config.kv_cache_shape = [
                1,
                int(language_model.config.num_key_value_heads),
                int(self.config.context_max_length),
                int(language_model.config.head_dim),
            ]
        pad_token_id = getattr(language_model.config, "pad_token_id", None)
        if pad_token_id is None:
            eos_token_id = language_model.config.eos_token_id
            pad_token_id = eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id
        self.pad_token_id = pad_token_id

    def create_export_metadata(self, output_dir: str):
        """Include the checkpoint's remote-code modules in the HMONNX bundle."""
        meta_info = super().create_export_metadata(output_dir)
        hf_config_dir = Path(output_dir) / meta_info.hf_config
        source_dir = Path(self.config.hf_model)
        remote_code_files = sorted(source_dir.glob("*.py"))
        if not remote_code_files:
            raise FileNotFoundError(f"MiniCPM5 export requires remote-code files in {source_dir}")
        for source in remote_code_files:
            shutil.copy2(source, hf_config_dir / source.name)
        return meta_info


__all__ = ["XHMiniCPM5Model", "XHMiniCPM5ModelConfig"]
