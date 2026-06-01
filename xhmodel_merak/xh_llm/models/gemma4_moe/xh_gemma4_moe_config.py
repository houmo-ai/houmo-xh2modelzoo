from __future__ import annotations

import json
from pathlib import Path

from xhmodel_merak.configuration_utils import HFModelConfig
from xhquant.api import QuantScheme

from ...vision_llm_model import VisionLLMModelConfig


class XHGemma4MoeVisualConfig(HFModelConfig):
    def __init__(
        self,
        *,
        max_size_w: int = 448,
        max_size_h: int = 448,
        upsample_token: bool = False,
        fuse_norm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_size_w = max_size_w
        self.max_size_h = max_size_h
        self.upsample_token = upsample_token
        self.fuse_norm = fuse_norm


class XHGemma4MoeWithMaskConfig(VisionLLMModelConfig):
    def __init__(
        self,
        *,
        model_name: str,
        chip_arch: str = "XH2a",
        model_type: str | None = None,
        quant_scheme: dict | QuantScheme | None = None,
        quant_weight: str | None = None,
        hf_model: str | None = None,
        fallback_hf_model: str | None = None,
        batch_size: int = 1,
        context_max_length: int = 2048,
        prefill_chunk_length: int = 256,
        num_logits_to_keep: int | None = 1,
        mix_search: bool = False,
        use_cache: bool = True,
        visual_config: dict | XHGemma4MoeVisualConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            chip_arch=chip_arch,
            model_type=model_type,
            hf_model=hf_model,
            quant_scheme=quant_scheme,
            quant_weight=quant_weight,
            batch_size=batch_size,
            context_max_length=context_max_length,
            prefill_chunk_length=prefill_chunk_length,
            num_logits_to_keep=num_logits_to_keep,
            mix_search=mix_search,
            use_cache=use_cache,
            **kwargs,
        )
        if isinstance(visual_config, dict):
            if "model_name" not in visual_config:
                visual_config["model_name"] = f"{model_name}_visual"
            if "hf_model" not in visual_config:
                visual_config["hf_model"] = hf_model
            visual_config = XHGemma4MoeVisualConfig(**visual_config)
        self.visual_config = visual_config
        self.fallback_hf_model = fallback_hf_model
        hf_config = self._load_hf_config(hf_model)
        text_config = hf_config.get("text_config", {})
        layer_types = text_config.get("layer_types", [])
        self.image_token_id: int | None = hf_config.get("image_token_id")
        self.sliding_window: int | None = text_config.get("sliding_window")
        self.local_attention_window_size: int | None = text_config.get("sliding_window")
        self.global_attention_window_size: int | None = context_max_length
        self.has_local_attention: bool = any(layer_type == "sliding_attention" for layer_type in layer_types)
        self.has_global_attention: bool = any(layer_type == "full_attention" for layer_type in layer_types)
        self.num_hidden_layers: int | None = text_config.get("num_hidden_layers")
        self.num_key_value_heads: int | None = text_config.get("num_key_value_heads")
        self.num_global_key_value_heads: int | None = text_config.get("num_global_key_value_heads")
        self.head_dim: int | None = text_config.get("head_dim")

    @staticmethod
    def _load_hf_config(hf_model: str | None) -> dict:
        if hf_model is None:
            return {}
        config_path = Path(hf_model) / "config.json"
        if not config_path.exists():
            return {}
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)