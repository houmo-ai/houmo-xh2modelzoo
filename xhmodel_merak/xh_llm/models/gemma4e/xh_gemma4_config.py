import json
from collections.abc import Mapping
from pathlib import Path

from transformers import AutoConfig
from xhmodel_merak.configuration_utils import BaseConfig, HFModelConfig
from xhquant.api import QuantScheme

from ...types import LLMModelMeta
from ...vision_llm_model import VisionLLMModelConfig


def _load_hf_config(hf_model: str | None):
    if hf_model is None:
        return None
    try:
        return AutoConfig.from_pretrained(hf_model)
    except Exception:
        return None


def _load_processor_config(hf_model: str | None) -> dict:
    if hf_model is None:
        return {}
    processor_config_path = Path(hf_model) / "processor_config.json"
    if not processor_config_path.exists():
        return {}
    try:
        return json.loads(processor_config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_visual_defaults(hf_model: str | None) -> dict[str, int]:
    defaults = {
        "max_size_w": 448,
        "max_size_h": 448,
        "patch_size": 16,
        "image_seq_length": 280,
        "pooling_kernel_size": 3,
    }
    processor_config = _load_processor_config(hf_model)
    if processor_config:
        defaults["image_seq_length"] = processor_config.get("image_seq_length", defaults["image_seq_length"])

    config = _load_hf_config(hf_model)
    if config is None:
        return defaults

    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        defaults["patch_size"] = getattr(vision_config, "patch_size", defaults["patch_size"])
        defaults["pooling_kernel_size"] = getattr(
            vision_config, "pooling_kernel_size", defaults["pooling_kernel_size"]
        )
    return defaults


def _normalize_visual_export_mode(export_mode: str | None) -> str:
    normalized = "full" if export_mode is None else export_mode.lower()
    if normalized not in {"full", "compact"}:
        raise ValueError(f"Unsupported Gemma4 visual export mode: {export_mode}")
    return normalized


def _get_audio_defaults(hf_model: str | None) -> dict[str, int]:
    defaults = {
        "sampling_rate": 16000,
        "feature_size": 128,
        "input_feature_length": 400,
    }
    processor_config = _load_processor_config(hf_model)
    if processor_config:
        defaults["sampling_rate"] = processor_config.get("audio_sampling_rate", defaults["sampling_rate"])

    config = _load_hf_config(hf_model)
    if config is None:
        return defaults

    audio_config = getattr(config, "audio_config", None)
    if audio_config is not None:
        defaults["feature_size"] = getattr(audio_config, "feature_size", defaults["feature_size"])
    return defaults


class XHGemma4VisualConfig(HFModelConfig):
    def __init__(
        self,
        *,
        export_mode: str | None = None,
        max_size_w: int | None = None,
        max_size_h: int | None = None,
        patch_size: int | None = None,
        image_seq_length: int | None = None,
        pooling_kernel_size: int | None = None,
        **kwargs,
    ):
        export_mode = _normalize_visual_export_mode(export_mode)
        defaults = _get_visual_defaults(kwargs.get("hf_model"))
        if export_mode == "compact":
            defaults["max_size_w"] = 448
            defaults["max_size_h"] = 448
            defaults["image_seq_length"] = 256
            defaults["pooling_kernel_size"] = 1
        super().__init__(**kwargs)
        self.export_mode = export_mode
        self.max_size_w = defaults["max_size_w"] if max_size_w is None else max_size_w
        self.max_size_h = defaults["max_size_h"] if max_size_h is None else max_size_h
        self.patch_size = defaults["patch_size"] if patch_size is None else patch_size
        self.image_seq_length = defaults["image_seq_length"] if image_seq_length is None else image_seq_length
        self.pooling_kernel_size = (
            defaults["pooling_kernel_size"] if pooling_kernel_size is None else pooling_kernel_size
        )


class XHGemma4AudioConfig(HFModelConfig):
    def __init__(
        self,
        *,
        sampling_rate: int | None = None,
        feature_size: int | None = None,
        input_feature_length: int | None = None,
        **kwargs,
    ):
        defaults = _get_audio_defaults(kwargs.get("hf_model"))
        super().__init__(**kwargs)
        self.sampling_rate = defaults["sampling_rate"] if sampling_rate is None else sampling_rate
        self.feature_size = defaults["feature_size"] if feature_size is None else feature_size
        self.input_feature_length = (
            defaults["input_feature_length"] if input_feature_length is None else input_feature_length
        )


class Gemma4AudioModelMeta(BaseConfig):
    def __init__(
        self,
        *,
        sampling_rate: int | None = None,
        feature_size: int | None = None,
        input_feature_length: int | None = None,
        hmonnx: str | None = None,
        onnx: str | None = None,
    ):
        self.sampling_rate = sampling_rate
        self.feature_size = feature_size
        self.input_feature_length = input_feature_length
        self.hmonnx = hmonnx
        self.onnx = onnx


class Gemma4ModelMeta(LLMModelMeta):
    def __init__(
        self,
        *,
        visual_config: dict | XHGemma4VisualConfig | None = None,
        audio_config: dict | Gemma4AudioModelMeta | None = None,
        per_layer_input_builder: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_config = visual_config
        self.audio_config = audio_config
        self.per_layer_input_builder = per_layer_input_builder
        if "_meta_path_" in kwargs:
            meta_path = Path(kwargs["_meta_path_"]).parent
            if self.visual_config is not None and getattr(self.visual_config, "hmonnx", None):
                self.visual_config.hmonnx = str(meta_path / self.visual_config.hmonnx)
            if self.visual_config is not None and getattr(self.visual_config, "onnx", None):
                self.visual_config.onnx = str(meta_path / self.visual_config.onnx)
            if self.audio_config is not None and getattr(self.audio_config, "hmonnx", None):
                self.audio_config.hmonnx = str(meta_path / self.audio_config.hmonnx)
            if self.audio_config is not None and getattr(self.audio_config, "onnx", None):
                self.audio_config.onnx = str(meta_path / self.audio_config.onnx)
            if self.per_layer_input_builder is not None:
                self.per_layer_input_builder = str(meta_path / self.per_layer_input_builder)


class XHGemma4ModelConfig(VisionLLMModelConfig):
    def __init__(
        self,
        *,
        model_name: str,
        chip_arch: str = "XH2a",
        model_type: str | None = None,
        quant_scheme: dict | QuantScheme | None = None,
        quant_weight: str | None = None,
        hf_model: str | None = None,
        batch_size: int = 1,
        context_max_length: int = 2048,
        prefill_chunk_length: int = 256,
        num_logits_to_keep: int | None = 1,
        mix_search: bool = False,
        use_cache: bool = True,
        use_explicit_attention_mask: bool = True,
        visual_config: dict | XHGemma4VisualConfig | None = None,
        audio_config: dict | XHGemma4AudioConfig | None = None,
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
        if isinstance(visual_config, Mapping) and not isinstance(visual_config, XHGemma4VisualConfig):
            visual_config = dict(visual_config)
            if "model_name" not in visual_config:
                visual_config["model_name"] = f"{model_name}_visual"
            if "hf_model" not in visual_config:
                visual_config["hf_model"] = hf_model
            visual_config = XHGemma4VisualConfig(**visual_config)
        if isinstance(audio_config, Mapping) and not isinstance(audio_config, XHGemma4AudioConfig):
            audio_config = dict(audio_config)
            if "model_name" not in audio_config:
                audio_config["model_name"] = f"{model_name}_audio"
            if "hf_model" not in audio_config:
                audio_config["hf_model"] = hf_model
            audio_config = XHGemma4AudioConfig(**audio_config)
        self.visual_config = visual_config
        self.audio_config = audio_config
        self.use_explicit_attention_mask = use_explicit_attention_mask

        self.image_token_id: int | None = None
        self.audio_token_id: int | None = None
        self.video_token_id: int | None = None
        self.boi_token_id: int | None = None
        self.eoi_token_id: int | None = None
        self.boa_token_id: int | None = None
        self.eoa_token_id: int | None = None
