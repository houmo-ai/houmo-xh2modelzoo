"""Unified Gemma4 Series config/meta classes."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

from xhmodel_merak.configuration_utils import BaseConfig, HFModelConfig
from xhquant.api import QuantScheme

from ...types import VLLMModelMeta
from ...vision_llm_model import VisionLLMModelConfig
from .attention_visibility import Gemma4AttentionVisibilitySpec
from .modality_contract import Gemma4SeriesModalityContract
from .variants import Gemma4SeriesVariantSpec, resolve_gemma4_series_variant


_DEFAULT_VIDEO_VISUAL_IMAGE_SEQ_LENGTH = 70
_DEFAULT_VIDEO_VISUAL_MAX_PATCHES = 630


def _load_processor_config(hf_model: str | None) -> dict:
    if hf_model is None:
        return {}
    processor_config_path = Path(hf_model) / "processor_config.json"
    if not processor_config_path.exists():
        return {}
    return json.loads(processor_config_path.read_text(encoding="utf-8"))


def _get_audio_defaults(hf_model: str | None) -> dict[str, int]:
    defaults = {
        "sampling_rate": 16000,
        "feature_size": 128,
        # HF Gemma4 processor emits 2999 log-mel frames for the default 30s
        # audio window (16000 Hz, 40ms/token, audio_seq_length=750).
        "input_feature_length": 2999,
    }
    processor_config = _load_processor_config(hf_model)
    feature_extractor = processor_config.get("feature_extractor", {})
    if processor_config:
        defaults["sampling_rate"] = int(
            processor_config.get(
                "audio_sampling_rate",
                feature_extractor.get("sampling_rate", defaults["sampling_rate"]),
            )
        )
        defaults["feature_size"] = int(feature_extractor.get("feature_size", defaults["feature_size"]))
        if "audio_feature_length" in processor_config:
            defaults["input_feature_length"] = int(processor_config["audio_feature_length"])
    hf_config = XHGemma4SeriesModelConfig._load_hf_config(hf_model)
    audio_config = hf_config.get("audio_config") if isinstance(hf_config, dict) else None
    if isinstance(audio_config, dict):
        defaults["feature_size"] = int(audio_config.get("feature_size", defaults["feature_size"]))
        defaults["attention_chunk_size"] = int(audio_config.get("attention_chunk_size", 12))
        defaults["attention_context_left"] = int(audio_config.get("attention_context_left", 13))
        defaults["attention_context_right"] = int(audio_config.get("attention_context_right", 0))
    return defaults


class XHGemma4SeriesVisualConfig(HFModelConfig):
    """Gemma4 Series padded ViT config for image/video subgraphs."""

    def __init__(
        self,
        *,
        export_mode: str = "padded",
        image_seq_length: int = 280,
        max_patches: int = 2520,
        patch_size: int = 16,
        pooling_kernel_size: int = 3,
        input_modality: str = "image",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.export_mode = export_mode
        self.image_seq_length = image_seq_length
        self.max_patches = max_patches
        self.patch_size = patch_size
        self.pooling_kernel_size = pooling_kernel_size
        self.input_modality = input_modality


class XHGemma4SeriesAudioConfig(HFModelConfig):
    """Gemma4 Series fixed-window audio encoder config."""

    def __init__(
        self,
        *,
        sampling_rate: int | None = None,
        feature_size: int | None = None,
        input_feature_length: int | None = None,
        attention_chunk_size: int | None = None,
        attention_context_left: int | None = None,
        attention_context_right: int | None = None,
        **kwargs,
    ):
        defaults = _get_audio_defaults(kwargs.get("hf_model"))
        super().__init__(**kwargs)
        self.sampling_rate = defaults["sampling_rate"] if sampling_rate is None else sampling_rate
        self.feature_size = defaults["feature_size"] if feature_size is None else feature_size
        self.input_feature_length = (
            defaults["input_feature_length"] if input_feature_length is None else input_feature_length
        )
        self.attention_chunk_size = (
            defaults.get("attention_chunk_size", 12) if attention_chunk_size is None else attention_chunk_size
        )
        self.attention_context_left = (
            defaults.get("attention_context_left", 13) if attention_context_left is None else attention_context_left
        )
        self.attention_context_right = (
            defaults.get("attention_context_right", 0) if attention_context_right is None else attention_context_right
        )


class XHGemma4UnifiedVisualConfig(HFModelConfig):
    """Fixed encoder-free image or per-frame video graph config."""

    def __init__(
        self,
        *,
        image_seq_length: int,
        input_dim: int,
        position_capacity: int,
        input_modality: str = "image",
        frontend_kind: str = "encoder_free",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_seq_length = int(image_seq_length)
        self.input_dim = int(input_dim)
        self.position_capacity = int(position_capacity)
        self.input_modality = str(input_modality)
        self.frontend_kind = str(frontend_kind)


class XHGemma4UnifiedAudioConfig(HFModelConfig):
    """Fixed raw-waveform-frame graph config for Gemma4 Unified."""

    def __init__(
        self,
        *,
        input_feature_length: int,
        feature_size: int,
        sampling_rate: int,
        frontend_kind: str = "encoder_free",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_feature_length = int(input_feature_length)
        self.feature_size = int(feature_size)
        self.sampling_rate = int(sampling_rate)
        self.frontend_kind = str(frontend_kind)


class Gemma4SeriesAudioModelMeta(BaseConfig):
    """Gemma4 Series audio runtime meta."""

    def __init__(
        self,
        *,
        sampling_rate: int | None = None,
        feature_size: int | None = None,
        input_feature_length: int | None = None,
        attention_chunk_size: int | None = None,
        attention_context_left: int | None = None,
        attention_context_right: int | None = None,
        frontend_kind: str = "tower",
        hmonnx: str | None = None,
        onnx: str | None = None,
    ):
        self.sampling_rate = sampling_rate
        self.feature_size = feature_size
        self.input_feature_length = input_feature_length
        self.attention_chunk_size = attention_chunk_size
        self.attention_context_left = attention_context_left
        self.attention_context_right = attention_context_right
        self.frontend_kind = frontend_kind
        self.hmonnx = hmonnx
        self.onnx = onnx


class Gemma4SeriesModelMeta(VLLMModelMeta):
    """Runtime meta with explicit Gemma4 Series variant/capability facts."""

    _COMPATIBLE_CLASS_NAMES = {"Gemma4ModelMeta", "Gemma4SeriesModelMeta"}

    def __init__(
        self,
        *,
        visual_config: dict | BaseConfig | None = None,
        video_visual_config: dict | BaseConfig | None = None,
        audio_config: dict | Gemma4SeriesAudioModelMeta | None = None,
        per_layer_input_embedding: str | None = None,
        variant: str | None = None,
        capabilities: Mapping[str, bool] | None = None,
        attention_contract_version: int = 1,
        attention_lowering: str | None = None,
        attention_visibility_spec: Mapping | None = None,
        max_mm_ranges_per_chunk: int = 1,
        uses_sliding_flash_attention_v2: bool = False,
        frontend_kind: str = "tower",
        hf_architecture: str = "Gemma4ForConditionalGeneration",
        modality_contract: Mapping | None = None,
        **kwargs,
    ):
        super().__init__(visual_config=visual_config, **kwargs)
        self.video_visual_config = video_visual_config
        self.audio_config = audio_config
        self.per_layer_input_embedding = per_layer_input_embedding
        self.variant = variant
        self.capabilities = dict(capabilities or {})
        self.attention_contract_version = int(attention_contract_version)
        self.attention_lowering = attention_lowering or (
            "flash_attention" if self.attention_contract_version >= 2 else "legacy_attention"
        )
        self.attention_visibility_spec = (
            dict(attention_visibility_spec) if attention_visibility_spec is not None else None
        )
        self.max_mm_ranges_per_chunk = int(max_mm_ranges_per_chunk)
        self.uses_sliding_flash_attention_v2 = bool(uses_sliding_flash_attention_v2)
        self.frontend_kind = str(frontend_kind)
        self.hf_architecture = str(hf_architecture)
        self.modality_contract = dict(modality_contract) if modality_contract is not None else None
        if "_meta_path_" in kwargs:
            meta_path = Path(kwargs["_meta_path_"]).parent
            if self.visual_config is not None and getattr(self.visual_config, "onnx", None):
                self.visual_config.onnx = str(meta_path / self.visual_config.onnx)
            if self.video_visual_config is not None and getattr(self.video_visual_config, "hmonnx", None):
                self.video_visual_config.hmonnx = str(meta_path / self.video_visual_config.hmonnx)
            if self.video_visual_config is not None and getattr(self.video_visual_config, "onnx", None):
                self.video_visual_config.onnx = str(meta_path / self.video_visual_config.onnx)
            if self.audio_config is not None and getattr(self.audio_config, "hmonnx", None):
                self.audio_config.hmonnx = str(meta_path / self.audio_config.hmonnx)
            if self.audio_config is not None and getattr(self.audio_config, "onnx", None):
                self.audio_config.onnx = str(meta_path / self.audio_config.onnx)
            if self.per_layer_input_embedding is not None:
                self.per_layer_input_embedding = str(meta_path / self.per_layer_input_embedding)

    @classmethod
    def from_dict(cls, config_dict: dict):
        meta_info = config_dict.get("meta")
        class_name = meta_info.get("class_name") if isinstance(meta_info, Mapping) else None
        if class_name in cls._COMPATIBLE_CLASS_NAMES and class_name != cls.__name__:
            config_dict = copy.deepcopy(config_dict)
            config_dict["meta"]["class_name"] = cls.__name__
        return super().from_dict(config_dict)


class XHGemma4SeriesModelConfig(VisionLLMModelConfig):
    """Single public config for E4B, 31B dense, and 26B-A4B MoE."""

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
        prefill_chunk_length: int = 320,
        mm_prefill_chunk_length: int | None = None,
        num_logits_to_keep: int | None = 1,
        mix_search: bool = False,
        use_cache: bool = True,
        enable_mtp_outputs: bool = False,
        spec_decode_mode: str | None = None,
        num_draft_tokens: int | None = None,
        output_post_norm_hidden: bool = False,
        mtp_config: dict | BaseConfig | None = None,
        visual_config: dict | XHGemma4SeriesVisualConfig | None = None,
        video_visual_config: dict | XHGemma4SeriesVisualConfig | None = None,
        audio_config: dict | XHGemma4SeriesAudioConfig | None = None,
        sliding_kv_cache_input_mode: str = "slice_window",
        attention_contract_version: int | None = None,
        flash_attention: Mapping | None = None,
        max_mm_ranges_per_chunk: int = 1,
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
        if mm_prefill_chunk_length is not None:
            raise ValueError(
                "Gemma4 Series no longer supports mm_prefill_chunk_length/prefill_mm; "
                "use prefill_chunk_length instead."
            )
        self.spec_decode_mode = str(spec_decode_mode).lower() if spec_decode_mode is not None else None
        sliding_kv_cache_input_mode = self._normalize_sliding_kv_cache_input_mode(sliding_kv_cache_input_mode)
        self.flash_attention = dict(flash_attention) if flash_attention is not None else None
        # A Gemma FlashAttention block is enabled by presence for compatibility
        # with existing family YAMLs; callers may explicitly set enable=false.
        flash_attention_enabled = (
            self.flash_attention is not None
            and bool(self.flash_attention.get("enable", True))
        )
        derived_attention_contract = 2 if flash_attention_enabled else 1
        if (
            attention_contract_version is not None
            and int(attention_contract_version) != derived_attention_contract
        ):
            raise ValueError(
                "Gemma4 attention_contract_version must agree with "
                "flash_attention.enable; omit the version and let the graph "
                "contract be derived automatically"
            )
        self.attention_contract_version = derived_attention_contract
        self.attention_lowering = (
            "flash_attention" if flash_attention_enabled else "legacy_attention"
        )
        self.max_mm_ranges_per_chunk = int(max_mm_ranges_per_chunk)
        if self.max_mm_ranges_per_chunk <= 0:
            raise ValueError("Gemma4 max_mm_ranges_per_chunk must be positive")
        if self.attention_lowering == "flash_attention" and sliding_kv_cache_input_mode != "slice_window":
            raise ValueError("Gemma4 contract-v2 requires sliding_kv_cache_input_mode='slice_window'")
        if self.spec_decode_mode == "mtp" and sliding_kv_cache_input_mode != "slice_window":
            raise ValueError(
                "Gemma4 Series MTP requires sliding_kv_cache_input_mode='slice_window'; "
                f"got {sliding_kv_cache_input_mode!r}."
            )
        self.enable_mtp_outputs = bool(enable_mtp_outputs or self.spec_decode_mode == "mtp")
        self.num_draft_tokens = num_draft_tokens
        self.output_post_norm_hidden = bool(output_post_norm_hidden)
        if isinstance(mtp_config, Mapping):
            mtp_config = dict(mtp_config)
            mtp_config["context_max_length"] = context_max_length
            mtp_config = BaseConfig(**mtp_config)
        elif mtp_config is not None:
            mtp_config.context_max_length = context_max_length
        self.mtp_config = mtp_config
        hf_config = self._load_hf_config(hf_model)
        text_config = hf_config.get("text_config", {})
        self._variant_spec: Gemma4SeriesVariantSpec = resolve_gemma4_series_variant(hf_config)
        self.variant = self._variant_spec.name
        self.capabilities = self._variant_spec.capabilities
        self.frontend_kind = self._variant_spec.frontend_kind
        self.hf_architecture = self._variant_spec.hf_architecture
        unified_contract = (
            Gemma4SeriesModalityContract.from_pretrained(hf_model)
            if self.frontend_kind == "encoder_free" and hf_model is not None
            else None
        )
        self.modality_contract = (
            {
                "image_soft_tokens": unified_contract.image_soft_tokens,
                "video_soft_tokens_per_frame": unified_contract.video_soft_tokens_per_frame,
                "audio_soft_tokens": unified_contract.audio_soft_tokens,
                "vision_patch_dim": unified_contract.vision_patch_dim,
                "audio_feature_dim": unified_contract.audio_feature_dim,
                "position_capacity": unified_contract.position_capacity,
                "sampling_rate": unified_contract.sampling_rate,
                "config_hash": unified_contract.config_hash,
            }
            if unified_contract is not None
            else None
        )
        self.use_bidirectional_attention: str | None = text_config.get("use_bidirectional_attention")
        self.bidirectional_vision_attention: bool = self._variant_spec.bidirectional_vision_attention
        if int(prefill_chunk_length) <= 0:
            raise ValueError(
                "Gemma4 Series prefill_chunk_length must be positive; "
                f"got {prefill_chunk_length}."
            )
        if self.bidirectional_vision_attention and int(prefill_chunk_length) < 280:
            raise ValueError(
                "Gemma4 Series bidirectional vision attention requires prefill_chunk_length >= 280; "
                f"got {prefill_chunk_length}."
            )

        if unified_contract is not None:
            visual_config = self._build_unified_visual_config(
                visual_config,
                model_name=model_name,
                hf_model=hf_model,
                contract=unified_contract,
                input_modality="image",
                quant_scheme=quant_scheme,
            )
            if video_visual_config is None and visual_config is not None:
                video_visual_config = {}
            video_visual_config = self._build_unified_visual_config(
                video_visual_config,
                model_name=model_name,
                hf_model=hf_model,
                contract=unified_contract,
                input_modality="video",
                quant_scheme=quant_scheme,
            )
            audio_config = self._build_unified_audio_config(
                audio_config,
                model_name=model_name,
                hf_model=hf_model,
                contract=unified_contract,
                quant_scheme=quant_scheme,
            )
        elif isinstance(visual_config, dict):
            if "model_name" not in visual_config:
                visual_config["model_name"] = f"{model_name}_visual"
            if "hf_model" not in visual_config:
                visual_config["hf_model"] = hf_model
            visual_config = XHGemma4SeriesVisualConfig(**visual_config)
        if unified_contract is None and video_visual_config is None and visual_config is not None:
            video_visual_config = {
                "model_name": f"{model_name}_video_visual",
                "hf_model": hf_model,
                "export_mode": visual_config.export_mode,
                "image_seq_length": _DEFAULT_VIDEO_VISUAL_IMAGE_SEQ_LENGTH,
                "max_patches": _DEFAULT_VIDEO_VISUAL_MAX_PATCHES,
                "patch_size": visual_config.patch_size,
                "pooling_kernel_size": visual_config.pooling_kernel_size,
                "input_modality": "video",
                "quant_scheme": copy.deepcopy(visual_config.quant_scheme),
            }
        if unified_contract is None and isinstance(video_visual_config, dict):
            if "model_name" not in video_visual_config:
                video_visual_config["model_name"] = f"{model_name}_video_visual"
            if "hf_model" not in video_visual_config:
                video_visual_config["hf_model"] = hf_model
            video_visual_config.setdefault("image_seq_length", _DEFAULT_VIDEO_VISUAL_IMAGE_SEQ_LENGTH)
            video_visual_config.setdefault("max_patches", _DEFAULT_VIDEO_VISUAL_MAX_PATCHES)
            if visual_config is not None:
                video_visual_config.setdefault("export_mode", visual_config.export_mode)
                video_visual_config.setdefault("patch_size", visual_config.patch_size)
                video_visual_config.setdefault("pooling_kernel_size", visual_config.pooling_kernel_size)
                video_visual_config.setdefault("quant_scheme", copy.deepcopy(visual_config.quant_scheme))
            video_visual_config.setdefault("input_modality", "video")
            video_visual_config = XHGemma4SeriesVisualConfig(**video_visual_config)
        if unified_contract is None and audio_config is None and isinstance(hf_config.get("audio_config"), dict):
            audio_config = {
                "model_name": f"{model_name}_audio",
                "hf_model": hf_model,
            }
        if (
            unified_contract is None
            and isinstance(audio_config, Mapping)
            and not isinstance(audio_config, XHGemma4SeriesAudioConfig)
        ):
            audio_config = dict(audio_config)
            if "model_name" not in audio_config:
                audio_config["model_name"] = f"{model_name}_audio"
            if "hf_model" not in audio_config:
                audio_config["hf_model"] = hf_model
            audio_config = XHGemma4SeriesAudioConfig(**audio_config)
        self.visual_config = visual_config
        self.video_visual_config = video_visual_config
        self.audio_config = audio_config

        self.image_token_id: int | None = None
        self.video_token_id: int | None = None
        self.audio_token_id: int | None = None
        self.boi_token_id: int | None = None
        self.eoi_token_id: int | None = None
        self.mm_token_type_ids_enabled: bool = True
        self.fallback_hf_model = fallback_hf_model or hf_model
        self.sliding_kv_cache_input_mode = sliding_kv_cache_input_mode

        layer_types = text_config.get("layer_types", [])
        self.enable_moe_block: bool = bool(text_config.get("enable_moe_block", False))
        self.image_token_id = hf_config.get("image_token_id", self.image_token_id)
        self.video_token_id = hf_config.get("video_token_id", self.video_token_id)
        self.audio_token_id = hf_config.get("audio_token_id", self.audio_token_id)
        self.boi_token_id = hf_config.get("boi_token_id", self.boi_token_id)
        self.eoi_token_id = hf_config.get("eoi_token_id", self.eoi_token_id)
        self.sliding_window: int | None = text_config.get("sliding_window")
        self.local_attention_window_size: int | None = text_config.get("sliding_window")
        self.global_attention_window_size: int | None = context_max_length
        self.has_local_attention: bool = any(layer_type == "sliding_attention" for layer_type in layer_types)
        self.has_global_attention: bool = any(layer_type == "full_attention" for layer_type in layer_types)
        self.num_hidden_layers: int | None = text_config.get("num_hidden_layers")
        self.num_key_value_heads: int | None = text_config.get("num_key_value_heads")
        self.num_global_key_value_heads: int | None = text_config.get("num_global_key_value_heads")
        self.head_dim: int | None = text_config.get("head_dim")
        self.hidden_size_per_layer_input: int = int(text_config.get("hidden_size_per_layer_input", 0) or 0)
        self.vocab_size_per_layer_input: int | None = text_config.get("vocab_size_per_layer_input")
        visibility_spec = Gemma4AttentionVisibilitySpec.from_checkpoint_semantics(
            layer_types=layer_types,
            sliding_window=self.sliding_window,
            bidirectional_vision_attention=self.bidirectional_vision_attention,
            max_mm_ranges_per_chunk=self.max_mm_ranges_per_chunk,
        )
        self.attention_visibility_spec = visibility_spec.to_dict()
        # Serialized compatibility alias.  New code should read the visibility
        # spec for semantics and attention_lowering for graph selection.
        self.uses_sliding_flash_attention_v2 = (
            self.attention_lowering == "flash_attention" and visibility_spec.has_sliding_attention
        )

    @staticmethod
    def _build_unified_visual_config(
        value,
        *,
        model_name: str,
        hf_model: str,
        contract: Gemma4SeriesModalityContract,
        input_modality: str,
        quant_scheme,
    ):
        if value is None:
            return None
        if isinstance(value, XHGemma4UnifiedVisualConfig):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"Gemma4 Unified {input_modality} visual_config must be a mapping")
        cfg = dict(value)
        expected_tokens = (
            contract.image_soft_tokens
            if input_modality == "image"
            else contract.video_soft_tokens_per_frame
        )
        numeric_contract = {
            "image_seq_length": expected_tokens,
            "max_soft_tokens": expected_tokens,
            "max_patches": expected_tokens,
            "input_dim": contract.vision_patch_dim,
            "position_capacity": contract.position_capacity,
        }
        for key, expected_value in numeric_contract.items():
            if key in cfg and int(cfg[key]) != expected_value:
                raise ValueError(
                    f"Gemma4 Unified {key} is checkpoint-owned: expected {expected_value}, got {cfg[key]}."
                )
            cfg.pop(key, None)
        if "input_modality" in cfg and str(cfg["input_modality"]) != input_modality:
            raise ValueError(
                "Gemma4 Unified input_modality is graph-owned: "
                f"expected {input_modality!r}, got {cfg['input_modality']!r}."
            )
        cfg.pop("input_modality", None)
        if "frontend_kind" in cfg and str(cfg["frontend_kind"]) != "encoder_free":
            raise ValueError(
                "Gemma4 Unified frontend_kind is checkpoint-owned: "
                f"expected 'encoder_free', got {cfg['frontend_kind']!r}."
            )
        cfg.pop("frontend_kind", None)
        cfg.setdefault("model_name", f"{model_name}_{'visual' if input_modality == 'image' else 'video_visual'}")
        cfg.setdefault("hf_model", hf_model)
        cfg.setdefault("quant_scheme", copy.deepcopy(quant_scheme))
        return XHGemma4UnifiedVisualConfig(
            image_seq_length=expected_tokens,
            input_dim=contract.vision_patch_dim,
            position_capacity=contract.position_capacity,
            input_modality=input_modality,
            **cfg,
        )

    @staticmethod
    def _build_unified_audio_config(
        value,
        *,
        model_name: str,
        hf_model: str,
        contract: Gemma4SeriesModalityContract,
        quant_scheme,
    ):
        if isinstance(value, XHGemma4UnifiedAudioConfig):
            return value
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise TypeError("Gemma4 Unified audio_config must be a mapping")
        cfg = dict(value)
        expected = {
            "input_feature_length": contract.audio_soft_tokens,
            "audio_seq_length": contract.audio_soft_tokens,
            "feature_size": contract.audio_feature_dim,
            "sampling_rate": contract.sampling_rate,
        }
        for key, expected_value in expected.items():
            if key in cfg and int(cfg[key]) != expected_value:
                raise ValueError(
                    f"Gemma4 Unified {key} is checkpoint-owned: expected {expected_value}, got {cfg[key]}."
                )
            cfg.pop(key, None)
        cfg.setdefault("model_name", f"{model_name}_audio")
        cfg.setdefault("hf_model", hf_model)
        cfg.setdefault("quant_scheme", copy.deepcopy(quant_scheme))
        return XHGemma4UnifiedAudioConfig(
            input_feature_length=contract.audio_soft_tokens,
            feature_size=contract.audio_feature_dim,
            sampling_rate=contract.sampling_rate,
            **cfg,
        )

    @staticmethod
    def _normalize_sliding_kv_cache_input_mode(mode: str | None) -> str:
        normalized = str(mode or "slice_window").lower()
        if normalized not in {"slice_window", "legacy_full"}:
            raise ValueError(
                "Unsupported Gemma4 sliding_kv_cache_input_mode: "
                f"{mode!r}. Expected 'slice_window' or 'legacy_full'."
            )
        return normalized

    @staticmethod
    def _load_hf_config(hf_model: str | None) -> dict:
        if hf_model is None:
            return {}
        config_path = Path(hf_model) / "config.json"
        if not config_path.exists():
            return {}
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)


# Compatibility names used by existing config serializers/tests while the
# public package name moves to gemma4_series.
XHGemma4ModelConfig = XHGemma4SeriesModelConfig
XHGemma4VisualConfig = XHGemma4SeriesVisualConfig
XHGemma4AudioConfig = XHGemma4SeriesAudioConfig
Gemma4ModelMeta = Gemma4SeriesModelMeta
Gemma4AudioModelMeta = Gemma4SeriesAudioModelMeta


__all__ = [
    "Gemma4AudioModelMeta",
    "Gemma4ModelMeta",
    "Gemma4SeriesAudioModelMeta",
    "Gemma4SeriesModelMeta",
    "XHGemma4AudioConfig",
    "XHGemma4ModelConfig",
    "XHGemma4SeriesAudioConfig",
    "XHGemma4SeriesModelConfig",
    "XHGemma4SeriesVisualConfig",
    "XHGemma4UnifiedAudioConfig",
    "XHGemma4UnifiedVisualConfig",
    "XHGemma4VisualConfig",
]
