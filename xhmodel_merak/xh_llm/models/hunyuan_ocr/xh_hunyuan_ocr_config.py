# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from safetensors import safe_open

from xhmodel_merak.configuration_utils import HFModelConfig
from xhquant.api import QuantScheme

from ...vision_llm_model import VisionLLMModelConfig


HUNYUAN_OCR_MODEL_TYPE = "HunYuanVLForConditionalGeneration"
HUNYUAN_OCR_VISUAL_MODEL_TYPE = f"{HUNYUAN_OCR_MODEL_TYPE}_visual"
HUNYUAN_OCR_FIXED_IMAGE_SIZE_W = 896
HUNYUAN_OCR_FIXED_IMAGE_SIZE_H = 1152


def _load_json(hf_model: str | None, filename: str) -> dict[str, Any]:
    if hf_model is None:
        return {}
    path = Path(hf_model) / filename
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _normalise_generation_eos_token_id(value: Any) -> int | list[int]:
    if type(value) is int:
        return value
    if isinstance(value, (list, tuple)) and value and all(type(token_id) is int for token_id in value):
        return list(dict.fromkeys(value))
    raise ValueError("generation_config.eos_token_id must be an integer or a non-empty integer list")


def _load_dflash_target_contract(
    *,
    target_config: Mapping[str, Any],
    dflash_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(dflash_config, Mapping):
        raise ValueError("dflash_config is required when spec_decode_mode='dflash'")
    dflash_model = dflash_config.get("hf_model")
    if not isinstance(dflash_model, str) or not dflash_model:
        raise ValueError("dflash_config.hf_model must be a non-empty checkpoint path")

    dflash_root = Path(dflash_model)
    config_path = dflash_root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"HunyuanOCR DFlash config does not exist: {config_path}")
    config_bytes = config_path.read_bytes()
    raw_dflash_config = json.loads(config_bytes)
    values = raw_dflash_config.get("dflash_config")
    target_layer_ids = values.get("target_layer_ids") if isinstance(values, Mapping) else None
    if not isinstance(target_layer_ids, list) or not target_layer_ids:
        raise ValueError("dflash_config.target_layer_ids must be a non-empty integer list")
    if any(type(layer_id) is not int for layer_id in target_layer_ids):
        raise ValueError("dflash_config.target_layer_ids must contain only integers")
    if len(set(target_layer_ids)) != len(target_layer_ids):
        raise ValueError("dflash_config.target_layer_ids must not contain duplicates")
    if any(left >= right for left, right in zip(target_layer_ids, target_layer_ids[1:], strict=False)):
        raise ValueError("dflash_config.target_layer_ids must be strictly increasing")

    text_config = target_config.get("text_config", {})
    target_layers = int(text_config.get("num_hidden_layers", 0))
    draft_target_layers = int(raw_dflash_config.get("num_target_layers", 0))
    if any(layer_id < 0 or layer_id >= min(target_layers, draft_target_layers) for layer_id in target_layer_ids):
        raise ValueError("dflash_config.target_layer_ids contains an id outside the target layer range")

    target_hidden_size = int(text_config.get("hidden_size", 0))
    if int(raw_dflash_config.get("hidden_size", 0)) != target_hidden_size:
        raise ValueError("HunyuanOCR DFlash hidden_size mismatch")
    weights_path = dflash_root / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"HunyuanOCR DFlash weights do not exist: {weights_path}")
    with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
        if "fc.weight" not in checkpoint.keys():
            raise ValueError(f"HunyuanOCR DFlash checkpoint is missing fc.weight: {weights_path}")
        fc_shape = list(checkpoint.get_slice("fc.weight").get_shape())
    expected_shape = [target_hidden_size, len(target_layer_ids) * target_hidden_size]
    if fc_shape != expected_shape:
        raise ValueError(f"HunyuanOCR DFlash fc.weight shape mismatch: expected={expected_shape}, got={fc_shape}")

    return {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "target_layer_ids": target_layer_ids,
        "target_hidden_size": target_hidden_size,
        "target_hidden_concat_size": expected_shape[1],
        "hidden_layout": "concat_last_dim",
        "reference_dtype": "bfloat16",
        "deployment_dtype": "float16",
    }


class XHHunYuanOCRVisualConfig(HFModelConfig):
    """Architecture-only configuration for the HunyuanOCR vision tower."""

    def __init__(
        self,
        *,
        model_type: str = HUNYUAN_OCR_VISUAL_MODEL_TYPE,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        spatial_patch_size: int | None = None,
        spatial_merge_size: int | None = None,
        hidden_size: int | None = None,
        text_hidden_size: int | None = None,
        num_hidden_layers: int | None = None,
        cat_extra_token: bool | None = None,
        image_size_w: int = HUNYUAN_OCR_FIXED_IMAGE_SIZE_W,
        image_size_h: int = HUNYUAN_OCR_FIXED_IMAGE_SIZE_H,
        approved_bucket_sizes: list[list[int]] | tuple[tuple[int, int], ...] | None = None,
        **kwargs: Any,
    ) -> None:
        vision = _load_json(kwargs.get("hf_model"), "config.json").get("vision_config", {})
        super().__init__(model_type=model_type, **kwargs)
        self.patch_size = int(patch_size if patch_size is not None else vision.get("patch_size", 16))
        self.temporal_patch_size = int(
            temporal_patch_size if temporal_patch_size is not None else vision.get("temporal_patch_size", 1)
        )
        self.spatial_patch_size = int(
            spatial_patch_size if spatial_patch_size is not None else vision.get("spatial_patch_size", 1)
        )
        self.spatial_merge_size = int(
            spatial_merge_size if spatial_merge_size is not None else vision.get("spatial_merge_size", 2)
        )
        self.hidden_size = int(hidden_size if hidden_size is not None else vision.get("hidden_size", 1152))
        self.text_hidden_size = int(
            text_hidden_size if text_hidden_size is not None else vision.get("text_hidden_size", 1024)
        )
        self.num_hidden_layers = int(
            num_hidden_layers if num_hidden_layers is not None else vision.get("num_hidden_layers", 27)
        )
        self.cat_extra_token = bool(
            cat_extra_token if cat_extra_token is not None else vision.get("cat_extra_token", True)
        )
        self.image_size_w = int(image_size_w)
        self.image_size_h = int(image_size_h)
        self.approved_bucket_sizes = [
            [int(width), int(height)] for width, height in (approved_bucket_sizes or [])
        ]
        allowed_sizes = {
            (HUNYUAN_OCR_FIXED_IMAGE_SIZE_W, HUNYUAN_OCR_FIXED_IMAGE_SIZE_H),
            *(tuple(size) for size in self.approved_bucket_sizes),
        }
        if (self.image_size_w, self.image_size_h) not in allowed_sizes:
            raise ValueError(
                "HunyuanOCR visual bucket is not approved: "
                f"{self.image_size_w}x{self.image_size_h}"
            )
        if self.image_size_w % self.patch_size or self.image_size_h % self.patch_size:
            raise ValueError("HunyuanOCR visual bucket dimensions must be divisible by patch_size")
        if not self.cat_extra_token:
            raise ValueError("HunyuanOCR visual bucket requires cat_extra_token=True")

        self.grid_h = self.image_size_h // self.patch_size
        self.grid_w = self.image_size_w // self.patch_size
        if self.grid_h % self.spatial_merge_size or self.grid_w % self.spatial_merge_size:
            raise ValueError("HunyuanOCR visual grid must be divisible by spatial_merge_size")
        self.num_patches = self.grid_h * self.grid_w
        merged_h = self.grid_h // self.spatial_merge_size
        merged_w = self.grid_w // self.spatial_merge_size
        self.image_token_count = merged_h * (merged_w + 1) + 2


class XHHunYuanOCRDFlashConfig(HFModelConfig):
    """Static export configuration for the three HunyuanOCR DFlash graphs."""

    _SEQUENCE_LENGTHS = {"context": 256, "context_decode": 16, "decode": 16}

    def __init__(
        self,
        *,
        mode: str,
        target_model_dir: str,
        context_max_length: int,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        if mode not in self._SEQUENCE_LENGTHS:
            raise ValueError(f"Unsupported HunyuanOCR DFlash mode: {mode!r}")
        if batch_size != 1:
            raise ValueError("HunyuanOCR DFlash export requires batch_size=1")
        if type(context_max_length) is not int or context_max_length <= 0:
            raise ValueError("HunyuanOCR DFlash context_max_length must be positive")
        quant_scheme = kwargs.pop("quant_scheme", None)
        if quant_scheme is None:
            quant_scheme = {"quant_type": "w16a16h0_sefp", "ops": {}}
        super().__init__(batch_size=batch_size, quant_scheme=quant_scheme, **kwargs)
        self.mode = mode
        self.target_model_dir = target_model_dir
        self.context_max_length = context_max_length
        self.max_sequence_length = context_max_length
        self.input_sequence_length = self._SEQUENCE_LENGTHS[mode]
        self.num_hidden_layers = 5
        self.quantization_mode = "w16a16h0_sefp"

    @property
    def dflash_model_dir(self) -> str:
        return self.hf_model


class XHHunYuanOCRModelConfig(VisionLLMModelConfig):
    """Merak construction config for the official HunyuanOCR checkpoint."""

    def __init__(
        self,
        *,
        model_name: str,
        chip_arch: str = "XH2a",
        model_type: str = HUNYUAN_OCR_MODEL_TYPE,
        quant_scheme: dict | QuantScheme | None = None,
        quant_weight: str | None = None,
        hf_model: str | None = None,
        batch_size: int = 1,
        context_max_length: int | None = None,
        prefill_chunk_length: int = 256,
        num_logits_to_keep: int | None = 1,
        mix_search: bool = False,
        use_cache: bool = True,
        smoke_only: bool = True,
        calibration: Mapping[str, Any] | None = None,
        resolution_bucket_manifest: str | None = None,
        spec_decode_mode: str | None = None,
        dflash_config: Mapping[str, Any] | None = None,
        num_draft_tokens: int = 15,
        visual_config: Mapping[str, Any] | XHHunYuanOCRVisualConfig | None = None,
        **kwargs: Any,
    ) -> None:
        raw_config = _load_json(hf_model, "config.json")
        architectures = raw_config.get("architectures")
        if architectures is not None and HUNYUAN_OCR_MODEL_TYPE not in architectures:
            raise ValueError(
                f"HunyuanOCR checkpoint architectures must contain {HUNYUAN_OCR_MODEL_TYPE!r}, "
                f"got {architectures!r}"
            )
        text_config = raw_config.get("text_config", {})
        if context_max_length is None:
            context_max_length = int(text_config.get("max_position_embeddings", 131072))
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
        if visual_config is None:
            visual_config = {}
        if isinstance(visual_config, Mapping):
            visual_values = dict(visual_config)
            self.resolution_bucket_manifest = resolution_bucket_manifest
            self.resolution_bucket_contract = {}
            if resolution_bucket_manifest is not None:
                from examples_merak.llm.hunyuan_ocr.hunyuan_ocr_resolution_buckets import (
                    load_resolution_bucket_manifest,
                )

                self.resolution_bucket_contract = load_resolution_bucket_manifest(
                    resolution_bucket_manifest,
                    require_approved=True,
                )
                buckets = self.resolution_bucket_contract["buckets"]
                visual_values.setdefault(
                    "approved_bucket_sizes",
                    [[int(bucket["width"]), int(bucket["height"])] for bucket in buckets],
                )
                first_bucket = buckets[0]
                visual_values.setdefault("image_size_w", int(first_bucket["width"]))
                visual_values.setdefault("image_size_h", int(first_bucket["height"]))
            visual_values.setdefault("model_name", f"{model_name}_visual")
            visual_values.setdefault("model_type", HUNYUAN_OCR_VISUAL_MODEL_TYPE)
            visual_values.setdefault("hf_model", hf_model)
            visual_config = XHHunYuanOCRVisualConfig(**visual_values)
        else:
            self.resolution_bucket_manifest = resolution_bucket_manifest
            self.resolution_bucket_contract = {}
        self.visual_config = visual_config

        generation_config = _load_json(hf_model, "generation_config.json")
        self.image_token_id = int(raw_config.get("image_token_id", 120120))
        self.image_start_token_id = int(raw_config.get("image_start_token_id", raw_config.get("im_start_id", 120118)))
        self.image_end_token_id = int(raw_config.get("image_end_token_id", raw_config.get("im_end_id", 120119)))
        self.bos_token_id = int(text_config.get("bos_token_id", 120000))
        self.pad_token_id = int(text_config.get("pad_token_id", 120002))
        self.tokenizer_eos_token_id = int(text_config.get("eos_token_id", 120007))
        self.generation_eos_token_id = _normalise_generation_eos_token_id(
            generation_config.get("eos_token_id", 120020)
        )
        self.smoke_only = bool(smoke_only)
        self.calibration = copy.deepcopy(dict(calibration or {}))
        if spec_decode_mode not in (None, "dflash"):
            raise ValueError(f"Unsupported HunyuanOCR spec_decode_mode: {spec_decode_mode!r}")
        self.spec_decode_mode = spec_decode_mode
        self.dflash_config = copy.deepcopy(dict(dflash_config or {}))
        if type(num_draft_tokens) is not int or num_draft_tokens != 15:
            raise ValueError("HunyuanOCR DFlash num_draft_tokens must be 15")
        self.num_draft_tokens = num_draft_tokens
        self.verify_input_length = num_draft_tokens + 1
        self.dflash_target_contract = (
            _load_dflash_target_contract(target_config=raw_config, dflash_config=dflash_config)
            if spec_decode_mode == "dflash"
            else None
        )
