# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.hunyuan_vl.modeling_hunyuan_vl import (
    HunYuanVLForConditionalGeneration,
)

from xhquant.api import to_export_graph, to_export_hmonnx_v2
from xhquant.utils.registry import _DMRegistryCls

from ....utils import calculate_file_md5
from ...builder import register_llm_model
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...types import ExportData, LLMModelState, ModelSwitcher, VisualModelMeta, VLLMModelMeta
from ...utils import unfold_args
from ...vision_llm_model import VisionLLMModel
from . import _llm_model_impl as _hunyuan_ocr_text_impl  # noqa: F401
from .data_preprocess import HunyuanOCRTextDataPreprocess
from .hunyuan_ocr_visual_model import XHHunYuanOCRVisualModel
from .xh_hunyuan_ocr_config import XHHunYuanOCRModelConfig


def _copy_model_shared_params(model: nn.Module) -> nn.Module:
    memo: dict[int, Any] = {}
    for parameter in model.parameters():
        memo.setdefault(id(parameter), nn.Parameter(parameter.data, requires_grad=parameter.requires_grad))
    for buffer in model.buffers():
        memo.setdefault(id(buffer), buffer)
    return copy.deepcopy(model, memo)


class HunyuanOCRVisualMeta(VisualModelMeta):
    def __init__(
        self,
        *,
        bucket_id: str = "fixed_896x1152",
        hmonnx: str = "",
        external_data: str = "",
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        input_shape: list[int] | None = None,
        output_shape: list[int] | None = None,
        input_dtype: str = "float16",
        output_dtype: str = "float16",
        image_grid_thw: list[int] | None = None,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        merged_grid_h: int | None = None,
        merged_grid_w: int | None = None,
        num_patches: int | None = None,
        image_token_count: int | None = None,
        hidden_size: int = 1024,
        output_layout: str = "begin,row_major_merged_patches_with_per_row_newline,end",
        image_size_w: int | None = None,
        image_size_h: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(image_size_w=image_size_w, image_size_h=image_size_h)
        self.meta = {"class_name": type(self).__name__}
        self.bucket_id = bucket_id
        self.hmonnx = hmonnx
        self.external_data = external_data
        self.input_names = input_names or ["pixel_values"]
        self.output_names = output_names or ["image_embeds"]
        self.input_shape = input_shape or [4032, 768]
        self.output_shape = output_shape or [1, 1046, hidden_size]
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.image_grid_thw = image_grid_thw or [1, 72, 56]
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.merged_grid_h = merged_grid_h or self.image_grid_thw[1] // spatial_merge_size
        self.merged_grid_w = merged_grid_w or self.image_grid_thw[2] // spatial_merge_size
        self.num_patches = num_patches or self.image_grid_thw[1] * self.image_grid_thw[2]
        self.image_token_count = image_token_count or self.merged_grid_h * (self.merged_grid_w + 1) + 2
        self.hidden_size = hidden_size
        self.output_layout = output_layout
        for name, value in kwargs.items():
            if not name.startswith("_") and name != "meta":
                setattr(self, name, value)

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=4), encoding="utf-8")
        return str(path)

    @classmethod
    def from_file(cls, path: str | Path) -> "HunyuanOCRVisualMeta":
        path = Path(path)
        values = json.loads(path.read_text(encoding="utf-8"))
        values.pop("meta", None)
        meta = cls.from_dict(values)
        root = path.parent
        if meta.hmonnx:
            meta.hmonnx = str(root / meta.hmonnx)
        if meta.external_data:
            meta.external_data = str(root / meta.external_data)
        return meta


class HunyuanOCRTextExportMeta(VLLMModelMeta):
    POSITION_ID_NAMES = (
        "sequence_position_ids",
        "width_position_ids",
        "height_position_ids",
        "image_position_ids",
    )
    DRAFT_ARTIFACT_FIELDS = (
        "dflash_context_hmonnx",
        "dflash_context_external_data",
        "dflash_context_decode_hmonnx",
        "dflash_context_decode_external_data",
        "dflash_decode_hmonnx",
        "dflash_decode_external_data",
    )

    def __init__(
        self,
        *,
        schema_version: int = 1,
        cache_update_mode: str = "in_place",
        output_names: list[str] | None = None,
        spec_decode: dict[str, Any] | None = None,
        verify_hmonnx: str = "",
        verify_external_data: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.schema_version = schema_version
        self.cache_update_mode = cache_update_mode
        self.position_id_names = list(self.POSITION_ID_NAMES)
        self.output_names = output_names or ["logits"]
        self.spec_decode = dict(spec_decode or {})
        self.verify_hmonnx = verify_hmonnx
        self.verify_external_data = verify_external_data
        meta_path = kwargs.get("_meta_path_")
        if meta_path:
            root = Path(meta_path).parent
            if self.verify_hmonnx:
                self.verify_hmonnx = str(root / self.verify_hmonnx)
            if self.verify_external_data:
                self.verify_external_data = str(root / self.verify_external_data)
            for field in self.DRAFT_ARTIFACT_FIELDS:
                value = getattr(self, field, "")
                if value:
                    setattr(self, field, str(root / value))

    @classmethod
    def _validate_serialized_contract(cls, data: dict[str, Any]) -> None:
        output_names = data.get("output_names", ["logits"])
        if int(data.get("schema_version", 1)) == 1 and output_names != ["logits"]:
            raise ValueError("HunyuanOCR metadata schema v1 exposes only the logits output")
        if output_names == ["logits", "target_hidden"]:
            spec_decode = data.get("spec_decode")
            if not isinstance(spec_decode, dict) or spec_decode.get("mode") != "dflash":
                raise ValueError("HunyuanOCR target_hidden output requires DFlash spec_decode metadata")

    @staticmethod
    def _runtime_ready(metadata: Any) -> bool:
        spec_decode = getattr(metadata, "spec_decode", None)
        if not isinstance(spec_decode, dict) or spec_decode.get("status") != "speculative_runtime_ready":
            return False
        capabilities = spec_decode.get("capabilities")
        if not isinstance(capabilities, dict) or capabilities.get("speculative_runtime") is not True:
            return False
        if capabilities.get("target_verify") is not True or capabilities.get("draft_graphs") is not True:
            return False
        draft = spec_decode.get("draft")
        if not isinstance(draft, dict):
            return False
        cache = draft.get("cache")
        output_head = draft.get("output_head")
        return isinstance(cache, dict) and isinstance(output_head, dict)

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        self._validate_serialized_contract(data)
        path.write_text(json.dumps(data, indent=4), encoding="utf-8")
        return str(path)

    @classmethod
    def from_file(cls, path: str | Path) -> "HunyuanOCRTextExportMeta":
        path = Path(path)
        values = json.loads(path.read_text(encoding="utf-8"))
        cls._validate_serialized_contract(values)
        values["_meta_path_"] = str(path)
        return cls.from_dict(values)


class HunyuanOCRModelMeta(HunyuanOCRTextExportMeta):
    def __init__(
        self,
        *,
        schema_version: int = 1,
        resolution_bucket_manifest: dict[str, Any] | None = None,
        visual_buckets: dict[str, HunyuanOCRVisualMeta] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(schema_version=schema_version, **kwargs)
        self.resolution_bucket_manifest = resolution_bucket_manifest
        self.visual_buckets = visual_buckets or {}
        meta_path = kwargs.get("_meta_path_")
        if meta_path:
            root = Path(meta_path).parent
            visual_artifacts = [self.visual_config, *self.visual_buckets.values()]
            for visual_meta in visual_artifacts:
                if visual_meta is None:
                    continue
                for field in ("hmonnx", "external_data"):
                    value = getattr(visual_meta, field, "")
                    if value:
                        setattr(visual_meta, field, str(root / value))

    def to_dict(self):
        data = super().to_dict()
        data["visual_buckets"] = {
            bucket_id: visual_meta.to_dict() if hasattr(visual_meta, "to_dict") else dict(visual_meta)
            for bucket_id, visual_meta in self.visual_buckets.items()
        }
        return data


class _HunyuanOCRHFCompatible(TextLLMHFCompatible):
    """Route the HF generation lifecycle through the Merak text backend."""

    def _setup(self: HunYuanVLForConditionalGeneration, xh_model: "XHHunYuanOCRModel"):
        model = super()._setup(xh_model)
        if model is not None:
            if hasattr(model, "model"):
                del model.model
            if hasattr(model, "lm_head"):
                del model.lm_head
        return model

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        has_past = self._past_seq_length > 0
        if not has_past:
            has_past = cache_position is not None and int(cache_position[0].item()) != 0
        if not has_past and past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            has_past = past_key_values.get_seq_length() != 0
        if has_past:
            input_ids = input_ids[:, -1:]
            inputs_embeds = None
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "cache_position": cache_position,
            "position_ids": position_ids,
            "use_cache": use_cache,
        }
        if not has_past:
            multimodal_inputs = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "mm_token_type_ids": mm_token_type_ids,
            }
            model_inputs.update({name: value for name, value in multimodal_inputs.items() if value is not None})
        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        attention_mask = model_kwargs.get("attention_mask")
        if attention_mask is None:
            positions = torch.arange(inputs_tensor.shape[1], device=inputs_tensor.device).view(1, -1)
            positions = positions.expand(inputs_tensor.shape[0], -1)
        else:
            positions = attention_mask.long().cumsum(-1) - 1
            positions.masked_fill_(attention_mask == 0, 0)
        return positions.unsqueeze(0).expand(4, -1, -1)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if self._past_seq_length > 0 and input_ids is not None:
            input_ids = input_ids[:, -1:]
        if self._past_seq_length == 0 and pixel_values is not None:
            data = self._prepare_multimodal_prefill(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
        elif self._past_seq_length == 0:
            data = self._prepare_text_prefill(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
            )
        else:
            if any(value is not None for value in (pixel_values, image_grid_thw, mm_token_type_ids)):
                raise ValueError("HunyuanOCR decode must not carry image inputs")
            data = self._prepare_decode_step(input_ids=input_ids, inputs_embeds=inputs_embeds)

        output = self._llm_model.run_generation_step(data)
        extract_logits = getattr(self._llm_model, "_extract_logits", None)
        logits = extract_logits(output) if callable(extract_logits) else output
        if past_key_values is None:
            past_key_values = DynamicCache(config=self.config.text_config)
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def _prepare_multimodal_prefill(
        self,
        *,
        input_ids: torch.LongTensor | None,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None,
    ) -> dict[str, Any]:
        if input_ids is None:
            raise ValueError("HunyuanOCR multimodal prefill requires input_ids")
        if image_grid_thw is None:
            raise ValueError("HunyuanOCR multimodal prefill requires image_grid_thw")
        if mm_token_type_ids is None:
            raise ValueError("HunyuanOCR multimodal prefill requires mm_token_type_ids")

        run_visual_bucket = getattr(self._llm_model, "run_visual_bucket", None)
        if callable(run_visual_bucket):
            image_embeds, visual_meta = run_visual_bucket(pixel_values, image_grid_thw)
            expected_shape = tuple(int(value) for value in visual_meta.output_shape)
        else:
            visual = self._llm_model.visual
            visual.validate_single_bucket_inputs(pixel_values, image_grid_thw)
            image_embeds = visual(pixel_values)
            if hasattr(image_embeds, "pooler_output"):
                image_embeds = image_embeds.pooler_output
            expected_shape = (1, visual.config.image_token_count, visual.config.text_hidden_size)
        if tuple(image_embeds.shape) != expected_shape:
            raise ValueError(
                "HunyuanOCR visual output does not match the fixed bucket contract: "
                f"expected={expected_shape}, got={tuple(image_embeds.shape)}"
            )
        return {
            "input_ids": input_ids,
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
            "past_seq_length": 0,
        }

    def _prepare_text_prefill(
        self,
        *,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor | None,
        position_ids: torch.LongTensor | None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"past_seq_length": 0}
        if input_ids is not None:
            data["input_ids"] = input_ids
        else:
            data["inputs_embeds"] = inputs_embeds
        if position_ids is not None and position_ids.ndim == 3 and position_ids.shape[0] == 4:
            data["position_ids"] = position_ids
        return data

    def _prepare_decode_step(
        self,
        *,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor | None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"past_seq_length": self._past_seq_length}
        if input_ids is not None:
            data["input_ids"] = input_ids[:, -1:]
        else:
            data["inputs_embeds"] = inputs_embeds[:, -1:]
        return data


def build_hunyuan_ocr_hf_compatible_model(
    hf_model: HunYuanVLForConditionalGeneration,
    xh_model: "XHHunYuanOCRModel",
) -> HunYuanVLForConditionalGeneration:
    compatible_modules = _DMRegistryCls("HunyuanOCRCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in compatible_modules:
        compatible_modules.register_module({hf_model_cls: hf_model_cls.__name__}, _HunyuanOCRHFCompatible)
    compatible_modules.convert(hf_model, xh_model=xh_model)
    return hf_model


@register_llm_model("HunYuanVLForConditionalGeneration")
class XHHunYuanOCRModel(VisionLLMModel):
    """Merak registration shell around the official HunyuanOCR HF model."""

    transformers_min_version = "5.13.0"
    transformers_max_version = "5.13.0"
    HF_MODEL_CLS = HunYuanVLForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    HF_MODEL_DTYPE = torch.bfloat16
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_hunyuan_ocr_hf_compatible_model)
    HMONNXINFERENCE_CLS = None
    META_CLS = HunyuanOCRModelMeta
    CONFIG_CLS = XHHunYuanOCRModelConfig
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.hunyuan_ocr.workflow:HunyuanOCRWorkflow"

    def __init__(self, config: XHHunYuanOCRModelConfig) -> None:
        super().__init__(config)
        self.visual = XHHunYuanOCRVisualModel(config.visual_config)
        self.image_token_id = int(config.image_token_id)
        self.image_start_token_id = int(config.image_start_token_id)
        self.image_end_token_id = int(config.image_end_token_id)
        self._verify_frontend_model = None
        self._verify_quanted_model = None
        if config.dflash_target_contract is not None:
            self.wrap_cfg["output_hidden_state_indices"] = list(
                config.dflash_target_contract["target_layer_ids"]
            )

    @classmethod
    def get_hmonnx_inference_cls(cls):
        from .hunyuan_ocr_hmonnx_inference import XHHunYuanOCRHMONNXModel

        XHHunYuanOCRHMONNXModel.LLM_MODEL_CLS = cls
        return XHHunYuanOCRHMONNXModel

    def _get_language_model(self, hf_model: HunYuanVLForConditionalGeneration) -> Any:
        return hf_model.model.language_model

    def _wraped_post(self, hf_model: HunYuanVLForConditionalGeneration) -> None:
        super()._wraped_post(hf_model)
        self.pad_token_id = int(hf_model.config.text_config.pad_token_id)
        self.image_token_id = int(hf_model.config.image_token_id)
        self.image_start_token_id = int(
            getattr(hf_model.config, "image_start_token_id", hf_model.config.im_start_id)
        )
        self.image_end_token_id = int(getattr(hf_model.config, "image_end_token_id", hf_model.config.im_end_id))

    def _get_data_preprocessor(self) -> HunyuanOCRTextDataPreprocess:
        return HunyuanOCRTextDataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=int(self.wrap_cfg.input_sequence_length),
            context_max_length=int(self.config.context_max_length),
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            pad_token_id=int(self.pad_token_id),
            image_token_id=self.image_token_id,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
            spatial_merge_size=int(self.config.visual_config.spatial_merge_size),
        )

    def _set_device(self, device: torch.device | str | None):
        super()._set_device(device)
        self.visual._set_device(device)
        return self

    def _set_dtype(self, dtype: torch.dtype | str | None):
        super()._set_dtype(dtype)
        self.visual._set_dtype(dtype)
        return self

    def _to_quanted(self, frontend_model, state, **kwargs):
        from ...types import ModelSwitcher

        if not isinstance(frontend_model, ModelSwitcher):
            return super()._to_quanted(frontend_model, state, **kwargs)
        self.set_prefill()
        prefill = super()._to_quanted(frontend_model.prefill, state, **kwargs)
        self.set_decode()
        decode = super()._to_quanted(frontend_model.decode, state, **kwargs)
        if self._verify_frontend_model is not None:
            regular = self._regular_graph_config()
            self._set_verify_graph_config()
            try:
                self._verify_quanted_model = super()._to_quanted(self._verify_frontend_model, state, **kwargs)
            finally:
                self._restore_graph_config(regular)
        self.set_prefill()
        quanted_model = ModelSwitcher({"prefill": prefill, "decode": decode})
        quanted_model.set_activate_model("prefill")
        return quanted_model

    def _to_fronted(self, wrap_model):
        self.set_prefill()
        prefill_wrap_model = wrap_model
        decode_wrap_model = _copy_model_shared_params(wrap_model)
        verify_wrap_model = (
            _copy_model_shared_params(wrap_model)
            if self.config.dflash_target_contract is not None
            else None
        )
        self._wrap_model = prefill_wrap_model
        prefill_frontend = super()._to_fronted(prefill_wrap_model)
        self._wrap_model = decode_wrap_model
        self.set_decode()
        decode_frontend = super()._to_fronted(decode_wrap_model)
        if verify_wrap_model is not None:
            self._wrap_model = verify_wrap_model
            regular = self._regular_graph_config()
            self._set_verify_graph_config(verify_wrap_model)
            try:
                self._verify_frontend_model = super()._to_fronted(verify_wrap_model)
            finally:
                self._restore_graph_config(regular)
        self._wrap_model = prefill_wrap_model
        self.set_prefill()
        frontend = ModelSwitcher({"prefill": prefill_frontend, "decode": decode_frontend})
        frontend.set_activate_model("prefill")
        return frontend

    def _regular_graph_config(self) -> dict[str, int]:
        return {
            "input_sequence_length": int(self.config.prefill_chunk_length),
            "num_logits_to_keep": int(self.config.num_logits_to_keep),
        }

    def _set_verify_graph_config(self, model: nn.Module | None = None) -> None:
        self._llm_prefill = False
        self.wrap_cfg["input_sequence_length"] = int(self.config.verify_input_length)
        self.wrap_cfg["num_logits_to_keep"] = 0
        if model is not None:
            def apply_verify_config(module: nn.Module) -> None:
                if hasattr(module, "num_logits_to_keep"):
                    module.num_logits_to_keep = 0
                if hasattr(module, "_update_cfg"):
                    module._update_cfg(self.wrap_cfg)

            model.apply(apply_verify_config)
        if self._data_processor is not None:
            self._data_processor.input_sequence_length = int(self.config.verify_input_length)

    def _restore_graph_config(self, values: dict[str, int]) -> None:
        self._llm_prefill = True
        self.wrap_cfg["input_sequence_length"] = values["input_sequence_length"]
        self.wrap_cfg["num_logits_to_keep"] = values["num_logits_to_keep"]
        if self._data_processor is not None:
            self._data_processor.input_sequence_length = values["input_sequence_length"]

    def get_export_cfg(self) -> dict[str, list[str]]:
        input_names = [
            "inputs_embeds",
            "sequence_position_ids",
            "width_position_ids",
            "height_position_ids",
            "image_position_ids",
            "past_seq_length",
            "current_input_length",
        ]
        input_names.extend(f"past_key_cache_{layer_index}" for layer_index in range(self.kvcache_config.num_layers))
        input_names.extend(f"past_value_cache_{layer_index}" for layer_index in range(self.kvcache_config.num_layers))
        output_names = ["logits"]
        if self.config.dflash_target_contract is not None:
            output_names.append("target_hidden")
        return {"input_names": input_names, "output_names": output_names}

    def _spec_decode_metadata(self) -> dict[str, Any]:
        contract = self.config.dflash_target_contract
        if contract is None:
            return {}
        metadata = {
            "status": "target_verify_ready",
            "mode": "dflash",
            "hidden_output_name": "target_hidden",
            "capabilities": {
                "target_hidden": True,
                "target_verify": True,
                "draft_graphs": False,
                "speculative_runtime": False,
            },
            "verify": {
                "input_length": int(self.config.verify_input_length),
                "max_draft_tokens": int(self.config.num_draft_tokens),
                "valid_length_input_name": "current_input_length",
                "output_names": ["logits", "target_hidden"],
                "cache_commit_mode": "in_place_prefix_length",
            },
            **{key: value for key, value in contract.items() if key != "config_path"},
        }
        draft_graphs = self.config.dflash_config.get("draft_graphs")
        if not isinstance(draft_graphs, dict):
            return metadata
        graph_paths = [draft_graphs.get(mode) for mode in ("context", "context_decode", "decode")]
        draft_contract = draft_graphs.get("contract")
        if not all(isinstance(path, str) and path for path in graph_paths) or not isinstance(draft_contract, dict):
            return metadata
        metadata["status"] = "speculative_runtime_ready"
        metadata["capabilities"] = {
            "target_hidden": True,
            "target_verify": True,
            "draft_graphs": True,
            "speculative_runtime": True,
        }
        metadata["draft"] = dict(draft_contract)
        return metadata

    def _create_text_export_info(self, output_dir: str) -> ExportData:
        text_dir = Path(output_dir) / "text"
        text_dir.mkdir(parents=True, exist_ok=True)
        base_meta = self.create_export_metadata(str(text_dir))
        text_model_config = self.config.to_dict()
        text_model_config.pop("visual_config", None)
        text_meta = HunyuanOCRTextExportMeta(
            schema_version=2 if self.config.dflash_target_contract is not None else 1,
            create_time=base_meta.create_time,
            model_config=text_model_config,
            hf_config=base_meta.hf_config,
            quant_embedding=base_meta.quant_embedding,
            quant_embedding_md5=base_meta.quant_embedding_md5,
            kv_cache=base_meta.kv_cache,
            pad_token_id=base_meta.pad_token_id,
            max_sequence_length=int(self.config.context_max_length),
            prefill_chunk_length=int(self.config.prefill_chunk_length),
            output_names=self.get_export_cfg()["output_names"],
            spec_decode=self._spec_decode_metadata(),
        )
        exported_info = ExportData()
        exported_info.exported_dir = str(text_dir)
        exported_info.meta = text_meta
        exported_info.model_name = f"{self.config.model_name.lower()}_text"
        return exported_info

    def _create_bundle_export_info(self, output_dir: str) -> ExportData:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        exported_info = ExportData()
        exported_info.exported_dir = str(root)
        exported_info.meta = self.create_export_metadata(str(root))
        exported_info.model_name = self.config.model_name.lower()
        return exported_info

    def _fix_text_graphs(self) -> None:
        if isinstance(self._quanted_model, ModelSwitcher):
            self._quanted_model.prefill.fixed()
            self._quanted_model.decode.fixed()
        else:
            self._quanted_model.fixed()

    def _export_text_graphs(self, exported_info: ExportData):
        with self._text_only_lifecycle():
            if self._state != LLMModelState.QUANTED_ALIGNED:
                self.to_quanted_aligned()
            self._fix_text_graphs()
            self._export_hmonnx(exported_info)
            if getattr(self.config, "dflash_target_contract", None) is not None:
                self._export_verify_graph(exported_info)
        return exported_info.meta

    def export_text_hmonnx(self, output_dir: str) -> HunyuanOCRTextExportMeta:
        exported_info = self._create_text_export_info(output_dir)
        metadata = self._export_text_graphs(exported_info)
        metadata.save(Path(exported_info.exported_dir) / "text_meta_info.json")
        return metadata

    def _export_additional_visual_bucket(self, bucket: dict[str, Any], output_dir: Path):
        visual_values = copy.deepcopy(self.config.visual_config.to_dict())
        visual_values.update(
            {
                "model_name": f"{self.config.model_name}_visual_{bucket['width']}x{bucket['height']}",
                "image_size_w": int(bucket["width"]),
                "image_size_h": int(bucket["height"]),
            }
        )
        visual = XHHunYuanOCRVisualModel(type(self.config.visual_config)(**visual_values))
        visual.to_wrap(self.get_native_model())
        visual.to_quanted_aligned()
        return visual.export_hmonnx(str(output_dir))

    def export_hmonnx(self, output_dir: str) -> HunyuanOCRModelMeta:
        if self._state == LLMModelState.NONE:
            with self._text_only_lifecycle():
                self.to_wrap()
        exported_info = self._create_bundle_export_info(output_dir)
        metadata = self._export_text_graphs(exported_info)
        if not isinstance(metadata, HunyuanOCRModelMeta):
            raise TypeError(f"Expected HunyuanOCRModelMeta, got {type(metadata).__name__}")

        root = Path(exported_info.exported_dir)
        manifest = copy.deepcopy(dict(self.config.resolution_bucket_contract))
        buckets = manifest.get("buckets", [])
        visual_buckets = {}
        default_size = (int(self.visual.config.image_size_w), int(self.visual.config.image_size_h))
        for bucket in buckets:
            bucket_id = bucket["id"]
            bucket_dir = root / "visual" / bucket_id
            bucket_size = (int(bucket["width"]), int(bucket["height"]))
            if bucket_size == default_size:
                if self.visual._state == LLMModelState.NONE:
                    self.visual.to_wrap(self.get_native_model())
                visual_meta = self.visual.export_hmonnx(str(bucket_dir))
            else:
                visual_meta = self._export_additional_visual_bucket(bucket, bucket_dir)
            visual_meta.bucket_id = bucket_id
            visual_meta.hmonnx = Path(visual_meta.hmonnx).relative_to(root).as_posix()
            visual_meta.external_data = Path(visual_meta.external_data).relative_to(root).as_posix()
            visual_buckets[bucket_id] = visual_meta
        if not visual_buckets:
            raise ValueError("HunyuanOCR bundle export requires approved visual buckets")
        metadata.resolution_bucket_manifest = manifest
        metadata.visual_buckets = visual_buckets
        metadata.visual_config = visual_buckets[buckets[0]["id"]]

        final_path = root / "golden_meta_info.json"
        temporary_path = root / ".golden_meta_info.json.tmp"
        try:
            metadata.save(temporary_path)
            temporary_path.replace(final_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return metadata

    def _export_verify_graph(self, exported_info: ExportData) -> None:
        if self._verify_quanted_model is None:
            raise RuntimeError("HunyuanOCR DFlash verify graph was not quantized")
        if not self._verify_quanted_model.is_fixed():
            self._verify_quanted_model.fixed()
        output_root = Path(exported_info.exported_dir)
        verify_dir = output_root / "verify"
        verify_dir.mkdir(parents=True, exist_ok=True)
        regular = self._regular_graph_config()
        try:
            with self.get_kvcache_mixin().kv_cache_scope(device="meta"):
                self._set_verify_graph_config()
                processor = self.get_data_preprocessor()
                processor.input_sequence_length = int(self.config.verify_input_length)
                inputs = processor(
                    {
                        "input_ids": torch.zeros((1, self.config.verify_input_length), dtype=torch.long),
                        "past_seq_length": int(self.config.prefill_chunk_length),
                        "current_input_length": int(self.config.verify_input_length),
                    }
                )
                inputs = unfold_args(inputs)
                exported_model = to_export_graph(self._verify_quanted_model, inputs)
                verify_file = verify_dir / f"{exported_info.model_name}_verify.onnx"
                export_cfg = copy.deepcopy(self.get_export_cfg())
                export_cfg["input_names"] = self.xh1_hmonnx_compatible(export_cfg["input_names"])
                verify_hmonnx = to_export_hmonnx_v2(
                    exported_model,
                    inputs,
                    str(verify_file),
                    export_cfg,
                    normalize_onnx_name=True,
                )
        finally:
            self._restore_graph_config(regular)
        exported_info.meta.verify_hmonnx = Path(verify_hmonnx).relative_to(output_root).as_posix()
        exported_info.meta.verify_hmonnx_md5 = calculate_file_md5(verify_hmonnx)
        external = Path(verify_hmonnx).with_name(
            f"{Path(verify_hmonnx).stem.removesuffix('_with_act')}_external_data"
        )
        if external.exists():
            exported_info.meta.verify_external_data = external.relative_to(output_root).as_posix()

    def _extra_export_metadata(self, output_dir: str, meta_info: HunyuanOCRModelMeta) -> HunyuanOCRModelMeta:
        del output_dir
        meta_info.tokenizer_eos_token_id = int(self.config.tokenizer_eos_token_id)
        meta_info.generation_eos_token_id = self.config.generation_eos_token_id
        meta_info.image_token_id = int(self.config.image_token_id)
        meta_info.schema_version = 2 if self.config.dflash_target_contract is not None else 1
        meta_info.spec_decode = self._spec_decode_metadata()
        return meta_info

    @contextmanager
    def _text_only_lifecycle(self):
        registered_models = self._models.copy()
        self._models.pop("visual", None)
        try:
            yield
        finally:
            self._models.clear()
            self._models.update(registered_models)
