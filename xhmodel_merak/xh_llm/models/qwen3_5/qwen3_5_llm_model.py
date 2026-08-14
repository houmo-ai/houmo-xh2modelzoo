# -*- coding: utf-8 -*-
# Copyright 2025 The Qwen Team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 HOUMO AI. All rights reserved.
#
# Modifications:
# - Portions of this file have been modified by HOUMO AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File: qwen3_5_llm_model.py
# Description:
#   Qwen3.5 LLM model adapted for the xh2 model zoo (xh2modelzoo).
import copy
import gc
import json
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union, cast

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoModelForImageTextToText
from transformers.cache_utils import Cache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5CausalLMOutputWithPast

from xhmodel_merak.utils import calculate_file_md5
from xhmodel_merak.xh_llm.base_model import get_model_param_buffer_size_gb
from xhmodel_merak.xh_llm.llm_data_processor import BaseLLMInputProcessor
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_processor import XHQwen3_5Processor
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_vision_model import XHQwen3_5VisionModel
from xhquant import nn as xhnn
from xhquant.api import CacheTensor, get_xhquant_logger
from xhquant.utils import ConfigDict, log_function_call
from xhquant.utils.registry import _DMRegistryCls

from ...builder import register_llm_model
from ...kv_cache_mixin import KVCacheWithLinearMixin
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...types import (
    CacheList,
    ExportData,
    KVCacheWithLinearConfig,
    LLMModelState,
    ModelSwitcher,
    VLLMModelMeta,
)
from ...utils import get_cpu_memory_mb, is_huge_model_export_enabled
from ...vision_llm_model import VisionLLMModel
from ._gdr_ops import GDRChunkScan
from .data_preprocess import Qwen3_5_DataPreprocess
from .lora import (
    LoRAAdapterSpec,
    apply_lora_to_frontend,
    attach_lora_buffers,
    finalize_lora_metadata,
    inspect_lora_adapters,
)
from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as XHQwen3_5ForConditionalGeneration
from .modeling_qwen3_5_patch import qwen3_5_patch
from .qwen3_5_hmonnx_inference import XHQwen3_5_HMONNXModel
from .split_conv_cache_utils import (
    _is_grouped_split_conv_cache,
    _regroup_flat_split_conv_cache,
)
from .xh_qwen3_5_config import XHQwen3_5ModelConfig


try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    no_init_weights = init_empty_weights

from ._llm_model_impl import register_wrap_modules


def build_qwen35_spec_decode_contract(
    config: Any,
    meta_info: Any,
) -> dict[str, Any]:
    """Build the deployment ABI consumed by the Merak vLLM proposer."""

    mode = str(getattr(config, "spec_decode_mode", "") or "").lower()
    if mode not in {"mtp", "dflash"}:
        raise ValueError(f"Qwen3.5 speculative contract requires mtp/dflash, got {mode!r}")
    num_draft_tokens = int(getattr(config, "num_draft_tokens", 0))
    if num_draft_tokens <= 0:
        raise ValueError("Qwen3.5 num_draft_tokens must be positive")
    draft_config = config.mtp_config if mode == "mtp" else config.dflash_config
    if draft_config is None:
        raise ValueError(f"Qwen3.5 {mode} export is missing its draft config")
    draft_head_weight_bits = int(
        getattr(
            draft_config,
            "draft_head_weight_bits",
            getattr(config, "spec_draft_head_weight_bits", 4),
        )
    )
    hidden_output_name = "target_hidden" if mode == "dflash" else "post_norm_hidden"
    contract: dict[str, Any] = {
        "runtime_contract_version": 2,
        "mode": mode,
        # Retain the legacy DFlash block-size convention for old tools.
        "block_size": (num_draft_tokens + 1 if mode == "dflash" else num_draft_tokens),
        "num_draft_tokens": num_draft_tokens,
        "verify_length": num_draft_tokens + 1,
        "draft_head_weight_bits": draft_head_weight_bits,
        "hidden_output_name": hidden_output_name,
        "target": {
            "hidden_output_name": hidden_output_name,
            "linear_state_outputs": "per_step",
        },
    }

    def graph_path(attribute: str) -> str:
        graph_config = getattr(meta_info, attribute, None)
        path = getattr(graph_config, "hmonnx", None)
        if not isinstance(path, str) or not path:
            raise ValueError(f"Qwen3.5 {mode} export is missing {attribute}.hmonnx")
        return path

    if mode == "mtp":
        prefill_path = graph_path("mtp_prefill_config")
        decode_path = graph_path("mtp_decode_config")
        contract.update(
            draft_prefill_onnx=prefill_path,
            mtp_draft_prefill_onnx=prefill_path,
            draft_decode_onnx=decode_path,
            mtp_draft_decode_onnx=decode_path,
        )
        contract["draft"] = {
            "abi": "qwen_mtp_paged_v2",
            "prefill_hmonnx": prefill_path,
            "decode_hmonnx": decode_path,
            "cache_mutation": "page_attention",
            "cache_binding": "private_draft",
            "page_cache_block_size": 64,
        }
    else:
        context_path = graph_path("dflash_context_config")
        context_decode_path = graph_path("dflash_context_decode_config")
        decode_path = graph_path("dflash_decode_config")
        noise_token_id_value = getattr(draft_config, "noise_token_id", None)
        if noise_token_id_value is None:
            raise ValueError(
                "Qwen3.5 DFlash deployment metadata requires the assistant checkpoint dflash_config.mask_token_id"
            )
        noise_token_id = int(noise_token_id_value)
        if noise_token_id < 0:
            raise ValueError("Qwen3.5 DFlash noise_token_id must be non-negative")
        flash_attention = getattr(draft_config, "flash_attention", None) or {}
        flash_attention_enabled = bool(
            flash_attention.get("enable", False)
            if hasattr(flash_attention, "get")
            else getattr(flash_attention, "enable", False)
        )
        contract.update(
            draft_context_onnx=context_path,
            dflash_draft_context_onnx=context_path,
            draft_context_decode_onnx=context_decode_path,
            dflash_draft_context_decode_onnx=context_decode_path,
            draft_decode_onnx=decode_path,
            dflash_draft_decode_onnx=decode_path,
        )
        contract["draft"] = {
            "abi": ("qwen_dflash_paged_shared_v3" if flash_attention_enabled else "qwen_dflash_v1"),
            "context_hmonnx": context_path,
            "context_decode_hmonnx": context_decode_path,
            "decode_hmonnx": decode_path,
            "cache_mutation": ("page_attention" if flash_attention_enabled else "in_place"),
            "cache_binding": "private_draft",
            "noise_token_id": noise_token_id,
        }
        if flash_attention_enabled:
            contract["draft"]["page_cache_block_size"] = 64
    return contract


class _Qwen3_5KVCacheMixin(KVCacheWithLinearMixin):  # noqa: N801
    """Extended mixin supporting split_conv_cache (3 separate q/k/v caches per layer)."""

    def __init__(self, kv_cache_config: KVCacheWithLinearConfig) -> None:
        super().__init__(kv_cache_config)
        self.split_conv_cache: bool = False
        self._linear_key_dim: int = -1
        self._linear_value_dim: int = -1

    def prepare_other_cache(self):
        if not self.split_conv_cache:
            return super().prepare_other_cache()
        linear_cfg = self.kvcache_config.linear_kv_cache_config
        cache_dtype = linear_cfg.cache_torch_dtype
        batch_size = linear_cfg.batch_size
        kernel_size = linear_cfg.conv_kernel_size
        for _i in range(linear_cfg.num_layers):
            conv_cache_q = CacheTensor(
                torch.zeros(
                    batch_size,
                    self._linear_key_dim,
                    kernel_size,
                    dtype=cache_dtype,
                    device=self._device,
                )
            )
            conv_cache_k = CacheTensor(
                torch.zeros(
                    batch_size,
                    self._linear_key_dim,
                    kernel_size,
                    dtype=cache_dtype,
                    device=self._device,
                )
            )
            conv_cache_v = CacheTensor(
                torch.zeros(
                    batch_size,
                    self._linear_value_dim,
                    kernel_size,
                    dtype=cache_dtype,
                    device=self._device,
                )
            )
            self.past_conv_caches.append(CacheList([conv_cache_q, conv_cache_k, conv_cache_v]))
            recurrent_cache_shape = [
                batch_size,
                linear_cfg.num_v_heads,
                linear_cfg.head_k_dim,
                linear_cfg.head_v_dim,
            ]
            self.past_recurrent_states.append(
                CacheTensor(torch.zeros(recurrent_cache_shape, dtype=cache_dtype, device=self._device))
            )


def _copy_model_shared_params(model: nn.Module) -> nn.Module:
    """深拷贝模型结构，所有 tensor/parameter/buffer 与原模型共享数据。"""

    def _share_tensors(obj: Any, memo: dict[int, Any], seen: set[int]) -> None:
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, nn.Parameter):
            memo.setdefault(obj_id, nn.Parameter(obj.data, requires_grad=obj.requires_grad))
            return
        if torch.is_tensor(obj):
            memo.setdefault(obj_id, obj)
            return
        if isinstance(obj, nn.Module):
            for value in obj.__dict__.values():
                _share_tensors(value, memo, seen)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                _share_tensors(key, memo, seen)
                _share_tensors(value, memo, seen)
            return
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                _share_tensors(item, memo, seen)

    memo: dict[int, Any] = {}
    _share_tensors(model, memo, set())
    return copy.deepcopy(model, memo)


class _Qwen3_5HFCompatible(TextLLMHFCompatible):  # noqa: N801
    def _setup(self: XHQwen3_5ForConditionalGeneration, text_llm_model: "XHQwen3_5Model"):
        model = super()._setup(text_llm_model)
        if model is not None:
            # if hasattr(model, "model"):
            #     del model.model
            del model.model.visual
            del model.model.language_model
            if hasattr(model, "lm_head"):
                del model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, Qwen3_5CausalLMOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = None

        if pixel_values is not None:
            if hasattr(self._llm_model.visual, "encode_many"):
                image_embed_list = self._llm_model.visual.encode_many(pixel_values, image_grid_thw)
                normalized_image_embeds = []
                for embedding in image_embed_list:
                    if embedding.ndim == 3 and embedding.shape[0] == 1:
                        embedding = embedding[0]
                    elif embedding.ndim != 2:
                        raise ValueError(
                            "Qwen3.5 visual token-gear output must have shape [N, D] "
                            f"or [1, N, D], got {tuple(embedding.shape)}"
                        )
                    normalized_image_embeds.append(embedding)
                image_embeds = torch.cat(
                    normalized_image_embeds,
                    dim=0,
                ).to(inputs_embeds.device, inputs_embeds.dtype)
            else:
                image_embeds = list()
                for i in range(len(pixel_values)):
                    image_embeds_i = self._llm_model.visual.forward(
                        pixel_values[i].type(self._llm_model.visual.dtype).to(self._llm_model.visual.device),
                    )
                    image_embeds.append(image_embeds_i)

                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_embeds = image_embeds.squeeze(0)

        seq_length = inputs_embeds.shape[1]
        data_processor = self._llm_model.get_data_preprocessor()
        net_input_seq_len = self._llm_model.get_input_sequence_length()

        if seq_length <= net_input_seq_len:
            data_batch = {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds if input_ids is None else None,
                "image_embeds": image_embeds,
                "past_seq_length": self._past_seq_length,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
            }
            data_input = data_processor(data_batch)
            (
                inputs_embeds,
                time_position_ids,
                height_position_ids,
                width_position_ids,
                past_seq_length,
                current_seq_length,
                linear_mask,
                past_key_values,
                past_value_caches,
                past_conv_caches,
                past_recurrent_states,
            ) = data_input

            result = self._llm_model.forward(
                inputs_embeds,
                time_position_ids,
                height_position_ids,
                width_position_ids,
                past_seq_length,
                current_seq_length,
                linear_mask,
                past_key_values,
                past_value_caches,
                past_conv_caches,
                past_recurrent_states,
            )
            logits = result[0]
        else:
            device = inputs_embeds.device

            if image_embeds is not None and input_ids is not None:
                image_token_id = data_processor.image_token_id
                n_image_tokens = int((input_ids == image_token_id).sum().item())
                if n_image_tokens > 0:
                    n_image_features = int(image_embeds.shape[0])
                    if n_image_features != n_image_tokens:
                        raise ValueError(
                            "Image features and image tokens do not match: "
                            f"tokens={n_image_tokens}, features={n_image_features}"
                        )
                    image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                    image_embeds_merged = image_embeds.to(device=device, dtype=inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_merged)

            position_ids, rope_deltas = data_processor.get_rope_index(
                input_ids, inputs_embeds, image_grid_thw, video_grid_thw, attention_mask
            )
            data_processor.rope_deltas = rope_deltas

            steps = (seq_length + net_input_seq_len - 1) // net_input_seq_len
            pad_len = steps * net_input_seq_len - seq_length
            if pad_len > 0:
                padding_embeds = self.get_input_embeddings()(torch.zeros((1, pad_len), dtype=torch.long, device=device))
                inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)
                last_pos = position_ids[:, :, -1:].expand(-1, -1, pad_len)
                position_ids = torch.cat([position_ids, last_pos], dim=2)

            past_key_caches = data_processor.past_key_caches
            past_value_caches = data_processor.past_value_caches
            past_conv_caches = data_processor.past_conv_caches
            past_recurrent_states = data_processor.past_recurrent_states
            running_past_seq = self._past_seq_length
            outputs_logits = []

            for step_index in range(steps):
                start = step_index * net_input_seq_len
                end = (step_index + 1) * net_input_seq_len
                sub_current_len = min(end, seq_length) - start
                sub_embeds = inputs_embeds[:, start:end, :]
                sub_time_pos = position_ids[0, 0, start:end].to(torch.int64)
                sub_height_pos = position_ids[1, 0, start:end].to(torch.int64)
                sub_width_pos = position_ids[2, 0, start:end].to(torch.int64)
                linear_mask = (
                    torch.cat(
                        [
                            torch.ones(sub_current_len, device=device),
                            torch.zeros(net_input_seq_len - sub_current_len, device=device),
                        ]
                    )
                    .unsqueeze(0)
                    .to(dtype=torch.float16)
                )

                chunk_result = self._llm_model.forward(
                    sub_embeds,
                    sub_time_pos,
                    sub_height_pos,
                    sub_width_pos,
                    torch.tensor([running_past_seq], dtype=torch.int32, device=device),
                    torch.tensor([sub_current_len], dtype=torch.int32, device=device),
                    linear_mask,
                    past_key_caches,
                    past_value_caches,
                    past_conv_caches,
                    past_recurrent_states,
                )
                outputs_logits.append(chunk_result[0])
                running_past_seq += sub_current_len

            if self._llm_model.get_num_logits_to_keep() == 0:
                logits = torch.cat(outputs_logits, dim=1)[:, :seq_length, :]
            else:
                logits = outputs_logits[-1]

        return Qwen3_5CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            rope_deltas=data_processor.rope_deltas,
        )


def build_qwen3_5_hf_compatible_model(
    hf_model: Qwen3_5ForConditionalGeneration,
    xh_model: "XHQwen3_5Model",
):
    llm_compatible_modules = _DMRegistryCls("XHCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in llm_compatible_modules:
        llm_compatible_modules.register_module({hf_model_cls: hf_model_cls.__name__}, _Qwen3_5HFCompatible)
    return llm_compatible_modules.convert(hf_model, text_llm_model=xh_model)


def _prefill_recurrent_state_uses_cache_tensor(wrap_cfg: ConfigDict, config: XHQwen3_5ModelConfig) -> bool:
    fuse_gdr_ops = bool(wrap_cfg.get("fuse_gdr_ops", getattr(config, "fuse_gdr_ops", False)))
    return fuse_gdr_ops and bool(getattr(GDRChunkScan, "state_is_cache", False))


def _enforce_split_conv_cache_wrap_cfg(llm_model: nn.Module, wrap_cfg: ConfigDict) -> None:
    """Ensure traceable TextModel/GatedDeltaNet modules see split cache mode.

    Some conversion paths pass a reduced cfg into child ``_setup`` calls even
    though the exported model signature is already flat q/k/v. Re-applying the
    split flag after wrapping keeps the per-layer trace path consistent with
    ``get_export_cfg()`` and the runtime cache mixin.
    """
    if not bool(wrap_cfg.get("split_conv_cache", False)):
        return

    if hasattr(llm_model, "split_conv_cache"):
        llm_model.split_conv_cache = True
    if hasattr(llm_model, "_setup"):
        llm_model._setup(wrap_cfg)

    for decoder_layer in getattr(llm_model, "layers", []):
        linear_attn = getattr(decoder_layer, "linear_attn", None)
        if linear_attn is None:
            continue
        if hasattr(linear_attn, "split_conv_cache"):
            linear_attn.split_conv_cache = True
        if hasattr(linear_attn, "_setup"):
            linear_attn._setup(wrap_cfg)


class Qwen3_5_ModelMeta(VLLMModelMeta):  # noqa: N801
    KVCACHE_CONFOG_CLS = KVCacheWithLinearConfig


def _ensure_gptq_desc_act_default(hf_config_dir: str | Path) -> None:
    """Keep copied GPTQ metadata loadable by runtimes requiring ``desc_act``."""
    config_path = Path(hf_config_dir) / "config.json"
    if not config_path.is_file():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        return
    if str(quantization_config.get("quant_method", "")).lower() != "gptq":
        return
    if "desc_act" in quantization_config:
        return

    # Hugging Face/GPTQ treats an omitted desc_act as the non-act-order mode,
    # while current vLLM requires the field to be explicit during validation.
    quantization_config["desc_act"] = False
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _resolve_qwen_export_pad_token_id(hf_model_dir: str | Path) -> int | None:
    """Resolve the Qwen padding id before low-memory wrapping starts.

    Qwen target models use the language-model EOS id for graph padding in
    ``_wraped_post``.  Huge-model export creates metadata before that hook, so
    reproduce the same lookup directly from the source HF config.
    """
    config_path = Path(hf_model_dir) / "config.json"
    if not config_path.is_file():
        return None

    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    config_sections = (text_config, config) if isinstance(text_config, dict) else (config,)
    for section in config_sections:
        eos_token_id = section.get("eos_token_id")
        if type(eos_token_id) is int:
            return eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            for token_id in eos_token_id:
                if type(token_id) is int:
                    return token_id
    return None


@register_llm_model("Qwen3_5ForConditionalGeneration")
class XHQwen3_5Model(VisionLLMModel):  # noqa: N801
    HF_MODEL_CLS = XHQwen3_5ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = Qwen3_5_ModelMeta
    HMONNXINFERENCE_CLS = XHQwen3_5_HMONNXModel
    CONFIG_CLS = XHQwen3_5ModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_qwen3_5_hf_compatible_model)
    transformers_min_version = "5.5.0"
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen3_5.workflow:Qwen35Workflow"

    def __init__(self, config: XHQwen3_5ModelConfig):
        super().__init__(config)

        if hasattr(config, "visual_config") and config.visual_config is not None and config.visual_config.enable:
            self.visual = XHQwen3_5VisionModel(config.visual_config)
            self.visual.config.model_name = f"{self.config.model_name}_visual"
        self.full_attention_layer_indices: list[int] = []
        self.linear_attention_layer_indices: list[int] = []
        self._kvcache_config = KVCacheWithLinearConfig()
        self._kvcache_config.use_cache = self.config.use_cache
        self._kvcache_mixin = _Qwen3_5KVCacheMixin(self.kvcache_config)
        self.config = cast(XHQwen3_5ModelConfig, self.config)
        self.wrap_cfg["linear_attention_mode"] = "auto"
        self.wrap_cfg["linear_chunk_size"] = self.config.linear_chunk_size
        self.wrap_cfg["flash_attention"] = self.config.flash_attention
        self.wrap_cfg["fuse_gdr_ops"] = self.config.fuse_gdr_ops
        self.wrap_cfg["fuse_gdr_block_recurrent_ops"] = self.config.fuse_gdr_block_recurrent_ops
        self.wrap_cfg["split_conv_cache"] = self.config.split_conv_cache
        self.wrap_cfg["use_manual_depthwise_conv1d"] = self.config.use_manual_depthwise_conv1d
        self._set_recurrent_state_output_contract(prefill=True)
        if self.config.spec_decode_mode == "dflash":
            self.wrap_cfg["output_hidden_state_indices"] = self._get_dflash_target_layer_ids()

    def create_export_metadata(self, output_dir: str) -> VLLMModelMeta:
        pad_token_id = _resolve_qwen_export_pad_token_id(self.hf_model_dir)
        if pad_token_id is not None:
            # Match _wraped_post even if tokenizer loading previously installed
            # a different tokenizer-level padding id.
            self.pad_token_id = pad_token_id
        meta_info = super().create_export_metadata(output_dir)
        _ensure_gptq_desc_act_default(Path(output_dir) / meta_info.hf_config)
        return meta_info

    def _get_dflash_target_layer_ids(self) -> list[int]:
        dflash_config = self.config.dflash_config
        if dflash_config is None:
            raise ValueError("dflash_config is required when spec_decode_mode='dflash'")

        dflash_model_dir = Path(dflash_config.dflash_model_dir)
        config_path = dflash_model_dir / "config.json"
        with open(config_path, encoding="utf-8") as f:
            target_layer_ids = json.load(f).get("dflash_config", {}).get("target_layer_ids")
        if not target_layer_ids:
            raise ValueError(f"Failed to read dflash_config.target_layer_ids from {config_path}")
        target_layer_ids = list(target_layer_ids)

        # DFlash is trained against hidden states from these exact target
        # layers.  A workflow-level override is therefore not a tuning knob:
        # selecting different layers still exports a shape-compatible graph,
        # but silently destroys draft acceptance.  Keep the checkpoint as the
        # source of truth and fail before a multi-hour export if a legacy YAML
        # duplicates the contract incorrectly.
        configured = getattr(self.config, "output_hidden_state_indices", None)
        if configured is not None and list(configured) != target_layer_ids:
            raise ValueError(
                "DFlash output_hidden_state_indices must match the assistant "
                f"checkpoint {config_path}: configured={list(configured)}, "
                f"checkpoint={target_layer_ids}"
            )
        return target_layer_ids

    @VisionLLMModel.work_dir.setter
    def work_dir(self, work_dir: str):
        self.config.work_dir = work_dir
        if hasattr(self, "visual") and self.visual is not None:
            self.visual.work_dir = str(Path(work_dir) / "visual")

    def is_support_dynamic_input(self) -> bool:
        if self._state in [
            LLMModelState.EAGER_ALIGNED,
            LLMModelState.EAGER_FAST,
        ]:
            return True
        if self._state in [
            LLMModelState.WRAP,
            LLMModelState.FRONTED,
            LLMModelState.QUANTED_ALIGNED,
            LLMModelState.QUANTED_FAST,
        ]:
            return False
        return False

    def _prefill_recurrent_state_uses_cache(self) -> bool:
        return _prefill_recurrent_state_uses_cache_tensor(self.wrap_cfg, self.config)

    def _set_recurrent_state_output_contract(self, prefill: Optional[bool] = None) -> bool:
        if prefill is None:
            prefill = self.is_prefill()
        prefill_uses_cache = self._prefill_recurrent_state_uses_cache()
        suppress_outputs = bool(prefill and prefill_uses_cache)
        self.wrap_cfg["suppress_recurrent_state_outputs"] = suppress_outputs
        # Persist the concrete export/runtime contract in metadata for HMONNX
        # runtimes. The config already records fuse_gdr_ops, but this flag also
        # captures whether the selected GDRChunkScan implementation is the new
        # CacheTensor-in-place variant.
        self.config.prefill_recurrent_state_uses_cache = prefill_uses_cache
        return suppress_outputs

    def get_dummy_inputs(self):
        data_batch = super().get_dummy_inputs()
        if hasattr(self, "visual") and self.visual is not None:
            visual_dummy_inputs = self.visual._get_dummy_inputs()
            data_batch["image_grid_thw"] = visual_dummy_inputs.get("image_grid_thw", None)
            data_batch["video_grid_thw"] = visual_dummy_inputs.get("video_grid_thw", None)
        return data_batch

    def _sync_split_conv_cache_state(self) -> None:
        split_conv_cache = bool(self.wrap_cfg.get("split_conv_cache", self.config.split_conv_cache))
        self._kvcache_mixin.split_conv_cache = split_conv_cache
        if not split_conv_cache:
            return

        language_model = self._get_language_model(self._wrap_model) if self._wrap_model is not None else None
        if language_model is not None:
            _enforce_split_conv_cache_wrap_cfg(language_model, self.wrap_cfg)

        if self.linear_attention_layer_indices and language_model is not None:
            linear_attn = language_model.layers[self.linear_attention_layer_indices[0]].linear_attn
            self._kvcache_mixin._linear_key_dim = linear_attn.key_dim
            self._kvcache_mixin._linear_value_dim = linear_attn.value_dim

        # ``kv_cache_scope`` prepares auxiliary caches before callers ask this
        # model for a data processor/export cfg.  If split_conv_cache was not
        # synchronized before that scope was entered, the cache list can already
        # contain legacy merged qkv tensors.  Rebuild only the linear-attention
        # auxiliary caches so the traced graph receives grouped q/k/v tensors
        # and the flattened export signature stays aligned.
        if (
            len(self._kvcache_mixin.past_conv_caches) > 0
            and not _is_grouped_split_conv_cache(self._kvcache_mixin.past_conv_caches)
            and self._kvcache_mixin._linear_key_dim > 0
            and self._kvcache_mixin._linear_value_dim > 0
        ):
            self._kvcache_mixin.clear_other_cache()
            self._kvcache_mixin.prepare_other_cache()

    @property
    def past_conv_caches(self):
        self._sync_split_conv_cache_state()
        return self._kvcache_mixin.past_conv_caches

    @property
    def past_recurrent_states(self):
        return self._kvcache_mixin.past_recurrent_states

    def get_data_preprocessor(self) -> BaseLLMInputProcessor:
        # self._sync_split_conv_cache_state()
        # The base class caches data processors, but Qwen3.5 export scopes
        # recreate KV/linear caches repeatedly (prefill/decode/frontend/export).
        # Recreate the lightweight processor so it captures the current cache
        # objects instead of stale merged conv-cache snapshots. Preserve
        # rope_deltas across prefill -> decode processor refreshes; decode
        # dummy/export inputs need the delta computed by the prefill pass.
        rope_deltas = getattr(self._data_processor, "rope_deltas", None)
        self._data_processor = None
        data_processor = super().get_data_preprocessor()
        if rope_deltas is not None and getattr(data_processor, "rope_deltas", None) is None:
            data_processor.rope_deltas = rope_deltas
        return data_processor

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        native_model = super().get_hf_model(hf_model_dir, quant_weight, **kwargs)
        return qwen3_5_patch(native_model)

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        native_hf_model = super().get_empty_hf_model(hf_model_dir, **kwargs)
        native_hf_model = qwen3_5_patch(native_hf_model)
        return native_hf_model

    def _get_language_model(self, hf_model: Any) -> Any:
        return hf_model.model.language_model

    def get_tf_processor(self):
        if hasattr(self, "visual") and self.visual is not None:
            return self.visual.get_tf_processor()
        else:
            return XHQwen3_5Processor.from_pretrained(self.hf_model_dir)

    def get_inference_model(self):
        inference_model = super().get_inference_model()
        if isinstance(inference_model, (list, tuple)):
            if self.is_prefill():
                return inference_model[0]
            else:
                return inference_model[1]
        else:
            return inference_model

    def set_prefill(self):
        self.wrap_cfg["linear_attention_mode"] = "chunk"
        self._set_recurrent_state_output_contract(prefill=True)
        if self._state == LLMModelState.FRONTED:
            self._frontend_model.set_activate_model("prefill")
        elif self._state in [LLMModelState.QUANTED_ALIGNED, LLMModelState.QUANTED_FAST, LLMModelState.QUANTED_DISABLE]:
            self._quanted_model.set_activate_model("prefill")
        super().set_prefill()
        self._set_recurrent_state_output_contract(prefill=True)
        self.update_cfg(self.wrap_cfg)

    def set_decode(self):
        self.wrap_cfg["linear_attention_mode"] = "recurrent"
        self._set_recurrent_state_output_contract(prefill=False)
        if self._state == LLMModelState.FRONTED:
            self._frontend_model.set_activate_model("decode")
        elif self._state in [LLMModelState.QUANTED_ALIGNED, LLMModelState.QUANTED_FAST, LLMModelState.QUANTED_DISABLE]:
            self._quanted_model.set_activate_model("decode")
        super().set_decode()
        self._set_recurrent_state_output_contract(prefill=False)
        self.update_cfg(self.wrap_cfg)

    def get_quant_cfg(self):
        quant_cfg = super().get_quant_cfg()
        quant_cfg.setdefault("ops_cfg", ConfigDict())
        quant_cfg["ops_cfg"]["Normalize"] = ConfigDict(force_fp32=self.config.normalize_force_fp32)

        cumsum_quant_cfg = self.config.cumsum_matmul_quant_config
        if cumsum_quant_cfg is None:
            cumsum_quant_cfg = dict(
                act_schema=dict(fp_mode="sefp", man_bit=16),
                act_schema_2=dict(fp_mode="sefp", man_bit=8),
            )
        quant_cfg.setdefault("nodes_cfg", ConfigDict())
        wrap_model = self._wrap_model or self.get_inference_model()
        if wrap_model is not None:
            for name, _ in wrap_model.named_modules():
                if "cumsum_matmul" not in name:
                    continue
                for key in (name, name.replace(".", "_")):
                    if key not in quant_cfg["nodes_cfg"]:
                        quant_cfg["nodes_cfg"][key] = ConfigDict(dict(cumsum_quant_cfg))
        return quant_cfg

    def _to_quanted(self, frontend_model, state):
        prefill_fronted_model = frontend_model.prefill
        self.set_prefill()
        prefill_quanted_model = super()._to_quanted(prefill_fronted_model, state, infer_shape=True)

        decode_fronted_model = frontend_model.decode
        self.set_decode()

        spec_decode_mode = self.config.spec_decode_mode
        if spec_decode_mode in ("mtp", "dflash"):
            spec_seq_len = self.config.num_draft_tokens + 1
            self.set_input_sequence_length(spec_seq_len)

        decode_quanted_model = super()._to_quanted(decode_fronted_model, state)
        if spec_decode_mode in ("mtp", "dflash"):
            self._restore_prefill_wrap_cfg()
        self.set_prefill()

        model = ModelSwitcher({"prefill": prefill_quanted_model, "decode": decode_quanted_model})
        model.set_activate_model("prefill")
        return model

    def _restore_prefill_wrap_cfg(self):
        self.wrap_cfg["verify_output_intermediates"] = False
        self.wrap_cfg["num_logits_to_keep"] = self.config.num_logits_to_keep
        self.wrap_cfg["input_sequence_length"] = self.config.prefill_chunk_length
        self._set_recurrent_state_output_contract(prefill=True)
        output_post_norm_hidden = getattr(self.config, "output_post_norm_hidden", False)
        if output_post_norm_hidden:
            self.wrap_cfg["output_post_norm_hidden"] = output_post_norm_hidden
        else:
            self.wrap_cfg.pop("output_post_norm_hidden", None)

    def _to_fronted(self, wrap_model):
        # 将模型转换成前端图，准备进行量化
        self.set_prefill()
        prefill_wrap_model = wrap_model
        decode_wrap_model = wrap_model
        logger = get_xhquant_logger()
        memory_info = get_cpu_memory_mb()
        logger.info(f"Initial CPU memory usage 1: {str(memory_info)}")
        decode_wrap_model = _copy_model_shared_params(wrap_model)
        self._wrap_model = prefill_wrap_model
        self._sync_split_conv_cache_state()

        memory_info = get_cpu_memory_mb()
        logger.info(f"Initial CPU memory usage 2: {str(memory_info)}")
        prefill_frontend_model = super()._to_fronted(prefill_wrap_model)

        memory_info = get_cpu_memory_mb()
        logger.info(f"Initial CPU memory usage 3: {str(memory_info)}")
        self._wrap_model = decode_wrap_model
        self.set_decode()
        self._sync_split_conv_cache_state()

        spec_decode_mode = self.config.spec_decode_mode
        if spec_decode_mode in ("mtp", "dflash"):
            spec_seq_len = self.config.num_draft_tokens + 1
            self.wrap_cfg["verify_output_intermediates"] = True
            self.wrap_cfg["num_logits_to_keep"] = 0
            if spec_decode_mode == "mtp":
                self.wrap_cfg["output_post_norm_hidden"] = True
            self.set_input_sequence_length(spec_seq_len)

            def _apply_spec_flags(module):
                if hasattr(module, "num_logits_to_keep"):
                    module.num_logits_to_keep = 0
                if hasattr(module, "output_post_norm_hidden") and spec_decode_mode == "mtp":
                    module.output_post_norm_hidden = True

            decode_wrap_model.apply(_apply_spec_flags)

        decode_frontend_model = super()._to_fronted(decode_wrap_model)

        if spec_decode_mode in ("mtp", "dflash"):
            self._restore_prefill_wrap_cfg()

        self._wrap_model = prefill_wrap_model
        self.set_prefill()
        memory_info = get_cpu_memory_mb()
        logger.info(f"Initial CPU memory usage 4: {str(memory_info)}")
        name2modules = {}
        reuse_types = (nn.Linear, nn.Conv2d, nn.Conv2d, xhnn.MoeBlock)
        for name, module in prefill_frontend_model.named_modules():
            if isinstance(module, reuse_types):
                name2modules[name] = module

        for name, module in decode_frontend_model.named_modules():
            if isinstance(module, reuse_types):
                if name in name2modules:
                    decode_frontend_model.set_submodule(name, name2modules[name])

        fronted_model = ModelSwitcher({"prefill": prefill_frontend_model, "decode": decode_frontend_model})
        fronted_model.set_activate_model("prefill")

        return fronted_model

    def _wraped_pre(self, hf_model: XHQwen3_5ForConditionalGeneration):
        super()._wraped_pre(hf_model)
        llm_model = self._get_language_model(hf_model)
        text_config = llm_model.config
        self.layer_types = list(text_config.layer_types)

        if self.config.only_first_block:
            self.linear_attention_layer_indices = []
            for idx, layer_type in enumerate(self.layer_types):
                if layer_type == "full_attention":
                    self.full_attention_layer_indices = [idx]
                    break
                else:
                    self.linear_attention_layer_indices.append(idx)
            self.config.max_layers = len(self.full_attention_layer_indices) + len(self.linear_attention_layer_indices)
            self.config.only_first_block = False
        else:
            self.full_attention_layer_indices = [
                idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"
            ]
            self.linear_attention_layer_indices = [
                idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"
            ]
        del hf_model.model.visual

        logger = get_xhquant_logger()
        language_count, language_bytes = get_model_param_buffer_size_gb(hf_model)

        logger.info(f"_wraped_pre model parameters: {language_count} B ({language_bytes:.2f} GB)")
        memory_info = get_cpu_memory_mb()
        logger.info(f"CPU memory usage _wraped_pre: {str(memory_info)}")
        return hf_model

    def _wraped_post(self, hf_model: XHQwen3_5ForConditionalGeneration):
        self.config.image_token_id = hf_model.config.image_token_id
        self.config.video_token_id = hf_model.config.video_token_id
        self.config.vision_start_token_id = hf_model.config.vision_start_token_id

        self.config.vision_end_token_id = hf_model.config.vision_end_token_id
        self.config.spatial_merge_size = hf_model.config.vision_config.spatial_merge_size

        hf_model = self._wrap_model
        llm_model = self._get_language_model(hf_model)
        _enforce_split_conv_cache_wrap_cfg(llm_model, self.wrap_cfg)
        # self.embed_tokens.weight 和 lm_head.weight 可能是相同对象
        self.embed_tokens = copy.deepcopy(llm_model.get_input_embeddings())
        text_config = llm_model.config
        self.pad_token_id = llm_model.config.eos_token_id
        self.layer_types = list(text_config.layer_types)

        self_attn = llm_model.layers[self.full_attention_layer_indices[0]].self_attn
        linear_attn = llm_model.layers[self.linear_attention_layer_indices[0]].linear_attn

        linear_kv_cache_config = self.kvcache_config.linear_kv_cache_config
        linear_kv_cache_config.conv_dim = linear_attn.conv_dim
        linear_kv_cache_config.conv_kernel_size = linear_attn.conv_kernel_size
        linear_kv_cache_config.num_v_heads = linear_attn.num_v_heads
        linear_kv_cache_config.head_k_dim = linear_attn.head_k_dim
        linear_kv_cache_config.head_v_dim = linear_attn.head_v_dim
        linear_kv_cache_config.num_layers = len(self.linear_attention_layer_indices)

        split_conv_cache = self.wrap_cfg.get("split_conv_cache", False)
        self._kvcache_mixin.split_conv_cache = split_conv_cache
        if split_conv_cache:
            self._kvcache_mixin._linear_key_dim = linear_attn.key_dim
            self._kvcache_mixin._linear_value_dim = linear_attn.value_dim

        if self.use_cache:
            num_decoder_layers = len(self.full_attention_layer_indices)
            head_dim = self_attn.head_dim

            self.kvcache_config.num_layers = num_decoder_layers
            self.kvcache_config.kv_cache_shape = [
                1,
                text_config.num_key_value_heads,
                self.config.context_max_length,
                head_dim,
            ]
        fp32_tensors = {}
        for name, param in self._wrap_model.named_parameters():
            if param.dtype in [torch.float32]:
                fp32_tensors[name] = param
        for name, buffer in self._wrap_model.named_buffers():
            if buffer.dtype in [torch.float32]:
                fp32_tensors[name] = buffer

        logger = get_xhquant_logger()
        for name, tensor in fp32_tensors.items():
            logger.info(
                f"Tensor {name} is {tensor.dtype}, consider converting it to lower precision for better performance."
            )
        language_count, language_bytes = get_model_param_buffer_size_gb(self._wrap_model)

        logger.info(f"_wraped_post model parameters: {language_count} B ({language_bytes:.2f} GB)")

        memory_info = get_cpu_memory_mb()
        logger.info(f"CPU memory usage _wraped_post: {str(memory_info)}")

        hf_model = None

    def init_wrap_model(self, hf_model: XHQwen3_5ForConditionalGeneration) -> Any:
        register_wrap_modules()
        wrap_model = super().init_wrap_model(hf_model)
        return wrap_model

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        logits = result[0]
        conv_cache_out_list = result[1]
        recurrent_state_out_list = result[2]
        spec_decode_hidden = result[3] if len(result) > 3 else None

        verify_output_intermediates = bool(self.wrap_cfg.get("verify_output_intermediates", False))
        if not verify_output_intermediates:
            past_conv_caches = self._kvcache_mixin.past_conv_caches
            past_recurrent_states = self._kvcache_mixin.past_recurrent_states
            if self._kvcache_mixin.split_conv_cache:
                grouped_conv_cache_out_list = _regroup_flat_split_conv_cache(conv_cache_out_list)
                for (pq, pk, pv), (oq, ok, ov) in zip(past_conv_caches, grouped_conv_cache_out_list, strict=True):
                    pq[:] = oq[:]
                    pk[:] = ok[:]
                    pv[:] = ov[:]
            else:
                for past_conv_cache, conv_cache_out in zip(past_conv_caches, conv_cache_out_list, strict=True):
                    past_conv_cache[:] = conv_cache_out[:]
            if recurrent_state_out_list:
                for past_recurrent_state, recurrent_state_out in zip(
                    past_recurrent_states, recurrent_state_out_list, strict=True
                ):
                    past_recurrent_state[:] = recurrent_state_out[:]
            elif past_recurrent_states and not self._prefill_recurrent_state_uses_cache():
                raise RuntimeError("Missing recurrent state outputs for non-CacheTensor GDR prefill/decode path.")

        if spec_decode_hidden is not None:
            return logits, conv_cache_out_list, recurrent_state_out_list, spec_decode_hidden
        return logits, conv_cache_out_list, recurrent_state_out_list

    def _set_device(self, device):
        super()._set_device(device)
        if hasattr(self, "visual") and self.visual is not None:
            self.visual._set_device(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        if hasattr(self, "visual") and self.visual is not None:
            self.visual._set_dtype(dtype)
        return self

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        # self._sync_split_conv_cache_state()
        data_preprocess = Qwen3_5_DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            image_size_w=self.config.visual_config.max_size_w,
            image_size_h=self.config.visual_config.max_size_h,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            patch_size=self.config.visual_config.patch_size,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            spatial_merge_size=self.config.spatial_merge_size,
        )
        return data_preprocess

    def get_export_cfg(self) -> dict[str, list[str]]:
        self._sync_split_conv_cache_state()
        export_cfg = {
            "input_names": [
                "inputs_embeds",
                "time_position_ids",
                "hight_position_ids",
                "width_position_ids",
                "past_seq_length",
                "current_input_length",
                "linear_attn_mask",
            ],
            "output_names": ["logits"],
        }
        for layer_idx in range(self.kvcache_config.num_layers):
            export_cfg["input_names"].append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(self.kvcache_config.num_layers):
            export_cfg["input_names"].append(f"past_value_cache_{layer_idx}")

        linear_num_layers = self.kvcache_config.linear_kv_cache_config.num_layers
        split_conv_cache = bool(self.wrap_cfg.get("split_conv_cache", self.config.split_conv_cache))
        self._kvcache_mixin.split_conv_cache = split_conv_cache
        suppress_recurrent_state_outputs = self._set_recurrent_state_output_contract()

        if split_conv_cache:
            for cache_idx in range(linear_num_layers):
                for branch in ("q", "k", "v"):
                    export_cfg["input_names"].append(f"past_conv_cache_{branch}_{cache_idx}")
        else:
            for cache_idx in range(linear_num_layers):
                export_cfg["input_names"].append(f"past_conv_cache_{cache_idx}")
        for cache_idx in range(linear_num_layers):
            export_cfg["input_names"].append(f"past_recurrent_state_{cache_idx}")

        verify_steps = 1
        if (
            bool(self.wrap_cfg.get("verify_output_intermediates", False))
            and int(self.wrap_cfg.get("input_sequence_length", 1)) > 1
        ):
            verify_steps = int(self.wrap_cfg.input_sequence_length)

        if verify_steps > 1:
            if split_conv_cache:
                for cache_idx in range(linear_num_layers):
                    for branch in ("q", "k", "v"):
                        for step_idx in range(verify_steps):
                            export_cfg["output_names"].append(f"conv_cache_out_{branch}_{cache_idx}_{step_idx}")
            else:
                for cache_idx in range(linear_num_layers):
                    for step_idx in range(verify_steps):
                        export_cfg["output_names"].append(f"conv_cache_out_{cache_idx}_{step_idx}")
            if not suppress_recurrent_state_outputs:
                for cache_idx in range(linear_num_layers):
                    for step_idx in range(verify_steps):
                        export_cfg["output_names"].append(f"recurrent_state_out_{cache_idx}_{step_idx}")
        else:
            if split_conv_cache:
                for cache_idx in range(linear_num_layers):
                    for branch in ("q", "k", "v"):
                        export_cfg["output_names"].append(f"conv_cache_out_{branch}_{cache_idx}")
            else:
                for cache_idx in range(linear_num_layers):
                    export_cfg["output_names"].append(f"conv_cache_out_{cache_idx}")
            if not suppress_recurrent_state_outputs:
                for cache_idx in range(linear_num_layers):
                    export_cfg["output_names"].append(f"recurrent_state_out_{cache_idx}")

        if self.wrap_cfg.get("output_hidden_state_indices") is not None:
            export_cfg["output_names"].append("target_hidden")
        elif self.wrap_cfg.get("output_post_norm_hidden", False):
            export_cfg["output_names"].append("post_norm_hidden")

        return export_cfg

    def export_llm_hmonnx(self, output_dir):
        super().export_hmonnx(output_dir)

    def export_visual_hmonnx(self, output_dir):
        assert hasattr(self, "visual") and self.visual is not None, (
            "Visual model is not initialized, cannot export visual hmonnx."
        )
        self.visual.export_hmonnx(output_dir)

    def get_export_info(self, output_dir) -> ExportData:
        str_datetime = datetime.now().strftime("%Y%m%d")
        model_name = self.config.model_name.lower()
        if model_name is None or len(model_name) == 0:
            raise ValueError("Model name is not specified in config, please set model_name in config before exporting.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if getattr(self, "visual", None) is not None:
            image_size_h = self.visual.config.max_size_h
            image_size_w = self.visual.config.max_size_w
            model_name = f"hmquant_{model_name}_{image_size_w}x{image_size_h}_{str_datetime}"
        else:
            model_name = f"hmquant_{model_name}_{str_datetime}"
        output_dir = Path(output_dir) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_info = self.create_export_metadata(output_dir)
        export_data = ExportData()
        export_data.exported_dir = str(output_dir)
        export_data.meta = meta_info
        export_data.model_name = model_name
        export_data.str_datetime = str_datetime
        return export_data

    def _quantize_wrap_variant(
        self,
        wrap_model: nn.Module,
        adapter: LoRAAdapterSpec | None = None,
    ) -> ModelSwitcher:
        """Trace and quantize one base/LoRA target pair without reloading HF weights."""

        self._state = LLMModelState.WRAP
        self._wrap_model = wrap_model
        # The previous target export finishes in decode mode and may leave
        # spec-decode-only values in wrap_cfg.  Each structural variant must
        # trace its prefill graph from the canonical prefill configuration.
        self._restore_prefill_wrap_cfg()
        if adapter is not None:
            attach_lora_buffers(wrap_model, adapter)

        frontend_model = self._to_fronted(wrap_model)
        self._frontend_model = frontend_model
        self.release_wraped_model()
        self._state = LLMModelState.FRONTED

        if adapter is not None:
            lora_config = self.config.lora
            assert lora_config is not None
            apply_lora_to_frontend(frontend_model, adapter, lora_config.w_schema)

        quanted_model = self._to_quanted(frontend_model, LLMModelState.QUANTED_ALIGNED)
        self._quanted_model = quanted_model
        del self._frontend_model
        self._frontend_model = None
        self._state = LLMModelState.QUANTED_ALIGNED
        quanted_model.prefill.fixed()
        quanted_model.decode.fixed()
        return quanted_model

    def _configure_spec_decode_target_export(self) -> None:
        spec_decode_mode = self.config.spec_decode_mode
        if spec_decode_mode not in ("mtp", "dflash"):
            return

        self._decode_input_sequence_length = self.config.num_draft_tokens + 1
        self._decode_wrap_cfg_overrides = {
            "verify_output_intermediates": True,
        }
        self.wrap_cfg["num_logits_to_keep"] = 0
        if spec_decode_mode == "mtp":
            self.wrap_cfg["output_post_norm_hidden"] = True

        def _apply_spec_decode_flags(module):
            if hasattr(module, "num_logits_to_keep"):
                module.num_logits_to_keep = 0
            if hasattr(module, "output_post_norm_hidden") and spec_decode_mode == "mtp":
                module.output_post_norm_hidden = True

        self._quanted_model.prefill.apply(_apply_spec_decode_flags)
        self._quanted_model.decode.apply(_apply_spec_decode_flags)

    def _export_lora_variants(
        self,
        exported_info: ExportData,
        wrap_template: nn.Module,
        adapters: list[LoRAAdapterSpec],
    ) -> list[tuple[LoRAAdapterSpec, ExportData]]:
        logger = get_xhquant_logger()
        exported_adapters: list[tuple[LoRAAdapterSpec, ExportData]] = []
        root_dir = Path(exported_info.exported_dir)

        # The root/base quant graph has already been exported and can be
        # released before processing the first adapter.
        del self._quanted_model
        self._quanted_model = None
        gc.collect()

        for adapter in adapters:
            logger.info(
                f"Start exporting Qwen3.5 LoRA adapter {adapter.name!r} ({len(adapter.pairs)} target Linear modules)"
            )
            adapter_wrap_model = _copy_model_shared_params(wrap_template)
            self._quantize_wrap_variant(adapter_wrap_model, adapter)
            self._configure_spec_decode_target_export()

            adapter_dir = root_dir / "lora" / adapter.name
            adapter_dir.mkdir(parents=True, exist_ok=False)
            adapter_meta = copy.deepcopy(exported_info.meta)
            adapter_export = ExportData()
            adapter_export.exported_dir = str(adapter_dir)
            adapter_export.meta = adapter_meta
            adapter_export.model_name = f"{exported_info.model_name}_{adapter.name}"
            adapter_export.str_datetime = exported_info.str_datetime
            self._export_hmonnx(adapter_export)
            exported_adapters.append((adapter, adapter_export))

            del self._quanted_model
            self._quanted_model = None
            gc.collect()
            logger.info(f"Finished exporting Qwen3.5 LoRA adapter {adapter.name!r} to {adapter_dir}")

        return exported_adapters

    def _export_visual_hmonnx_impl(self, exported_info: ExportData) -> None:
        """导出视觉子模型并将相对路径写入统一的 VLLM 元数据。"""
        if getattr(self, "visual", None) is None:
            return

        meta_info = cast(VLLMModelMeta, exported_info.meta)
        visual_output_dir = str(Path(exported_info.exported_dir) / "visual")
        logger = get_xhquant_logger()
        memory_info = get_cpu_memory_mb()
        logger.info(f"Initial CPU memory usage before exporting visual hmonnx: {str(memory_info)}")
        logger.info(f"Start exporting visual hmonnx to {visual_output_dir}")

        self.visual.config.model_name = f"{exported_info.model_name}_visual"
        self.visual.to_quanted_aligned()
        visual_meta = self.visual.export_hmonnx(visual_output_dir)
        visual_meta.hmonnx = str(Path(visual_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix())
        if getattr(visual_meta, "gears", None):
            for gear in visual_meta.gears:
                gear["hmonnx"] = str(
                    (Path(visual_output_dir) / gear["hmonnx"]).relative_to(exported_info.exported_dir).as_posix()
                )
        if getattr(visual_meta, "gear_manifest", None):
            visual_meta.gear_manifest = str(
                Path(visual_meta.gear_manifest).relative_to(exported_info.exported_dir).as_posix()
            )
        meta_info.visual_config = visual_meta
        json.dump(
            meta_info.to_dict(),
            open(str(Path(exported_info.exported_dir) / "golden_meta_info.json"), "w"),
            indent=4,
        )

        memory_info = get_cpu_memory_mb()
        logger.info(f"Initial CPU memory usage after exporting visual hmonnx: {str(memory_info)}")

    def _get_big_language_placeholder_export_components(self):
        from ._qwen3_5_big_export import Qwen3_5BigHFModel

        return Qwen3_5BigHFModel, Qwen3_5BigHFModel.PLACEHOLDER_TYPES

    def _check_big_language_placeholder_export_supported(self, empty_hf_model: Any) -> None:
        hf_model_type = str(getattr(empty_hf_model.config, "model_type", "")).lower()
        if "moe" in hf_model_type or "moe" in type(empty_hf_model).__name__.lower():
            raise NotImplementedError(
                "Qwen3.5 MoE big-model placeholder export must use the MoE-specific placeholder components."
            )

        language_model = self._get_language_model(empty_hf_model)
        if not hasattr(language_model, "modules"):
            raise NotImplementedError("Qwen3.5 big-model placeholder export requires an nn.Module language model.")

        found_types = {type(module).__name__ for module in language_model.modules()}

        _, placeholder_types = self._get_big_language_placeholder_export_components()
        missing_types = sorted(set(placeholder_types) - found_types)
        if missing_types:
            raise NotImplementedError(
                "Qwen3.5 big-model placeholder export requires placeholder modules "
                f"{placeholder_types}, but missing {missing_types}."
            )

    def _export_big_language_hmonnx(self, exported_info: ExportData) -> VLLMModelMeta:
        """使用 placeholder 分层导出 Qwen3.5 文本模型，降低峰值内存占用。

        主图仅保留 attention、GatedDeltaNet、MLP/MoE 等大模块的输入输出契约；
        每个 placeholder 模块随后按需从 safetensors 加载权重，并分别导出 prefill
        和 decode 子图。最后将主图中的 placeholder 节点替换为对应子图。

        Args:
            output_dir: 导出产物的根目录。

        Returns:
            包含 prefill/decode HMONNX 路径及模型配置的导出元数据。

        Note:
            该流程会推进当前模型的 wrap、frontend、quant 和 export 状态，并移除
            ``self.visual``；视觉模型不在此文本大模型分层导出流程中处理。
        """
        from ...wrap_model import traceable_module_placeholder_context

        logger = get_xhquant_logger()
        big_hf_model_cls, placeholder_types = self._get_big_language_placeholder_export_components()
        # 主图模型保持 placeholder 子树为 meta tensor，仅加载 embedding、norm、
        # lm_head 等非 placeholder 权重，避免一次性物化完整语言模型。
        empty_hf_model = self.get_empty_hf_model(self.hf_model_dir)
        empty_hf_model.model.visual = None
        self._check_big_language_placeholder_export_supported(empty_hf_model)
        main_placeholder_prefixes = big_hf_model_cls.resolve_placeholder_prefixes(
            empty_hf_model,
            placeholder_types,
        )

        # 子图模型专用于逐模块加载真实权重。必须与主图模型分离，避免子图导出时的
        # 反量化、wrap 和释放操作污染仍在构建中的主图模型。
        empty_hf_model_for_placeholder = copy.deepcopy(empty_hf_model)
        # 将nn.Linear替换成量化版本的Linear
        big_hf_model_cls._preprocess_quantized_hf_model(
            empty_hf_model_for_placeholder,
            self.hf_model_dir,
        )
        big_hf_model_cls._preprocess_quantized_hf_model(
            empty_hf_model,
            self.hf_model_dir,
            skip_module_prefixes=main_placeholder_prefixes,
        )

        big_hf_model = big_hf_model_cls(self.hf_model_dir, empty_hf_model, placeholder_types)
        big_hf_model.replace_runtime_placeholder_modules(empty_hf_model)

        # 首次注册原生 HF 模块类型，使 TorchFX 在 wrap 阶段将其视为叶子节点，
        # 防止 tracing 提前展开大模块内部计算。
        big_hf_model.register_layer_as_placeholder(empty_hf_model)
        placeholder_callback = partial(big_hf_model_cls.register_placeholder, hf_model=empty_hf_model)
        with traceable_module_placeholder_context(placeholder_types, callback=placeholder_callback):
            self.to_wrap(empty_hf_model)

        # wrap 会把原生模块转换为 DynamicModule/XH 模块，因此需要再次解析并注册
        # 转换后的实际类型，确保 frontend tracing 仍保留相同的模块边界。
        big_hf_model.strip_unwrapped_placeholder_members(empty_hf_model)
        big_hf_model.register_layer_as_placeholder(empty_hf_model)
        self.to_fronted()

        # 先创建统一导出目录和元数据；placeholder 子图分别存放在 prefill/decode
        # 主图目录下，文件名由完整 module target 唯一确定。

        exported_dir = exported_info.exported_dir
        # 视觉模型与文本 PlaceHolder 主/子图相互独立，优先导出并释放视觉权重，
        # 后续大模型文本分层导出即可保持较低的峰值内存。

        target_device = self.config.chip_arch
        quant_cfg = self.get_quant_cfg()
        prefill_placeholder_exported_dir = str(Path(exported_dir) / "prefill" / "placeholders")

        # prefill 与 decode 的输入 shape、线性注意力模式和缓存契约不同，需在各自
        # 模式下保存 wrap 配置，并将两张前端图中的目标模块替换为 PlaceHolderModule。
        self.set_prefill()
        prefill_wrap_cfg = copy.deepcopy(self.get_wrap_cfg())
        big_hf_model.register_layer_as_place_holder(self._frontend_model.prefill)
        self.set_decode()
        decode_wrap_cfg = copy.deepcopy(self.get_wrap_cfg())
        big_hf_model.register_layer_as_place_holder(self._frontend_model.decode)
        decode_placeholder_exported_dir = str(Path(exported_dir) / "decode" / "placeholders")

        # 同一 module target 的真实权重只加载一次，再分别按 prefill/decode 配置导出；
        # 每个模块完成后立即释放，控制 CPU/GPU 峰值内存。外层 MemoryTracker 会递归
        # 采样这里创建的所有 placeholder 导出子进程。
        big_hf_model.export_prefill_decode_placeholder_layers(
            self._frontend_model.prefill,
            self._frontend_model.decode,
            empty_hf_model_for_placeholder,
            target_device,
            prefill_wrap_cfg,
            decode_wrap_cfg,
            quant_cfg,
            prefill_placeholder_exported_dir,
            decode_placeholder_exported_dir,
            empty_hf_model_factory=type(self).get_empty_hf_model,
        )

        # 导出包含 PlaceHolder 节点的整体 prefill/decode 主图。主图量化阶段不会再次
        # 展开已经独立导出的模块，因此无需同时持有所有层的真实权重。
        self.set_prefill()
        self._export_language_hmonnx_impl(exported_info)
        export_meta = cast(VLLMModelMeta, exported_info.meta)
        exported_dir = exported_info.exported_dir

        # 根据 PlaceHolder 节点的 content（完整 module target）定位对应子图，并回填
        # prefill/decode 主图，形成不再依赖 placeholder 自定义算子的最终 HMONNX。
        prefill_hmonnx_file = str(Path(exported_dir) / export_meta.prefill_hmonnx)
        decode_hmonnx_file = str(Path(exported_dir) / export_meta.decode_hmonnx)
        big_hf_model.replace_hmonnx_placeholders_with_subgraphs(prefill_hmonnx_file)
        big_hf_model.replace_hmonnx_placeholders_with_subgraphs(decode_hmonnx_file)
        export_meta.prefill_hmonnx_md5 = calculate_file_md5(prefill_hmonnx_file)
        export_meta.decode_hmonnx_md5 = calculate_file_md5(decode_hmonnx_file)
        json.dump(
            export_meta.to_dict(),
            open(str(Path(exported_dir) / "golden_meta_info.json"), "w"),
            indent=4,
        )
        logger.info(f"Exporting completed! Exported model is saved at: {exported_info.exported_dir}")
        return export_meta

    @log_function_call()
    def export_hmonnx(self, output_dir: str) -> VLLMModelMeta:
        self.work_dir = str(output_dir)

        # Inspect every adapter before loading/tracing the base model. This
        # rejects unsupported PEFT features and visual/VIT tensors before any
        # partial LoRA artifacts are created.
        lora_adapters = inspect_lora_adapters(getattr(self.config, "lora", None))

        exported_info = self.get_export_info(output_dir)
        self._export_visual_hmonnx_impl(exported_info)
        if getattr(self, "visual", None) is not None:
            del self.visual
            self._trim_cpu_allocator()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            memory_info = get_cpu_memory_mb()
            get_xhquant_logger().info(f"CPU memory usage after releasing visual model: {str(memory_info)}")

        if is_huge_model_export_enabled():
            if lora_adapters:
                raise NotImplementedError("Qwen3.5 LoRA export is not supported with big-model placeholder export.")
            meta_info = self._export_big_language_hmonnx(exported_info)
        else:
            self.to_wrap()
            meta_info = self._export_language_hmonnx_impl(exported_info, lora_adapters=lora_adapters)
        return meta_info

    def _export_language_hmonnx_impl(
        self,
        exported_info: ExportData,
        lora_adapters: Optional[list[LoRAAdapterSpec]] = None,
    ) -> VLLMModelMeta:
        logger = get_xhquant_logger()
        lora_adapters = lora_adapters or []
        meta_info = exported_info.meta
        meta_info = cast(VLLMModelMeta, meta_info)
        assert isinstance(meta_info, VLLMModelMeta), f"meta_info expected VLLMModelMeta, but get {type(meta_info)}"

        wrap_template = None
        if lora_adapters:
            if self._wrap_model is None:
                raise RuntimeError("Qwen3.5 LoRA export requires a wrapped base model template.")
            # Keep one structural template with shared base tensors. Base and
            # adapter frontends are traced from independent structural clones,
            # so applying one adapter cannot mutate another graph.
            wrap_template = self._wrap_model
            base_wrap_model = _copy_model_shared_params(wrap_template)
            self._quantize_wrap_variant(base_wrap_model)
        elif self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.prefill.fixed()
        self._quanted_model.decode.fixed()
        memory_info = get_cpu_memory_mb()
        logger.info(f"Initial CPU memory usage after quantization: {str(memory_info)}")

        self.config.model_name = exported_info.model_name

        spec_decode_mode = self.config.spec_decode_mode
        self._configure_spec_decode_target_export()

        self._export_hmonnx(exported_info)

        exported_lora_adapters: list[tuple[LoRAAdapterSpec, ExportData]] = []
        if lora_adapters:
            assert wrap_template is not None
            exported_lora_adapters = self._export_lora_variants(
                exported_info,
                wrap_template,
                lora_adapters,
            )
            del wrap_template
            gc.collect()

        # 导出 draft 模型 (MTP / DFlash)
        spec_decode_mode = self.config.spec_decode_mode
        if spec_decode_mode == "mtp" and self.config.mtp_config is not None:
            from .qwen3_5_mtp_model import XHQwen3_5MTPDraftModel

            mtp_base_cfg = self.config.mtp_config

            mtp_prefill_cfg = copy.deepcopy(mtp_base_cfg)
            mtp_prefill_cfg.model_name = f"{exported_info.model_name}_mtp_draft_prefill"
            mtp_prefill_cfg.input_sequence_length = self.config.prefill_chunk_length
            mtp_prefill_cfg.work_dir = str(Path(exported_info.exported_dir) / "mtp_draft_prefill")
            mtp_prefill_model = XHQwen3_5MTPDraftModel(mtp_prefill_cfg)
            mtp_prefill_model.to_quanted_aligned()
            mtp_prefill_meta = mtp_prefill_model.export_hmonnx(mtp_prefill_cfg.work_dir)
            mtp_prefill_meta.hmonnx = str(
                Path(mtp_prefill_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix()
            )

            mtp_decode_cfg = copy.deepcopy(mtp_base_cfg)
            mtp_decode_cfg.model_name = f"{exported_info.model_name}_mtp_draft_decode"
            mtp_decode_cfg.input_sequence_length = 1
            mtp_decode_cfg.work_dir = str(Path(exported_info.exported_dir) / "mtp_draft_decode")
            mtp_decode_model = XHQwen3_5MTPDraftModel(mtp_decode_cfg)
            mtp_decode_model.to_quanted_aligned()
            mtp_decode_meta = mtp_decode_model.export_hmonnx(mtp_decode_cfg.work_dir)
            mtp_decode_meta.hmonnx = str(
                Path(mtp_decode_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix()
            )

            meta_info.mtp_prefill_config = mtp_prefill_meta
            meta_info.mtp_decode_config = mtp_decode_meta
            logger.info(f"MTP draft models exported to: {mtp_prefill_cfg.work_dir}, {mtp_decode_cfg.work_dir}")

        elif spec_decode_mode == "dflash" and self.config.dflash_config is not None:
            from .qwen3_5_dflash_model import XHQwen3_5DFlashDraftModel

            dflash_base_cfg = self.config.dflash_config
            dflash_verify_seq_len = self.config.num_draft_tokens + 1

            # context mode
            ctx_cfg = copy.deepcopy(dflash_base_cfg)
            ctx_cfg.model_name = f"{exported_info.model_name}_dflash_draft_context"
            ctx_cfg.mode = "context"
            ctx_cfg.work_dir = str(Path(exported_info.exported_dir) / "dflash_draft_context")
            ctx_model = XHQwen3_5DFlashDraftModel(ctx_cfg)
            ctx_model.to_quanted_aligned()
            ctx_meta = ctx_model.export_hmonnx(ctx_cfg.work_dir)
            ctx_meta.hmonnx = str(Path(ctx_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix())

            # context_decode mode uses context inputs over the verify length.
            ctx_dec_cfg = copy.deepcopy(dflash_base_cfg)
            ctx_dec_cfg.model_name = f"{exported_info.model_name}_dflash_draft_context_decode"
            ctx_dec_cfg.mode = "context"
            ctx_dec_cfg.input_sequence_length = dflash_verify_seq_len
            ctx_dec_cfg.work_dir = str(Path(exported_info.exported_dir) / "dflash_draft_context_decode")
            ctx_dec_model = XHQwen3_5DFlashDraftModel(ctx_dec_cfg)
            ctx_dec_model.to_quanted_aligned()
            ctx_dec_meta = ctx_dec_model.export_hmonnx(ctx_dec_cfg.work_dir)
            ctx_dec_meta.hmonnx = str(Path(ctx_dec_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix())

            # decode mode also uses the verify length for draft-token generation.
            dec_cfg = copy.deepcopy(dflash_base_cfg)
            dec_cfg.model_name = f"{exported_info.model_name}_dflash_draft_decode"
            dec_cfg.mode = "decode"
            dec_cfg.input_sequence_length = dflash_verify_seq_len
            dec_cfg.work_dir = str(Path(exported_info.exported_dir) / "dflash_draft_decode")
            dec_model = XHQwen3_5DFlashDraftModel(dec_cfg)
            dec_model.to_quanted_aligned()
            dec_meta = dec_model.export_hmonnx(dec_cfg.work_dir)
            dec_meta.hmonnx = str(Path(dec_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix())

            meta_info.dflash_context_config = ctx_meta
            meta_info.dflash_context_decode_config = ctx_dec_meta
            meta_info.dflash_decode_config = dec_meta
            logger.info(
                f"DFlash draft models exported to: {ctx_cfg.work_dir}, {ctx_dec_cfg.work_dir}, {dec_cfg.work_dir}"
            )

        # 兼容 spec_decode test/bench 脚本的 meta 格式
        meta_info.prefill_onnx = meta_info.prefill_hmonnx
        meta_info.decode_onnx = meta_info.decode_hmonnx
        meta_info.token_embedding_file = meta_info.quant_embedding
        meta_info.max_context_tokens = self.config.context_max_length
        if spec_decode_mode in ("mtp", "dflash"):
            spec_decode_section = build_qwen35_spec_decode_contract(
                self.config,
                meta_info,
            )
            meta_info.spec_decode_draft_head_weight_bits = spec_decode_section["draft_head_weight_bits"]
            meta_info.spec_decode = spec_decode_section

        if exported_lora_adapters:
            lora_config = self.config.lora
            assert lora_config is not None
            finalize_lora_metadata(meta_info, exported_info, exported_lora_adapters, lora_config)

        json.dump(
            meta_info.to_dict(), open(str(Path(exported_info.exported_dir) / "golden_meta_info.json"), "w"), indent=4
        )
        logger.info(f"Exporting completed! Exported model is saved at: {exported_info.exported_dir}")
        return meta_info
