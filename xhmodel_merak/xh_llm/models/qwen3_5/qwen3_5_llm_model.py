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
from pathlib import Path
from re import I
from typing import Any, Optional, Union, cast

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoModelForImageTextToText
from transformers.cache_utils import Cache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5CausalLMOutputWithPast

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
from ...types import ExportData, KVCacheWithLinearConfig, LLMModelState, ModelSwitcher, VLLMModelMeta
from ...utils import get_cpu_memory_mb
from ...vision_llm_model import VisionLLMModel
from .data_preprocess import Qwen3_5_DataPreprocess
from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as XHQwen3_5ForConditionalGeneration
from .modeling_qwen3_5_patch import qwen3_5_patch
from .qwen3_5_hmonnx_inference import XHQwen3_5_HMONNXModel
from .split_conv_cache_utils import (
    _flatten_split_conv_cache_outputs,
    _is_grouped_split_conv_cache,
    _regroup_flat_split_conv_cache,
)
from .xh_qwen3_5_config import XHQwen3_5ModelConfig


try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    no_init_weights = init_empty_weights


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
            self.past_conv_caches.append((conv_cache_q, conv_cache_k, conv_cache_v))
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
            image_embeds = list()
            for i in range(len(pixel_values)):
                image_embeds_i = self._llm_model.visual.forward(
                    pixel_values[i].type(self._llm_model.visual.dtype).to(self._llm_model.visual.device),
                )
                image_embeds.append(image_embeds_i)

            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_embeds = image_embeds.squeeze(0)

        # seq_length = inputs_embeds.shape[1]
        data_processor = self._llm_model.get_data_preprocessor()
        # net_input_seq_len = self._llm_model.get_input_sequence_length()
        # steps = (seq_length + net_input_seq_len - 1) // net_input_seq_len

        data_batch = {
            "input_ids": input_ids,
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


@register_llm_model("Qwen3_5ForConditionalGeneration")
class XHQwen3_5Model(VisionLLMModel):  # noqa: N801
    HF_MODEL_CLS = XHQwen3_5ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = Qwen3_5_ModelMeta
    HMONNXINFERENCE_CLS = XHQwen3_5_HMONNXModel
    CONFIG_CLS = XHQwen3_5ModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_qwen3_5_hf_compatible_model)
    transformers_min_version = "5.5.0"

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
        self.wrap_cfg["fuse_gdr_ops"] = self.config.fuse_gdr_ops
        self.wrap_cfg["split_conv_cache"] = self.config.split_conv_cache
        self.wrap_cfg["use_manual_depthwise_conv1d"] = self.config.use_manual_depthwise_conv1d
        if self.config.spec_decode_mode == "dflash":
            self.wrap_cfg["output_hidden_state_indices"] = self._get_dflash_target_layer_ids()

    def _get_dflash_target_layer_ids(self) -> list[int]:
        configured = getattr(self.config, "output_hidden_state_indices", None)
        if configured is not None:
            return list(configured)

        dflash_config = self.config.dflash_config
        if dflash_config is None:
            raise ValueError("dflash_config is required when spec_decode_mode='dflash'")

        dflash_model_dir = Path(dflash_config.dflash_model_dir)
        config_path = dflash_model_dir / "config.json"
        with open(config_path, encoding="utf-8") as f:
            target_layer_ids = json.load(f).get("dflash_config", {}).get("target_layer_ids")
        if not target_layer_ids:
            raise ValueError(f"Failed to read dflash_config.target_layer_ids from {config_path}")
        return list(target_layer_ids)

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
        if self._kvcache_mixin.split_conv_cache:
            return _flatten_split_conv_cache_outputs(self._kvcache_mixin.past_conv_caches)
        return self._kvcache_mixin.past_conv_caches

    @property
    def past_recurrent_states(self):
        return self._kvcache_mixin.past_recurrent_states

    def get_data_preprocessor(self) -> BaseLLMInputProcessor:
        self._sync_split_conv_cache_state()
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
        native_hf_model = super().get_empty_hf_model(hf_model_dir)
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
        if self._state == LLMModelState.FRONTED:
            self._frontend_model.set_activate_model("prefill")
        elif self._state in [LLMModelState.QUANTED_ALIGNED, LLMModelState.QUANTED_FAST, LLMModelState.QUANTED_DISABLE]:
            self._quanted_model.set_activate_model("prefill")
        super().set_prefill()

    def set_decode(self):
        self.wrap_cfg["linear_attention_mode"] = "recurrent"
        if self._state == LLMModelState.FRONTED:
            self._frontend_model.set_activate_model("decode")
        elif self._state in [LLMModelState.QUANTED_ALIGNED, LLMModelState.QUANTED_FAST, LLMModelState.QUANTED_DISABLE]:
            self._quanted_model.set_activate_model("decode")
        super().set_decode()

    def get_quant_cfg(self):
        quant_cfg = super().get_quant_cfg()
        quant_cfg.setdefault("ops_cfg", ConfigDict())
        quant_cfg["ops_cfg"]["Normalize"] = ConfigDict(force_fp32=self.config.normalize_force_fp32)

        cumsum_quant_cfg = self.config.cumsum_matmul_quant_config
        if cumsum_quant_cfg is None:
            cumsum_quant_cfg = dict(
                act_schema=dict(fp_mode="sefp", man_bit=16),
                act_schema_2=dict(fp_mode="fp16", man_bit=8),
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
        prefill_quanted_model = super()._to_quanted(prefill_fronted_model, state, infer_shape=False)

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
        from ._llm_model_impl import register_wrap_modules

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
            for past_recurrent_state, recurrent_state_out in zip(
                past_recurrent_states, recurrent_state_out_list, strict=True
            ):
                past_recurrent_state[:] = recurrent_state_out[:]

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
        self._sync_split_conv_cache_state()
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
        assert hasattr(self, "visual") and self.visual is not None, (
            "Visual model is not initialized, cannot get visual config for export."
        )
        image_size_h = self.visual.config.max_size_h
        image_size_w = self.visual.config.max_size_w
        model_name = f"hmquant_{model_name}_{image_size_w}x{image_size_h}_{str_datetime}"
        output_dir = Path(output_dir) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_info = self.create_export_metadata(output_dir)
        export_data = ExportData()
        export_data.exported_dir = str(output_dir)
        export_data.meta = meta_info
        export_data.model_name = model_name
        export_data.str_datetime = str_datetime
        return export_data

    @log_function_call()
    def export_hmonnx(self, output_dir: str) -> VLLMModelMeta:
        logger = get_xhquant_logger()
        self.work_dir = str(output_dir)

        self.to_wrap()

        exported_info = self.get_export_info(output_dir)
        meta_info = exported_info.meta
        meta_info = cast(VLLMModelMeta, meta_info)
        assert isinstance(meta_info, VLLMModelMeta), f"meta_info expected VLLMModelMeta, but get {type(meta_info)}"

        visual_output_dir = str(Path(exported_info.exported_dir) / "visual")
        # 导出visual
        assert hasattr(self, "visual") and self.visual is not None, (
            "Visual model is not initialized, cannot export hmonnx."
        )
        memory_info = get_cpu_memory_mb()
        logger.info(f"Initial CPU memory usage before exporting visual hmonnx: {str(memory_info)}")
        logger.info(f"Start exporting visual hmonnx to {visual_output_dir}")
        self.visual.config.model_name = (
            f"{exported_info.model_name}_{self.visual.config.max_size_w}x{self.visual.config.max_size_h}"
        )
        self.visual.to_quanted_aligned()
        visual_meta = self.visual.export_hmonnx(visual_output_dir)
        visual_meta.hmonnx = str(Path(visual_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix())
        meta_info.visual_config = visual_meta
        json.dump(
            meta_info.to_dict(), open(str(Path(exported_info.exported_dir) / "golden_meta_info.json"), "w"), indent=4
        )

        del self.visual
        gc.collect()

        memory_info = get_cpu_memory_mb()
        logger.info(f"Initial CPU memory usage after exporting visual hmonnx: {str(memory_info)}")

        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.prefill.fixed()
        self._quanted_model.decode.fixed()
        memory_info = get_cpu_memory_mb()
        logger.info(f"Initial CPU memory usage after quantization: {str(memory_info)}")

        self.config.model_name = exported_info.model_name

        spec_decode_mode = self.config.spec_decode_mode
        if spec_decode_mode in ("mtp", "dflash"):
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

        self._export_hmonnx(exported_info)

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
            num_draft_tokens = getattr(self.config, "num_draft_tokens", 4)
            hidden_output_name = "target_hidden" if spec_decode_mode == "dflash" else "post_norm_hidden"
            spec_block_size = num_draft_tokens + 1 if spec_decode_mode == "dflash" else num_draft_tokens
            hidden_output_name = "target_hidden" if spec_decode_mode == "dflash" else "post_norm_hidden"
            spec_decode_section = {
                "mode": spec_decode_mode,
                "block_size": spec_block_size,
                "num_draft_tokens": num_draft_tokens,
                "hidden_output_name": hidden_output_name,
            }
            if spec_decode_mode == "mtp":
                if hasattr(meta_info, "mtp_prefill_config"):
                    spec_decode_section["draft_prefill_onnx"] = meta_info.mtp_prefill_config.hmonnx
                    spec_decode_section["mtp_draft_prefill_onnx"] = meta_info.mtp_prefill_config.hmonnx
                if hasattr(meta_info, "mtp_decode_config"):
                    spec_decode_section["draft_decode_onnx"] = meta_info.mtp_decode_config.hmonnx
                    spec_decode_section["mtp_draft_decode_onnx"] = meta_info.mtp_decode_config.hmonnx
            elif spec_decode_mode == "dflash":
                if hasattr(meta_info, "dflash_context_config"):
                    spec_decode_section["draft_context_onnx"] = meta_info.dflash_context_config.hmonnx
                    spec_decode_section["dflash_draft_context_onnx"] = meta_info.dflash_context_config.hmonnx
                if hasattr(meta_info, "dflash_context_decode_config"):
                    spec_decode_section["draft_context_decode_onnx"] = meta_info.dflash_context_decode_config.hmonnx
                    spec_decode_section["dflash_draft_context_decode_onnx"] = (
                        meta_info.dflash_context_decode_config.hmonnx
                    )
                if hasattr(meta_info, "dflash_decode_config"):
                    spec_decode_section["draft_decode_onnx"] = meta_info.dflash_decode_config.hmonnx
                    spec_decode_section["dflash_draft_decode_onnx"] = meta_info.dflash_decode_config.hmonnx
            meta_info.spec_decode = spec_decode_section

        json.dump(
            meta_info.to_dict(), open(str(Path(exported_info.exported_dir) / "golden_meta_info.json"), "w"), indent=4
        )
        logger.info(f"Exporting completed! Exported model is saved at: {output_dir}")
        return meta_info
