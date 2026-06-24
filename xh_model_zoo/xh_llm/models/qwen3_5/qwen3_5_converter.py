# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import shutil
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ForCausalLM,
)
from xhquant.api import CacheTensor

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
from .qwen3_5_convert_config import Qwen3_5ConvertConfig

from xhquant.api import (  # type: ignore # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
)


def _extract_quant_method(config: AutoConfig) -> Optional[str]:
    quantization_config = getattr(config, "quantization_config", None)
    quant_method = getattr(quantization_config, "quant_method", None)
    if isinstance(quantization_config, dict):
        quant_method = quantization_config.get("quant_method", quant_method)
    return None if quant_method is None else str(quant_method).lower()


def _flatten_cache_outputs(self: nn.Module, *args, **kwargs):
    result = self._qwen3_5_original_forward(*args, **kwargs)
    batch_size = int(getattr(getattr(self, "cfg", None), "get", lambda *_: 1)("batch_size", 1))

    def _append_batch(outputs: List[torch.Tensor], value):
        if isinstance(value, (list, tuple)):
            outputs.extend(value)
            return
        if batch_size > 1:
            outputs.extend(value[batch_idx : batch_idx + 1] for batch_idx in range(batch_size))
        else:
            outputs.append(value)

    # Handle 3-tuple (logits, conv_caches, recurrent_states) and
    # 4-tuple (+ post_norm_hidden or target_hidden for spec decode).
    if len(result) == 4:
        logits, conv_cache_out_list, recurrent_state_out_list, hidden_output = result
    else:
        logits, conv_cache_out_list, recurrent_state_out_list = result
        hidden_output = None

    outputs: List[torch.Tensor] = []
    _append_batch(outputs, logits)
    if conv_cache_out_list is not None:
        outputs.extend(list(conv_cache_out_list))
    if recurrent_state_out_list is not None:
        outputs.extend(list(recurrent_state_out_list))
    if hidden_output is not None:
        _append_batch(outputs, hidden_output)
    return tuple(outputs)


def _get_text_model(native_model):
    """Get the text model from either ForConditionalGeneration or ForCausalLM."""
    if isinstance(native_model, Qwen3_5ForConditionalGeneration):
        return native_model.model.language_model
    elif isinstance(native_model, Qwen3_5ForCausalLM):
        return native_model.model
    else:
        raise ValueError(f"Unsupported model type: {type(native_model)}")


def _get_text_config(native_model):
    """Get text_config from the model config."""
    config = native_model.config
    if hasattr(config, "text_config"):
        return config.text_config
    return config


def _load_dflash_target_layer_ids(dflash_model_dir: str) -> List[int]:
    config_path = Path(dflash_model_dir) / "config.json"
    with open(config_path, encoding="utf-8") as f:
        dflash_config = json.load(f)
    target_layer_ids = dflash_config.get("dflash_config", {}).get("target_layer_ids")
    if not target_layer_ids:
        raise ValueError(f"Failed to read dflash_config.target_layer_ids from {config_path}")
    return list(target_layer_ids)


DRAFT_BASE_QUANT_TYPE = "w8a8h1_sefp"


def _build_spec_draft_quant_config(head_weight_bits: int) -> ConfigDict:
    if head_weight_bits == 8:
        return ConfigDict(quant_type=DRAFT_BASE_QUANT_TYPE)
    if head_weight_bits != 4:
        raise ValueError(f"Unsupported spec draft head weight bits: {head_weight_bits}. Expected 4 or 8.")
    return ConfigDict(
        quant_type=DRAFT_BASE_QUANT_TYPE,
        nodes_cfg=dict(
            lm_head=dict(
                w_schema=dict(
                    bits=4,
                    fp_mode="ssfp",
                    hidden_bit=False,
                )
            )
        ),
    )


_LINEAR_CONV_CACHE_BRANCHES = ("q", "k", "v")


def _split_batch_tensor(tensor: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
    return [tensor[i : i + 1] for i in range(batch_size)]


def _validate_spec_decode_batch_size(spec_decode_mode: Optional[str], batch_size: int) -> None:
    if spec_decode_mode in {"mtp", "dflash"} and int(batch_size) != 1:
        raise ValueError(
            "Qwen3.5 dense multi-batch export does not support "
            f"spec_decode_mode={spec_decode_mode!r}; use batch_size=1 for MTP/DFlash "
            "or disable spec_decode_mode for batch_size>1."
        )


class Qwen3_5ConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: Qwen3_5ConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        native_model = AutoModelForCausalLM.from_pretrained(
            hf_model_dir,
            trust_remote_code=True,
            **kwargs,
        )
        native_model = self.dequantize_hf_model(native_model)
        assert isinstance(native_model, (Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM)), (
            f"Expected Qwen3_5ForConditionalGeneration or Qwen3_5ForCausalLM, got {type(native_model)}"
        )
        native_model = native_model.eval()

        # Handle tied embeddings
        tie_word_embeddings = getattr(native_model.config, "tie_word_embeddings", False)
        text_config = _get_text_config(native_model)
        if not tie_word_embeddings:
            tie_word_embeddings = getattr(text_config, "tie_word_embeddings", False)
        if tie_word_embeddings:
            old_torchscript = native_model.config.torchscript
            native_model.config.torchscript = True
            native_model.tie_weights()
            native_model.config.tie_word_embeddings = False
            if hasattr(text_config, "tie_word_embeddings"):
                text_config.tie_word_embeddings = False
            native_model.config.torchscript = old_torchscript

        self.hf_model_path = hf_model_dir
        return native_model

    def _copy_hf_config(self, hf_model_path: str, work_dir: Path):
        logger = get_root_logger()
        hf_config_dir = work_dir / "hf_config"
        hf_config_dir.mkdir(exist_ok=True, parents=True)
        hf_config_files = [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "merges.txt",
            "tokenizer.model",
        ]
        for cfg_file in hf_config_files:
            src_file = Path(hf_model_path) / cfg_file
            dst_file = hf_config_dir / cfg_file
            if src_file.exists():
                shutil.copyfile(src_file, dst_file)
            else:
                logger.warning(f"{src_file} not exists, skip copy")
        return hf_config_dir

    def _build_quant_config(self, wraped_model: nn.Module):
        quant_config = ConfigDict(create_quant_config(self.config.quant_scheme))
        quant_config.setdefault("inputs", {})
        quant_config["inputs"].setdefault(
            "linear_attn_mask",
            dict(
                quantizer=dict(
                    qspec=dict(fake_dtype="float16"),
                )
            ),
        )

        cumsum_quant_cfg = self.config.cumsum_matmul_quant_config
        if cumsum_quant_cfg is None:
            cumsum_quant_cfg = dict(
                act_schema=dict(
                    fp_mode="sefp",
                    man_bit=16,
                ),
                act_schema_2=dict(
                    fp_mode="fp16",
                    man_bit=8,
                ),
            )

        quant_config.setdefault("nodes_cfg", {})
        for name, module in wraped_model.named_modules():
            if isinstance(module, nn.Linear) and name == "lm_head" and not hasattr(module, "quant_weight"):
                quant_config["nodes_cfg"].setdefault(
                    name,
                    dict(
                        w_schema=dict(
                            bits=8,
                            fp_mode="sefp",
                        )
                    ),
                )

            if "cumsum_matmul" not in name:
                continue
            key_candidates = {
                name,
                name.replace(".", "_"),
            }
            for key in key_candidates:
                quant_config["nodes_cfg"][key] = dict(cumsum_quant_cfg)

        quant_config.setdefault("ops_cfg", {})
        quant_config["ops_cfg"]["Normalize"] = dict(force_fp32=self.config.normalize_force_fp32)

        return quant_config

    def _prepare_wrap_model(self, native_model):
        from ._model import register_wrap_modules as qwen3_5_register_wrap_modules

        qwen3_5_register_wrap_modules()
        spec_decode_mode = getattr(self.config, "spec_decode_mode", None)
        _validate_spec_decode_batch_size(spec_decode_mode, self.config.batch_size)
        output_post_norm_hidden = spec_decode_mode == "mtp"
        output_hidden_state_indices = None
        if spec_decode_mode == "dflash":
            dflash_model_dir = getattr(self.config, "dflash_model_dir", None)
            if not dflash_model_dir:
                raise ValueError("dflash_model_dir is required when spec_decode_mode='dflash'")
            output_hidden_state_indices = _load_dflash_target_layer_ids(dflash_model_dir)
        wrap_cfg = Config(
            dict(
                batch_size=self.config.batch_size,
                max_sequence_length=self.config.context_length,
                input_sequence_length=self.config.input_sequence_length,
                use_cache=True,
                num_logits_to_keep=self.config.num_logits_to_keep,
                linear_attention_mode=self.config.linear_attention_mode,
                linear_chunk_size=self.config.linear_chunk_size,
                enable_rope=self.config.enable_rope,
                max_pe_length=getattr(self.config, "max_pe_length", 262144),
                support_long_context_over_fp16_limit=getattr(
                    self.config, "support_long_context_over_fp16_limit", True
                ),
                output_hidden_state_indices=output_hidden_state_indices,
                output_post_norm_hidden=output_post_norm_hidden,
                split_conv_cache=self.config.split_conv_cache,
                use_manual_depthwise_conv1d=self.config.use_manual_depthwise_conv1d,
                fuse_gdr_ops=self.config.fuse_gdr_ops,
                fuse_gdr_block_recurrent_ops=self.config.fuse_gdr_block_recurrent_ops,
                kv_cache=dict(
                    cache_axis=2,
                ),
            )
        )
        wraped_model = wrap_llm_model(native_model, wrap_cfg)
        if not hasattr(wraped_model, "_qwen3_5_original_forward"):
            wraped_model._qwen3_5_original_forward = wraped_model.forward
            wraped_model.forward = types.MethodType(_flatten_cache_outputs, wraped_model)
        return wraped_model, wrap_cfg

    def _build_cache_inputs(
        self,
        wraped_model: nn.Module,
        native_model,
        context_length: int,
    ):
        text_config = _get_text_config(native_model)
        text_model = _get_text_model(wraped_model)
        layer_types = list(text_config.layer_types)
        full_attention_layer_indices = [i for i, layer_type in enumerate(layer_types) if layer_type == "full_attention"]
        linear_attention_layer_indices = [
            i for i, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
        ]

        head_dim = text_config.head_dim
        batch_size = int(self.config.batch_size)
        if batch_size > 1:
            kv_cache_shape = [1, text_config.num_key_value_heads, context_length, head_dim]
            past_key_caches = []
            past_value_caches = []
            for _ in full_attention_layer_indices:
                for _batch_idx in range(batch_size):
                    past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
                    past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
        else:
            kv_cache_shape = [
                batch_size,
                text_config.num_key_value_heads,
                context_length,
                head_dim,
            ]
            past_key_caches = [
                CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
                for _ in full_attention_layer_indices
            ]
            past_value_caches = [
                CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
                for _ in full_attention_layer_indices
            ]

        past_conv_caches = []
        past_recurrent_states = []
        linear_cache_meta = []
        for layer_idx in linear_attention_layer_indices:
            linear_attn = text_model.layers[layer_idx].linear_attn
            if self.config.split_conv_cache:
                cache_dtype = (
                    linear_attn.conv1d_q.weight.dtype
                    if hasattr(linear_attn, "conv1d_q")
                    else linear_attn.conv1d.weight.dtype
                )
                conv_shape_q = [
                    1 if batch_size > 1 else batch_size,
                    linear_attn.head_k_dim * linear_attn.num_v_heads,
                    linear_attn.conv_kernel_size,
                ]
                conv_shape_k = [
                    1 if batch_size > 1 else batch_size,
                    linear_attn.head_k_dim * linear_attn.num_v_heads,
                    linear_attn.conv_kernel_size,
                ]
                conv_shape_v = [
                    1 if batch_size > 1 else batch_size,
                    linear_attn.head_v_dim * linear_attn.num_v_heads,
                    linear_attn.conv_kernel_size,
                ]
                conv_shapes = [conv_shape_q, conv_shape_k, conv_shape_v]
                if batch_size > 1:
                    expanded_conv_shapes = []
                    for shape in conv_shapes:
                        for _batch_idx in range(batch_size):
                            past_conv_caches.append(CacheTensor(torch.zeros(shape, dtype=cache_dtype)))
                            expanded_conv_shapes.append(list(shape))
                else:
                    expanded_conv_shapes = conv_shapes
                    for shape in conv_shapes:
                        past_conv_caches.append(CacheTensor(torch.zeros(shape, dtype=cache_dtype)))
                recurrent_shape = [
                    1 if batch_size > 1 else batch_size,
                    linear_attn.num_v_heads,
                    linear_attn.head_k_dim,
                    linear_attn.head_v_dim,
                ]
                if batch_size > 1:
                    recurrent_shapes = []
                    for _batch_idx in range(batch_size):
                        past_recurrent_states.append(CacheTensor(torch.zeros(recurrent_shape, dtype=cache_dtype)))
                        recurrent_shapes.append(list(recurrent_shape))
                else:
                    recurrent_shapes = [recurrent_shape]
                    past_recurrent_states.append(CacheTensor(torch.zeros(recurrent_shape, dtype=cache_dtype)))
                linear_cache_meta.append(
                    dict(
                        layer_idx=layer_idx,
                        conv_shapes=expanded_conv_shapes,
                        recurrent_shapes=recurrent_shapes,
                        recurrent_shape=recurrent_shape,
                        per_batch=batch_size > 1,
                    )
                )
            else:
                cache_dtype = linear_attn.conv1d.weight.dtype
                conv_shape = [
                    1 if batch_size > 1 else batch_size,
                    linear_attn.conv_dim,
                    linear_attn.conv_kernel_size,
                ]
                if batch_size > 1:
                    expanded_conv_shapes = []
                    for _batch_idx in range(batch_size):
                        past_conv_caches.append(CacheTensor(torch.zeros(conv_shape, dtype=cache_dtype)))
                        expanded_conv_shapes.append(list(conv_shape))
                else:
                    expanded_conv_shapes = [conv_shape]
                    past_conv_caches.append(CacheTensor(torch.zeros(conv_shape, dtype=cache_dtype)))
                recurrent_shape = [
                    1 if batch_size > 1 else batch_size,
                    linear_attn.num_v_heads,
                    linear_attn.head_k_dim,
                    linear_attn.head_v_dim,
                ]
                if batch_size > 1:
                    recurrent_shapes = []
                    for _batch_idx in range(batch_size):
                        past_recurrent_states.append(CacheTensor(torch.zeros(recurrent_shape, dtype=cache_dtype)))
                        recurrent_shapes.append(list(recurrent_shape))
                else:
                    recurrent_shapes = [recurrent_shape]
                    past_recurrent_states.append(CacheTensor(torch.zeros(recurrent_shape, dtype=cache_dtype)))
                linear_cache_meta.append(
                    dict(
                        layer_idx=layer_idx,
                        conv_shapes=expanded_conv_shapes,
                        conv_shape=conv_shape,
                        recurrent_shapes=recurrent_shapes,
                        recurrent_shape=recurrent_shape,
                        per_batch=batch_size > 1,
                    )
                )

        return (
            full_attention_layer_indices,
            linear_attention_layer_indices,
            kv_cache_shape,
            past_key_caches,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
            linear_cache_meta,
        )

    def _convert(self, hf_model_path: str, output_dir: str):
        logger = get_root_logger()
        native_model = self.get_hf_model(
            hf_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        if self.config.quant_weight is not None:
            self.load_quant_weight(self.config.quant_weight, native_model)

        work_dir = Path(output_dir)
        work_dir.mkdir(exist_ok=True, parents=True)
        model_name = Path(hf_model_path).name
        context_length = self.config.context_length
        input_sequence_length = self.config.input_sequence_length
        quant_type = self.config.quant_scheme.quant_type
        target_device = self.config.quant_scheme.target_device
        text_config = _get_text_config(native_model)

        meta_info: Dict[str, Any] = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            device=str(target_device),
            model_name=model_name,
            hf_model_path=hf_model_path,
            architecture="Qwen3_5ForConditionalGeneration",
            quant_scheme=self.config.quant_scheme.to_dict(),
            quant_weight=self.config.quant_weight,
            source_quant_method=_extract_quant_method(native_model.config),
            pad_token_id=getattr(native_model.config, "eos_token_id", None)
            or getattr(text_config, "eos_token_id", 151645),
            max_context_tokens=context_length,
        )

        hf_config_dir = self._copy_hf_config(hf_model_path, work_dir)
        meta_info["hf_config"] = str(hf_config_dir.relative_to(work_dir))

        text_model = _get_text_model(native_model)
        token_embedding = text_model.get_input_embeddings()
        token_embedding_file = work_dir / "token_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file))
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        wraped_model, wrap_cfg = self._prepare_wrap_model(native_model)
        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        (
            full_attention_layer_indices,
            linear_attention_layer_indices,
            kv_cache_shape,
            past_key_caches,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
            linear_cache_meta,
        ) = self._build_cache_inputs(wraped_model, native_model, context_length)

        meta_info["kv_cache"] = dict(
            shape=kv_cache_shape,
            num_decoder_layers=len(full_attention_layer_indices),
            layer_indices=full_attention_layer_indices,
        )
        meta_info["linear_cache"] = dict(
            num_decoder_layers=len(linear_attention_layer_indices),
            layer_indices=linear_attention_layer_indices,
            layers=linear_cache_meta,
            num_conv_caches=len(past_conv_caches),
        )

        # Build inputs with M-RoPE position IDs
        input_ids_t = torch.randint(0, 1000, (self.config.batch_size, input_sequence_length), dtype=torch.long)
        inputs_embeds = token_embedding(input_ids_t)
        past_seq_length_t = torch.zeros(self.config.batch_size, dtype=torch.int32)
        current_input_length_t = torch.full(
            (self.config.batch_size,),
            input_sequence_length,
            dtype=torch.int32,
        )
        linear_attn_mask_t = torch.ones(
            self.config.batch_size,
            input_sequence_length,
            dtype=inputs_embeds.dtype,
        )

        # M-RoPE position IDs: for text-only, all three are sequential
        position_ids = torch.arange(input_sequence_length, dtype=torch.long).unsqueeze(0).expand(
            self.config.batch_size, -1
        )
        time_position_ids = position_ids
        hight_position_ids = position_ids
        width_position_ids = position_ids

        batch_size = int(self.config.batch_size)
        if batch_size > 1:
            # Export every batch item as an independent graph input.  The
            # wrapper concatenates those single-batch tensors before compute and
            # splits logits/cache outputs back to single-batch tensors so the
            # runtime can maintain per-request KV/linear caches.
            graph_inputs_embeds = _split_batch_tensor(inputs_embeds, batch_size)
            graph_time_position_ids = _split_batch_tensor(time_position_ids, batch_size)
            graph_hight_position_ids = _split_batch_tensor(hight_position_ids, batch_size)
            graph_width_position_ids = _split_batch_tensor(width_position_ids, batch_size)
            graph_past_seq_length = _split_batch_tensor(past_seq_length_t, batch_size)
            graph_current_input_length = _split_batch_tensor(current_input_length_t, batch_size)
            graph_linear_attn_mask = _split_batch_tensor(linear_attn_mask_t, batch_size)
        else:
            graph_inputs_embeds = inputs_embeds
            graph_time_position_ids = time_position_ids
            graph_hight_position_ids = hight_position_ids
            graph_width_position_ids = width_position_ids
            graph_past_seq_length = past_seq_length_t
            graph_current_input_length = current_input_length_t
            graph_linear_attn_mask = linear_attn_mask_t

        inputs = (
            graph_inputs_embeds,
            graph_time_position_ids,
            graph_hight_position_ids,
            graph_width_position_ids,
            graph_past_seq_length,
            graph_current_input_length,
            graph_linear_attn_mask,
            past_key_caches,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
        )

        if batch_size > 1:
            input_names = []
            for base_name in (
                "inputs_embeds",
                "time_position_ids",
                "hight_position_ids",
                "width_position_ids",
                "past_seq_length",
                "current_input_length",
                "linear_attn_mask",
            ):
                for batch_idx in range(batch_size):
                    input_names.append(f"{base_name}_batch_{batch_idx}")
        else:
            input_names = [
                "inputs_embeds",
                "time_position_ids",
                "hight_position_ids",
                "width_position_ids",
                "past_seq_length",
                "current_input_length",
                "linear_attn_mask",
            ]
        for layer_idx in range(len(full_attention_layer_indices)):
            if batch_size > 1:
                for batch_idx in range(batch_size):
                    input_names.append(f"past_key_cache_{layer_idx}_batch_{batch_idx}")
            else:
                input_names.append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(len(full_attention_layer_indices)):
            if batch_size > 1:
                for batch_idx in range(batch_size):
                    input_names.append(f"past_value_cache_{layer_idx}_batch_{batch_idx}")
            else:
                input_names.append(f"past_value_cache_{layer_idx}")
        for layer_idx in range(len(linear_attention_layer_indices)):
            if self.config.split_conv_cache:
                for branch in _LINEAR_CONV_CACHE_BRANCHES:
                    if batch_size > 1:
                        for batch_idx in range(batch_size):
                            input_names.append(f"past_conv_cache_{branch}_{layer_idx}_batch_{batch_idx}")
                    else:
                        input_names.append(f"past_conv_cache_{branch}_{layer_idx}")
            else:
                if batch_size > 1:
                    for batch_idx in range(batch_size):
                        input_names.append(f"past_conv_cache_{layer_idx}_batch_{batch_idx}")
                else:
                    input_names.append(f"past_conv_cache_{layer_idx}")
        for layer_idx in range(len(linear_attention_layer_indices)):
            if batch_size > 1:
                for batch_idx in range(batch_size):
                    input_names.append(f"past_recurrent_state_{layer_idx}_batch_{batch_idx}")
            else:
                input_names.append(f"past_recurrent_state_{layer_idx}")

        output_names_base = (
            [f"logits_batch_{batch_idx}" for batch_idx in range(batch_size)]
            if batch_size > 1
            else ["logits"]
        )
        for layer_idx in range(len(linear_attention_layer_indices)):
            if self.config.split_conv_cache:
                for branch in _LINEAR_CONV_CACHE_BRANCHES:
                    if batch_size > 1:
                        for batch_idx in range(batch_size):
                            output_names_base.append(f"conv_cache_out_{branch}_{layer_idx}_batch_{batch_idx}")
                    else:
                        output_names_base.append(f"conv_cache_out_{branch}_{layer_idx}")
            else:
                if batch_size > 1:
                    for batch_idx in range(batch_size):
                        output_names_base.append(f"conv_cache_out_{layer_idx}_batch_{batch_idx}")
                else:
                    output_names_base.append(f"conv_cache_out_{layer_idx}")
        for layer_idx in range(len(linear_attention_layer_indices)):
            if batch_size > 1:
                for batch_idx in range(batch_size):
                    output_names_base.append(f"recurrent_state_out_{layer_idx}_batch_{batch_idx}")
            else:
                output_names_base.append(f"recurrent_state_out_{layer_idx}")

        spec_decode_mode = getattr(self.config, "spec_decode_mode", None)
        num_draft_tokens = int(getattr(self.config, "num_draft_tokens", 4))
        verify_length = num_draft_tokens + 1
        output_hidden_state_indices = None
        if spec_decode_mode == "dflash":
            dflash_model_dir = getattr(self.config, "dflash_model_dir", None)
            if not dflash_model_dir:
                raise ValueError("dflash_model_dir is required when spec_decode_mode='dflash'")
            output_hidden_state_indices = _load_dflash_target_layer_ids(dflash_model_dir)
        output_post_norm_hidden = spec_decode_mode == "mtp"
        extra_hidden_output_name = None
        if output_hidden_state_indices is not None:
            extra_hidden_output_name = "target_hidden"
        elif output_post_norm_hidden:
            extra_hidden_output_name = "post_norm_hidden"

        prefill_output_names = list(output_names_base)
        # Spec decode verify runs draft tokens + the current token in one decode
        # pass and emits per-step linear-attention intermediates for accept/reject.
        if spec_decode_mode:
            decode_output_names = (
                [f"logits_batch_{batch_idx}" for batch_idx in range(batch_size)]
                if batch_size > 1
                else ["logits"]
            )
            for layer_idx in range(len(linear_attention_layer_indices)):
                if self.config.split_conv_cache:
                    for branch in _LINEAR_CONV_CACHE_BRANCHES:
                        for step_idx in range(verify_length):
                            if batch_size > 1:
                                for batch_idx in range(batch_size):
                                    decode_output_names.append(
                                        f"conv_cache_out_{branch}_{layer_idx}_{step_idx}_batch_{batch_idx}"
                                    )
                            else:
                                decode_output_names.append(f"conv_cache_out_{branch}_{layer_idx}_{step_idx}")
                else:
                    for step_idx in range(verify_length):
                        if batch_size > 1:
                            for batch_idx in range(batch_size):
                                decode_output_names.append(
                                    f"conv_cache_out_{layer_idx}_{step_idx}_batch_{batch_idx}"
                                )
                        else:
                            decode_output_names.append(f"conv_cache_out_{layer_idx}_{step_idx}")
            for layer_idx in range(len(linear_attention_layer_indices)):
                for step_idx in range(verify_length):
                    if batch_size > 1:
                        for batch_idx in range(batch_size):
                            decode_output_names.append(
                                f"recurrent_state_out_{layer_idx}_{step_idx}_batch_{batch_idx}"
                            )
                    else:
                        decode_output_names.append(f"recurrent_state_out_{layer_idx}_{step_idx}")
        else:
            decode_output_names = list(output_names_base)
        if extra_hidden_output_name is not None:
            if batch_size > 1:
                extra_hidden_output_names = [
                    f"{extra_hidden_output_name}_batch_{batch_idx}"
                    for batch_idx in range(batch_size)
                ]
                prefill_output_names.extend(extra_hidden_output_names)
                decode_output_names.extend(extra_hidden_output_names)
            else:
                prefill_output_names.append(extra_hidden_output_name)
                decode_output_names.append(extra_hidden_output_name)

        quant_config = self._build_quant_config(wraped_model)
        wraped_model_prefill = copy.deepcopy(wraped_model)
        if spec_decode_mode:
            wrap_cfg_prefill = copy.deepcopy(wrap_cfg)
            wrap_cfg_prefill.num_logits_to_keep = 0

            def _apply_prefill_update_cfg(module):
                if hasattr(module, "_update_cfg"):
                    module._update_cfg(wrap_cfg_prefill)

            wraped_model_prefill.apply(_apply_prefill_update_cfg)
        quanted_prefill_model = convert_fx_model_to_quanted_model(
            wraped_model_prefill,
            inputs,
            target_device,
            quant_config=quant_config,
        )

        prefix = f"{model_name}-{target_device}-{context_length // 1024}k-{quant_type}"
        prefill_onnx_file = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))
        logger.info("********************* start export prefill model *********************")
        convert_quanted_model_to_hmonnx(
            quanted_prefill_model,
            inputs,
            str(prefill_onnx_file),
            BaseConverter.xh1_hmonnx_compatible(input_names),
            prefill_output_names,
        )
        logger.info(f"Export Prefill model to {prefill_onnx_file}")
        del quanted_prefill_model, wraped_model_prefill

        # Re-trace decode after changing sequence length. In spec mode the
        # target decode verifies num_draft_tokens + 1 positions.
        decode_seq_len = verify_length if spec_decode_mode else 1
        wrap_cfg.input_sequence_length = decode_seq_len
        if spec_decode_mode:
            wrap_cfg.num_logits_to_keep = 0
            wrap_cfg.verify_output_intermediates = True
            wrap_cfg.linear_attention_mode = "recurrent"

        def _apply_update_cfg(module):
            if hasattr(module, "_update_cfg"):
                module._update_cfg(wrap_cfg)

        wraped_model.apply(_apply_update_cfg)

        decode_position_ids = torch.zeros(self.config.batch_size, decode_seq_len, dtype=torch.long)
        decode_current_input_length_t = torch.full_like(current_input_length_t, decode_seq_len)
        decode_linear_attn_mask_t = torch.ones(batch_size, decode_seq_len, dtype=inputs_embeds.dtype)
        if batch_size > 1:
            decode_graph_inputs_embeds = _split_batch_tensor(inputs_embeds[:, :decode_seq_len, :], batch_size)
            decode_graph_position_ids = _split_batch_tensor(decode_position_ids, batch_size)
            decode_graph_past_seq_length = _split_batch_tensor(past_seq_length_t, batch_size)
            decode_graph_current_input_length = _split_batch_tensor(decode_current_input_length_t, batch_size)
            decode_graph_linear_attn_mask = _split_batch_tensor(decode_linear_attn_mask_t, batch_size)
        else:
            decode_graph_inputs_embeds = inputs_embeds[:, :decode_seq_len, :]
            decode_graph_position_ids = decode_position_ids
            decode_graph_past_seq_length = past_seq_length_t
            decode_graph_current_input_length = decode_current_input_length_t
            decode_graph_linear_attn_mask = decode_linear_attn_mask_t
        decode_inputs = (
            decode_graph_inputs_embeds,
            decode_graph_position_ids,
            decode_graph_position_ids,
            decode_graph_position_ids,
            decode_graph_past_seq_length,
            decode_graph_current_input_length,
            decode_graph_linear_attn_mask,
            past_key_caches,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
        )

        quanted_decode_model = convert_fx_model_to_quanted_model(
            wraped_model,
            decode_inputs,
            target_device,
            quant_config=quant_config,
        )

        decode_onnx_file = work_dir / "hmonnx" / "decode" / f"{prefix}_decoder.onnx"
        decode_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["decode_onnx"] = str(decode_onnx_file.relative_to(work_dir))
        logger.info("********************* start export decode model *********************")
        convert_quanted_model_to_hmonnx(
            quanted_decode_model,
            decode_inputs,
            str(decode_onnx_file),
            BaseConverter.xh1_hmonnx_compatible(input_names),
            decode_output_names,
        )
        logger.info(f"Export decode model to {decode_onnx_file}")

        if spec_decode_mode == "mtp":
            draft_onnx_files = self._export_mtp_draft_model(
                hf_model_path=hf_model_path,
                work_dir=work_dir,
                prefix=prefix,
                context_length=context_length,
                verify_length=verify_length,
                target_device=target_device,
            )
            self._add_spec_decode_meta(
                meta_info=meta_info,
                spec_decode_mode=spec_decode_mode,
                draft_onnx_files=draft_onnx_files,
                work_dir=work_dir,
                hidden_output_name="post_norm_hidden",
                verify_length=verify_length,
            )
        elif spec_decode_mode == "dflash":
            draft_onnx_files = self._export_dflash_draft_model(
                hf_model_path=hf_model_path,
                dflash_model_dir=self.config.dflash_model_dir,
                work_dir=work_dir,
                prefix=prefix,
                context_length=context_length,
                verify_length=verify_length,
                target_device=target_device,
            )
            self._add_spec_decode_meta(
                meta_info=meta_info,
                spec_decode_mode=spec_decode_mode,
                draft_onnx_files=draft_onnx_files,
                work_dir=work_dir,
                hidden_output_name="target_hidden",
                verify_length=verify_length,
            )

        with open(work_dir / "meta.json", "w", encoding="utf-8") as fout:
            json.dump(meta_info, fout, ensure_ascii=False, indent=4)

    def _export_mtp_draft_model(
        self,
        hf_model_path: str,
        work_dir: Path,
        prefix: str,
        context_length: int,
        verify_length: int,
        target_device: str,
    ) -> dict:
        """Export separate MTP draft prefill/decode ONNX files for dense Qwen3.5."""
        import xh_model_zoo.xh_llm.models.qwen3_5.qwen3_5_mtp_model  # noqa: F401
        from xh_model_zoo.xh_llm.models.builder import MODELS
        from xhquant.api import (  # type: ignore
            PrecisionMode,
            ptq_quantize,
        )

        logger = get_root_logger()
        draft_onnx_dir = work_dir / "draft_onnx"
        draft_onnx_dir.mkdir(exist_ok=True, parents=True)
        max_pe_length = 262144
        draft_head_weight_bits = int(getattr(self.config, "spec_draft_head_weight_bits", 4))
        logger.info(f"MTP draft quant config: base={DRAFT_BASE_QUANT_TYPE}, lm_head_w_bits={draft_head_weight_bits}")

        def _export_one(wrap_cfg_extra: dict, name_suffix: str) -> str:
            model_cfg = dict(
                type="XHMTPDraftModel",
                hf_model=None,
                wrap_cfg=ConfigDict(
                    max_sequence_length=context_length,
                    max_pe_length=max_pe_length,
                    dtype="float16",
                    batch_size=1,
                    **wrap_cfg_extra,
                ),
                quant_config=_build_spec_draft_quant_config(draft_head_weight_bits),
                export_cfg=ConfigDict(),
                target_model_dir=hf_model_path,
            )
            draft_model = MODELS.build(model_cfg)
            draft_model.init_wrap_model()
            logger.info(
                f"[{name_suffix}] MTP draft params: "
                f"{sum(p.numel() for p in draft_model._wrap_model.parameters()) / 1e6:.1f}M"
            )
            dummy_data = draft_model.prepare_inputs(None)
            draft_model.convert_to_fronted_graph(dummy_data)
            draft_model.convert_to_quant_graph(target_device)
            ptq_quantize(
                draft_model._quanted_model,
                [draft_model.prepare_inputs(None)],
                PrecisionMode.ALIGNED,
                [torch.device("cpu")],
            )
            draft_model.convert_to_export_graph(dummy_data)
            onnx_file = draft_model.to_export_onnx(dummy_data, str(draft_onnx_dir), f"{prefix}_{name_suffix}")[0]
            draft_model.release_exported_model()
            draft_model.release_quanted_model()
            draft_model.release_frontend_model()
            draft_model.release_wraped_model()
            del draft_model
            return onnx_file

        prefill_onnx = _export_one({"input_sequence_length": self.config.input_sequence_length}, "mtp_prefill")
        decode_onnx = _export_one({"input_sequence_length": 1}, "mtp_decode")
        return {
            "draft_prefill_onnx": prefill_onnx,
            "draft_decode_onnx": decode_onnx,
        }

    def _export_dflash_draft_model(
        self,
        hf_model_path: str,
        dflash_model_dir: str,
        work_dir: Path,
        prefix: str,
        context_length: int,
        verify_length: int,
        target_device: str,
    ) -> dict:
        import xh_model_zoo.xh_llm.models.qwen3_5.qwen3_5_dflash_model  # noqa: F401
        from xh_model_zoo.xh_llm.models.builder import MODELS
        from xhquant.api import (  # type: ignore
            PrecisionMode,
            ptq_quantize,
        )

        logger = get_root_logger()
        draft_onnx_dir = work_dir / "draft_onnx"
        draft_onnx_dir.mkdir(exist_ok=True, parents=True)
        max_pe_length = 262144
        draft_decode_seq_len = int(verify_length)
        draft_head_weight_bits = int(getattr(self.config, "spec_draft_head_weight_bits", 4))
        logger.info(f"DFlash draft quant config: base={DRAFT_BASE_QUANT_TYPE}, lm_head_w_bits={draft_head_weight_bits}")
        with open(Path(dflash_model_dir) / "config.json", encoding="utf-8") as f:
            dflash_cfg = json.load(f)
        model_block_size = int(dflash_cfg.get("block_size", draft_decode_seq_len))
        if draft_decode_seq_len > model_block_size:
            raise ValueError(
                f"DFlash draft decode input length ({draft_decode_seq_len} = num_draft_tokens + 1) exceeds "
                f"model block_size ({model_block_size}) from {Path(dflash_model_dir) / 'config.json'}"
            )
        if draft_decode_seq_len < model_block_size:
            logger.info(
                "DFlash draft decode input_sequence_length reduced from model "
                f"block_size={model_block_size} to verify_length={draft_decode_seq_len}"
            )

        def _export_one(mode: str, input_sequence_length: int, name_suffix: str) -> str:
            model_cfg = dict(
                type="XHDFlashDraftModel",
                hf_model=None,
                wrap_cfg=ConfigDict(
                    mode=mode,
                    input_sequence_length=input_sequence_length,
                    max_sequence_length=context_length,
                    max_pe_length=max_pe_length,
                    dtype="float16",
                    batch_size=1,
                ),
                quant_config=_build_spec_draft_quant_config(draft_head_weight_bits),
                export_cfg=ConfigDict(),
                dflash_model_dir=dflash_model_dir,
                target_model_dir=hf_model_path,
            )
            draft_model = MODELS.build(model_cfg)
            draft_model.init_wrap_model()
            logger.info(
                f"[{name_suffix}] DFlash draft params: "
                f"{sum(p.numel() for p in draft_model._wrap_model.parameters()) / 1e6:.1f}M"
            )
            dummy_data = draft_model.prepare_inputs(None)
            draft_model.convert_to_fronted_graph(dummy_data)
            draft_model.convert_to_quant_graph(target_device)
            ptq_quantize(
                draft_model._quanted_model,
                [draft_model.prepare_inputs(None)],
                PrecisionMode.ALIGNED,
                [torch.device("cpu")],
            )
            draft_model.convert_to_export_graph(dummy_data)
            onnx_file = draft_model.to_export_onnx(dummy_data, str(draft_onnx_dir), f"{prefix}_{name_suffix}")[0]
            draft_model.release_exported_model()
            draft_model.release_quanted_model()
            draft_model.release_frontend_model()
            draft_model.release_wraped_model()
            del draft_model
            return onnx_file

        return {
            "draft_context_onnx": _export_one("context", self.config.input_sequence_length, "dflash_context"),
            "draft_context_decode_onnx": _export_one("context", draft_decode_seq_len, "dflash_context_decode"),
            "draft_decode_onnx": _export_one("decode", draft_decode_seq_len, "dflash_decode"),
        }

    @staticmethod
    def _resolve_existing_meta_path(existing_work_dir: Path, meta_info: Dict[str, Any], *keys: str) -> Path:
        for key in keys:
            path_value = meta_info.get(key)
            if path_value:
                path = Path(str(path_value))
                if not path.is_absolute():
                    path = (existing_work_dir / path).resolve()
                if not path.exists():
                    raise FileNotFoundError(f"Resolved {key} does not exist: {path}")
                return path
        raise FileNotFoundError(f"None of {keys} found in {existing_work_dir / 'meta.json'}")

    @staticmethod
    def _path_relative_to_work_dir(path_value: str, work_dir: Path) -> str:
        path = Path(path_value)
        if path.is_absolute():
            return str(path.resolve().relative_to(work_dir.resolve()))
        try:
            return str(path.relative_to(work_dir))
        except ValueError:
            pass
        try:
            return str(path.resolve().relative_to(work_dir.resolve()))
        except ValueError:
            return str(path)

    def _add_spec_decode_meta(
        self,
        meta_info: Dict[str, Any],
        spec_decode_mode: str,
        draft_onnx_files: Dict[str, str],
        work_dir: Path,
        hidden_output_name: str,
        verify_length: int,
    ) -> None:
        block_size = int(getattr(self.config, "num_draft_tokens", 4))
        draft_head_weight_bits = int(getattr(self.config, "spec_draft_head_weight_bits", 4))
        spec_decode: Dict[str, Any] = dict(
            mode=spec_decode_mode,
            block_size=block_size,
            hidden_output_name=hidden_output_name,
            verify_length=verify_length,
            draft_head_weight_bits=draft_head_weight_bits,
        )
        for key, value in draft_onnx_files.items():
            relative_path = self._path_relative_to_work_dir(value, work_dir)
            meta_info[f"{key}_file"] = relative_path
            spec_decode[key] = relative_path
        if "draft_decode_onnx" in spec_decode:
            meta_info["draft_onnx_file"] = spec_decode["draft_decode_onnx"]
            spec_decode["draft_onnx"] = spec_decode["draft_decode_onnx"]
        meta_info.update(
            dict(
                spec_decode_mode=spec_decode_mode,
                spec_decode_block_size=block_size,
                spec_decode_hidden_output_name=hidden_output_name,
                spec_decode_verify_length=verify_length,
                spec_decode_draft_head_weight_bits=draft_head_weight_bits,
                spec_decode=spec_decode,
            )
        )

    def export_draft_only(self, hf_model_path: str, existing_work_dir: str, output_dir: str):
        logger = get_root_logger()
        existing_work_dir_path = Path(existing_work_dir).resolve()
        work_dir = Path(output_dir).resolve()
        meta_path = existing_work_dir_path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in existing work_dir: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as file:
            existing_meta = json.load(file)

        spec_decode_mode = getattr(self.config, "spec_decode_mode", None)
        if spec_decode_mode not in {"mtp", "dflash"}:
            raise ValueError("draft-only dense Qwen3.5 export requires spec_decode_mode to be one of {'mtp', 'dflash'}")
        if spec_decode_mode == "dflash" and not getattr(self.config, "dflash_model_dir", None):
            raise ValueError("dflash_model_dir is required when spec_decode_mode='dflash'")

        work_dir.mkdir(exist_ok=True, parents=True)
        hf_model_path = str(existing_meta.get("hf_model_path") or hf_model_path)
        context_length = int(existing_meta.get("max_context_tokens", self.config.context_length))
        wrap_cfg = existing_meta.get("wrap_cfg", {})
        existing_batch_size = int(wrap_cfg.get("batch_size", 1)) if isinstance(wrap_cfg, dict) else 1
        _validate_spec_decode_batch_size(spec_decode_mode, existing_batch_size)
        if isinstance(wrap_cfg, dict) and wrap_cfg.get("input_sequence_length") is not None:
            self.config.input_sequence_length = int(wrap_cfg["input_sequence_length"])
        self.config.context_length = context_length

        quant_scheme = existing_meta.get("quant_scheme", {})
        quant_type = (
            quant_scheme.get("quant_type", self.config.quant_scheme.quant_type)
            if isinstance(quant_scheme, dict)
            else self.config.quant_scheme.quant_type
        )
        target_device = self.config.quant_scheme.target_device
        model_name = str(existing_meta.get("model_name") or Path(hf_model_path).name)
        prefix = f"{model_name}-{target_device}-{context_length // 1024}k-{quant_type}"
        verify_length = int(getattr(self.config, "num_draft_tokens", 4)) + 1

        logger.info(f"Draft-only dense Qwen3.5 export: reusing target work_dir={existing_work_dir_path}")
        prefill_onnx = self._resolve_existing_meta_path(
            existing_work_dir_path, existing_meta, "prefill_onnx", "prefill_onnx_file"
        )
        decode_onnx = self._resolve_existing_meta_path(
            existing_work_dir_path, existing_meta, "decode_onnx", "decode_onnx_file"
        )
        hf_config = self._resolve_existing_meta_path(existing_work_dir_path, existing_meta, "hf_config")
        token_embedding = self._resolve_existing_meta_path(
            existing_work_dir_path, existing_meta, "token_embedding_file"
        )
        logger.info(f"Reused target prefill ONNX: {prefill_onnx}")
        logger.info(f"Reused target decode ONNX: {decode_onnx}")

        if spec_decode_mode == "mtp":
            draft_onnx_files = self._export_mtp_draft_model(
                hf_model_path=hf_model_path,
                work_dir=work_dir,
                prefix=prefix,
                context_length=context_length,
                verify_length=verify_length,
                target_device=target_device,
            )
            hidden_output_name = "post_norm_hidden"
        else:
            draft_onnx_files = self._export_dflash_draft_model(
                hf_model_path=hf_model_path,
                dflash_model_dir=self.config.dflash_model_dir,
                work_dir=work_dir,
                prefix=prefix,
                context_length=context_length,
                verify_length=verify_length,
                target_device=target_device,
            )
            hidden_output_name = "target_hidden"

        new_meta = copy.deepcopy(existing_meta)
        new_meta.update(
            dict(
                create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                hf_model_path=hf_model_path,
                prefill_onnx=str(prefill_onnx),
                decode_onnx=str(decode_onnx),
                hf_config=str(hf_config),
                token_embedding_file=str(token_embedding),
            )
        )
        self._add_spec_decode_meta(
            meta_info=new_meta,
            spec_decode_mode=spec_decode_mode,
            draft_onnx_files=draft_onnx_files,
            work_dir=work_dir,
            hidden_output_name=hidden_output_name,
            verify_length=verify_length,
        )

        with (work_dir / "meta.json").open("w", encoding="utf-8") as fout:
            json.dump(new_meta, fout, ensure_ascii=False, indent=4)
        with (work_dir / "export_meta_info.json").open("w", encoding="utf-8") as fout:
            json.dump(new_meta, fout, ensure_ascii=False, indent=4)
        logger.info(f"Draft-only dense Qwen3.5 export done. New artifacts in: {work_dir}")

    @classmethod
    def convert(cls, hf_model_path: str, config: Qwen3_5ConvertConfig, output_dir: str):
        cls(config)._convert(hf_model_path, output_dir)
