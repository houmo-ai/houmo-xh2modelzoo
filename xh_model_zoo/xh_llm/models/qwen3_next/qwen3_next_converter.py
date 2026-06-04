# Copyright 2025 HOUMO AI
#
# File: qwen3_next_converter.py
# Description:
#   Qwen3 Next Converter implementation.
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

import json
import shutil
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM
from xhquant.api import CacheTensor

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
from .qwen3_next_convert_config import Qwen3NextConvertConfig

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




def _linear_split_conv_dims(linear_attn) -> tuple[int, int, int]:
    def _module_channels(module_name: str, fallback: int) -> int:
        module = getattr(linear_attn, module_name, None)
        if module is not None:
            if hasattr(module, "in_channels"):
                return int(module.in_channels)
            if hasattr(module, "out_features"):
                return int(module.out_features)
        return int(fallback)

    qk_fallback = int(linear_attn.head_k_dim) * int(linear_attn.num_v_heads)
    v_fallback = int(linear_attn.head_v_dim) * int(linear_attn.num_v_heads)
    return (
        _module_channels("conv1d_q", qk_fallback),
        _module_channels("conv1d_k", qk_fallback),
        _module_channels("conv1d_v", v_fallback),
    )

def _flatten_cache_outputs(self: nn.Module, *args, **kwargs):
    logits, conv_cache_out_list, recurrent_state_out_list = self._qwen3_next_original_forward(*args, **kwargs)
    outputs: List[torch.Tensor] = [logits]
    if conv_cache_out_list is not None:
        outputs.extend(list(conv_cache_out_list))
    if recurrent_state_out_list is not None:
        outputs.extend(list(recurrent_state_out_list))
    return tuple(outputs)


_LINEAR_CONV_CACHE_BRANCHES = ("q", "k", "v")


class Qwen3NextConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: Qwen3NextConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
        native_model = AutoModelForCausalLM.from_pretrained(
            hf_model_dir,
            trust_remote_code=True,
            **kwargs,
        )
        native_model = self.dequantize_hf_model(native_model)
        assert isinstance(native_model, Qwen3NextForCausalLM), (
            f"The model is not Qwen3NextForCausalLM, but {type(native_model)}"
        )
        native_model = native_model.eval()

        if native_model.config.tie_word_embeddings:  # type: ignore[attr-defined]
            old_torchscript = native_model.config.torchscript  # type: ignore[attr-defined]
            native_model.config.torchscript = True  # type: ignore[attr-defined]
            native_model.tie_weights()  # type: ignore[attr-defined]
            native_model.config.tie_word_embeddings = False  # type: ignore[attr-defined]
            native_model.config.torchscript = old_torchscript  # type: ignore[attr-defined]

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

    def _build_quant_config(self, wraped_qwen_model: nn.Module):
        quant_config = ConfigDict(create_quant_config(self.config.quant_scheme))
        quant_config.setdefault("inputs", {})
        quant_config.setdefault("ops_cfg", {})
        quant_config["ops_cfg"]["Normalize"] = dict(force_fp32=self.config.normalize_force_fp32)
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
        for name, module in wraped_qwen_model.named_modules():
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

        return quant_config

    def _prepare_wrap_model(self, native_model: Qwen3NextForCausalLM):
        from ._model import register_wrap_modules as qwen3_next_register_wrap_modules  # noqa: F401

        qwen3_next_register_wrap_modules()
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
                alpha_scaling_layers=list(self.config.alpha_scaling_layers),
                chunk_inverse_alpha=self.config.chunk_inverse_alpha,
                normalize_force_fp32=self.config.normalize_force_fp32,
                split_conv_cache=self.config.split_conv_cache,
                use_manual_depthwise_conv1d=self.config.use_manual_depthwise_conv1d,
                fuse_gdr_ops=self.config.fuse_gdr_ops,
                kv_cache=dict(
                    cache_axis=2,
                ),
            )
        )
        wraped_qwen_model = wrap_llm_model(native_model, wrap_cfg)
        if not hasattr(wraped_qwen_model, "_qwen3_next_original_forward"):
            wraped_qwen_model._qwen3_next_original_forward = wraped_qwen_model.forward
            wraped_qwen_model.forward = types.MethodType(_flatten_cache_outputs, wraped_qwen_model)
        return wraped_qwen_model, wrap_cfg

    def _build_cache_inputs(
        self,
        wraped_qwen_model: nn.Module,
        context_length: int,
    ):
        layer_types = list(wraped_qwen_model.model.config.layer_types)
        full_attention_layer_indices = [i for i, layer_type in enumerate(layer_types) if layer_type == "full_attention"]
        linear_attention_layer_indices = [
            i for i, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
        ]

        head_dim = wraped_qwen_model.config.head_dim
        kv_cache_shape = [
            self.config.batch_size,
            wraped_qwen_model.model.config.num_key_value_heads,
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
            linear_attn = wraped_qwen_model.model.layers[layer_idx].linear_attn
            if self.config.split_conv_cache:
                cache_dtype = (
                    linear_attn.conv1d_q.weight.dtype
                    if hasattr(linear_attn, "conv1d_q")
                    else linear_attn.conv1d.weight.dtype
                )
                q_dim, k_dim, v_dim = _linear_split_conv_dims(linear_attn)
                conv_shapes = [
                    [self.config.batch_size, q_dim, linear_attn.conv_kernel_size],
                    [self.config.batch_size, k_dim, linear_attn.conv_kernel_size],
                    [self.config.batch_size, v_dim, linear_attn.conv_kernel_size],
                ]
                for shape in conv_shapes:
                    past_conv_caches.append(CacheTensor(torch.zeros(shape, dtype=cache_dtype)))
            else:
                cache_dtype = linear_attn.conv1d.weight.dtype
                conv_shape = [
                    self.config.batch_size,
                    linear_attn.conv_dim,
                    linear_attn.conv_kernel_size,
                ]
                past_conv_caches.append(CacheTensor(torch.zeros(conv_shape, dtype=cache_dtype)))
            recurrent_shape = [
                self.config.batch_size,
                linear_attn.num_v_heads,
                linear_attn.head_k_dim,
                linear_attn.head_v_dim,
            ]
            past_recurrent_states.append(CacheTensor(torch.zeros(recurrent_shape, dtype=cache_dtype)))
            meta = dict(
                layer_idx=layer_idx,
                recurrent_shape=recurrent_shape,
            )
            if self.config.split_conv_cache:
                meta["conv_shapes"] = conv_shapes
            else:
                meta["conv_shape"] = conv_shape
            linear_cache_meta.append(meta)

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

        meta_info: Dict[str, Any] = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            device=str(target_device),
            model_name=model_name,
            hf_model_path=hf_model_path,
            architecture="Qwen3NextForCausalLM",
            quant_scheme=self.config.quant_scheme.to_dict(),
            quant_weight=self.config.quant_weight,
            source_quant_method=_extract_quant_method(native_model.config),
            pad_token_id=native_model.config.eos_token_id,
            max_context_tokens=context_length,
        )

        hf_config_dir = self._copy_hf_config(hf_model_path, work_dir)
        meta_info["hf_config"] = str(hf_config_dir.relative_to(work_dir))

        token_embedding = native_model.model.get_input_embeddings()
        token_embedding_file = work_dir / "token_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file))
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        wraped_qwen_model, wrap_cfg = self._prepare_wrap_model(native_model)
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
        ) = self._build_cache_inputs(wraped_qwen_model, context_length)

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

        inputs = (
            inputs_embeds,
            past_seq_length_t,
            current_input_length_t,
            linear_attn_mask_t,
            past_key_caches,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
        )

        input_names = [
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "linear_attn_mask",
        ]
        for layer_idx in range(len(full_attention_layer_indices)):
            input_names.append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(len(full_attention_layer_indices)):
            input_names.append(f"past_value_cache_{layer_idx}")
        for layer_idx in range(len(linear_attention_layer_indices)):
            if self.config.split_conv_cache:
                for branch in _LINEAR_CONV_CACHE_BRANCHES:
                    input_names.append(f"past_conv_cache_{branch}_{layer_idx}")
            else:
                input_names.append(f"past_conv_cache_{layer_idx}")
        for layer_idx in range(len(linear_attention_layer_indices)):
            input_names.append(f"past_recurrent_state_{layer_idx}")

        output_names = ["logits"]
        for layer_idx in range(len(linear_attention_layer_indices)):
            if self.config.split_conv_cache:
                for branch in _LINEAR_CONV_CACHE_BRANCHES:
                    output_names.append(f"conv_cache_out_{branch}_{layer_idx}")
            else:
                output_names.append(f"conv_cache_out_{layer_idx}")
        for layer_idx in range(len(linear_attention_layer_indices)):
            output_names.append(f"recurrent_state_out_{layer_idx}")

        quant_config = self._build_quant_config(wraped_qwen_model)
        quanted_model = convert_fx_model_to_quanted_model(
            wraped_qwen_model,
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
            quanted_model,
            inputs,
            str(prefill_onnx_file),
            BaseConverter.xh1_hmonnx_compatible(input_names),
            output_names,
        )
        logger.info(f"Export Prefill model to {prefill_onnx_file}")

        wrap_cfg.input_sequence_length = 1
        quanted_model.update_cfg(wrap_cfg)

        decode_inputs = (
            inputs_embeds[:, :1, :],
            past_seq_length_t,
            torch.ones_like(current_input_length_t),
            torch.ones(self.config.batch_size, 1, dtype=inputs_embeds.dtype),
            past_key_caches,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
        )
        decode_onnx_file = work_dir / "hmonnx" / "decode" / f"{prefix}_decoder.onnx"
        decode_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["decode_onnx"] = str(decode_onnx_file.relative_to(work_dir))
        logger.info("********************* start export decode model *********************")
        convert_quanted_model_to_hmonnx(
            quanted_model,
            decode_inputs,
            str(decode_onnx_file),
            BaseConverter.xh1_hmonnx_compatible(input_names),
            output_names,
        )
        logger.info(f"Export decode model to {decode_onnx_file}")

        with open(work_dir / "meta.json", "w", encoding="utf-8") as fout:
            json.dump(meta_info, fout, ensure_ascii=False, indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config: Qwen3NextConvertConfig, output_dir: str):
        cls(config)._convert(hf_model_path, output_dir)
