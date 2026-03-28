# Copyright 2025 HOUMO AI
#
# File: qwen3_5_moe_converter.py
# Description:
#   Qwen3.5-MoE Converter implementation.
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
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)
from xhquant.api import CacheTensor

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
from .qwen3_5_moe_convert_config import Qwen3_5MoeConvertConfig

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
    logits, conv_cache_out_list, recurrent_state_out_list = self._qwen3_5_moe_original_forward(*args, **kwargs)
    outputs: List[torch.Tensor] = [logits]
    if conv_cache_out_list is not None:
        outputs.extend(list(conv_cache_out_list))
    if recurrent_state_out_list is not None:
        outputs.extend(list(recurrent_state_out_list))
    return tuple(outputs)


def _get_text_model(native_model):
    """Get the text model from either ForConditionalGeneration or ForCausalLM."""
    if isinstance(native_model, Qwen3_5MoeForConditionalGeneration):
        return native_model.model.language_model
    elif isinstance(native_model, Qwen3_5MoeForCausalLM):
        return native_model.model
    else:
        raise ValueError(f"Unsupported model type: {type(native_model)}")


def _get_text_config(native_model):
    """Get text_config from the model config."""
    config = native_model.config
    if hasattr(config, "text_config"):
        return config.text_config
    return config


class Qwen3_5MoeConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: Qwen3_5MoeConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None

    @staticmethod
    def _is_gptqmodel_checkpoint(hf_model_dir: str) -> bool:
        """Detect gptqmodel-format checkpoints (checkpoint_format == 'gptq' in config)."""
        cfg_path = Path(hf_model_dir) / "config.json"
        if not cfg_path.exists():
            return False
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            qc = cfg.get("quantization_config", {})
            return isinstance(qc, dict) and qc.get("checkpoint_format") == "gptq"
        except Exception:
            return False

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        logger = get_root_logger()

        if self._is_gptqmodel_checkpoint(hf_model_dir):
            # gptqmodel-format GPTQ checkpoints store weights as packed qweight/scales/qzeros.
            # AutoModelForCausalLM.from_pretrained() treats those as UNEXPECTED keys and
            # leaves Linear weights uninitialized, producing garbage outputs.
            # GPTQModel.load() properly unpacks and dequantizes back to float16 nn.Linear.
            try:
                from gptqmodel import GPTQModel  # type: ignore

                logger.info(
                    f"Detected gptqmodel checkpoint; using GPTQModel.load() for dequantization: {hf_model_dir}"
                )
                torch_dtype = kwargs.get("torch_dtype", torch.float16)
                qmodel = GPTQModel.load(
                    hf_model_dir,
                    device="cpu",
                    dtype=torch_dtype,
                )
                # GPTQModel.load() wraps the HF model in a BaseQModel; extract the inner model.
                # The inner model is the properly dequantized PreTrainedModel (nn.Linear weights).
                from gptqmodel.models.base import BaseQModel  # type: ignore

                if isinstance(qmodel, BaseQModel):
                    native_model = qmodel.model
                    logger.info(f"Extracted inner HF model: {type(native_model).__name__}")
                else:
                    native_model = qmodel
                # Clear quantization metadata so downstream dequantize_hf_model() is a no-op.
                if hasattr(native_model, "config"):
                    native_model.config.quantization_config = None
                    if hasattr(native_model.config, "quantization_method"):
                        native_model.config.quantization_method = None
                    if hasattr(native_model, "quantization_method"):
                        native_model.quantization_method = None
            except ImportError:
                logger.warning("gptqmodel not available; falling back to AutoModelForCausalLM")
                native_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_dir,
                    trust_remote_code=True,
                    **kwargs,
                )
                native_model = self.dequantize_hf_model(native_model)
        else:
            native_model = AutoModelForCausalLM.from_pretrained(
                hf_model_dir,
                trust_remote_code=True,
                **kwargs,
            )
            native_model = self.dequantize_hf_model(native_model)
        assert isinstance(native_model, (Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeForCausalLM)), (
            f"Expected Qwen3_5MoeForConditionalGeneration or Qwen3_5MoeForCausalLM, got {type(native_model)}"
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

        return quant_config

    def _prepare_wrap_model(self, native_model):
        from ._moe_model import register_wrap_modules as qwen3_5_moe_register_wrap_modules

        qwen3_5_moe_register_wrap_modules()
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
                kv_cache=dict(
                    cache_axis=2,
                ),
            )
        )
        wraped_model = wrap_llm_model(native_model, wrap_cfg)
        if not hasattr(wraped_model, "_qwen3_5_moe_original_forward"):
            wraped_model._qwen3_5_moe_original_forward = wraped_model.forward
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
        kv_cache_shape = [
            self.config.batch_size,
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
            cache_dtype = linear_attn.conv1d.weight.dtype
            conv_shape = [
                self.config.batch_size,
                linear_attn.conv_dim,
                linear_attn.conv_kernel_size,
            ]
            recurrent_shape = [
                self.config.batch_size,
                linear_attn.num_v_heads,
                linear_attn.head_k_dim,
                linear_attn.head_v_dim,
            ]
            past_conv_caches.append(CacheTensor(torch.zeros(conv_shape, dtype=cache_dtype)))
            past_recurrent_states.append(CacheTensor(torch.zeros(recurrent_shape, dtype=cache_dtype)))
            linear_cache_meta.append(
                dict(
                    layer_idx=layer_idx,
                    conv_shape=conv_shape,
                    recurrent_shape=recurrent_shape,
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
        # When quant_weight is a directory, it is a pre-quantized GPTQ model.
        # Load from that directory (dequantize_hf_model is called inside get_hf_model).
        # When quant_weight is a single file (.safetensors / .pt), load base model
        # first and then apply the weight file via load_quant_weight.
        quant_weight = self.config.quant_weight
        if quant_weight is not None and Path(quant_weight).is_dir():
            logger.info(f"Loading GPTQ model from: {quant_weight}")
            native_model = self.get_hf_model(
                quant_weight,
                torch_dtype=torch.float16,
                device_map="cpu",
            )
        else:
            native_model = self.get_hf_model(
                hf_model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
            )
            if quant_weight is not None:
                self.load_quant_weight(quant_weight, native_model)

        work_dir = Path(output_dir)
        work_dir.mkdir(exist_ok=True, parents=True)
        model_name = Path(hf_model_path).name
        context_length = self.config.context_length
        input_sequence_length = self.config.input_sequence_length
        quant_type = self.config.quant_scheme.quant_type
        target_device = self.config.quant_scheme.target_device
        text_config = _get_text_config(native_model)

        # Determine architecture string
        if isinstance(native_model, Qwen3_5MoeForConditionalGeneration):
            architecture_str = "Qwen3_5MoeForConditionalGeneration"
        else:
            architecture_str = "Qwen3_5MoeForCausalLM"

        meta_info: Dict[str, Any] = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            device=str(target_device),
            model_name=model_name,
            hf_model_path=hf_model_path,
            architecture=architecture_str,
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
        )

        # Build calibration input_ids: use real text if tokenizer is available
        # (essential for W4A8 quantization — random tokens give wrong activation scales).
        try:
            from transformers import AutoTokenizer as _AutoTokenizer

            _tok = _AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
            _calib_text = (
                "The Qwen3.5-MoE model is a large language model based on the Mixture-of-Experts "
                "architecture. It combines dense attention layers with sparse MoE feed-forward blocks. "
                "Each token is routed to a small subset of experts, reducing compute cost while "
                "maintaining high model capacity. The model supports both Chinese and English. "
                "这是一个基于专家混合架构的大语言模型，支持中英双语，并具备强大的推理能力。"
            )
            _enc = _tok(
                _calib_text,
                return_tensors="pt",
                truncation=True,
                max_length=input_sequence_length,
            )
            raw_ids = _enc["input_ids"][0]
            seq_actual = raw_ids.shape[0]
            if seq_actual < input_sequence_length:
                # Repeat + truncate to fill the window
                repeat_times = (input_sequence_length + seq_actual - 1) // seq_actual
                raw_ids = raw_ids.repeat(repeat_times)[:input_sequence_length]
            else:
                raw_ids = raw_ids[:input_sequence_length]
            input_ids_t = raw_ids.unsqueeze(0).expand(self.config.batch_size, -1)
            del _tok, _calib_text, _enc, raw_ids, seq_actual
        except Exception:
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

        inputs = (
            inputs_embeds,
            time_position_ids,
            hight_position_ids,
            width_position_ids,
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
            "time_position_ids",
            "hight_position_ids",
            "width_position_ids",
            "past_seq_length",
            "current_input_length",
            "linear_attn_mask",
        ]
        for layer_idx in range(len(full_attention_layer_indices)):
            input_names.append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(len(full_attention_layer_indices)):
            input_names.append(f"past_value_cache_{layer_idx}")
        for layer_idx in range(len(linear_attention_layer_indices)):
            input_names.append(f"past_conv_cache_{layer_idx}")
        for layer_idx in range(len(linear_attention_layer_indices)):
            input_names.append(f"past_recurrent_state_{layer_idx}")

        output_names = ["logits"]
        for layer_idx in range(len(linear_attention_layer_indices)):
            output_names.append(f"conv_cache_out_{layer_idx}")
        for layer_idx in range(len(linear_attention_layer_indices)):
            output_names.append(f"recurrent_state_out_{layer_idx}")

        # ── PREFILL quantisation & export ────────────────────────────────────
        # convert_fx_model_to_quanted_model modifies wraped_model in-place
        # (FrontendGraph is built on top of wraped_model; PTQ inserts QBaseModules
        # directly into the shared module tree). We deepcopy wraped_model for
        # prefill so the original remains clean for decode re-trace.
        quant_config = self._build_quant_config(wraped_model)
        wraped_model_prefill = copy.deepcopy(wraped_model)
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
            output_names,
        )
        logger.info(f"Export Prefill model to {prefill_onnx_file}")
        del quanted_prefill_model, wraped_model_prefill

        # ── DECODE quantisation & export ─────────────────────────────────────
        # Re-trace from clean wraped_model after updating to decode mode so that
        # baked-in FX slice constants use seq_len=1 (not the prefill value).
        wrap_cfg.input_sequence_length = 1

        def _apply_update_cfg(module):
            if hasattr(module, "_update_cfg"):
                module._update_cfg(wrap_cfg)

        wraped_model.apply(_apply_update_cfg)

        decode_position_ids = torch.zeros(self.config.batch_size, 1, dtype=torch.long)
        decode_inputs = (
            inputs_embeds[:, :1, :],
            decode_position_ids,
            decode_position_ids,
            decode_position_ids,
            past_seq_length_t,
            torch.ones_like(current_input_length_t),
            torch.ones(self.config.batch_size, 1, dtype=inputs_embeds.dtype),
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
            output_names,
        )
        logger.info(f"Export decode model to {decode_onnx_file}")

        with open(work_dir / "meta.json", "w", encoding="utf-8") as fout:
            json.dump(meta_info, fout, ensure_ascii=False, indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config: Qwen3_5MoeConvertConfig, output_dir: str):
        Qwen3_5MoeConverterXH2a(config)._convert(hf_model_path, output_dir)
