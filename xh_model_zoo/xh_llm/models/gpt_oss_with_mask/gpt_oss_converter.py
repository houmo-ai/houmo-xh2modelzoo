# Copyright 2025 HOUMO AI
#
# File: gpt_oss_converter.py
# Description:
#   Gpt Oss Converter implementation.
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, GptOssForCausalLM
from xhquant.api import CacheTensor

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
from .gpt_oss_convert_config import GptOssWithMaskConvertConfig

from xhquant.api import (  # type: ignore # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    CacheTensor,
)


def aligned(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align


def _gen_mask_v2(x: torch.Tensor, valid_length, attention_max_length: int = -1):
    if isinstance(valid_length, int):
        valid_length = torch.tensor(valid_length).to(x.device)
    valid_length = valid_length.reshape(-1)
    bsz, nq, nk = x.size(0), x.size(-2), x.size(-1)
    masks = []
    for i in range(bsz):
        b_valid_length = int(valid_length[i].item())
        if attention_max_length > 0:
            b_valid_length = min(b_valid_length, attention_max_length - 1)
        attention_mask = torch.tril(
            torch.ones(nq, nk, dtype=torch.bool, device=x.device),
            diagonal=b_valid_length,
        ).logical_not()
        if attention_max_length > 0:
            sliding_window_mask = torch.tril(
                torch.ones_like(attention_mask, dtype=torch.bool),
                diagonal=b_valid_length - attention_max_length,
            )
            attention_mask = torch.where(sliding_window_mask, True, attention_mask)
        masks.append(attention_mask.unsqueeze(0).unsqueeze(0))
    return torch.cat(masks, dim=0)


def prepare_casual_mask(x: torch.Tensor, valid_length, attention_max_length: int):
    mask = _gen_mask_v2(x, valid_length, attention_max_length)
    attention_mask = torch.zeros_like(mask, dtype=x.dtype, device=x.device)
    return attention_mask.masked_fill(mask, torch.finfo(x.dtype).min)


class GptOssWithMaskConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: GptOssWithMaskConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        hf_config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)

        # 如果用户指定了 rope_max_length，在加载模型前更新 HF config 的
        # max_position_embeddings 和 rope_scaling.factor，
        # 使 RotaryEmbedding 在 __init__ 时即使用正确的 inv_freq
        rope_max_length = self.config.rope_max_length
        if rope_max_length is not None:
            logger = get_root_logger()
            old_max_pos = getattr(hf_config, "max_position_embeddings", None)
            if old_max_pos is not None and rope_max_length != old_max_pos:
                hf_config.max_position_embeddings = rope_max_length
                logger.info(
                    f"Updated HF config max_position_embeddings: {old_max_pos} -> {rope_max_length}"
                )
            rope_scaling = getattr(hf_config, "rope_scaling", None)
            if rope_scaling is not None and isinstance(rope_scaling, dict):
                orig = rope_scaling.get("original_max_position_embeddings", None)
                if orig is not None and orig > 0:
                    new_factor = rope_max_length / orig
                    old_factor = rope_scaling.get("factor", None)
                    rope_scaling["factor"] = float(new_factor)
                    hf_config.rope_scaling = rope_scaling
                    logger.info(
                        f"Updated HF config rope_scaling.factor: {old_factor} -> {new_factor} "
                        f"(rope_max_length={rope_max_length} / original_max_position_embeddings={orig})"
                    )

        native_model = AutoModelForCausalLM.from_pretrained(
            hf_model_dir, config=hf_config, **kwargs
        )
        native_model = self.dequantize_hf_model(native_model)
        assert isinstance(native_model, GptOssForCausalLM), (
            f"The model is not GptOssForCausalLM, but {type(native_model)}"
        )
        native_model: GptOssForCausalLM = native_model  # type: ignore

        self.hf_model_path = hf_model_dir
        return native_model

    def _convert(self, hf_model_path: str, output_dir: str):
        logger = get_root_logger()
        config = self.config

        native_model = self.load_hf_model(
            hf_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu"
        )
        config.num_experts_per_tok = native_model.config.num_experts_per_tok

        # 自动检测rope_max_length：优先使用用户配置，否则从HF config中推断
        if config.rope_max_length is None:
            hf_cfg = native_model.config
            rope_max_length = getattr(hf_cfg, "max_position_embeddings", None)
            if rope_max_length is not None:
                config.rope_max_length = rope_max_length
                logger.info(f"Auto-detected rope_max_length={config.rope_max_length} from HF config")

        # 确保rope_max_length不小于context_length
        if config.rope_max_length is not None and config.rope_max_length < config.context_length:
            logger.warning(
                f"rope_max_length({config.rope_max_length}) < context_length({config.context_length}), "
                f"auto adjusting rope_max_length to {config.context_length}"
            )
            config.rope_max_length = config.context_length

        # 融合GPTQ权重
        resume_from = self.config.quant_weight
        if resume_from is not None:
            self.load_quant_weight(resume_from, native_model)
        lm_head = native_model.lm_head
        if not hasattr(lm_head, "quant_weight"):
            config.quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"

        model_name = Path(hf_model_path).name
        target_device = config.quant_scheme.target_device
        batch_size = config.batch_size
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length
        sliding_window = config.sliding_window
        assert target_device == DeviceType.XH2a, f"Only support convert to XH2a, but got {target_device}"
        quant_type = config.quant_scheme.quant_type
        quant_config = create_quant_config(config.quant_scheme)
        quant_config = ConfigDict(quant_config)

        meta_info: Dict[str, Any] = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )
        meta_info["device"] = str(target_device)
        meta_info["model_name"] = model_name
        meta_info["hf_model_path"] = hf_model_path
        meta_info["quant_scheme"] = config.quant_scheme.to_dict()
        meta_info["quant_weight"] = resume_from

        work_dir = Path(output_dir)
        hf_config_dir = Path(work_dir) / "hf_config"
        hf_config_dir.mkdir(exist_ok=True, parents=True)
        hf_config_files = [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "tokenizer.json",
        ]
        for cfg_file in hf_config_files:
            src_file = Path(hf_model_path) / cfg_file
            dst_file = Path(hf_config_dir) / cfg_file
            if src_file.exists():
                shutil.copyfile(src_file, dst_file)
            else:
                logger.warning(f"{src_file} not exists, skip copy")
        meta_info["hf_config"] = str(hf_config_dir.relative_to(work_dir))

        token_embedding = native_model.model.get_input_embeddings()

        token_embedding_file = Path(work_dir) / "token_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file))
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        from ._model import register_wrap_modules as gpt_oss_register_wrap_modules  # noqa: F403, F401

        gpt_oss_register_wrap_modules(native_model)

        # 改写原模型, 以适合torch.fx导出
        wrap_cfg = Config(
            dict(
                batch_size=batch_size,
                max_sequence_length=context_length,  # 最大上下文长度
                input_sequence_length=input_sequence_length,  # prefill时输入的序列长度
                use_cache=True,
                num_logits_to_keep=config.num_logits_to_keep,
                sliding_window=config.sliding_window,
                kv_cache=dict(
                    cache_axis=2,
                ),
                enable_rope=True,
                num_experts_per_tok=config.num_experts_per_tok,
                rope_max_length=config.rope_max_length,
            )
        )

        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        wraped_gpt_oss_model = wrap_llm_model(native_model, wrap_cfg)
        # 设置kv cache
        num_hidden_layers = wraped_gpt_oss_model.model.config.num_hidden_layers
        head_dim = wraped_gpt_oss_model.model.layers[0].self_attn.head_dim
        num_decoder_layers = num_hidden_layers
        kv_cache_shape = [
            1,
            wraped_gpt_oss_model.model.config.num_key_value_heads,
            wrap_cfg.max_sequence_length,
            head_dim,
        ]
        meta_info["kv_cache"] = dict(
            shape=kv_cache_shape,
            num_decoder_layers=num_decoder_layers,
        )

        past_key_caches = []
        past_value_caches = []
        for _ in range(num_decoder_layers):
            past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
            past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))

        # 检查是否有 sliding window attention
        has_local_attention = False
        has_global_attention = True
        if sliding_window is not None and sliding_window > 0:
            # 检查模型层是否有 sliding_window 属性
            for layer in wraped_gpt_oss_model.model.layers:
                if hasattr(layer.self_attn, "sliding_window") and layer.self_attn.sliding_window > 0:
                    has_local_attention = True
                    break

        # 导出Prefill模型
        input_ids = []
        current_input_length = []
        for _ in range(1):
            input_id = torch.randint(0, 1000, (input_sequence_length,), dtype=torch.long)
            seq_length = input_id.shape[0]
            past_seq_length = 0
            current_input_length.append(seq_length)
            input_id = input_id.unsqueeze(0)
            input_ids.append(input_id)

        input_ids_t = torch.cat(input_ids, dim=0)
        inputs_embeds = token_embedding(input_ids_t)
        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor(current_input_length, dtype=torch.int32)

        # 准备 attention masks
        local_attention_mask = None
        global_attention_mask = None
        bz, nq = inputs_embeds.shape[:2]

        if has_global_attention:
            width = kv_cache_shape[2]
            x = torch.empty((bz, nq, width), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            global_attention_mask = prepare_casual_mask(x, past_seq_length_t, -1)

        if has_local_attention and sliding_window is not None and sliding_window > 0:
            local_window = sliding_window + nq - 1
            local_window = aligned(local_window, 16)
            x = torch.empty((bz, nq, local_window), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            local_attention_mask = prepare_casual_mask(x, past_seq_length_t, sliding_window)

        inputs = (
            inputs_embeds,
            past_seq_length_t,
            current_input_length_t,
            local_attention_mask,
            global_attention_mask,
            past_key_caches,
            past_value_caches,
        )
        input_names = [
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
        ]
        if local_attention_mask is not None:
            input_names.append("local_attention_mask")
        if global_attention_mask is not None:
            input_names.append("global_attention_mask")
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_value_cache_{layer_idx}")
        output_names = ["logits"]

        prefix = f"{model_name}-{target_device}-{context_length // 1024}k-{quant_type}"
        prefill_onnx_file = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))

        logger.info(f"********************* start export prefill model *********************")

        quanted_model = convert_fx_model_to_quanted_model(
            wraped_gpt_oss_model,
            inputs,
            target_device,
            quant_config=quant_config,
        )

        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(quanted_model, inputs, str(prefill_onnx_file), input_names, output_names)
        logger.info(f"Export Prefill model to {prefill_onnx_file}")

        logger.info(f"********************* start export decode model *********************")
        # 准备 decode 阶段的 attention masks
        decode_local_attention_mask = None
        decode_global_attention_mask = None
        decode_nq = 1
        decode_past_seq_length = input_sequence_length

        if has_global_attention:
            width = kv_cache_shape[2]
            x = torch.empty((bz, decode_nq, width), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            decode_global_attention_mask = prepare_casual_mask(
                x, torch.tensor([decode_past_seq_length], dtype=torch.int32), -1
            )

        if has_local_attention and sliding_window is not None and sliding_window > 0:
            local_window = sliding_window + decode_nq - 1
            local_window = aligned(local_window, 16)
            x = torch.empty((bz, decode_nq, local_window), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            decode_local_attention_mask = prepare_casual_mask(
                x, torch.tensor([decode_past_seq_length], dtype=torch.int32), sliding_window
            )

        decode_inputs = (
            inputs_embeds[:, :1, :],
            torch.tensor([decode_past_seq_length], dtype=torch.int32),
            torch.ones_like(current_input_length_t),
            decode_local_attention_mask,
            decode_global_attention_mask,
            past_key_caches,
            past_value_caches,
        )

        # 更新与input_sequence_length相关的Module
        wrap_cfg.input_sequence_length = 1
        quanted_model.update_cfg(wrap_cfg)

        decode_onnx_file = work_dir / "hmonnx" / "decode" / f"{prefix}_decoder.onnx"
        decode_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["decode_onnx"] = str(decode_onnx_file.relative_to(work_dir))
        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(quanted_model, decode_inputs, str(decode_onnx_file), input_names, output_names)

        logger.info(f"Export decode model to {decode_onnx_file}")
        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config: GptOssWithMaskConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp:
            hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
            if not hasattr(hf_config, "quantization_config"):
                assert config.quant_weight is not None and Path(config.quant_weight).exists()
        GptOssWithMaskConverterXH2a(config)._convert(hf_model_path, output_dir)
