# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from xhquant.api import CacheTensor

from ....datasets.preprocess.mix_search_preprocess import ms_data_preprocess
from ..base_converter import BaseConverter
from ..builder import wrap_llm_model
from ..qwen3_legacy import Qwen3LegacyConvertConfig
from ..qwen3_legacy.qwen3_converter import Qwen3LegacyConverterXH2a as _BaseQwen3LegacyConverterXH2a
from xhquant.api import (  # type: ignore # isort:skip
    Config,
    ConfigDict,
    DeviceType,
    PrecisionMode,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
    is_ssfp_quant_config,
)

__all__ = ["Qwen3LegacyConverterXH2a"]


class Qwen3LegacyConverterXH2a(_BaseQwen3LegacyConverterXH2a):
    target_device = DeviceType.XH2a

    def __init__(self, config: Qwen3LegacyConvertConfig):
        super().__init__(config)
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        raise RuntimeError("FunAudioChat converter expects an already loaded `model.language_model`.")

    def _convert(self, native_model: Qwen3ForCausalLM, output_dir: str):
        logger = get_root_logger()
        config = self.config

        assert isinstance(native_model, Qwen3ForCausalLM), f"Expected Qwen3ForCausalLM, but got {type(native_model)}"

        if native_model.config.tie_word_embeddings:  # type: ignore
            old_torchscript = native_model.config.torchscript  # type: ignore
            native_model.config.torchscript = True  # type: ignore
            native_model.tie_weights()  # type: ignore
            native_model.config.tie_word_embeddings = False  # type: ignore
            native_model.config.torchscript = old_torchscript  # type: ignore

        resume_from = self.config.quant_weight
        if resume_from is not None:
            self.load_quant_weight(resume_from, native_model)
        lm_head = native_model.lm_head
        if not hasattr(lm_head, "quant_weight"):
            config.quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"

        model_name = Path(self.hf_model_path).name if self.hf_model_path else "funaudiochat_llm"
        target_device = config.quant_scheme.target_device
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length
        assert target_device == DeviceType.XH2a, f"Only support convert to XH2a, but got {target_device}"
        quant_type = config.quant_scheme.quant_type
        quant_config = ConfigDict(create_quant_config(config.quant_scheme))

        meta_info: Dict[str, Any] = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )
        meta_info["device"] = str(target_device)
        meta_info["model_name"] = model_name
        meta_info["hf_model_path"] = self.hf_model_path
        meta_info["quant_scheme"] = config.quant_scheme.to_dict()
        meta_info["quant_weight"] = resume_from

        work_dir = Path(output_dir)
        hf_config_dir = Path(work_dir) / "hf_config"
        hf_config_dir.mkdir(exist_ok=True, parents=True)
        if self.hf_model_path:
            hf_config_files = [
                "config.json",
                "generation_config.json",
                "tokenizer_config.json",
                "vocab.json",
                "tokenizer.json",
                "chat_template.jinja",
            ]
            for cfg_file in hf_config_files:
                src_file = Path(self.hf_model_path) / cfg_file
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

        from ._model import register_wrap_modules as qwen3_register_wrap_modules

        qwen3_register_wrap_modules(native_model)

        wrap_cfg = Config(
            dict(
                max_sequence_length=context_length,
                input_sequence_length=input_sequence_length,
                use_cache=True,
                num_logits_to_keep=config.num_logits_to_keep,
                kv_cache=dict(
                    cache_axis=2,
                ),
            )
        )

        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        wraped_qwen_model = wrap_llm_model(native_model, wrap_cfg)
        num_hidden_layers = wraped_qwen_model.model.config.num_hidden_layers
        head_dim = wraped_qwen_model.model.layers[0].self_attn.head_dim
        num_decoder_layers = num_hidden_layers
        kv_cache_shape = [
            1,
            wraped_qwen_model.model.config.num_key_value_heads,
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

        input_ids = []
        current_input_length = []
        for _ in range(1):
            input_id = torch.randint(0, 1000, (input_sequence_length,), dtype=torch.long)
            seq_length = input_id.shape[0]
            current_input_length.append(seq_length)
            input_ids.append(input_id.unsqueeze(0))

        input_ids_t = torch.cat(input_ids, dim=0)
        inputs_embeds = token_embedding(input_ids_t)
        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor(current_input_length, dtype=torch.int32)
        attention_mask = torch.zeros(
            (1, 1, input_sequence_length, input_sequence_length),
            dtype=torch.float16,
        )

        inputs = (
            inputs_embeds,
            attention_mask,
            past_seq_length_t,
            current_input_length_t,
            past_key_caches,
            past_value_caches,
        )
        input_names = [
            "inputs_embeds",
            "attention_mask",
            "past_seq_length",
            "current_input_length",
        ]
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_value_cache_{layer_idx}")
        output_names = ["logits", "last_hidden_state"]

        prefix = f"{model_name}-{target_device}-{context_length // 1024}k-{quant_type}"
        prefill_onnx_file = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))

        logger.info("********************* start export prefill model *********************")

        if os.path.exists(prefill_onnx_file):
            logger.warning(f"{prefill_onnx_file} already exists, skip export")
        else:
            quanted_model = convert_fx_model_to_quanted_model(
                wraped_qwen_model,
                inputs,
                target_device,
                quant_config=quant_config,
            )
            if self.config.mix_search is not None:
                quanted_model = quanted_model.cuda()
                quanted_model.enable_fast_precision_mode()
                from xhquant.mix_precision.mix_precision import MixPrecisionSearch

                with open(self.config.mix_search, "r") as f:
                    ms_cfg = yaml.safe_load(f)
                ms = MixPrecisionSearch(quanted_model, ms_cfg)
                if not self.hf_model_path:
                    raise RuntimeError("mix_search requires hf_model_path for tokenizer loading.")
                tokenizer = AutoTokenizer.from_pretrained(self.hf_model_path)
                gpu_loader, label_dataloader = ms_data_preprocess(
                    wraped_qwen_model, tokenizer, past_key_caches, past_value_caches
                )

                precision_mode = PrecisionMode.FAST
                ms.search(gpu_loader, precision_mode, label_dataloader, wrap_cfg)
                quanted_model.enable_aligned_precision_mode()
                quanted_model = quanted_model.to("cpu")
            input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(quanted_model, inputs, str(prefill_onnx_file), input_names, output_names)
            logger.info(f"Export Prefill model to {prefill_onnx_file}")

        logger.info("********************* start export decode model *********************")
        decode_inputs = (
            inputs_embeds[:, :1, :],
            attention_mask[:, :, :1, :],
            past_seq_length_t,
            torch.ones_like(current_input_length_t),
            past_key_caches,
            past_value_caches,
        )

        wrap_cfg.input_sequence_length = 1
        quanted_model.update_cfg(wrap_cfg)

        decode_onnx_file = work_dir / "hmonnx" / "decode" / f"{prefix}_decoder.onnx"
        decode_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["decode_onnx"] = str(decode_onnx_file.relative_to(work_dir))
        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        if os.path.exists(decode_onnx_file):
            logger.warning(f"{decode_onnx_file} already exists, skip export")
        else:
            convert_quanted_model_to_hmonnx(
                quanted_model, decode_inputs, str(decode_onnx_file), input_names, output_names
            )
            logger.info(f"Export decode model to {decode_onnx_file}")
        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    @classmethod
    def convert(cls, native_model: Qwen3ForCausalLM, config: Qwen3LegacyConvertConfig, output_dir: str):
        if not isinstance(native_model, Qwen3ForCausalLM):
            raise TypeError(f"Expected Qwen3ForCausalLM, but got {type(native_model)}")

        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)

        converter = cls(config)
        model_path = getattr(native_model.config, "_name_or_path", None)
        if model_path and Path(model_path).exists():
            converter.hf_model_path = model_path

        if is_ssfp and converter.hf_model_path:
            hf_config = AutoConfig.from_pretrained(converter.hf_model_path, trust_remote_code=True)
            if not hasattr(hf_config, "quantization_config"):
                assert config.quant_weight is not None and Path(config.quant_weight).exists()

        converter._convert(native_model, output_dir)