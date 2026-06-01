# Copyright 2025 HOUMO AI
#
# File: deepseekv2_converter.py
# Description:
#   Deepseekv2 Converter implementation.
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

from pyparsing import str_type
import torch
import torch.nn as nn
import yaml
from torch import Tensor
import time
from functools import cached_property
from typing import List,Optional,Dict,Any
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
)
from xh_model_zoo.xh_llm.llm_converter import LLMConverter
from xh_model_zoo.xh_llm.llm_convert_config import LLMConvertConfig
from dataclasses import dataclass
from pathlib import Path
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
from transformers.utils.quantization_config import QuantizationMethod
from xhquant.utils import TimeProfiler
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.datasets.preprocess.mix_search_preprocess import (
    ms_data_preprocess,
)
import data
from dataclasses import dataclass, field
import shutil,json
from copy import deepcopy
from ..base_converter import BaseConverter, HFTransfromersConverter
from xhquant.api import (  # type: ignore # isort:skip
    Config,
    HMONNXGoldenInference,
    PrecisionMode,
    QuantGraph,
    CacheTensor,
    DeviceType,
    ConfigDict,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    CacheTensor,
)
from .modeling_deepseekv2 import DeepseekV2ForCausalLM

@dataclass
class DeepseekV2ConverterConfig(LLMConvertConfig):
    mix_search: Optional[str] = None
    num_logits_to_keep: Optional[int] = 1
    




class DeepSeekV2ConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: DeepseekV2ConverterConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
        assert not hasattr(config, "quantization_config")
        native_model = DeepseekV2ForCausalLM.from_pretrained(hf_model_dir, **kwargs)
        assert not hasattr(native_model, "hf_quantizer")
        assert isinstance(
            native_model, DeepseekV2ForCausalLM
        ), f"The model is not Qwen2ForCausalLM, but {type(native_model)}"
        native_model: DeepseekV2ForCausalLM = native_model  # type: ignore

        if native_model.config.tie_word_embeddings:  # type: ignore
            old_torchscript = native_model.config.torchscript  # type: ignore
            native_model.config.torchscript = True  # type: ignore
            native_model.tie_weights()  # type: ignore
            native_model.config.tie_word_embeddings = False  # type: ignore
            native_model.config.torchscript = old_torchscript  # type: ignore

        self.hf_model_path = hf_model_dir
        return native_model

    def _convert(self, hf_model_path: str, output_dir: str):
        logger = get_root_logger()
        config = self.config

        native_model = self.load_hf_model(
            hf_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu"
        )

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

        from ._model_deepseekv2 import register_wrap_modules as deepseekv2_register_wrap_modules  # noqa: F403, F401

        deepseekv2_register_wrap_modules(native_model)

        # 改写原模型, 以适合torch.fx导出
        wrap_cfg = Config(
            dict(
                batch_size=batch_size,
                max_sequence_length=context_length,  # 最大上下文长度
                input_sequence_length=input_sequence_length,  # prefill时输入的序列长度
                use_cache=True,
                num_logits_to_keep=config.num_logits_to_keep,
                kv_cache=dict(
                    cache_axis=2,
                ),
            )
        )

        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        wraped_model = wrap_llm_model(native_model, wrap_cfg)

        # self.test_model(wraped_model)

        # 设置kv cache
        num_hidden_layers = wraped_model.model.config.num_hidden_layers
        head_dim = wraped_model.model.layers[0].self_attn.head_dim
        # pad_token_id = wraped_model.config.eos_token_id

        # head_dim = wraped_model.model.config.hidden_size // wraped_model.model.config.num_attention_heads
        # num_hidden_layers = wraped_model.model.config.num_hidden_layers
        num_decoder_layers = num_hidden_layers
        kv_cache_shape = [
            1, 
            wraped_model.model.config.num_key_value_heads,
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
        # 导出Prefill模型
        input_ids = []
        current_input_length = []
        position_ids = []
        for _ in range(1):
            input_id = torch.randint(0, 1000, (input_sequence_length,), dtype=torch.long)
            seq_length = input_id.shape[0]
            past_seq_length = 0
            position_id = torch.arange(past_seq_length, past_seq_length + seq_length, dtype=torch.long)
            current_input_length.append(seq_length)

            input_id = input_id.unsqueeze(0)
            position_id = position_id.unsqueeze(0)
            position_ids.append(position_id)
            input_ids.append(input_id)

        input_ids_t = torch.cat(input_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)

        inputs_embeds = token_embedding(input_ids_t)
        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor(current_input_length, dtype=torch.int32)

        inputs = (
            inputs_embeds,
            past_seq_length_t,
            current_input_length_t,
            # position_ids,
            past_key_caches,
            past_value_caches,
        )
        input_names = [
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            # "position_ids",
        ]
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_value_cache_{layer_idx}")
        output_names = ["logits"]

        prefix = f"{model_name}-{target_device}-{context_length//1024}k-{quant_type}"
        prefill_onnx_file = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))

        logger.info(f"********************* start export prefill model *********************")

        quanted_model = convert_fx_model_to_quanted_model(
            wraped_model,
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
            tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
            # data preprocess
            gpu_loader, label_dataloader = ms_data_preprocess(
                wraped_model, tokenizer, past_key_caches, past_value_caches
            )

            precision_mode = PrecisionMode.FAST
            ms.search(gpu_loader, precision_mode, label_dataloader, wrap_cfg)
            quanted_model.enable_aligned_precision_mode()  # adjust precision mode
            quanted_model = quanted_model.to("cpu")
        # quant_info_onnx_file = str(Path(output_dir) / "quant_info.onnx")
        # quanted_model.dump_quant_info_to_onnx(quant_info_onnx_file)
        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(quanted_model, inputs, str(prefill_onnx_file), input_names, output_names)
        logger.info(f"Export Prefill model to {prefill_onnx_file}")

        logger.info(f"********************* start export decode model *********************")
        decode_inputs = (
            inputs_embeds[:, :1, :],
            past_seq_length_t,
            torch.ones_like(current_input_length_t),
            # position_ids[:, :1],
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
    def convert(cls, hf_model_path: str, config: DeepseekV2ConverterConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        # if is_ssfp:
        #     assert config.quant_weight is not None and Path(config.quant_weight).exists()
        DeepSeekV2ConverterXH2a(config)._convert(hf_model_path, output_dir)



    def test_model(self, model, **kwargs):
        prompt = "<image>\nFree OCR. "
        # prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        image_file = 'examples/llm/deepseek_ocr/data/img1.png'
        output_path = 'examples/llm/deepseek_ocr/data'
        hf_model_path = self.hf_model_path
        from xh_model_zoo.xh_llm.models.deepseek_ocr.modeling_deepseekocr import DeepseekOCRForCausalLM
        native_model = DeepseekOCRForCausalLM.from_pretrained(hf_model_path)
        native_model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
        token_embedding = native_model.model.get_input_embeddings()
        num_hidden_layers = native_model.model.config.num_hidden_layers
        kv_cache_shape = [
            1,
            native_model.model.config.num_key_value_heads,
            8192,
            128,
        ]
        input_data: List[torch.Tensor] = []
        original_forward = native_model.model._forward

        def forward_with_hook(*args, **kwargs):
            input_data.append(kwargs['inputs_embeds'])
            raise Exception("Stop Forwarding to get the first llm input")
            return original_forward(*args, **kwargs)

        native_model.model._forward = forward_with_hook
        with torch.no_grad():
            try:
                native_model.infer(tokenizer, 
                            prompt=prompt, 
                            image_file=image_file, 
                            output_path = output_path, 
                            base_size = 512, 
                            image_size = 512, 
                            crop_mode=False, 
                            save_results = True, 
                            test_compress = True)
            except:
                print("搜集到了第一个llm输入数据")

        native_model.model._forward = original_forward
        del native_model



        past_key_caches = []
        past_value_caches = []
        for _ in range(num_hidden_layers):
            past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
            past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
        
        input_ids = []
        current_input_length = []
        position_ids = []
        for _ in range(1):
            input_id = torch.randint(0, 1000, (256,), dtype=torch.long)
            seq_length = input_id.shape[0]
            past_seq_length = 0
            position_id = torch.arange(past_seq_length, past_seq_length + seq_length, dtype=torch.long)
            current_input_length.append(seq_length)

            input_id = input_id.unsqueeze(0)
            position_id = position_id.unsqueeze(0)
            position_ids.append(position_id)
            input_ids.append(input_id)

        input_ids_t = torch.cat(input_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)

        inputs_embeds = torch.nn.functional.pad(
            input_data[0],
            pad=(0, 0, 256-input_data[0].shape[1], 0, 0, 0),  # 对应：dim2(0,0) | dim1(pad_len,0) | dim0(0,0)
            value=0.0
        )
        # inputs_embeds = input_data[0]
        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor(inputs_embeds.shape[1], dtype=torch.int32)

        inputs = (
            inputs_embeds.half(),
            past_seq_length_t,
            current_input_length_t,
            # position_ids,
            past_key_caches,
            past_value_caches,
        )

        with TimeProfiler("test_convert_model", get_root_logger()):
            outputs = model.cuda().half()(*inputs)
            print("convert model test pass")