# Copyright 2025 HOUMO AI
#
# File: cogvlm2_vl_converter.py
# Description:
#   Cogvlm2 Vl Converter implementation.
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
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import onnx
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from xhquant.api import (
    convert_fx_model_to_hmonnx,
    convert_fx_model_to_quanted_model,
    convert_onnx_to_hmonnx,
    convert_quanted_model_to_hmonnx,
)

from ..base_converter import HFTransfromersConverter
from ..builder import wrap_llm_model
from .cogvlm2_convert_config import CogVLM2ConvertConfig

from xhquant.api import (  # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
)


class CogVLM2ConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: CogVLM2ConvertConfig):
        super().__init__()
        self.config: CogVLM2ConvertConfig = config
        self.work_dir = None
        self.pad_token_id = 0

    def load_hf_model(
        self, hf_model_path, **kwargs
    ) -> "transformers_modules.cogvlm2-llama3-chat-19B.modeling_cogvlm.CogVLMForCausalLM":
        native_model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",
        ).eval()
        return native_model

    def _export_vision(
        self,
        hf_model: "transformers_modules.cogvlm2-llama3-chat-19B.modeling_cogvlm.CogVLMForCausalLM",
        vision_hmonnx_file: str,
    ):
        logger = get_root_logger()
        logger.info("********************* start export vision model *********************")
        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls

        vision = hf_model.model.vision
        vision.cpu()
        wrap_cfg = ConfigDict(
            dict(
                image_size=self.config.vision_config.image_size,
                patch_size=self.config.vision_config.patch_size,
            )
        )
        wraped_vision_model = wrap_llm_model(vision, wrap_cfg)
        wraped_vision_model.cpu()

        out_hmonnx_file = vision_hmonnx_file
        target_device = self.config.quant_scheme.target_device

        quant_config = dict(
            inputs=dict(
                images=dict(
                    quantizer=dict(
                        qspec=dict(fake_dtype="float16"),
                    )
                ),
            )
        )
        quant_config = ConfigDict(quant_config)
        dummy_input = torch.randn(
            [
                1,
                3,
                wrap_cfg.image_size,
                wrap_cfg.image_size,
            ],
            dtype=torch.float16,
        )
        convert_fx_model_to_hmonnx(
            wraped_vision_model,
            [dummy_input],
            target_device,
            out_hmonnx_file,
            quant_config,
            input_names=[
                "images",
            ],
            output_names=[
                "features",
            ],
        )

        logger.info(f"Export vision model to {vision_hmonnx_file}")

    def _convert(self, hf_model_path: str, output_dir: str):
        logger = get_root_logger()
        config = self.config

        native_model = self.load_hf_model(
            hf_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu"
        )

        token_embedding = deepcopy(native_model.model.get_input_embeddings())

        model_name = Path(hf_model_path).name
        target_device = config.quant_scheme.target_device
        batch_size = config.batch_size
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length
        assert target_device == DeviceType.XH2a, f"Only support convert to XH2a, but got {target_device}"

        lm_head = native_model.lm_head
        if not hasattr(lm_head, "quant_weight"):
            config.quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"

        quant_config = create_quant_config(config.quant_scheme)

        work_dir = Path(output_dir)
        self.work_dir = output_dir

        quant_config = ConfigDict(quant_config)

        wrap_cfg = Config(
            dict(
                batch_size=batch_size,
                max_sequence_length=context_length,  # 最大上下文长度
                input_sequence_length=input_sequence_length,  # prefill时输入的序列长度
                use_cache=True,
                num_logits_to_keep=1,
                kv_cache=dict(
                    cache_axis=2,
                ),
                vision=dict(
                    image_size=config.vision_config.image_size,
                    patch_size=config.vision_config.patch_size,
                ),
            )
        )

        meta_info = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )
        meta_info["device"] = str(target_device)
        meta_info["model_name"] = model_name
        meta_info["quant_scheme"] = config.quant_scheme.to_dict()
        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        hf_config_dir = Path(work_dir) / "hf_config"
        hf_config_dir.mkdir(exist_ok=True, parents=True)
        hf_config_files = [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "configuration.json",
        ]
        for cfg_file in hf_config_files:
            shutil.copyfile(
                Path(hf_model_path) / cfg_file,
                Path(hf_config_dir) / cfg_file,
            )
        meta_info["hf_config"] = str(hf_config_dir.relative_to(work_dir))

        token_embedding = native_model.model.get_input_embeddings()
        token_embedding_file = Path(work_dir) / "token_embedding.pt"
        torch.save(token_embedding.cpu().state_dict(), str(token_embedding_file))
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        Path(work_dir / "hmonnx").mkdir(exist_ok=True, parents=True)

        quant_type = config.quant_scheme.quant_type
        prefix = f"{model_name}-{target_device}-{quant_type}"
        vision_onnx_file = str(work_dir / "hmonnx" / f"{prefix}_vision.onnx")

        # Export vision model
        if not Path(vision_onnx_file).exists():
            self._export_vision(native_model, vision_onnx_file)
        else:
            logger.info(f"Vision model already exists, skip export")

        meta_info["vision_onnx"] = str(Path(vision_onnx_file).relative_to(work_dir))

        cogvlm2_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
        hidden_size = cogvlm2_config.hidden_size

        feature_size = self.config.vision_config.image_size // self.config.vision_config.patch_size // 2
        vision_token_num = feature_size * feature_size + 2  # 2306

        # # 设置kv cache
        # hf_model = native_model
        # self.num_hidden_layers = hf_model.model.config.num_hidden_layers

        # head_dim = hf_model.model.config.hidden_size // hf_model.model.config.num_attention_heads

        # head_dim = hf_model.model.config.hidden_size // hf_model.model.config.num_attention_heads
        # num_hidden_layers = hf_model.model.config.num_hidden_layers
        # num_decoder_layers = num_hidden_layers
        # kv_cache_shape = [
        #     wrap_cfg.batch_size,
        #     hf_model.model.config.num_key_value_heads,
        #     wrap_cfg.max_sequence_length,
        #     head_dim,
        # ]
        # meta_info["kv_cache"] = {
        #     "shape": kv_cache_shape,
        #     "num_decoder_layers": num_decoder_layers,
        # }

        # past_key_caches = []
        # past_value_caches = []
        # for _ in range(num_decoder_layers):
        #     past_key_caches.append(torch.zeros(kv_cache_shape, dtype=torch.float16))
        #     past_value_caches.append(torch.zeros(kv_cache_shape, dtype=torch.float16))

        # # Export LLM model, prefill
        # data_prefill = {
        #     "input_ids": inputs["input_ids"],
        #     "image_embeds": image_embeds.to(torch.float16),
        #     "past_seq_length": [0],
        #     "image_grid_thw": inputs["image_grid_thw"],
        # }

        # input_sequence_length = wrap_cfg.input_sequence_length

        # input_seq_len = data_prefill["input_ids"].shape[-1]
        # steps = (input_seq_len + input_sequence_length - 1) // input_sequence_length

        # data_preprocess = Qwen2VLDataPreprocess(
        #     token_embedding,
        #     input_sequence_length * steps,
        # )
        # data_input = data_preprocess(data_prefill)

        # inputs_embeds, past_seq_length, current_seq_length, position_ids = data_input

        # from ._llm_model_impl import register_wrap_cls as llm_register_wrap_cls  # noqa: F401

        # wraped_llm_model = wrap_llm_model(hf_model, wrap_cfg)
        # wraped_llm_model.cpu()
        # wraped_llm_model.to(torch.float16)

        # target_device = self.config.quant_scheme.target_device

        # ## prefill
        # prefill_inputs = (
        #     inputs_embeds[:, :input_sequence_length, :],
        #     past_seq_length,
        #     torch.tensor([input_sequence_length], dtype=torch.int32).to(inputs_embeds.device),
        #     position_ids[:, :, :input_sequence_length],
        #     past_key_caches,
        #     past_value_caches,
        # )

        # onnx_input_names = [
        #     "inputs_embeds",
        #     "past_seq_length",
        #     "current_input_length",
        #     "position_ids",
        # ]

        # for layer_idx in range(num_decoder_layers):
        #     onnx_input_names.append(f"past_key_cache_{layer_idx}")
        # for layer_idx in range(num_decoder_layers):
        #     onnx_input_names.append(f"past_value_cache_{layer_idx}")

        # onnx_output_names = ["logits"]

        # prefill_onnx_file = str(work_dir / "hmonnx" / f"{prefix}-llm-prefill.onnx")
        # quant_graph_model = None
        # if not Path(prefill_onnx_file).exists():
        #     logger.info("********************* start export llm prefill model *********************")
        #     quant_graph_model = convert_fx_model_to_quanted_model(
        #         wraped_llm_model, prefill_inputs, target_device, quant_config
        #     )
        #     convert_quanted_model_to_hmonnx(
        #         quant_graph_model, prefill_inputs, prefill_onnx_file, onnx_input_names, onnx_output_names
        #     )
        # else:
        #     logger.info(f"{prefill_onnx_file} exists, skip export prefill model.")

        # meta_info["prefill_onnx"] = str(Path(prefill_onnx_file).relative_to(work_dir))

        # ## decode
        # wrap_cfg.input_sequence_length = 1

        # def update_cfg_fn(module):
        #     if hasattr(module, "_update_cfg"):
        #         module._update_cfg(wrap_cfg)

        # past_seq_length[0] = wrap_cfg.input_sequence_length
        # decode_inputs = (
        #     inputs_embeds[:, :1, :],
        #     past_seq_length,
        #     torch.tensor([1], dtype=torch.int32).to(inputs_embeds.device),
        #     position_ids[:, :, :1],
        #     past_key_caches,
        #     past_value_caches,
        # )

        # decode_onnx_file = str(work_dir / "hmonnx" / f"{prefix}-llm-decode.onnx")
        # if not Path(decode_onnx_file).exists():
        #     logger.info("********************* start export llm decode model *********************")
        #     if quant_graph_model is None:
        #         quant_graph_model = convert_fx_model_to_quanted_model(
        #             wraped_llm_model, prefill_inputs, target_device, quant_config
        #         )
        #     quant_graph_model.apply(update_cfg_fn)
        #     convert_quanted_model_to_hmonnx(
        #         quant_graph_model, decode_inputs, decode_onnx_file, onnx_input_names, onnx_output_names
        #     )
        # else:
        #     logger.info(f"{decode_onnx_file} exists, skip export decode model.")

        # meta_info["decode_onnx"] = str(Path(decode_onnx_file).relative_to(work_dir))

        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config: CogVLM2ConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp:
            assert config.quant_weight is not None and Path(config.quant_weight).exists()

        CogVLM2ConverterXH2a(config)._convert(hf_model_path, output_dir)
