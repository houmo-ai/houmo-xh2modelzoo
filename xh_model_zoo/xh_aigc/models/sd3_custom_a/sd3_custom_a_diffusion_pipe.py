# Copyright 2025 HOUMO AI
#
# File: sd3_custom_a_diffusion_pipe.py
# Description:
#   Sd3 Custom A Diffusion Pipe implementation.
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

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import transformers
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from xhquant.api import get_root_logger


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def convert_sd_linear_layer_to_quantlinear_layer(model, cls, bit):
    import accelerate

    for child_name, child in model.named_children():
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            with accelerate.init_empty_weights():
                new_layer = cls(
                    bits=bit,
                    group_size=128,
                    infeatures=in_features,
                    outfeatures=out_features,
                    bias=child.bias != None,
                    use_cuda_fp16=True,
                    trainable=False,
                    weight_dtype=child.weight.dtype,
                )
            ori_layer_device = get_device(child)
            new_layer.device = ori_layer_device
            setattr(model, child_name, new_layer.to(ori_layer_device))
        else:
            convert_sd_linear_layer_to_quantlinear_layer(child, cls, bit)


def convert_layer_to_quantlinear_layer(model, cls, bit):
    import accelerate

    for child_name, child in model.named_children():
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            with accelerate.init_empty_weights():
                new_layer = cls(
                    bits=bit,
                    group_size=128,
                    infeatures=in_features,
                    outfeatures=out_features,
                    bias=child.bias != None,
                    use_cuda_fp16=True,
                    trainable=False,
                    weight_dtype=child.weight.dtype,
                )
            ori_layer_device = get_device(child)
            new_layer.device = ori_layer_device
            setattr(model, child_name, new_layer.to(ori_layer_device))
        else:
            convert_layer_to_quantlinear_layer(child, cls, bit)


class SD3CustomADiffusion3Pipe(StableDiffusion3Pipeline):
    @classmethod
    def from_pretrained(
        cls,
        hf_model_or_path: str,
        custom_a_model_or_path: str,
        mmdit_quant: bool = True,
        t5_quant: bool = True,
    ):
        logger = get_root_logger()
        logger.info("loading custom_a model")
        dtype = torch.float16

        pipe = StableDiffusion3Pipeline.from_pretrained(
            hf_model_or_path,
            transformer=None,
            text_encoder=None,
            text_encoder_2=None,
            text_encoder_3=None,
            torch_dtype=dtype,
        )

        import accelerate
        from safetensors.torch import load_file
        from transformers import AutoConfig
        from transformers.modeling_utils import _load_state_dict_into_meta_model

        from .qlinear_cuda_old import QuantLinear
        from .qlinear_cuda_old_non_zero import QuantLinear_non_zero

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
        )
        logger.info(f"bnb_config: {bnb_config}")
        name = "text_encoder"
        text_encoder = transformers.models.clip.modeling_clip.CLIPTextModelWithProjection.from_pretrained(
            str(Path(hf_model_or_path) / name), quantization_config=bnb_config, torch_dtype=dtype
        )
        logger.info("loaded text_encoder")

        name2 = "text_encoder_2"
        text_encoder_2 = transformers.models.clip.modeling_clip.CLIPTextModelWithProjection.from_pretrained(
            str(Path(hf_model_or_path) / name2), quantization_config=bnb_config, torch_dtype=dtype
        )
        logger.info(f"loaded text_encoder")

        # mmdit
        with accelerate.init_empty_weights():
            config = SD3Transformer2DModel.load_config(str(Path(hf_model_or_path) / "transformer"))
            mmdit_model = SD3Transformer2DModel.from_config(config).to(dtype)

        if mmdit_quant:
            sd_sym = False
            if sd_sym:
                cls = QuantLinear
            else:
                cls = QuantLinear_non_zero

            # qat 4bit no sym sd
            convert_sd_linear_layer_to_quantlinear_layer(mmdit_model, cls, bit=8)
            filename = str(Path(custom_a_model_or_path) / "mmdit_hypersd__non_sym_8bit_v1.3.0.1.safetensors")
            state_dict = load_file(filename)
            logger.info(f"loaded quanted mmdit:{filename}")
        else:
            filename = str(Path(custom_a_model_or_path) / "mmdit_hypersd_fp16_v1.3.0.0.safetensors")
            state_dict = load_file(filename)
            logger.info(f"loaded fp16 mmdit:{filename}")

        expected_keys = mmdit_model.state_dict().keys()

        unexpected_keys = set(list(state_dict.keys())) - set(list(expected_keys))
        load_state_dict = {}
        for key in state_dict:
            if key in unexpected_keys:
                continue
            else:
                load_state_dict[key] = state_dict[key]

        _load_state_dict_into_meta_model(
            mmdit_model,
            load_state_dict,
            start_prefix="",
            # expected_keys = list(load_state_dict.keys()),
            expected_keys=list(expected_keys),
            # device_map={"": 0},
            dtype=dtype,
        )

        name3 = "text_encoder_3"

        # with accelerate.init_empty_weights():
        config = AutoConfig.from_pretrained(Path(hf_model_or_path) / name3)
        text_encoder_3 = transformers.models.t5.modeling_t5.T5EncoderModel._from_config(config).to(dtype)

        if t5_quant:
            t5_sym = False
            if t5_sym:
                cls = QuantLinear
            else:
                cls = QuantLinear_non_zero

            # 2bit T5 sym
            convert_layer_to_quantlinear_layer(text_encoder_3, cls, bit=2)

            # gptq model always have bias, some layer do not have bias
            filename = str(Path(custom_a_model_or_path) / "T5_non_sym_2bit_v1.3.0.1.safetensors")
            state_dict = load_file(filename)  # 2bit T5

            expected_keys = text_encoder_3.state_dict().keys()
            unexpected_keys = set(list(state_dict.keys())) - set(list(expected_keys))
            load_state_dict = {}
            for key in state_dict:
                if key in unexpected_keys:
                    continue
                else:
                    load_state_dict[key] = state_dict[key]

            _load_state_dict_into_meta_model(
                text_encoder_3,
                load_state_dict,
                start_prefix="",
                # list(load_state_dict.keys()),
                expected_keys=list(expected_keys),
                # device_map={"": 0},
                dtype=dtype,
            )
            logger.info(f"loaded quanted t5:{filename}")

        pipe.transformer = mmdit_model
        pipe.text_encoder = text_encoder
        pipe.text_encoder_2 = text_encoder_2
        pipe.text_encoder_3 = text_encoder_3

        return pipe
