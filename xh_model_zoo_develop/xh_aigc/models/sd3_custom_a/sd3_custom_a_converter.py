# Copyright 2025 HOUMO AI
#
# File: sd3_custom_a_converter.py
# Description:
#   Custom SD3 conversion utilities for xh2modelzoo export flows.
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
# SPDX-License-Identifier: Apache-2.0

import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import onnx
import onnxsim
import torch
import transformers
from bitsandbytes.nn.modules import Linear4bit
from diffusers import StableDiffusion3Pipeline
from transformers.models.clip.modeling_clip import CLIPSdpaAttention, CLIPTextTransformer
from transformers.models.t5.modeling_t5 import T5Block, T5Stack
from xhquant.api import get_root_logger
from xhquant.utils import get_root_logger

from ..sd3 import SD3ConvertConfig, SD3Converter
from ..sd3.sd3_converter import CLIPSdpaAttention_forward, CLIPTextTransformer_forward, linear4bit_forward
from .sd3_custom_a_diffusion_pipe import SD3CustomADiffusion3Pipe


@dataclass
class SD3CustomAConvertConfig(SD3ConvertConfig):
    # 建议使用默认值
    mmdit_quant: bool = True
    t5_quant: bool = True


class SD3CustomAConverter(SD3Converter):
    def __init__(self, pretrained_model_path: str, custom_a_model_path: str, convert_config: SD3CustomAConvertConfig):
        super().__init__(pretrained_model_path, convert_config)
        self.custom_a_model_path = custom_a_model_path

    def load_model(self, pretrained_model_path, convert_config: SD3CustomAConvertConfig):  # type: ignore[override]
        pipe = SD3CustomADiffusion3Pipe.from_pretrained(
            pretrained_model_path, self.custom_a_model_path, convert_config.mmdit_quant, convert_config.t5_quant
        )
        return pipe

    @classmethod
    def export_onnx_clip(cls, hf_model: StableDiffusion3Pipeline, convert_config: SD3ConvertConfig, output_onnx_file):
        logger = get_root_logger()
        export_model = hf_model.text_encoder

        device = "cuda"
        assert torch.cuda.is_available(), "CUDA is not available, please check your environment"
        export_model.to(device)

        onnx_name = Path(output_onnx_file).stem
        legacy_onnx = True
        fuse_clip = True
        simplify_onnx = True

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = text_outputs[1]

            text_embeds = self.text_projection(pooled_output)

            if not return_dict:
                outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
                return tuple(output for output in outputs if output is not None)

            return text_embeds, text_outputs.hidden_states[-2]

        export_model.forward = types.MethodType(forward, export_model)

        for name, module in export_model.named_modules():
            if isinstance(module, CLIPTextTransformer):
                module.forward = types.MethodType(CLIPTextTransformer_forward, module)
            if isinstance(module, CLIPSdpaAttention):
                module.forward = types.MethodType(CLIPSdpaAttention_forward, module)
            if fuse_clip:
                if isinstance(module, Linear4bit):
                    module.forward = types.MethodType(linear4bit_forward, module)

        with torch.no_grad():
            export_model.eval()
            onnx_inputs_name = ["input_ids", "output_hidden_states"]
            onnx_args = (torch.randint(1000, (1, 77), dtype=torch.int32, device=device),)
            onnx_kwargs = dict(output_hidden_states=True)

            if Path(output_onnx_file).exists():
                pass
            else:
                # 执行一次前向推理，设置反量化权重
                export_model(*onnx_args, **onnx_kwargs)
                logger.info(f"exporting model {onnx_name} to onnx")

                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmp_onnx_file = str(Path(tmpdirname) / Path(output_onnx_file).name)
                    if legacy_onnx:
                        torch.onnx.export(
                            export_model,
                            onnx_args + (onnx_kwargs,),
                            tmp_onnx_file,
                            input_names=onnx_inputs_name,
                            export_params=True,
                            output_names=["prompt_embeds", "pooled_prompt_embeds"],
                            do_constant_folding=True,
                            opset_version=18,
                            # keep_initializers_as_inputs = True,
                            verbose=False,
                        )
                    else:
                        export_program = torch.export.export(
                            export_model,
                            onnx_args,
                            onnx_kwargs,
                        )
                        torch.onnx.dynamo_export(
                            export_program.module(),
                            *onnx_args,
                            **onnx_kwargs,
                        ).save(tmp_onnx_file)
                    onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)

            for i, input in enumerate(onnx_model.graph.input):
                logger.info(f"input[{i}]: {input.name}")
            for i, output in enumerate(onnx_model.graph.output):
                logger.info(f"output[{i}]: {output.name}")

            if simplify_onnx:
                model_opt, check_ok = onnxsim.simplify(
                    onnx_model,
                    skipped_optimizers=[
                        "fuse_pad_into_conv",
                        "fuse_consecutive_slices",
                        "eliminate_common_subexpression",
                        "fuse_qkv",
                    ],
                )

                if check_ok:
                    onnx_model = model_opt
            onnx.save(
                onnx_model,
                output_onnx_file,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{Path(output_onnx_file).stem}_external_data",
            )
            logger.info(f"save onnx to file {output_onnx_file}")

    @classmethod
    def export_onnx_clip_l(cls, hf_model: StableDiffusion3Pipeline, convert_config: SD3ConvertConfig, output_onnx_file):
        logger = get_root_logger()
        export_model = hf_model.text_encoder_2
        device = "cuda"
        assert torch.cuda.is_available(), "CUDA is not available, please check your environment"
        export_model.to(device)
        fuse_clip_l = True
        legacy_onnx = True
        simplify_onnx = True
        onnx_name = "clip_l"
        if fuse_clip_l:
            onnx_name = onnx_name + "_fused"

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # nn.GELU()
            pooled_output = text_outputs[1]

            text_embeds = self.text_projection(pooled_output)

            if not return_dict:
                outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
                return tuple(output for output in outputs if output is not None)

            return text_embeds, text_outputs.hidden_states[-2]

        export_model.forward = types.MethodType(forward, export_model)

        for name, module in export_model.named_modules():
            if isinstance(module, CLIPTextTransformer):
                module.forward = types.MethodType(CLIPTextTransformer_forward, module)
            if isinstance(module, CLIPSdpaAttention):
                module.forward = types.MethodType(CLIPSdpaAttention_forward, module)
            if fuse_clip_l:
                if isinstance(module, Linear4bit):
                    module.forward = types.MethodType(linear4bit_forward, module)

        with torch.no_grad():
            export_model.eval()
            onnx_inputs_name = ["input_ids", "output_hidden_states"]
            onnx_args = (torch.randint(1000, (1, 77), dtype=torch.int32, device=device),)
            onnx_kwargs = dict(output_hidden_states=True)

            if Path(output_onnx_file).exists():
                pass
            else:
                import math

                from transformers.activations import GELUActivation

                def _gelu_python(self, input):
                    return input * (1.0 + torch.erf(input / math.sqrt(2.0))) * 0.5
                    # return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

                for name, module in export_model.named_modules():
                    if isinstance(module, GELUActivation):
                        module.act = types.MethodType(_gelu_python, module)

                # 执行一次前向推理，设置反量化权重
                export_model(*onnx_args, **onnx_kwargs)
                logger.info(f"exporting model {onnx_name} to onnx")

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_onnx_file = str(Path(tmpdir) / Path(output_onnx_file).name)

                    if legacy_onnx:
                        torch.onnx.export(
                            export_model,
                            onnx_args + (onnx_kwargs,),
                            tmp_onnx_file,
                            input_names=onnx_inputs_name,
                            export_params=True,
                            output_names=["prompt_embeds", "pooled_prompt_embeds"],
                            do_constant_folding=True,
                            opset_version=18,
                            # keep_initializers_as_inputs = True,
                            verbose=False,
                        )
                    else:
                        export_program = torch.export.export(
                            export_model,
                            onnx_args,
                            onnx_kwargs,
                        )
                        torch.onnx.dynamo_export(
                            export_program.module(),
                            *onnx_args,
                            **onnx_kwargs,
                        ).save(tmp_onnx_file)
                    onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)

            for i, input in enumerate(onnx_model.graph.input):
                print(f"input[{i}]: {input.name}")
            for i, output in enumerate(onnx_model.graph.output):
                print(f"output[{i}]: {output.name}")

            if simplify_onnx:
                model_opt, check_ok = onnxsim.simplify(
                    onnx_model,
                    skipped_optimizers=[
                        "fuse_pad_into_conv",
                        "fuse_consecutive_slices",
                        "eliminate_common_subexpression",
                        "fuse_qkv",
                    ],
                )
                if check_ok:
                    onnx_model = model_opt

            onnx.save(
                onnx_model,
                output_onnx_file,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{Path(output_onnx_file).stem}_external_data",
            )
            logger.info(f"save onnx to file {output_onnx_file}")

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: str, convert_config: SD3CustomAConvertConfig, work_dir: str, **kwargs
    ):  # type: ignore[override]
        assert (
            transformers.__version__ == "4.46.0"
        ), "transformers version must be 4.46.0, please pip install transformers==4.46.0"
        custom_a_model = kwargs.get("custom_a_model", "")
        if convert_config.t5_quant:
            convert_config.no_clip_fp16_t5 = True
        converter = SD3CustomAConverter(pretrained_model_path, custom_a_model, convert_config)
        converter._convert(work_dir)
        return converter
