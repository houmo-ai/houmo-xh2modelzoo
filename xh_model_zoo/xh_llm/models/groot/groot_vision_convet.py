# Copyright 2025 HOUMO AI
#
# File: dit_converter.py
# Description:
#   Dit Converter implementation.
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
from typing import Callable, List, Optional, Tuple, Union

from numpy import dtype
import onnx
from sympy import false
import torch
import torch.nn as nn
from PIL import Image
from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
from xhquant.api import convert_fx_model_to_quanted_model, convert_onnx_to_hmonnx, convert_quanted_model_to_hmonnx, convert_dynamo_model_to_quanted_model
from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model

from xhquant.api import (  # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    CacheTensor,
)


class Groot_ConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.work_dir = None
        self.wraped_llm_model = None
        self.device = device

    def _convert(self, hf_model, output_dir: str, postprocess=None):
        logger = get_root_logger()
        config = self.config

        native_model = hf_model

        with torch.no_grad():
            torch.manual_seed(42)
            vision_input = torch.rand(1, 3, 252, 252).to(self.device)
            outoput_ori = native_model([vision_input])

        from ._vision_model import register_wrap_modules as vision_register_wrap_modules  # noqa: F401
        vision_register_wrap_modules()

        wrap_cfg = Config(
            dict(
                batch_size=1,
                token_len=256,
                num_logits_to_keep=0,
                input_sequence_length=324,
                use_cache=false,
                max_sequence_length=888888,
            )
        )

        wraped_llm_model = wrap_llm_model(hf_model, wrap_cfg)
        wraped_llm_model = wraped_llm_model.to(self.device)
        # wraped_llm_model.to(torch.float16)

        with torch.no_grad():
            # vision_input = torch.rand(1, 3, 252, 252).to(self.device)
            windows_tensor, win_meta_list, spatial_shapes, reverse_mapping = hf_model.vision_model.embeddings([vision_input])
            outoput_hm = wraped_llm_model(windows_tensor)
            outoput_hm = outoput_hm[:, reverse_mapping, :]

        model_name = "groot_vision"
        target_device = config.quant_scheme.target_device
        # batch_size = config.batch_size
        assert target_device == DeviceType.XH2a, f"Only support convert to XH2a, but got {target_device}"

        quant_config = create_quant_config(config.quant_scheme)
        work_dir = Path(output_dir)
        self.work_dir = output_dir
        quant_config = ConfigDict(quant_config)


        meta_info = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )
        meta_info["device"] = str(target_device)
        meta_info["model_name"] = model_name
        meta_info["quant_scheme"] = config.quant_scheme.to_dict()
        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        Path(work_dir / "hmonnx").mkdir(exist_ok=True, parents=True)
        quant_type = config.quant_scheme.quant_type
        prefix = f"{model_name}-{target_device}-{quant_type}"

        # latent = torch.rand(1, 3, 252, 252).to(self.device)

        with torch.no_grad():
            windows_tensor, win_meta_list, spatial_shapes, reverse_mapping = hf_model.vision_model.embeddings([vision_input])
            # last_hidden_state = last_hidden_state[:, reverse_mapping, :]

        windows_tensor = torch.rand(1, 324, 1152).to(self.device).half()
        target_device = self.config.quant_scheme.target_device
        input_names = ["windows_tensor"]
        inputs = [windows_tensor,]
 
        onnx_output_names = ["output_latent"]
        vison_onnx_file = str(work_dir / "hmonnx" / f"{prefix}.onnx")
        quant_graph_model = None
        wraped_llm_model = wraped_llm_model.to(self.device)
        if not Path(vison_onnx_file).exists():
            logger.info("********************* start export vision model *********************")
            

            quant_graph_model = convert_fx_model_to_quanted_model(
                wraped_llm_model, inputs, target_device, quant_config
            )
            convert_quanted_model_to_hmonnx(
                quant_graph_model, inputs, vison_onnx_file, input_names, onnx_output_names
            )
        else:
            logger.info(f"{vison_onnx_file} exists, skip export vision model.")

        meta_info["onnx"] = str(Path(vison_onnx_file).relative_to(work_dir))
        vison_golden_dir = str(work_dir / "golden" / f"{prefix}")

        if not Path(vison_golden_dir).exists():
            logger.info(f"start export vision model golden............")
            from xhquant.api import HMONNXGoldenInference

            vae_model = HMONNXGoldenInference(vison_onnx_file)
            vae_model.save_golden = True
            vae_model.exec_device = torch.device("cuda:0")

            Path(vison_golden_dir).mkdir(exist_ok=True, parents=True)
            vae_model.golden_dir = str(vison_golden_dir)

            inputs[0] = inputs[0].half()
            with torch.no_grad():
                vae_model.forward(*inputs)
            logger.info(f"Export vision model golden to {vison_golden_dir}")
        else:
            logger.info(f"{vison_golden_dir} exists, skip export vision model golden.")


        # meta_info["decoder_golden_dir"] = str(Path(vison_golden_dir))
        # json.dump(meta_info, open(work_dir / "meta_vision.json", "w"), indent=4)