# Copyright 2025 HOUMO AI
#
# File: vae_converter.py
# Description:
#   Vae Converter implementation.
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

import onnx
import torch
import torch.nn as nn
from PIL import Image
from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
from xhquant.api import convert_fx_model_to_quanted_model, convert_onnx_to_hmonnx, convert_quanted_model_to_hmonnx
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


def gptqmodel_torch_qlinear_converter(self: nn.Module):
    import torch as t  # conflict with torch.py

    if self.bits in [2, 4, 8]:
        zeros = t.bitwise_right_shift(
            t.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,  # self.wf.unsqueeze(0),
        ).to(self.dequant_dtype)
        zeros = t.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

        weight = t.bitwise_and(
            t.bitwise_right_shift(
                t.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,  # self.wf.unsqueeze(-1)
            ).to(self.dequant_dtype),
            self.maxq,
        )
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf_unsqueeze_zero  # self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = t.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(self.scales.shape)

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7  # self.wf.unsqueeze(-1)
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = t.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    quant_weight = weight - zeros[self.g_idx.long()]
    weight = self.scales[self.g_idx.long()] * quant_weight
    maxq = (2**self.bits) / 2

    assert quant_weight.max() < maxq and quant_weight.min() >= -maxq, f"{quant_weight.max()} {quant_weight}.min()"
    if hasattr(self, "qweight"):
        delattr(self, "qweight")
    if hasattr(self, "qzeros"):
        delattr(self, "qzeros")
    if hasattr(self, "scales"):
        delattr(self, "scales")
    if hasattr(self, "g_idx"):
        delattr(self, "g_idx")
    weight = weight.t()
    quant_weight = quant_weight.t()
    self.register_parameter("weight", nn.Parameter(weight))
    self.register_buffer("quant_weight", quant_weight)
    self.__class__ = nn.Linear


class VAE_ConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.work_dir = None

    def untied_weights(self, module: nn.Module) -> nn.Module:
        param_ids = {}
        duplicate_params = []
        for name, param in module.named_parameters(remove_duplicate=False):
            param_id = id(param)
            if param_id not in param_ids:
                param_ids[param_id] = param
            else:
                duplicate_params.append(name)

        duplicate_params = list(set(duplicate_params))
        for param_name in duplicate_params:
            fields = param_name.split(".")[:-1]
            m_name = ".".join(fields)
            attr_name = param_name.split(".")[-1]
            m = module.get_submodule(m_name)
            param = getattr(m, attr_name)
            setattr(m, attr_name, nn.Parameter(param.clone()))
        return module

    def load_gptq_model(self, hf_model_dir: str, **kwargs):
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_model_dir, **kwargs
        ).eval()  # quantization_config={"use_exllama": False}
        if hf_model.config.tie_word_embeddings:
            hf_model.config.torchscript = True
            hf_model.tie_weights()
            hf_model.config.tie_word_embeddings = False
            hf_model.config.torchscript = False

        hf_model = self.untied_weights(hf_model)

        assert hasattr(hf_model, "hf_quantizer")
        hf_quantizer: GptqHfQuantizer = hf_model.hf_quantizer

        from transformers.utils import is_auto_gptq_available, is_gptqmodel_available

        converter: Optional[Callable] = None

        QuantLinear = hf_quantizer.optimum_quantizer.quant_linear  # type: ignore
        if is_auto_gptq_available():
            from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear as GeneralQuantLinear
            from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear as CudaOldQuantLinear
            from auto_gptq.nn_modules.qlinear.qlinear_exllama import QuantLinear as ExllamaQuantLinear
            from auto_gptq.nn_modules.qlinear.qlinear_exllamav2 import QuantLinear as Exllamav2QuantLinear
            from auto_gptq.nn_modules.qlinear.qlinear_marlin import QuantLinear as MarlinQuantLinear

            if QuantLinear is GeneralQuantLinear:
                converter = general_qlinear_converter
            elif QuantLinear is CudaOldQuantLinear:
                converter = qlinear_cuda_old_converter
            elif QuantLinear is ExllamaQuantLinear:
                converter = None
            elif QuantLinear is Exllamav2QuantLinear:
                converter = None
            elif QuantLinear is MarlinQuantLinear:
                converter = None

        if is_gptqmodel_available():
            from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
            from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

            if QuantLinear is TorchQuantLinear:
                converter = gptqmodel_torch_qlinear_converter
            elif QuantLinear is MarlinQuantLinear:
                converter = None

        assert converter is not None, f"Not implemented for {QuantLinear} yet"

        for name, module in hf_model.named_modules():  # type: ignore
            if isinstance(module, QuantLinear):
                if converter is not None:
                    converter(module)
                else:
                    raise NotImplementedError(f"Not implemented for {type(QuantLinear)} yet")

        hf_model.quantization_method = None  # type: ignore
        hf_model._is_hf_initialized = False  # type: ignore
        return hf_model

    def _convert(self, hf_model, output_dir: str, postprocess=None):
        logger = get_root_logger()
        config = self.config

        native_model = hf_model

        # 融合GPTQ权重
        resume_from = self.config.quant_weight
        if resume_from is not None:
            self.load_quant_weight(resume_from, native_model)

        # lm_head = native_model.lm_head
        # if not hasattr(lm_head, "quant_weight"):
        #     config.quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"

        model_name = "zimage_vae"
        target_device = config.quant_scheme.target_device
        batch_size = config.batch_size
        assert target_device == DeviceType.XH2a, f"Only support convert to XH2a, but got {target_device}"

        quant_config = create_quant_config(config.quant_scheme)
        print(quant_config)
        work_dir = Path(output_dir)
        self.work_dir = output_dir
        quant_config = ConfigDict(quant_config)

        wrap_cfg = Config(
            dict(
                batch_size=batch_size,
            )
        )

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



        # Export LLM model, prefill

        # latent = torch.randn(1, 16, 128, 128).cuda()
        latent = torch.load("examples/llm/zimage/vae/latent.pt")

        # with torch.no_grad():
        #     output = hf_model.decode(latent.half().cuda())

        from ._vae_model_imp import register_wrap_cls as llm_register_wrap_cls  # noqa: F401
        llm_register_wrap_cls(hf_model)

        hf_model.encoder = None
        wraped_llm_model = wrap_llm_model(hf_model, wrap_cfg)
        wraped_llm_model.cuda()
        wraped_llm_model.to(torch.float16)

        if True:
            with torch.no_grad():
                output = wraped_llm_model(latent.half().cuda())
                image = postprocess(output, "pil")
                image[0].save("vae_export_img.png")

        target_device = self.config.quant_scheme.target_device

        ## vae
        onnx_output_names = ["image"]

        vae_onnx_file = str(work_dir / "hmonnx" / f"{prefix}.onnx")
        quant_graph_model = None
        if not Path(vae_onnx_file).exists():
            logger.info("********************* start export vae model *********************")
            quant_graph_model = convert_fx_model_to_quanted_model(
                wraped_llm_model, [latent.half().cuda()], target_device, quant_config
            )
            convert_quanted_model_to_hmonnx(
                quant_graph_model, [latent.half().cuda()], vae_onnx_file, ['latent'], onnx_output_names
            )
        else:
            logger.info(f"{vae_onnx_file} exists, skip export vae model.")

        meta_info["onnx"] = str(Path(vae_onnx_file).relative_to(work_dir))
        vae_golden_dir = str(work_dir / "golden" / f"{prefix}")

        if not Path(vae_golden_dir).exists():
            logger.info(f"start export vae model golden............")
            from xhquant.api import HMONNXGoldenInference

            vae_model = HMONNXGoldenInference(vae_onnx_file)
            vae_model.save_golden = True
            vae_model.exec_device = torch.device("cuda:0")

            Path(vae_golden_dir).mkdir(exist_ok=True, parents=True)
            vae_model.golden_dir = str(vae_golden_dir)

            with torch.no_grad():
                vae_model.forward(latent.half())
            logger.info(f"Export prefill model golden to {vae_golden_dir}")
        else:
            logger.info(f"{vae_golden_dir} exists, skip export prefill model golden.")


        meta_info["decoder_golden_dir"] = str(Path(vae_golden_dir).relative_to(work_dir))
        json.dump(meta_info, open(work_dir / "meta_vae.json", "w"), indent=4)