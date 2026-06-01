# Copyright 2025 HOUMO AI
#
# File: qwen3_gptq_converter.py
# Description:
#   Qwen3 Gptq Converter implementation.
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

from typing import Any

import torch
import torch.nn as nn
from awq.modules.linear.gemm import WQLinear_GEMM
from awq.utils.packing_utils import reverse_awq_order, unpack_awq
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from transformers.utils.quantization_config import QuantizationMethod

from .qwen3_convert_config import Qwen3ConvertConfig
from .qwen3_converter import Qwen3ConverterXH2a


class Qwen3GPTQConverterXH2a(Qwen3ConverterXH2a):
    def load_hf_model(self, hf_model_dir: str, **kwargs) -> Any:
        hf_model = super().load_hf_model(hf_model_dir, **kwargs)
        assert hf_model.config.quantization_config.quant_method == QuantizationMethod.GPTQ

        from auto_gptq.nn_modules.qlinear import qlinear_cuda, qlinear_cuda_old, qlinear_triton

        for name, module in hf_model.named_modules():
            if isinstance(module, qlinear_cuda_old.QuantLinear) or isinstance(module, qlinear_cuda.QuantLinear):
                if hasattr(module, "weight"):
                    continue

                bits = module.bits
                group_size = module.group_size
                iweight = module.qweight
                izeros = module.qzeros
                scales = module.scales
                wf = module.wf.to(izeros.device)

                if bits in [2, 4, 8]:
                    zeros = torch.bitwise_right_shift(
                        torch.unsqueeze(izeros, 2).expand(-1, -1, 32 // bits),
                        wf.unsqueeze(0),
                    ).to(torch.int16 if bits == 8 else torch.int8)

                    zeros = zeros + 1
                    zeros = torch.bitwise_and(
                        zeros, (2**bits) - 1
                    )  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.
                    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

                    scales = scales.reshape(-1, 1, scales.shape[-1])

                    weight = torch.bitwise_right_shift(
                        torch.unsqueeze(iweight, 1).expand(-1, 32 // bits, -1),
                        wf.unsqueeze(-1),
                    ).to(torch.int16 if bits == 8 else torch.int8)
                    weight = torch.bitwise_and(weight, (2**bits) - 1)
                    weight = weight.reshape(-1, group_size, weight.shape[2])

                elif bits in [3]:
                    zeros = izeros.reshape(izeros.shape[0], izeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
                    zeros = zeros >> wf.unsqueeze(0)
                    zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
                    zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
                    zeros = zeros & 0x7
                    zeros = torch.cat(
                        [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
                        dim=2,
                    )

                    zeros = zeros + 1
                    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
                    scales = scales.reshape(-1, 1, scales.shape[-1])

                    weight = iweight.reshape(iweight.shape[0] // 3, 3, 1, iweight.shape[1]).expand(-1, -1, 12, -1)
                    weight = (weight >> wf.unsqueeze(-1)) & 0x7
                    weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
                    weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
                    weight = weight & 0x7
                    weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
                    weight = weight.reshape(-1, self.group_size, weight.shape[2])

                quant_weight = weight - zeros
                weight = scales * quant_weight
                weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
                quant_weight = quant_weight.reshape(
                    quant_weight.shape[0] * quant_weight.shape[1], quant_weight.shape[2]
                )

                quant_weight = quant_weight.t().contiguous()
                weight = weight.t().contiguous()

                module.register_buffer("w_scales", scales)

                iweight = None
                izeros = None
                scales = None
                if hasattr(module, "qweight"):
                    delattr(module, "qweight")
                if hasattr(module, "qzeros"):
                    delattr(module, "qzeros")
                if hasattr(module, "scales"):
                    delattr(module, "scales")
                if hasattr(module, "wf"):
                    delattr(module, "wf")

                module.register_parameter("weight", nn.Parameter(weight))
                module.register_buffer("quant_weight", quant_weight)

                # module.register_buffer("in_features", module.infeatures)
                # module.register_buffer("out_features", module.outfeatures)
                module.in_features = module.infeatures
                module.out_features = module.outfeatures

                quant_weight = None
                weight = None
                # module.forward = types.MethodType(linear_forward, module)
                module.__class__ = nn.Linear
            # elif isinstance(module, nn.Linear):
                # module.register_buffer("w_scales", torch.zeros_like(module.weight) )
                # module.register_buffer("quant_weight", module.weight )

        if hf_model.config.tie_word_embeddings:
            hf_model.config.torchscript = True
            hf_model.tie_weights()
            hf_model.config.tie_word_embeddings = False

        return hf_model

    @classmethod
    def convert(cls, hf_model_path: str, config: Qwen3ConvertConfig, output_dir: str):
        Qwen3GPTQConverterXH2a(config)._convert(hf_model_path, output_dir)
