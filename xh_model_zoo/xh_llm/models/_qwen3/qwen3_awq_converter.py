from typing import Any

import torch
import torch.nn as nn
from awq.modules.linear.gemm import WQLinear_GEMM
from awq.utils.packing_utils import reverse_awq_order, unpack_awq
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from transformers.utils.quantization_config import QuantizationMethod

from .qwen3_convert_config import Qwen3ConvertConfig
from .qwen3_converter import Qwen3ConverterXH2a


class Qwen3AWQConverterXH2a(Qwen3ConverterXH2a):
    def load_hf_model(self, hf_model_dir: str, **kwargs) -> Any:
        hf_model = super().load_hf_model(hf_model_dir, **kwargs)
        assert hf_model.config.quantization_config.quant_method == QuantizationMethod.AWQ
        for name, module in hf_model.named_modules():
            if isinstance(module, WQLinear_GEMM):
                if hasattr(module, "weight"):
                    continue
                bits = module.w_bit
                group_size = module.group_size
                iweight = module.qweight
                izeros = module.qzeros
                scales = module.scales

                iweight, izeros = unpack_awq(iweight, izeros, bits)
                # Reverse the order of the iweight and izeros tensors
                iweight, izeros = reverse_awq_order(iweight, izeros, bits)

                # overflow checks
                iweight = torch.bitwise_and(iweight, (2**bits) - 1)
                izeros = torch.bitwise_and(izeros, (2**bits) - 1)

                # fp16 weights
                scales = scales.repeat_interleave(group_size, dim=0)
                izeros = izeros.repeat_interleave(group_size, dim=0)

                # quant weight and weight
                quant_weight = iweight - izeros
                weight = quant_weight * scales
                quant_weight = quant_weight.t().contiguous()
                weight = weight.t().contiguous()

                iweight = None
                izeros = None
                scales = None
                if hasattr(module, "qweight"):
                    delattr(module, "qweight")
                if hasattr(module, "qzeros"):
                    delattr(module, "qzeros")
                if hasattr(module, "scales"):
                    delattr(module, "scales")

                module.register_parameter("weight", nn.Parameter(weight))
                module.register_buffer("quant_weight", quant_weight)
                quant_weight = None
                weight = None
                # module.forward = types.MethodType(linear_forward, module)
                module.__class__ = nn.Linear

        if hf_model.config.tie_word_embeddings:
            hf_model.config.torchscript = True
            hf_model.tie_weights()
            hf_model.config.tie_word_embeddings = False

        return hf_model

    @classmethod
    def convert(cls, hf_model_path: str, config: Qwen3ConvertConfig, output_dir: str):
        Qwen3AWQConverterXH2a(config)._convert(hf_model_path, output_dir)
