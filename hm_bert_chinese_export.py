# Copyright 2025 HOUMO AI
#
# File: standalone_bert_export.py
# Description:
#   Standalone script for BERT model export with all xh_model_zoo dependencies included.
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
import time
import math
import copy
import re
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, Callable
from functools import partial
from types import MethodType
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope import AutoTokenizer, AutoModelForMaskedLM

try:
    from xhquant.api import (DeviceType, xhquant_init, QuantScheme, get_root_logger,
                             Config, ConfigDict, convert_fx_model_to_quanted_model,
                             convert_quanted_model_to_hmonnx, create_quant_config,
                             is_ssfp_quant_config)
    from xhquant.utils import digit_version
    from xhquant import nn as xhnn
    from xhquant.nn import LLMCache, MaskedSoftmax, RMSNorm, Rope
    from xhquant.utils.registry import DynamicModule, Registry
    from xhquant.utils.registry.dynamic_module import _DMRegistryCls
    from xhquant.utils.logger import get_root_logger as xh_get_root_logger
    from transformers.utils.quantization_config import QuantizationMethod
    from safetensors.torch import load_file as load_safetensors_file
    from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
except ImportError as e:
    print(f"Required xhquant library not found: {e}")
    print("Please install xhquant library first.")
    raise

try:
    import accelerate
except ImportError:
    accelerate = None
    print("Warning: accelerate not installed, some features may not work")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    print("Warning: tqdm not installed, progress bar will not be shown")

from transformers.models.bert.modeling_bert import (BertModel, BertEncoder, BertLayer,
                                                      BertSdpaSelfAttention, BertForMaskedLM, BertAttention)
from transformers.cache_utils import Cache


XHLLM_TRACEABLE_MODULES = _DMRegistryCls("XHTrace")
MODELS = Registry("xh_llm_models")


try:
    from awq.modules.linear.gemm import WQLinear_GEMM
    from awq.utils.packing_utils import reverse_awq_order, unpack_awq
except ImportError:
    WQLinear_GEMM = None
    reverse_awq_order = None
    unpack_awq = None


class DeviceDtypeMixin(nn.Module):
    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def _set_device(self, device: torch.device) -> None:
        self._device = device

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self._dtype = dtype

    def _set_exec_device(self, device: torch.device) -> None:
        self._exec_device = device

    def set_exec_device(self, device) -> None:
        def apply_fn(module):
            if not hasattr(module, "_set_exec_device"):
                return
            module._set_exec_device(device)
        self.apply(apply_fn)
        return self

    def to(self, *args, **kwargs) -> nn.Module:
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            def apply_fn(module):
                if not hasattr(module, "_set_device"):
                    return
                module._set_device(device)
            self.apply(apply_fn)
            return self
        if dtype is not None:
            def apply_fn(module):
                if not hasattr(module, "_set_dtype"):
                    return
                module._set_dtype(dtype)
            self.apply(apply_fn)
            return self

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None) -> nn.Module:
        if device is None or isinstance(device, int):
            device = torch.device("cuda", index=device)
        self._set_device(torch.device(device))
        return super().cuda(device)

    def cpu(self, *args, **kwargs) -> nn.Module:
        self._set_device(torch.device("cpu"))
        return super().cpu()


@dataclass
class VisualConfig:
    image_max_size_h: int = 364
    image_max_size_w: int = 644
    image_max_size_t: int = 2
    patch_size: int = 14
    temporal_patch_size: int = 2

    sample_image_path: str = field(default_factory=str)

    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)


@dataclass
class BaseConvertConfig:
    @classmethod
    def from_dict_or_other(cls, other: Union[dict, "BaseConvertConfig", Any]) -> "BaseConvertConfig":
        if isinstance(other, dict):
            return cls(**other)
        elif isinstance(other, BaseConvertConfig):
            if isinstance(other, cls):
                return other
            else:
                return cls(**other.to_dict())
        else:
            raise ValueError(f"Invalid type: {type(other)}")


@dataclass
class LLMConvertConfig(BaseConvertConfig):
    batch_size: int = 1
    context_length: int = 2048
    input_sequence_length: int = 256
    quant_scheme: QuantScheme = field(default_factory=QuantScheme)
    quant_weight: Optional[str] = None
    eval_ppl: bool = False

    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)


@dataclass
class Bert_ConvertConfig(LLMConvertConfig):
    visual_config: VisualConfig = field(default_factory=VisualConfig)
    gptqmodel_cfg: str = field(default_factory=str)
    quant_weight: str = field(default_factory=str)
    max_pe_length: int = 32768


def qlinear_cuda_old_converter(self: nn.Module):
    from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear as CudaOldQuantLinear
    assert isinstance(self, CudaOldQuantLinear)
    if self.bits in [2, 4, 8]:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)

        zeros = zeros + 1
        zeros = torch.bitwise_and(
            zeros, (2**self.bits) - 1
        )

        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)
        weight = weight.reshape(-1, self.group_size, weight.shape[2])
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        )

        zeros = zeros + 1
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
        weight = weight.reshape(-1, self.group_size, weight.shape[2])
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    quant_weight = weight - zeros
    weight = scales * quant_weight
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    quant_weight = quant_weight.reshape(quant_weight.shape[0] * quant_weight.shape[1], quant_weight.shape[2])

    max_val = 2 ** (self.bits - 1)
    min_val = -max_val

    assert quant_weight.max() < max_val and quant_weight.min() >= min_val, f"{quant_weight.max()} {quant_weight}.min()"
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
    self.out_features = self.outfeatures
    self.in_features = self.infeatures


def general_qlinear_converter(self: nn.Module):
    if self.bits in [2, 4, 8]:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        zeros = torch.bitwise_and(zeros, (2**self.bits) - 1)

        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        )
        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

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
    self.out_features = self.outfeatures
    self.in_features = self.infeatures


def gptqmodel_torch_qlinear_converter(self: nn.Module):
    import torch as t

    if self.bits in [2, 4, 8]:
        zeros = t.bitwise_right_shift(
            t.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,
        ).to(self.dequant_dtype)
        zeros = t.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

        weight = t.bitwise_and(
            t.bitwise_right_shift(
                t.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,
            ).to(self.dequant_dtype),
            self.maxq,
        )
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf_unsqueeze_zero
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = t.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(self.scales.shape)

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = t.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    quant_weight = weight - zeros[self.g_idx.long()]
    weight = self.scales[self.g_idx.long()] * quant_weight
    maxq = 2 ** (self.bits - 1)
    assert quant_weight.max() < maxq and quant_weight.min() >= -maxq, (
        f"min={quant_weight.min()}, max={quant_weight.max()}, not in [{-maxq}, {maxq})"
    )
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


class BaseConverter:
    target_device: DeviceType

    def __init__(self):
        pass

    @staticmethod
    def xh1_hmonnx_compatible(input_names: List[str]):
        input_names = copy.deepcopy(input_names)
        input_names_mapping = {
            "inputs_embeds": "input_1",
            "past_seq_length": "valid_length",
            "current_input_length": "current_length",
        }
        for idx in range(len(input_names)):
            in_name = input_names[idx]
            if in_name in input_names_mapping:
                input_names[idx] = input_names_mapping[in_name]
            else:
                kcache_pattern = r"^past_key_cache_\d+$"
                kcache_match = re.match(kcache_pattern, in_name)
                if kcache_match:
                    kcache_idx = kcache_match.group(0).split("_")[-1]
                    input_names[idx] = "model_layers_{}_self_attn_kcache_input".format(kcache_idx)
                else:
                    vcache_pattern = r"^past_value_cache_\d+$"
                    vcache_match = re.match(vcache_pattern, in_name)
                    if vcache_match:
                        vcache_idx = vcache_match.group(0).split("_")[-1]
                        input_names[idx] = "model_layers_{}_self_attn_vcache_input".format(vcache_idx)
        return input_names


class HFTransfromersConverter(BaseConverter):
    def __init__(self):
        super().__init__()

    def load_hf_model(self, hf_model_dir: str, **kwargs) -> Any:
        raise NotImplementedError()

    def load_quant_weight(self, quant_weight_path: str, native_hf_model: nn.Module) -> bool:
        logger = get_root_logger()
        archive_file = quant_weight_path
        logger.info(f"Load previously saved checkpoint from: {archive_file}")
        is_safetensors = archive_file.endswith(".safetensors")
        state_dict: Dict[str, torch.Tensor]
        if is_safetensors:
            state_dict = load_safetensors_file(archive_file, device="cpu")
        else:
            state_dict = torch.load(archive_file, weights_only=True, map_location="cpu")

        model_state_dict = native_hf_model.state_dict()
        unexpect_state_dict = []
        for k, v in state_dict.items():
            if k not in model_state_dict:
                unexpect_state_dict.append(k)

        for k in unexpect_state_dict:
            paths = k.split(".")
            if paths[-1] == "quant_weight":
                submodule_name = ".".join(paths[:-1])
                submodule = native_hf_model.get_submodule(submodule_name)
                v = state_dict[k]
                if v.min().item() >= -pow(2, 7) and v.max().item() <= pow(2, 7) - 1:
                    v = v.to(torch.int8)
                elif v.min().item() >= -pow(2, 15) and v.max().item() <= pow(2, 15) - 1:
                    v = v.to(torch.int16)
                else:
                    v = v.to(torch.float32)
                submodule.register_buffer("quant_weight", v, persistent=False)
                logger.debug(f"add quant_weight to {submodule_name}")
            else:
                logger.warning(f"ignore unexpect state dict: {k}")
            state_dict.pop(k)

        native_hf_model.load_state_dict(state_dict)
        del state_dict
        return True

    def _dequantize_awq_hf_model(self, native_hf_model: nn.Module):
        assert native_hf_model.config.quantization_config.quant_method == QuantizationMethod.AWQ
        hf_model = native_hf_model
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
                iweight, izeros = reverse_awq_order(iweight, izeros, bits)

                iweight = torch.bitwise_and(iweight, (2**bits) - 1)
                izeros = torch.bitwise_and(izeros, (2**bits) - 1)

                scales = scales.repeat_interleave(group_size, dim=0)
                izeros = izeros.repeat_interleave(group_size, dim=0)

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
                module.__class__ = nn.Linear

        if hf_model.config.tie_word_embeddings:
            hf_model.config.torchscript = True
            hf_model.tie_weights()
            hf_model.config.tie_word_embeddings = False

        hf_model.quantization_method = None
        hf_model._is_hf_initialized = False
        return hf_model

    def _dequantize_gptq_hf_model(self, native_hf_model: nn.Module):
        hf_model = native_hf_model
        assert hf_model.config.quantization_config.quant_method == QuantizationMethod.GPTQ
        hf_quantizer: GptqHfQuantizer = hf_model.hf_quantizer

        from transformers.utils import is_auto_gptq_available, is_gptqmodel_available

        converter: Optional[Callable] = None

        QuantLinear = hf_quantizer.optimum_quantizer.quant_linear

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

            torch_linear_cls = [TorchQuantLinear]
            try:
                from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedQuantLinear
                torch_linear_cls.append(TorchFusedQuantLinear)
            except Exception:
                pass

            if QuantLinear in torch_linear_cls:
                converter = gptqmodel_torch_qlinear_converter
            elif QuantLinear is MarlinQuantLinear:
                converter = None

        assert converter is not None, f"Not implemented for {QuantLinear} yet"

        dequant_linears = []
        for name, module in hf_model.named_modules():
            if isinstance(module, QuantLinear):
                dequant_linears.append((name, module))
        pbar = tqdm(dequant_linears, desc="Dequantizing GPTQ model")
        for name, module in pbar:
            pbar.set_description(f"Dequantizing GPTQ: {name}")
            converter(module)

        hf_model.quantization_method = None
        hf_model._is_hf_initialized = False
        return hf_model

    def _dequantize_compressed_tensors_hf_model(self, native_hf_model: nn.Module) -> nn.Module:
        import compressed_tensors.quantization.lifecycle.forward
        from compressed_tensors.linear.compressed_linear import CompressedLinear
        from compressed_tensors.quantization.quant_args import QuantizationArgs, QuantizationStrategy

        hf_model = native_hf_model
        for _, module in hf_model.named_modules():
            if isinstance(module, CompressedLinear):
                _process_quantization_orig = compressed_tensors.quantization.lifecycle.forward._process_quantization
                _dequantize_orig = compressed_tensors.quantization.lifecycle.forward._dequantize

                def _module_process_quantization(
                    self: nn.Module,
                    x: torch.Tensor,
                    scale: torch.Tensor,
                    zero_point: torch.Tensor,
                    args: QuantizationArgs,
                    g_idx: torch.Tensor | None = None,
                    dtype: torch.dtype | None = None,
                    do_quantize: bool = True,
                    do_dequantize: bool = True,
                    global_scale: torch.Tensor | None = None,
                ):
                    self._args = args
                    self._original_shape = x.shape
                    nonlocal _process_quantization_orig
                    return _process_quantization_orig(
                        x, scale, zero_point, args, g_idx, dtype, do_quantize, do_dequantize, global_scale
                    )

                def _module_dequantize(
                    self: nn.Module,
                    x_q: torch.Tensor,
                    scale: torch.Tensor,
                    zero_point: torch.Tensor | None = None,
                    dtype: torch.dtype | None = None,
                    global_scale: torch.Tensor | None = None,
                ):
                    quanted_strategy = self._args.strategy

                    quant_weight = x_q
                    if zero_point is not None:
                        quant_weight = x_q - zero_point
                    if quanted_strategy in (
                        QuantizationStrategy.GROUP,
                        QuantizationStrategy.TENSOR_GROUP,
                    ):
                        quant_weight = quant_weight.flatten(start_dim=-2)
                    elif quanted_strategy == QuantizationStrategy.BLOCK:
                        original_shape = self._original_shape
                        quant_weight = quant_weight.transpose(1, 2).reshape(original_shape)

                    self.register_buffer("quant_weight", quant_weight)
                    nonlocal _dequantize_orig
                    return _dequantize_orig(x_q, scale, zero_point, dtype, global_scale)

                compressed_tensors.quantization.lifecycle.forward._dequantize = partial(_module_dequantize, module)
                compressed_tensors.quantization.lifecycle.forward._process_quantization = partial(
                    _module_process_quantization, module
                )

                weight_data = module.compressor.decompress_module(module)
                compressed_tensors.quantization.lifecycle.forward._dequantize = _dequantize_orig
                compressed_tensors.quantization.lifecycle.forward._process_quantization = _process_quantization_orig
                param = nn.Parameter(weight_data, requires_grad=False)

                module.register_parameter("weight", param)
                module.__class__ = nn.Linear
                module.forward = MethodType(nn.Linear.forward, module)

        return hf_model

    def dequantize_hf_model(self, native_hf_model: nn.Module):
        if (
            not hasattr(native_hf_model.config, "quantization_config")
            or native_hf_model.config.quantization_config is None
        ):
            return native_hf_model

        hf_model = native_hf_model
        if hf_model.config.quantization_config.quant_method == QuantizationMethod.AWQ:
            hf_model = self._dequantize_awq_hf_model(hf_model)
        elif hf_model.config.quantization_config.quant_method == QuantizationMethod.GPTQ:
            hf_model = self._dequantize_gptq_hf_model(hf_model)
        elif hf_model.config.quantization_config.quant_method == QuantizationMethod.COMPRESSED_TENSORS:
            hf_model = self._dequantize_compressed_tensors_hf_model(hf_model)
        else:
            raise NotImplementedError(
                f"Dequantize not implemented for quantization method: {hf_model.config.quantization_config.quant_method}"
            )
        return hf_model

    def get_hf_model(self, hf_model_dir: str, **kwargs) -> Any:
        native_hf_model = self.load_hf_model(hf_model_dir, **kwargs)
        native_hf_model = self.dequantize_hf_model(native_hf_model)


class DynamicRegister(DynamicModule):
    @classmethod
    def register(cls: type, hf_cls: type):
        if hf_cls not in XHLLM_TRACEABLE_MODULES:
            XHLLM_TRACEABLE_MODULES.register_module(
                {
                    hf_cls: hf_cls.__name__,
                },
                cls,
            )


def wrap_llm_model(llm_model: nn.Module, config: Optional[Union[dict, ConfigDict]] = None) -> nn.Module:
    if accelerate is not None:
        llm_model = accelerate.hooks.remove_hook_from_module(llm_model, recurse=True)
    logger = get_root_logger()
    if config is None:
        config = ConfigDict()
    if isinstance(config, dict):
        config = ConfigDict(config)
    for name, module in list(llm_model.named_modules()):
        if type(module) in XHLLM_TRACEABLE_MODULES:
            logger.debug(f"Model {type(module)} will be wrapped")
            convert_module(module, config)
    wrap_llm_model = llm_model
    if type(llm_model) in XHLLM_TRACEABLE_MODULES:
        logger.debug(f"Model {type(llm_model)} will be wrapped")
        wrap_llm_model = convert_module(llm_model, config)
    return wrap_llm_model


def convert_module(module: nn.Module, config: ConfigDict) -> DynamicModule:
    if isinstance(module, DynamicModule):
        return module

    nn_cls = type(module)

    dm_cls = XHLLM_TRACEABLE_MODULES.get(nn_cls)
    if dm_cls is None:
        raise ValueError(f"Unsupported module: {nn_cls}")
    qmodule = dm_cls.convert(module, config)
    return qmodule


@XHLLM_TRACEABLE_MODULES.register_module({BertModel: "BertModel"})
class _BertModel(DynamicModule):
    def forward(
        self,
        token_embedding: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        token_type_embeddings: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ):
        output_attentions = False
        output_hidden_states = False
        use_cache = False

        embedding = token_embedding + token_type_embeddings + position_embeddings
        embedding_output = self.embeddings.LayerNorm(embedding)

        extended_attention_mask = None
        encoder_extended_attention_mask = None
        head_mask = None

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        return encoder_outputs

    def _setup(self, cfg: Optional[Dict] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module({BertEncoder: "BertEncoder"})
class _BertEncoder(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,
    ):
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                None,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

        return hidden_states

    def _setup(self, cfg: Optional[Dict] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module({BertLayer: "BertLayer"})
class _BertLayer(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        layer_output = self.feed_forward_chunk(attention_output)

        outputs = (layer_output,) + outputs
        return outputs

    def _setup(self, cfg: Optional[Dict] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module({BertSdpaSelfAttention: "BertSdpaSelfAttention"})
class _BertSdpaSelfAttention(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ):
        bsz = 1

        query_layer = self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        current_states = hidden_states

        key_layer = self.key(current_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_layer = self.value(current_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        query = query_layer * self.kv_scale
        key = key_layer.transpose(2, 3)
        attn_weights = torch.matmul(query, key)
        attn_weights = self.maskedadd(attn_weights, attention_mask)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_layer)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, -1, self.all_head_size)

        return attn_output, None

    def _setup(self, cfg: Optional[Dict] = None):
        _kv_scale = 1 / math.sqrt(self.attention_head_size)
        self.register_buffer("kv_scale", torch.tensor(_kv_scale, dtype=torch.float16), persistent=False)

        self.maskedadd = xhnn.MaskedAdd()
        return self

@XHLLM_TRACEABLE_MODULES.register_module({BertAttention: "BertAttention"})
class _BertAttention(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            past_key_values,
            output_attentions,
            cache_position,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    def _setup(self, cfg: Optional[Dict] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module({BertForMaskedLM: "BertForMaskedLM"})
class _BertForMaskedLM(DynamicModule):
    def forward(
        self,
        token_embedding: Optional[torch.Tensor] = None,
        token_type_embeddings = None,
        position_embeddings = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        outputs = self.bert(
            token_embedding,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
            token_type_embeddings=token_type_embeddings,
            position_embeddings=position_embeddings,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        return prediction_scores

    def _setup(self, cfg: Optional[Dict] = None):
        return self


def register_wrap_modules(hf_model=None):
    pass


class BertConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
        model = AutoModelForMaskedLM.from_pretrained(hf_model_dir)

        self.hf_model_path = hf_model_dir
        return model

    def _convert(self, native_model, output_dir: str, tokenizer=None):
        logger = get_root_logger()
        config = self.config
        device = "cuda"

        resume_from = self.config.quant_weight
        if resume_from is not None:
            self.load_quant_weight(resume_from, native_model)

        model_name = "bert_ch"
        target_device = config.quant_scheme.target_device
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
        meta_info["quant_scheme"] = config.quant_scheme.to_dict()
        meta_info["quant_weight"] = resume_from

        work_dir = Path(output_dir)

        token_embedding = native_model.bert.embeddings.word_embeddings

        token_type_ids = torch.zeros((1, context_length), dtype=torch.long, device=device)
        token_type_embeddings = native_model.bert.embeddings.token_type_embeddings(token_type_ids)

        position_ids = torch.arange(context_length, dtype=torch.long, device=device).unsqueeze(0)
        position_embeddings = native_model.bert.embeddings.position_embeddings(position_ids)

        atten_mask = torch.zeros((1, context_length), device=device)

        token_embedding_file = Path(work_dir) / "token_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file))
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        input_txt = "你好"
        input_ids = tokenizer(
            input_txt, return_tensors="pt", padding="max_length", max_length=context_length
        ).input_ids.cuda()
        output = native_model(input_ids)

        register_wrap_modules(native_model)
        wrap_cfg = Config(
            dict(
                max_sequence_length=context_length,
                input_sequence_length=input_sequence_length,
                use_cache=True,
                num_logits_to_keep=1,
                kv_cache=dict(
                    cache_axis=2,
                ),
            )
        )
        meta_info["wrap_cfg"] = wrap_cfg.to_dict()
        wraped_qwen_model = wrap_llm_model(native_model, wrap_cfg)

        input_emb = token_embedding(input_ids)

        warp_out = wraped_qwen_model(input_emb, token_type_embeddings, position_embeddings, atten_mask)

        inputs = [input_emb, token_type_embeddings, position_embeddings, atten_mask]

        input_names = [
            "input_emb",
            "token_type_embeddings",
            "position_embeddings",
            "atten_mask",
        ]
        output_names = ["logits"]

        prefix = f"{model_name}-{target_device}-{context_length // 1024}k-{quant_type}"
        prefill_onnx_file = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))

        logger.info(f"********************* start export prefill model *********************")

        quanted_model = convert_fx_model_to_quanted_model(
            wraped_qwen_model,
            inputs,
            target_device,
            quant_config=quant_config,
        )

        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(quanted_model, inputs, str(prefill_onnx_file), input_names, output_names)
        logger.info(f"Export Prefill model to {prefill_onnx_file}")

        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
            if not hasattr(hf_config, "quantization_config"):
                assert config.quant_weight is not None and Path(config.quant_weight).exists()
        BertConverterXH2a(config)._convert(hf_model_path, output_dir)


def main(args):
    xhquant_init()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model = model.to("cuda")

    input_txt = "你好"
    input_ids = tokenizer(
        input_txt, return_tensors="pt", padding="max_length", max_length=args.context_length
    ).input_ids
    output = model(input_ids.cuda())

    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    config = Bert_ConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        gptqmodel_cfg=args.use_gptqmodel,
        max_pe_length=args.max_pe_length,
    )

    BertConverterXH2a(config)._convert(
        model,
        "work_dirs/bert",
        tokenizer,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default="/data02/datasets/bert_chinese")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--context-length", type=int, default=256, help="max sequence length")
    parser.add_argument("--max_pe_length", type=int, default=32768, help="max pe length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--image_max_size_h", type=int, default=448, help="image max size height")
    parser.add_argument("--image_max_size_w", type=int, default=448, help="image max size width")
    parser.add_argument("--image_max_size_t", type=int, default=2, help="if image, temporal max size is 2, if video, temporal max size is fps")
    parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="temporal patch size")
    parser.add_argument("--sample_image_path", type=str, default="data/images/qwen2_vl_demo.jpeg", help="sample image path for generate golden")
    parser.add_argument("--use_gptqmodel", action="store_true", help="use gptqmodel quanted model")
    parser.add_argument(
        "--quant_weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    args = parser.parse_args()
    main(args)
