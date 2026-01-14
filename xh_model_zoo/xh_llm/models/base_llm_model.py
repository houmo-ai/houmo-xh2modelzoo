from functools import partial
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Union, cast

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel
from transformers.generation import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
from transformers.utils.quantization_config import QuantizationMethod
from xhquant.api import ConfigDict
from xhquant.core import CacheTensor

from .base_model import BaseModel
from .generation_mixin import BaseGenerationMixin


def qlinear_cuda_old_converter(self: nn.Module):
    from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
        QuantLinear as CudaOldQuantLinear,
    )

    assert isinstance(self, CudaOldQuantLinear)
    if self.bits in [2, 4, 8]:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)

        zeros = zeros + 1
        zeros = torch.bitwise_and(
            zeros,
            (2**self.bits) - 1,
            # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.
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
    # weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])

    quant_weight = weight - zeros[self.g_idx.long()]
    weight = self.scales[self.g_idx.long()] * quant_weight
    # weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    # quant_weight = quant_weight.reshape(quant_weight.shape[0] * quant_weight.shape[1], quant_weight.shape[2])

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
        # self.wf.unsqueeze(-1)
        weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = t.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    quant_weight = weight - zeros[self.g_idx.long()]
    weight = self.scales[self.g_idx.long()] * quant_weight
    maxq = 2 ** (self.bits - 1)
    # diff = quant_weight.to(torch.int32) - quant_weight
    # error = diff.abs().float()
    # assert torch.allclose(error, t.tensor(0.0), atol=1e-3), f"{error.max()}"
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


class LLMBaseModel(BaseModel):
    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type="TorchFX",
        allow_quant=True,
        export_cfg=None,
    ):
        super().__init__(hf_model, wrap_cfg, quant_config, frontend_type, allow_quant, export_cfg)
        self.use_cache = wrap_cfg.use_cache
        # Set up caching length based on configuration settings.
        if wrap_cfg.use_cache:
            self.cache_length = wrap_cfg.max_sequence_length
        else:
            self.cache_length = 0

        self.input_sequence_length = wrap_cfg.input_sequence_length
        self.wrap_cfg = wrap_cfg
        self.max_sequence_length = wrap_cfg.max_sequence_length
        self._default_pad_token_id = 0

        self.token_embedding: Optional[nn.Embedding] = None

        # 适配GenerationMixin的Generate方法
        self.main_input_name = "input_ids"
        self._supports_cache_class = False
        self.generation_config: Optional[GenerationConfig] = None
        self.config: Optional[PreTrainedModel.Config] = None

        if not self.use_cache:
            if "inputs" not in self.quant_cfg:
                self.quant_cfg.inputs = ConfigDict(dict())

            self.quant_cfg.inputs.past_key_cache = ConfigDict(dict(ignore=True))
            self.quant_cfg.inputs.past_value_cache = ConfigDict(dict(ignore=True))

    @property
    def pad_token_id(self) -> int:
        if self._default_pad_token_id is None or self._default_pad_token_id == 0:
            try:
                self._default_pad_token_id = self.tokenizer.pad_token_id
            except Exception:
                pass
        return self._default_pad_token_id

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._default_pad_token_id = value

    def _dequantize_awq_hf_model(self, native_hf_model: nn.Module) -> nn.Module:
        from awq.modules.linear.gemm import WQLinear_GEMM
        from awq.utils.packing_utils import reverse_awq_order, unpack_awq

        hf_model = native_hf_model
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

        hf_model.quantization_method = None  # type: ignore
        hf_model._is_hf_initialized = False  # type: ignore
        return hf_model

    def _dequantize_gptq_hf_model(self, native_hf_model: nn.Module) -> nn.Module:
        hf_model = native_hf_model
        assert hf_model.config.quantization_config.quant_method == QuantizationMethod.GPTQ
        hf_quantizer: GptqHfQuantizer = hf_model.hf_quantizer

        from transformers.utils import is_auto_gptq_available, is_gptqmodel_available

        converter: Optional[Callable] = None

        QuantLinear = hf_quantizer.optimum_quantizer.quant_linear  # type: ignore
        if is_auto_gptq_available():
            from auto_gptq.nn_modules.qlinear.qlinear_cuda import (
                QuantLinear as GeneralQuantLinear,
            )
            from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
                QuantLinear as CudaOldQuantLinear,
            )
            from auto_gptq.nn_modules.qlinear.qlinear_exllama import (
                QuantLinear as ExllamaQuantLinear,
            )
            from auto_gptq.nn_modules.qlinear.qlinear_exllamav2 import (
                QuantLinear as Exllamav2QuantLinear,
            )
            from auto_gptq.nn_modules.qlinear.qlinear_marlin import (
                QuantLinear as MarlinQuantLinear,
            )

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

    def _dequantize_compressed_tensors_hf_model(self, native_hf_model: nn.Module) -> nn.Module:
        from inspect import unwrap

        import compressed_tensors.quantization.lifecycle.forward
        from compressed_tensors.linear.compressed_linear import CompressedLinear
        from compressed_tensors.quantization.quant_args import (
            QuantizationArgs,
            QuantizationStrategy,
        )

        hf_model = native_hf_model
        for _, module in hf_model.named_modules():  # type: ignore
            if isinstance(module, CompressedLinear):
                _process_quantization_orig = compressed_tensors.quantization.lifecycle.forward._process_quantization
                _dequantize_orig = compressed_tensors.quantization.lifecycle.forward._dequantize

                def _module_process_quantization(
                    self: nn.Module,
                    x: torch.Tensor,
                    scale: torch.Tensor,
                    zero_point: torch.Tensor,
                    args: QuantizationArgs,
                    g_idx: Optional[torch.Tensor] = None,
                    dtype: Optional[torch.dtype] = None,
                    do_quantize: bool = True,
                    do_dequantize: bool = True,
                    global_scale: Optional[torch.Tensor] = None,
                ):
                    self._args = args
                    self._original_shape = x.shape
                    return _process_quantization_orig(
                        x,
                        scale,
                        zero_point,
                        args,
                        g_idx,
                        dtype,
                        do_quantize,
                        do_dequantize,
                        global_scale,
                    )

                def _module_dequantize(
                    self: nn.Module,
                    x_q: torch.Tensor,
                    scale: torch.Tensor,
                    zero_point: Optional[torch.Tensor] = None,
                    dtype: Optional[torch.dtype] = None,
                    global_scale: Optional[torch.Tensor] = None,
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

    def dequantize_hf_model(self, native_hf_model: nn.Module) -> nn.Module:
        hf_model = native_hf_model
        if not hasattr(hf_model.config, "quantization_config") or hf_model.config.quantization_config is None:
            return hf_model

        if hf_model.config.quantization_config.quant_method == QuantizationMethod.AWQ:
            hf_model = self._dequantize_awq_hf_model(hf_model)
        elif hf_model.config.quantization_config.quant_method == QuantizationMethod.GPTQ:
            hf_model = self._dequantize_gptq_hf_model(hf_model)
        elif hf_model.config.quantization_config.quant_method == QuantizationMethod.COMPRESSED_TENSORS:
            hf_model = self._dequantize_compressed_tensors_hf_model(hf_model)
        else:
            raise Exception(f"Unsupported quantization method: {hf_model.config.quantization_config.quant_method}")
        return hf_model

    def get_hf_model(self, device_map="cpu", **kwargs) -> Any:
        hf_model = super().get_hf_model(device_map, **kwargs)
        hf_model = self.dequantize_hf_model(hf_model)
        return hf_model

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        assert isinstance(data, Dict)
        input_ids = data.get("input_ids", None)
        inputs_embeds = data.get("inputs_embeds", None)

        if input_ids is not None:
            assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
            seq_length = input_ids.shape[1]
            input_ids = input_ids.to(self.execution_device)
            assert seq_length <= self.input_sequence_length, (
                f"Input sequence length is too long. max input sequence length is {self.input_sequence_length} but got {seq_length}"
            )
            if self.input_sequence_length > seq_length:
                padding_input_ids = torch.zeros((1, self.input_sequence_length - seq_length), dtype=torch.long).to(
                    self.execution_device
                )
                padding_input_ids.fill_(self.pad_token_id)
                input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)
            inputs_embeds = self.token_embedding.to(self.execution_device)(input_ids)
        elif inputs_embeds is not None:
            assert inputs_embeds.shape[0] == 1, "Batch size should be 1 in inference mode."
            seq_length = inputs_embeds.shape[1]
            inputs_embeds = inputs_embeds.to(self.execution_device)
            assert seq_length <= self.input_sequence_length, (
                "Input sequence length should be larger than input_sequence_length."
            )
            if self.input_sequence_length > seq_length:
                padding_token_id = self.pad_token_id
                padding_input_ids = (
                    torch.ones((1, self.input_sequence_length - seq_length), dtype=torch.long).to(self.execution_device)
                    * padding_token_id
                )
                padding_embedding = self.token_embedding.to(self.execution_device)(padding_input_ids)
                inputs_embeds = torch.cat([inputs_embeds, padding_embedding], dim=1)

        assert self.token_embedding is not None, "Token embedding is not available."

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."
        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches

        return (
            # position_ids.to(self.device),
            inputs_embeds.to(self.execution_device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(self.execution_device),
            torch.tensor([seq_length], dtype=torch.int32).to(self.execution_device),
            past_key_caches,
            past_value_caches,
        )

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        (
            inputs_embeds,
            past_seq_length,
            seg_length,
            past_key_caches,
            past_value_caches,
        ) = self.prepare_inputs(data)
        # 从CacheTensor转为Tensor
        # past_key_caches = [t.data for t in past_key_caches]
        # past_value_caches = [t.data for t in past_value_caches]
        return (
            # position_ids.to(self.device),
            inputs_embeds,
            past_seq_length,
            seg_length,
            past_key_caches,
            past_value_caches,
        )

    def _set_exec_device(self, device):
        device = torch.device(device)
        self.token_embedding.to(device)
        return super()._set_exec_device(device)

    def set_kv_cache_device(self, device):
        for i in range(len(self.past_key_caches)):
            self.past_key_caches[i].to(device)
            self.past_value_caches[i].to(device)

    def _set_device(self, device):
        device = torch.device(device)
        if self.use_cache and device != torch.device("meta"):
            self.past_key_caches = [t.to(device) for t in self.past_key_caches]
            self.past_value_caches = [t.to(device) for t in self.past_value_caches]

        if hasattr(self, "token_embedding") and self.token_embedding is not None:
            self.token_embedding.to(device)
        return super()._set_device(device)

    def _set_dtype(self, dtype):
        if self.use_cache:
            for i in range(len(self.past_key_caches)):
                self.past_key_caches[i] = self.past_key_caches[i].to(dtype)
                self.past_value_caches[i] = self.past_value_caches[i].to(dtype)
            #     setattr(self, f"past_k_cache_{i}", getattr(self, f"past_k_cache_{i}").to(dtype))
            #     setattr(self, f"past_v_cache_{i}", getattr(self, f"past_v_cache_{i}").to(dtype))

        if hasattr(self, "token_embedding") and self.token_embedding is not None:
            self.token_embedding.to(dtype)
        return super()._set_dtype(dtype)

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def _forward(
        self,
        # position_ids: Optional[Tensor] = None,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: List[Tensor],
        past_value_caches: List[Tensor],
    ):
        logits = self(
            inputs_embeds,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        )
        return CausalLMOutputWithPast(
            logits=logits,
        )

    def prepare_kv_cache(self, num_decoder_layers, kv_cache_shape, dif_value_shape=None):
        self.past_key_caches = []
        self.past_value_caches = []
        if self.use_cache:
            for i in range(num_decoder_layers):
                self.past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
                if dif_value_shape is not None:
                    self.past_value_caches.append(CacheTensor(torch.zeros(dif_value_shape, dtype=torch.float16)))
                else:
                    self.past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))

            for layer_idx in range(num_decoder_layers):
                self.export_cfg.input_names.append(f"past_key_cache_{layer_idx}")
            for layer_idx in range(num_decoder_layers):
                self.export_cfg.input_names.append(f"past_value_cache_{layer_idx}")
