from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from torchvision import transforms

# import transformers_modules
# from transformers import AutoModelForCausalLM, PaddleOCRVLForConditionalGeneration
from transformers.quantizers.quantizer_gptq import GptqHfQuantizer

from ..base_model import BaseModel
from ..builder import MODELS
from .modeling_paddleocr_vl import PaddleOCRVLForConditionalGeneration


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
        zeros = self.qzeros.reshape(
            self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1
        ).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf_unsqueeze_zero  # self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | (
            (zeros[:, :, 1, 0] << 2) & 0x4
        )
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | (
            (zeros[:, :, 2, 0] << 1) & 0x6
        )
        zeros = zeros & 0x7
        zeros = t.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(self.scales.shape)

        weight = self.qweight.reshape(
            self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]
        ).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7  # self.wf.unsqueeze(-1)
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = t.cat(
            [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1
        )
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    quant_weight = weight - zeros[self.g_idx.long()]
    weight = self.scales[self.g_idx.long()] * quant_weight
    maxq = (2**self.bits) / 2

    assert (
        quant_weight.max() < maxq and quant_weight.min() >= -maxq
    ), f"{quant_weight.max()} {quant_weight}.min()"
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


@MODELS.register_module()
class XHPaddleOCRVLVisionModel(BaseModel):
    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type="TorchFX",
        allow_quant=True,
        export_cfg=None,
        is_gptqmodel=False,
    ) -> None:
        super().__init__(
            hf_model=hf_model,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
            frontend_type=frontend_type,
        )
        self.is_gptqmodel = is_gptqmodel

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

    def get_hf_model(
        self, device_map="cpu", **kwargs
    ) -> PaddleOCRVLForConditionalGeneration:
        assert self.hf_model_dir is not None
        hf_model = PaddleOCRVLForConditionalGeneration.from_pretrained(
            self.hf_model_dir,
            torch_dtype=torch.float16,  ###
            trust_remote_code=True,
            device_map="cpu",
        ).eval()

        # if hf_model.config.tie_word_embeddings:
        #     hf_model.config.torchscript = True
        #     hf_model.tie_weights()
        #     hf_model.config.tie_word_embeddings = False
        #     hf_model.config.torchscript = False
        # if self.is_gptqmodel:
        #     hf_model = self.untied_weights(hf_model)

        #     assert hasattr(hf_model, "hf_quantizer"), "hf_model must have hf_quantizer"
        #     hf_quantizer: GptqHfQuantizer = hf_model.hf_quantizer

        #     from transformers.utils import is_auto_gptq_available, is_gptqmodel_available

        #     converter: Optional[Callable] = None

        #     QuantLinear = hf_quantizer.optimum_quantizer.quant_linear
        #     if is_gptqmodel_available():
        #         from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
        #         from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

        #         if QuantLinear is TorchQuantLinear:
        #             converter = gptqmodel_torch_qlinear_converter
        #         elif QuantLinear is MarlinQuantLinear:
        #             converter = None

        #     assert converter is not None, f"Not implemented for {QuantLinear} yet"

        #     for name, module in hf_model.named_modules():  # type: ignore
        #         if isinstance(module, QuantLinear):
        #             if converter is not None:
        #                 converter(module)
        #             else:
        #                 raise NotImplementedError(f"Not implemented for {type(QuantLinear)} yet")

        # hf_model.quantization_method = None  # type: ignore
        # hf_model._is_hf_initialized = False  # type: ignore
        return hf_model

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()

        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls

        vision_register_wrap_cls(hf_model)
        visual = hf_model.visual
        self.config = visual.config
        #
        wraped_model = super().init_wrap_model(visual)
        wraped_model.to(torch.float16)
        return wraped_model
