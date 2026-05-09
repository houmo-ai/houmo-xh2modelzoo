from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from ..base_model import BaseModel
from ..builder import MODELS


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


@MODELS.register_module()
class XHQwen3_5VisionModel(BaseModel):
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
            module_name = ".".join(fields)
            attr_name = param_name.split(".")[-1]
            module_ref = module.get_submodule(module_name)
            param = getattr(module_ref, attr_name)
            setattr(module_ref, attr_name, nn.Parameter(param.clone()))
        return module

    def _load_hf_model(self, device_map="cpu", **kwargs) -> Qwen3_5ForConditionalGeneration:
        config = AutoConfig.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        quantization_config = getattr(config, "quantization_config", None)
        quant_method = getattr(quantization_config, "quant_method", None)
        if isinstance(quantization_config, dict):
            quant_method = quantization_config.get("quant_method", quant_method)

        if self.is_gptqmodel or str(quant_method).lower() == "gptq":
            return self._load_gptqmodel(device_map, **kwargs)

        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = torch.float16

        return Qwen3_5ForConditionalGeneration.from_pretrained(
            self.hf_model_dir,
            trust_remote_code=True,
            device_map=device_map,
            **kwargs,
        ).eval()

    def _load_gptqmodel(self, device_map="cpu", **kwargs) -> Qwen3_5ForConditionalGeneration:
        from gptqmodel import GPTQModel
        
        trust_remote_code = bool(kwargs.pop("trust_remote_code", True))
        backend = kwargs.pop("backend", "torch")
        valid_string_device_maps = {"auto", "balanced", "balanced_low_0", "sequential"}
        load_kwargs = {
            "backend": backend,
            "trust_remote_code": trust_remote_code,
            **kwargs,
        }

        if isinstance(device_map, dict):
            load_kwargs["device_map"] = device_map
        elif isinstance(device_map, str):
            if device_map in valid_string_device_maps:
                load_kwargs["device_map"] = device_map
            elif device_map != "meta":
                load_kwargs["device"] = device_map
        elif device_map is not None:
            load_kwargs["device"] = device_map

        if "device_map" in load_kwargs and "device" not in load_kwargs:
            load_kwargs["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

        try:
            q_model = GPTQModel.load(self.hf_model_dir, **load_kwargs)
        except TypeError:
            load_kwargs.pop("backend", None)
            q_model = GPTQModel.load(self.hf_model_dir, **load_kwargs)

        hf_model = q_model.model.eval()
        qcfg = getattr(hf_model.config, "quantization_config", None)
        if isinstance(qcfg, dict):
            try:
                from transformers.utils.quantization_config import GPTQConfig

                hf_model.config.quantization_config = GPTQConfig.from_dict(qcfg)
            except Exception:
                pass
        return hf_model

    def get_hf_model(self, device_map="cpu", **kwargs) -> Qwen3_5ForConditionalGeneration:
        assert self.hf_model_dir is not None
        hf_model = self._load_hf_model(device_map, **kwargs)
        return hf_model

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()

        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls

        vision_register_wrap_cls(hf_model)

        if hasattr(hf_model, "model") and hasattr(hf_model.model, "visual"):
            visual = hf_model.model.visual
        elif hasattr(hf_model, "visual"):
            visual = hf_model.visual
        else:
            raise AttributeError("Cannot find visual module in hf_model")

        self.config = visual.config
        wraped_model = super().init_wrap_model(visual)
        wraped_model.to(torch.float16)
        return wraped_model
