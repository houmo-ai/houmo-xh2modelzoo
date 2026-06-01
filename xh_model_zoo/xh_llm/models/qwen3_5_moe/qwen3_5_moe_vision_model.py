import torch
import torch.nn as nn
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

from ..base_model import BaseModel
from ..builder import MODELS


@MODELS.register_module()
class XHQwen3_5MoeVisionModel(BaseModel):
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

    def get_hf_model(self, device_map="cpu", **kwargs) -> Qwen3_5MoeForConditionalGeneration:
        assert self.hf_model_dir is not None
        hf_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            self.hf_model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map,
            **kwargs,
        ).eval()
        return hf_model

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()

        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls

        vision_register_wrap_cls(hf_model)

        # Qwen3_5MoeForConditionalGeneration stores visual as model.visual
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
