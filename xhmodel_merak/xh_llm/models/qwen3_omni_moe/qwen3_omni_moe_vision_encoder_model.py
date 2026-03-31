import tempfile
import types
from pathlib import Path
from typing import Any, cast

import onnx
import torch
import torch.fx as fx
from transformers import (
    AutoConfig,
    AutoModelForTextToWaveform,
    Qwen3OmniMoeForConditionalGeneration,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionEncoder

from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph
from xhquant.utils.registry import DynamicModule, _DMRegistryCls

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...onnx_lazy_load import lazy_load_onnx
from .qwen3_omini_moe_processor import XHQwen3OmniMoeProcessor
from .xh_qwen3_omni_config import XHQwen3OmniVisualConfig


class _Qwen3OmniMoeThinkerForConditionalGenerationVisualHFCompatible(DynamicModule):
    def _setup(self: Qwen3OmniMoeVisionEncoder, xh_visual_model: "XHQwen3OmniMoeVisionEncoderModel"):
        # 清理原visual模型的权重，避免占用显存
        self.to(device="meta")
        self.visual = xh_visual_model

    def forward(self: Qwen3OmniMoeVisionEncoder, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.visual.forward(hidden_states, **kwargs)


def build_qwen3_omini_mode_visual_hf_compatible_model(
    hf_model: Qwen3OmniMoeForConditionalGeneration, xh_visual_model: "XHQwen3OmniMoeVisionEncoderModel"
):
    # 构建一个兼容HF的视觉模型，主要用于导出ONNX
    LLM_COMPATIBLE_MODULES = _DMRegistryCls("XHCompatible")  # noqa: N806
    hf_model_cls = type(hf_model.thinker.visual)
    if hf_model_cls not in LLM_COMPATIBLE_MODULES:
        LLM_COMPATIBLE_MODULES.register_module(
            {
                hf_model_cls: hf_model_cls.__name__,
            },
            _Qwen3OmniMoeThinkerForConditionalGenerationVisualHFCompatible,
        )
    LLM_COMPATIBLE_MODULES.convert(hf_model.thinker.visual, xh_visual_model=xh_visual_model)
    from xhquant_llm.models.qwen3_omni.monkey_patch import get_image_features

    hf_model.thinker.get_image_features = types.MethodType(get_image_features, hf_model.thinker)
    return hf_model


@register_llm_model("Qwen3OmniMoeForConditionalGeneration_visual", master=False)
class XHQwen3OmniMoeVisionEncoderModel(BaseVisionModel):
    transformers_min_version = "4.57.0"
    HF_MODEL_CLS = Qwen3OmniMoeForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForTextToWaveform
    CONFIG_CLS = XHQwen3OmniVisualConfig
    BUILD_HF_COMPATIBLE_FUNC = build_qwen3_omini_mode_visual_hf_compatible_model

    def __init__(self, config: XHQwen3OmniVisualConfig):
        super().__init__(config)
        qwen3moe_omini_config = AutoConfig.from_pretrained(self.hf_model_dir)
        assert qwen3moe_omini_config.thinker_config.vision_config.patch_size == config.patch_size
        assert qwen3moe_omini_config.thinker_config.vision_config.temporal_patch_size == config.temporal_patch_size
        self.config = cast(XHQwen3OmniVisualConfig, self.config)
        if self.config.model_type is None:
            self.config.model_type = "Qwen3OmniMoeForConditionalGeneration_visual"

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        native_model = super().get_hf_model(hf_model_dir, quant_weight, **kwargs)
        # native_model = qwen3_vl_patch(native_model)
        return native_model

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        native_hf_model = super().get_empty_hf_model(hf_model_dir)
        # native_hf_model = qwen3_vl_patch(native_hf_model)
        return native_hf_model

    def init_wrap_model(self, hf_model: Qwen3OmniMoeForConditionalGeneration) -> Any:
        from ._vision_model import register_wrap_modules

        register_wrap_modules()
        visual = hf_model.thinker.visual
        wrap_model = super().init_wrap_model(visual)

        return wrap_model

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.zeros(
                (
                    1,
                    3,
                    self.config.max_size_t,
                    self.config.max_size_h,
                    self.config.max_size_w,
                ),
                dtype=self.dtype,
            ),
        }

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": ["pixel_values"],
            "output_names": ["image_embeds", "deepstack_features"],
        }

    def get_tf_processor(self) -> XHQwen3OmniMoeProcessor:
        processor = XHQwen3OmniMoeProcessor.from_pretrained(self.hf_model_dir)
        config = cast(XHQwen3OmniVisualConfig, self.config)
        processor.config.patch_size = config.patch_size
        processor.config.max_size_h = config.max_size_h
        processor.config.max_size_w = config.max_size_w
        return processor

    def forward(self, *args, **kwargs):
        out = self._inference_model(*args, **kwargs)
        if isinstance(self._inference_model, fx.GraphModule):
            image_embeds_i, *deepstack_image_embeds = out
            out = (image_embeds_i, deepstack_image_embeds)
        return out

    def _to_fronted(self, wrap_model):
        logger = get_xhquant_logger()
        dummy_inputs = self.get_dummy_inputs()
        dummy_input = list(dummy_inputs.values())[0]
        work_dir = self.config.work_dir
        onnx_file = str(Path(work_dir) / "onnx" / "qwen3_vl_visual.onnx")
        Path(onnx_file).parent.mkdir(parents=True, exist_ok=True)
        if not Path(onnx_file).exists():
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                torch.onnx.export(
                    wrap_model.float().cpu(),
                    (dummy_input.float().cpu(),),
                    tmp_onnx_file,
                    export_params=True,
                    opset_version=18,
                    do_constant_folding=True,
                    input_names=["pixel_values"],
                    output_names=["image_embeds", "deepstack_feature_0", "deepstack_feature_1", "deepstack_feature_2"],
                    verbose=False,
                )
                onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)
                logger.info(f"onnx model size: {onnx_model.ByteSize()}")
                import onnxsim

                logger.info("simplify onnx model............")
                onnx_model_simp, check = onnxsim.simplify(onnx_model)
                if check:
                    onnx_model = onnx_model_simp
                onnx.save(
                    onnx_model,
                    onnx_file,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=f"{Path(onnx_file).stem}_external_data",
                )
            self._wrap_model.to(self.device, self.dtype)
        else:
            logger.info(f"from cached onnx: {onnx_file}")

        lazy_model = lazy_load_onnx(onnx_file)
        lazy_model.load_all_tensors()
        onnx_model = lazy_model.model
        lazy_model.close()
        return to_frontend_graph(onnx_model, FrontendType.ONNX, [dummy_input])
