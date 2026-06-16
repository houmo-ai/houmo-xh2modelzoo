import tempfile
from pathlib import Path
from typing import Any, cast

import onnx
import torch
import torch.fx as fx
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

from xhmodel_merak.xh_llm.models.qwen3_vl.qwen3_vl_processor import XHQwen3VLProcessor
from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph
from xhquant.utils.registry import DynamicModule, _DMRegistryCls

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...onnx_lazy_load import lazy_load_onnx
from ...types import VisualModelMeta
from ...utils import is_graph_module
from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration as XHQwen3VLForConditionalGeneration
from .modeling_qwen3_vl_patch import qwen3_vl_patch
from .xh_qwen3_vl_config import XHQwen3VLVisualConfig


class _Qwen3VLVisualHFCompatible(DynamicModule):
    def _setup(self, xh_visual_model: "XHQwen3VLVisualModel"):
        self.visual = xh_visual_model

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.visual.forward(hidden_states, **kwargs)


def build_qwen3_vl_visual_hf_compatible_model(
    hf_model: Qwen3VLForConditionalGeneration, xh_visual_model: "XHQwen3VLVisualModel"
):
    # 构建一个兼容HF的视觉模型，主要用于导出ONNX
    LLM_COMPATIBLE_MODULES = _DMRegistryCls("XHCompatible")  # noqa: N806
    hf_model_cls = type(hf_model.visual)
    if hf_model_cls not in LLM_COMPATIBLE_MODULES:
        LLM_COMPATIBLE_MODULES.register_module(
            {
                hf_model_cls: hf_model_cls.__name__,
            },
            _Qwen3VLVisualHFCompatible,
        )
    LLM_COMPATIBLE_MODULES.convert(hf_model.model.visual, xh_visual_model=xh_visual_model)
    return hf_model


@register_llm_model("Qwen3VLForConditionalGeneration_visual", master=False)
class XHQwen3VLVisualModel(BaseVisionModel):
    transformers_min_version = "4.57.6"
    HF_MODEL_CLS = XHQwen3VLForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    # HMONNXINFERENCE_CLS = Qwen3HMONNXModel
    BUILD_HF_COMPATIBLE_FUNC = build_qwen3_vl_visual_hf_compatible_model
    META_CLS = VisualModelMeta
    CONFIG_CLS = XHQwen3VLVisualConfig

    def __init__(self, config: XHQwen3VLVisualConfig):
        super().__init__(config)
        qwen3_vl_config = AutoConfig.from_pretrained(self.hf_model_dir)
        assert qwen3_vl_config.vision_config.patch_size == config.patch_size
        assert qwen3_vl_config.vision_config.temporal_patch_size == config.temporal_patch_size
        self.config = cast(XHQwen3VLVisualConfig, self.config)
        if self.config.model_type is None:
            self.config.model_type = "Qwen3VLForConditionalGeneration_visual"

    def _to_eager(self, aligned: bool = True):
        raise NotImplementedError("Eager mode is not implemented for vision model yet.")

    def _to_fronted(self, wrap_model):
        logger = get_xhquant_logger()
        dummy_inputs = self.get_dummy_inputs()
        dummy_input = list(dummy_inputs.values())[0][0]
        work_dir = self.config.work_dir
        _tmp_dir_ctx = None
        if not work_dir:
            _tmp_dir_ctx = tempfile.TemporaryDirectory()
            work_dir = _tmp_dir_ctx.name
        try:
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
                        output_names=[
                            "image_embeds",
                            "deepstack_feature_0",
                            "deepstack_feature_1",
                            "deepstack_feature_2",
                        ],
                        verbose=False,
                    )
                    onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)
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
        finally:
            if _tmp_dir_ctx is not None:
                _tmp_dir_ctx.cleanup()

    def get_dummy_inputs(self) -> Any:
        return {
            "image": self._get_dummy_inputs()["hm_pixel_values"],
        }

    def _get_dummy_inputs(self) -> Any:
        from .processing_qwen3_vl import Qwen3VLProcessor

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.new("RGB", (self.config.max_size_w, self.config.max_size_h)),
                        "resized_height": self.config.max_size_h,
                        "resized_width": self.config.max_size_w,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        processor = Qwen3VLProcessor.from_pretrained(self.hf_model_dir)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=self.config.patch_size)
        processor.image_processor.max_pixels = max(
            self.config.max_size_w * self.config.max_size_h + 1, processor.image_processor.max_pixels
        )

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {
            "hm_pixel_values": [v.float() for v in inputs["hm_pixel_values"]],
            "image_grid_thw": inputs["image_grid_thw"],
        }

    def init_wrap_model(self, hf_model: Qwen3VLForConditionalGeneration = None):
        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls

        vision_register_wrap_cls(hf_model)
        visual = hf_model.model.visual
        # self.config = visual.config
        wraped_model = super().init_wrap_model(visual)
        return wraped_model

    def get_tf_processor(self):
        processor = XHQwen3VLProcessor.from_pretrained(self.hf_model_dir)
        processor.config.patch_size = self.config.patch_size
        processor.config.max_size_h = self.config.max_size_h
        processor.config.max_size_w = self.config.max_size_w
        return processor

    def forward(self, *args, **kwargs):
        out = self._inference_model(*args, **kwargs)
        if len(out) > 2:
            image_embeds_i, *deepstack_image_embeds = out
            out = (image_embeds_i, deepstack_image_embeds)
        # if is_graph_module(self._inference_model):
        #     image_embeds_i, *deepstack_image_embeds = out
        #     out = (image_embeds_i, deepstack_image_embeds)
        return out

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        native_model = super().get_hf_model(hf_model_dir, quant_weight, **kwargs)
        return qwen3_vl_patch(native_model)

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        native_hf_model = super().get_empty_hf_model(hf_model_dir)
        native_hf_model = qwen3_vl_patch(native_hf_model)
        return native_hf_model

    @classmethod
    def get_compatible_model(cls, hf_model_dir, **kwargs):
        # 单独调试visual模型时，不能加载空模型
        return cls.get_hf_model(hf_model_dir, **kwargs)

    def get_export_cfg(self) -> dict[str, list[str]]:
        export_cfg = dict(
            input_names=[
                "pixel_values",
            ],
            output_names=[
                "image_embeds",
                "deepstack_feature_0",
                "deepstack_feature_1",
                "deepstack_feature_2",
            ],
        )
        return export_cfg

    def export_hmonnx(self, output_dir: str) -> VisualModelMeta:
        meta_info = self.create_export_metadata(output_dir)
        exported_hmonnx_file = super()._export_hmonnx(output_dir)
        meta_info.hmonnx = str(exported_hmonnx_file)
        return meta_info

    def create_export_metadata(self, output_dir: str) -> VisualModelMeta:
        meta_info = self.get_export_metadata_cls()()
        meta_info = cast(VisualModelMeta, meta_info)
        meta_info.image_size_w = self.config.max_size_w
        meta_info.image_size_h = self.config.max_size_h
        meta_info.patch_size = self.config.patch_size
        meta_info.max_size_t = self.config.max_size_t
        meta_info.temporal_patch_size = self.config.temporal_patch_size
        meta_info.spatial_merge_size = self.config.temporal_patch_size
        return meta_info
