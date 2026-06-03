import tempfile
from pathlib import Path
from typing import Any, cast

import onnx
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph
from xhquant.utils.registry import DynamicModule, _DMRegistryCls

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...onnx_lazy_load import lazy_load_onnx
from ...types import VisualModelMeta
from .qwen2_vl_processor import XHQwen2VLProcessor
from .xh_qwen2_vl_config import XHQwen2VLVisualConfig


class _Qwen2VLVisualHFCompatible(DynamicModule):
    def _setup(self, xh_visual_model: "XHQwen2VLVisualModel"):
        self.visual = xh_visual_model

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.visual.forward(hidden_states, **kwargs)


def build_qwen2_vl_visual_hf_compatible_model(
    hf_model: Qwen2VLForConditionalGeneration, xh_visual_model: "XHQwen2VLVisualModel"
):
    compatible_modules = _DMRegistryCls("XHCompatible")
    visual = hf_model.visual
    hf_model_cls = type(visual)
    if hf_model_cls not in compatible_modules:
        compatible_modules.register_module({hf_model_cls: hf_model_cls.__name__}, _Qwen2VLVisualHFCompatible)
    compatible_modules.convert(visual, xh_visual_model=xh_visual_model)
    return hf_model


@register_llm_model("Qwen2VLForConditionalGeneration_visual", master=False)
class XHQwen2VLVisualModel(BaseVisionModel):
    transformers_min_version = "4.57.0"
    HF_MODEL_CLS = Qwen2VLForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    BUILD_HF_COMPATIBLE_FUNC = build_qwen2_vl_visual_hf_compatible_model
    META_CLS = VisualModelMeta
    CONFIG_CLS = XHQwen2VLVisualConfig

    def __init__(self, config: XHQwen2VLVisualConfig):
        super().__init__(config)
        hf_config = AutoConfig.from_pretrained(self.hf_model_dir)
        assert hf_config.vision_config.patch_size == config.patch_size
        assert hf_config.vision_config.temporal_patch_size == config.temporal_patch_size
        self.config = cast(XHQwen2VLVisualConfig, self.config)
        if self.config.model_type is None:
            self.config.model_type = "Qwen2VLForConditionalGeneration_visual"

    def _to_eager(self, aligned: bool = True):
        raise NotImplementedError("Eager mode is not implemented for vision model yet.")

    def _to_fronted(self, wrap_model):
        logger = get_xhquant_logger()
        dummy_input = self.get_dummy_inputs()["image"][0]
        work_dir = self.config.work_dir
        tmp_dir_ctx = None
        if not work_dir:
            tmp_dir_ctx = tempfile.TemporaryDirectory()
            work_dir = tmp_dir_ctx.name
        try:
            onnx_file = str(Path(work_dir) / "onnx" / "qwen2_vl_visual.onnx")
            Path(onnx_file).parent.mkdir(parents=True, exist_ok=True)
            if not Path(onnx_file).exists():
                # The visual branch is converted through ONNX because the fixed
                # image graph is easier for xhquant to align than tracing the
                # original HF dynamic preprocessing path directly.
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
                        output_names=["image_embeds"],
                        verbose=False,
                    )
                    onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)
                    if not onnx_model.ir_version or onnx_model.ir_version > 10:
                        onnx_model.ir_version = 10
                    import onnxsim

                    logger.info("simplify onnx model............")
                    try:
                        onnx_model_simp, check = onnxsim.simplify(onnx_model)
                        if check:
                            onnx_model = onnx_model_simp
                    except RuntimeError as exc:
                        logger.warning(f"Skip onnx simplify for qwen2_vl visual model: {exc}")
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
            if tmp_dir_ctx is not None:
                tmp_dir_ctx.cleanup()

    def get_dummy_inputs(self) -> Any:
        return {"image": self._get_dummy_inputs()["hm_pixel_values"]}

    def _get_dummy_inputs(self) -> Any:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.new("RGB", (self.config.max_size_w, self.config.max_size_h)),
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        processor = self.get_tf_processor()
        return processor.apply_chat_template(messages)

    def init_wrap_model(self, hf_model: Qwen2VLForConditionalGeneration = None):
        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls

        vision_register_wrap_cls(hf_model)
        # Register only the HF visual module as the visual sub-model; the main LLM
        # wrapper owns language generation and calls this sibling explicitly.
        return super().init_wrap_model(hf_model.visual)

    def get_tf_processor(self):
        processor = XHQwen2VLProcessor.from_pretrained(self.hf_model_dir)
        processor.config.patch_size = self.config.patch_size
        processor.config.max_size_h = self.config.max_size_h
        processor.config.max_size_w = self.config.max_size_w
        return processor

    def forward(self, *args, **kwargs):
        return self._inference_model(*args, **kwargs)

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": ["pixel_values"],
            "output_names": ["image_embeds"],
        }

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
        meta_info.spatial_merge_size = 2
        return meta_info
