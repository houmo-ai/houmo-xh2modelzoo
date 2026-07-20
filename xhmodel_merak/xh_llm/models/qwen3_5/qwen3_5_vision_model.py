"""
XHQwen3_5VisionModel — Vision model wrapper for Qwen3.5 xhquant export.

Follows the same pattern as xhquant_llm/models/qwen3_vl/qwen3_vl_vision_model.py,
but adapted for Qwen3.5 (no deepstack features).
"""

import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, cast

import onnx
import torch
from accelerate import init_empty_weights
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForImageTextToText

from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph
from xhquant.utils.registry import DynamicModule, _DMRegistryCls

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...types import VisualModelMeta
from ...utils import get_cpu_memory_mb
from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as XHQwen3_5ForConditionalGeneration
from .modeling_qwen3_5 import Qwen3_5VisionModel as HFQwen3_5VisionModel
from .modeling_qwen3_5_patch import qwen3_5_patch
from .qwen3_5_processor import XHQwen3_5Processor
from .xh_qwen3_5_config import XHQwen3_5_VisualConfig


def _visual_simplified_onnx_needs_refresh(onnx_file: str | Path, simplified_onnx_file: str | Path) -> bool:
    onnx_path = Path(onnx_file)
    simplified_path = Path(simplified_onnx_file)
    return not simplified_path.exists() or simplified_path.stat().st_mtime < onnx_path.stat().st_mtime


class _Qwen3_5_VisualHFCompatible(DynamicModule):  # noqa: N801
    def _setup(self: HFQwen3_5VisionModel, xh_visual_model: "XHQwen3_5VisionModel"):
        dtype = self.dtype
        device = self.device

        del self.blocks
        del self.pos_embed
        del self.patch_embed
        del self.merger
        del self.rotary_pos_emb

        # 没有这行，self.dtype时会报错，因为所有参数都被移除了。
        self._dummy_param = torch.nn.Parameter(torch.empty(0, dtype=dtype, device=device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.visual = xh_visual_model

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.visual.forward(hidden_states, **kwargs)


def build_qwen3_5_visual_hf_compatible_model(
    hf_model: XHQwen3_5ForConditionalGeneration, xh_visual_model: "XHQwen3_5VisionModel"
):
    # 构建一个兼容HF的视觉模型，主要用于导出ONNX
    LLM_COMPATIBLE_MODULES = _DMRegistryCls("XHCompatible")  # noqa: N806
    hf_model_cls = type(hf_model.model.visual)
    if hf_model_cls not in LLM_COMPATIBLE_MODULES:
        LLM_COMPATIBLE_MODULES.register_module(
            {
                hf_model_cls: hf_model_cls.__name__,
            },
            _Qwen3_5_VisualHFCompatible,
        )
    LLM_COMPATIBLE_MODULES.convert(hf_model.model.visual, xh_visual_model=xh_visual_model)
    from .modeling_qwen3_5_patch import get_image_features

    hf_model.model.get_image_features = types.MethodType(get_image_features, hf_model.model)

    return hf_model


@register_llm_model("Qwen3_5ForConditionalGeneration_visual", master=False)
class XHQwen3_5VisionModel(BaseVisionModel):  # noqa: N801
    transformers_min_version = "5.2.0"
    HF_MODEL_CLS = XHQwen3_5ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    VISUAL_HF_MODEL_CLS = HFQwen3_5VisionModel
    VISUAL_WEIGHT_PREFIX = "model.visual."
    # HMONNXINFERENCE_CLS = Qwen3HMONNXModel
    BUILD_HF_COMPATIBLE_FUNC = build_qwen3_5_visual_hf_compatible_model

    META_CLS = VisualModelMeta
    CONFIG_CLS = XHQwen3_5_VisualConfig
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen3_5.workflow:Qwen35Workflow"

    def __init__(self, config: XHQwen3_5_VisualConfig):
        super().__init__(config)
        qwen3_5_config = AutoConfig.from_pretrained(self.hf_model_dir)
        assert qwen3_5_config.vision_config.patch_size == config.patch_size
        assert qwen3_5_config.vision_config.temporal_patch_size == config.temporal_patch_size
        self.config = cast(XHQwen3_5_VisualConfig, self.config)
        if self.config.model_type is None:
            self.config.model_type = "Qwen3_5ForConditionalGeneration_visual"

    def _to_eager(self, aligned: bool = True):
        raise NotImplementedError("Eager mode is not implemented for vision model yet.")

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
            onnx_file = str(Path(work_dir) / "onnx" / "qwen3_5_visual.onnx")
            Path(onnx_file).parent.mkdir(parents=True, exist_ok=True)
            if not Path(onnx_file).exists():
                logger.info("Visual ONNX memory before torch export: %s", get_cpu_memory_mb())
                torch.onnx.export(
                    wrap_model.float().cpu(),
                    (dummy_input.float().cpu(),),
                    onnx_file,
                    export_params=True,
                    external_data=True,
                    opset_version=18,
                    do_constant_folding=True,
                    input_names=["pixel_values"],
                    output_names=["image_embeds"],
                    verbose=False,
                )
                logger.info("Visual ONNX memory after torch export: %s", get_cpu_memory_mb())
            else:
                logger.info(f"from cached onnx: {onnx_file}")

            # ``to_fronted`` releases the wrapped model immediately after this
            # method returns; quantization/export only consume the frontend
            # graph. Release visual weights as soon as the ONNX file is fully
            # serialized instead of retaining another BF16/FP32 copy through
            # simplify and GraphSurgeon import.
            wrap_model.to_empty(device="meta")
            self._trim_cpu_allocator()
            logger.info("Visual ONNX memory after releasing wrap model: %s", get_cpu_memory_mb())

            simplified_onnx_file = str(Path(onnx_file).with_name("qwen3_5_visual_simplified.onnx"))
            if _visual_simplified_onnx_needs_refresh(onnx_file, simplified_onnx_file):
                # Qwen3.5 visual needs ONNX simplification (notably to make
                # Split sizes constant), but doing it inside the decoder keeps
                # the original ModelProto, simplified ModelProto and imported
                # GraphSurgeon graph alive in one process.  Run simplification
                # in a short-lived process so all of its protobuf/onnxsim
                # allocations are returned to the OS before frontend import.
                logger.info("Visual ONNX memory before simplify subprocess: %s", get_cpu_memory_mb())
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "xhmodel_merak.xh_llm.models.qwen3_5._visual_onnx_simplify",
                        onnx_file,
                        simplified_onnx_file,
                    ],
                    check=True,
                )
                self._trim_cpu_allocator()
                logger.info("Visual ONNX memory after simplify subprocess: %s", get_cpu_memory_mb())

            # Load once here and pass the ModelProto to xhquant. Passing the
            # path makes ONNXPythonDecoder load it and then run a second
            # file-based checker parse while the first 2+ GiB ModelProto is
            # still resident. The cached file is already simplified, so the
            # decoder only needs to import it into GraphSurgeon.
            logger.info("Visual ONNX memory before frontend conversion: %s", get_cpu_memory_mb())
            onnx_model = onnx.load(simplified_onnx_file)
            logger.info("Visual ONNX memory after simplified model load: %s", get_cpu_memory_mb())
            frontend_model = to_frontend_graph(
                onnx_model,
                FrontendType.ONNX,
                [dummy_input],
                simplify=False,
            )
            del onnx_model
            logger.info("Visual ONNX memory after frontend conversion: %s", get_cpu_memory_mb())
            return frontend_model
        finally:
            if _tmp_dir_ctx is not None:
                _tmp_dir_ctx.cleanup()

    def get_dummy_inputs(self) -> Any:
        return {
            "image": self._get_dummy_inputs()["pixel_values"],
        }

    def _get_dummy_inputs(self) -> Any:
        processor = self.get_tf_processor()

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

        inputs = processor.apply_chat_template(messages)

        return {
            "pixel_values": [v.float() for v in inputs["pixel_values"]],
            "image_grid_thw": inputs["image_grid_thw"],
        }

    def init_wrap_model(self, hf_model: XHQwen3_5ForConditionalGeneration = None):
        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls

        vision_register_wrap_cls(hf_model)
        if isinstance(hf_model, self.VISUAL_HF_MODEL_CLS):
            visual = hf_model
        else:
            visual = hf_model.model.visual
        # self.config = visual.config
        wraped_model = super().init_wrap_model(visual)
        return wraped_model

    @classmethod
    def _load_visual_state_dict(cls, hf_model_dir: str) -> dict[str, torch.Tensor]:
        """Load only ``model.visual.*`` tensors from a sharded HF checkpoint."""
        model_dir = Path(hf_model_dir)
        index_file = model_dir / "model.safetensors.index.json"
        if index_file.is_file():
            weight_map = json.loads(index_file.read_text(encoding="utf-8"))["weight_map"]
            weight_files = sorted(
                {
                    model_dir / filename
                    for name, filename in weight_map.items()
                    if name.startswith(cls.VISUAL_WEIGHT_PREFIX)
                }
            )
        else:
            weight_files = sorted(model_dir.glob("*.safetensors"))

        state_dict: dict[str, torch.Tensor] = {}
        for weight_file in weight_files:
            with safe_open(weight_file, framework="pt", device="cpu") as checkpoint:
                for full_name in checkpoint.keys():
                    if not full_name.startswith(cls.VISUAL_WEIGHT_PREFIX):
                        continue
                    name = full_name.removeprefix(cls.VISUAL_WEIGHT_PREFIX)
                    state_dict[name] = checkpoint.get_tensor(full_name)
        if not state_dict:
            raise RuntimeError(f"No visual weights with prefix {cls.VISUAL_WEIGHT_PREFIX!r} found in {hf_model_dir}")
        return state_dict

    def get_native_model(self):
        """Materialize only the visual tower instead of the complete VLM.

        Qwen3.5 AutoRound checkpoints keep the visual weights in BF16 while the
        language model is packed INT4.  Loading the root conditional-generation
        model would immediately dequantize the whole language model even though
        visual export only needs ``model.visual.*``.
        """
        hf_config = AutoConfig.from_pretrained(self.hf_model_dir)
        with init_empty_weights():
            visual = self.VISUAL_HF_MODEL_CLS._from_config(hf_config.vision_config)

        state_dict = self._load_visual_state_dict(self.hf_model_dir)
        visual.load_state_dict(state_dict, strict=True, assign=True)
        visual.eval()
        get_xhquant_logger().info(
            "Loaded visual-only checkpoint: %d tensors, %.3f GiB",
            len(state_dict),
            sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values()) / 1024**3,
        )
        return visual

    def get_tf_processor(self):
        processor = XHQwen3_5Processor.from_pretrained(self.hf_model_dir)
        processor.config.patch_size = self.config.patch_size
        processor.config.max_size_h = self.config.max_size_h
        processor.config.max_size_w = self.config.max_size_w
        return processor

    def forward(self, *args, **kwargs):
        out = self._inference_model(*args, **kwargs)
        return out

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        native_model = super().get_hf_model(hf_model_dir, quant_weight, **kwargs)
        return qwen3_5_patch(native_model)

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        native_hf_model = super().get_empty_hf_model(hf_model_dir)
        native_hf_model = qwen3_5_patch(native_hf_model)
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
