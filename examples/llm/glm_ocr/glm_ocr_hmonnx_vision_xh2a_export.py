import argparse
import sys
import tempfile
import time
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import xhquant.utils.suppress_printing
from PIL import Image, ImageOps
from xhquant.api import (
    ConfigDict,
    HMONNXInference,
    set_random_seed,
)
from torch import Tensor
from transformers import AutoModelForImageTextToText
from transformers.modeling_outputs import BaseModelOutputWithPooling

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import xhquant_llm_init, get_root_logger
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.glm_ocr import XHGlmOcrVisionModel, GlmOcrProcessor
from xh_model_zoo.xh_llm.models.glm_ocr.utils import build_inputs, build_messages
from xh_model_zoo.utils.memory_tracker import MemoryTracker
from xh_model_zoo.utils.time_profiler import TimeProfiler


def load_and_process_image(image_path: str, target_w: int, target_h: int):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    if (orig_w, orig_h) != (target_w, target_h):
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=(114, 114, 114))
    return image


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hf_model_dir", type=str, default="/data02/datasets/GLM-OCR",
                        help="HuggingFace model directory")
    parser.add_argument("--work_dir", type=str, default="work_dirs/glm_ocr_vision_xh2a_export_hmonnx",
                        help="output work directory")
    parser.add_argument("--image_path", type=str, default="examples/llm/glm_ocr/data/img3.png")
    parser.add_argument("--prompt", type=str, default="Text Recognition:")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--valid", default=True, action="store_true", help="validate the exported model")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--image_size_w", type=int, default=672, help="max image width")
    parser.add_argument("--image_size_h", type=int, default=672, help="max image height")
    parser.add_argument("--max_size_t", type=int, default=2, help="max temporal size")
    parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="temporal patch size")
    return parser


def _export_impl(cfg_name, work_dir, device, execution_device, dtype, model_cfg, quant_config, args):
    logger = get_root_logger()
    is_valid_model = args.valid
    max_size_w = args.image_size_w
    max_size_h = args.image_size_h

    # -------------------------------------------------------------------------
    # 1. Build vision model through MODELS registry (BaseModel workflow)
    # -------------------------------------------------------------------------
    glm_ocr_vision_model: XHGlmOcrVisionModel = MODELS.build(model_cfg)
    native_model = glm_ocr_vision_model.get_hf_model()
    hf_model_dir = glm_ocr_vision_model.hf_model_dir

    # -------------------------------------------------------------------------
    # 2. Prepare sample input using processor
    # -------------------------------------------------------------------------
    processor = GlmOcrProcessor.from_pretrained(hf_model_dir)
    image = load_and_process_image(args.image_path, target_w=max_size_w, target_h=max_size_h)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    inputs = build_inputs(processor, messages, device=execution_device)

    pixel_values = inputs["pixel_values"]

    if args.batch_size > 1:
        pixel_values = pixel_values.repeat(args.batch_size, 1)

    # -------------------------------------------------------------------------
    # 3. Validate native model BEFORE wrapping
    # -------------------------------------------------------------------------
    if is_valid_model:
        import accelerate
        accelerate.hooks.remove_hook_from_module(native_model, recurse=True)
        native_model.to(execution_device)
        with torch.no_grad():
            generated_ids = native_model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        native_output = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False
        )
        logger.info("***************** native model output *****************")
        logger.info(native_output)
        native_model.cpu()

    # -------------------------------------------------------------------------
    # 3.5. Wrap the vision model
    # -------------------------------------------------------------------------
    glm_ocr_vision_model.init_wrap_model(native_model)
    wraped_model = glm_ocr_vision_model.wrap_model

    # -------------------------------------------------------------------------
    # 4. Export ONNX via torch.onnx.export on wrapped model
    # -------------------------------------------------------------------------
    out_onnx_file = str(Path(work_dir) / "onnx" / f"visual_{args.batch_size}.onnx")
    Path(out_onnx_file).parent.mkdir(exist_ok=True, parents=True)

    if Path(out_onnx_file).exists():
        logger.info(f"onnx model already exists: {out_onnx_file}")
        onnx_model = onnx.load(out_onnx_file, load_external_data=True)
    else:
        wraped_model.float().eval().cpu()

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_file = str(Path(tmp_dir) / "visual.onnx")
            logger.info(f"exporting onnx model to {onnx_file}")
            torch.onnx.export(
                wraped_model,
                (pixel_values.float().cpu(),),
                onnx_file,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["pixel_values"],
                output_names=["image_embeds"],
                verbose=True,
            )
            onnx_model = onnx.load(onnx_file, load_external_data=True)

        import onnx_graphsurgeon as gs
        ir_version = onnx_model.ir_version
        graph = gs.import_onnx(onnx_model)
        graph.toposort()
        graph.fold_constants()
        graph.cleanup()
        onnx_model = gs.export_onnx(graph)
        onnx_model.ir_version = ir_version

        onnx.save(
            onnx_model,
            out_onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="visual_external_data",
            convert_attribute=True,
        )

    # -------------------------------------------------------------------------
    # 5. Convert ONNX to HMONNX
    # -------------------------------------------------------------------------
    out_hmonnx_file = Path(work_dir) / "vision" / f"{cfg_name}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)

    if not out_hmonnx_file.exists():
        from xhquant.api import DeviceType, convert_onnx_to_hmonnx

        input_args = [pixel_values.float().cpu()]
        convert_onnx_to_hmonnx(
            out_onnx_file,
            input_args,
            DeviceType.XH2a,
            str(out_hmonnx_file),
            quant_config,
        )
        logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}")

    # -------------------------------------------------------------------------
    # 6. Validate HMONNX model (optional)
    # -------------------------------------------------------------------------
    if not is_valid_model:
        return

    class HMONNXWrapModel(nn.Module):
        def __init__(self, model, device, spatial_merge_size=2):
            super().__init__()
            self._model = model
            self.device = device
            self.spatial_merge_size = spatial_merge_size

        @property
        def dtype(self):
            return torch.float16

        @torch.no_grad()
        def forward(self, pixel_values, grid_thw=None, return_dict=True, **kwargs):
            out = self._model(pixel_values.half())
            if return_dict:
                return BaseModelOutputWithPooling(last_hidden_state=out, pooler_output=out)
            return (out,)

    hm_session = HMONNXInference(out_hmonnx_file)
    hm_session.to("cpu")
    hm_session.exec_device = execution_device
    xh_model = HMONNXWrapModel(hm_session, device=execution_device)
    native_model.to(execution_device)
    native_model.model.visual = xh_model

    with torch.no_grad():
        generated_ids = native_model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    hmonnx_output = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False
    )
    logger.info("***************** hmonnx model output *****************")
    logger.info(hmonnx_output)


def main(args):
    work_dir = args.work_dir
    cfg_name = "glm_ocr_vision_xh2a_export_hmonnx"

    log_file = Path(work_dir) / f"{cfg_name}_debug.log"
    Path(work_dir).mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    execution_device = device
    dtype = getattr(torch, "float16")

    set_random_seed(args.seed)

    xhquant_llm_init(log_file, args.debug)
    logger = get_root_logger()

    # Build quant_config and model dict inline (no external config file)
    quant_config = dict(
        inputs=dict(
            pixel_values=dict(
                quantizer=dict(
                    qspec=dict(fake_dtype="float16"),
                )
            ),
        ),
        w_schema=dict(bits=8, fp_mode="sefp"),
        act_schema=dict(bits=16, fp_mode="sefp"),
    )

    model_cfg = ConfigDict(dict(
        type="XHGlmOcrVisionModel",
        hf_model=args.hf_model_dir,
        wrap_cfg=dict(
            max_size_w=args.image_size_w,
            max_size_h=args.image_size_h,
            max_size_t=args.max_size_t,
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
        ),
        quant_config=quant_config,
        export_cfg=dict(
            input_names=["pixel_values"],
            output_names=["image_embeds"],
        ),
    ))

    logger.info(f"Args: {args}")

    xhquant.utils.suppress_printing.disable_printing = True
    with TimeProfiler(f"{cfg_name} export", logger), MemoryTracker(0, "export", logger):
        _export_impl(cfg_name, work_dir, device, execution_device, dtype, model_cfg, quant_config, args)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)
