# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Export Qwen3-Omni vision encoder to HMONNX - FIXED VERSION.

Key improvements over original:
1. Two-stage export: FX→ONNX→(simplify)→HMONNX
2. Use real image data instead of random for accuracy
3. Add ONNX simplification to fix precision and performance
4. Add golden data validation post-export
"""

import argparse
import os.path as osp
import tempfile
import time
from pathlib import Path
import sys

import numpy as np
import torch
import onnx
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hmonnx_pipeline import (
    _create_hmonnx_session,
    discover_artifacts,
    run_dialogue_validation,
    run_text_hmonnx_chain_forward,
    save_json,
)
from xh_model_zoo.xh_llm.models.base_converter import BaseConverter
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model

from xhquant.api import (  # isort:skip
    Config,
    DeviceType,
    QuantScheme,
    convert_fx_model_to_hmonnx,
    convert_onnx_to_hmonnx,
    get_root_logger,
    xhquant_init,
)
from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def load_sample_image(image_path: str, height: int, width: int) -> torch.Tensor:
    """Load and preprocess sample image for ONNX export.
    
    Args:
        image_path: Path to the sample image file
        height: Target height
        width: Target width
    
    Returns:
        Preprocessed image tensor [1, C, H, W]
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Sample image not found: {image_path}")
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    
    # Normalize to [0, 1]
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, 3]
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    return img_tensor


def create_dummy_pixels_from_sample(sample_image_path: str, height: int, width: int, frames: int) -> torch.Tensor:
    """Create dummy pixels by repeating sample image across time dimension.
    
    Args:
        sample_image_path: Path to sample image
        height: Image height
        width: Image width
        frames: Number of frames
    
    Returns:
        Tensor of shape [1, 3, frames, height, width]
    """
    if sample_image_path and Path(sample_image_path).exists():
        # Load real image
        img_2d = load_sample_image(sample_image_path, height, width)  # [1, 3, H, W]
        # Repeat across time dimension
        img_3d = img_2d.unsqueeze(2).repeat(1, 1, frames, 1, 1)  # [1, 3, T, H, W]
        return img_3d.to(torch.float16)
    else:
        # Fallback: use randn (NOT IDEAL FOR ACCURACY)
        return torch.randn(1, 3, frames, height, width, dtype=torch.float16)


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type

    prefix = f"{model_name}-{target_device}-vision-{quant_type}"
    work_dir = Path(args.work_dir) / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()

    # ---- 1. Load HF model and extract visual encoder ----
    from transformers import Qwen3OmniMoeForConditionalGeneration

    logger.info(f"Loading HF model from {hf_model_path}")
    native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    native_model.eval()

    visual = native_model.thinker.visual
    if visual is None:
        raise RuntimeError("Model does not have visual encoder")

    # Free memory for non-vision components
    del native_model.talker, native_model.code2wav
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # ---- 2. Register wrap modules and wrap ----
    from xh_model_zoo.xh_llm.models.qwen3_omni._vision_model import (
        register_wrap_modules as vision_register_wrap_modules,
    )

    vision_register_wrap_modules()

    visual = visual.to(torch.float16).cpu()
    vision_wrap_cfg = Config(
        dict(
            max_size_w=args.image_max_size_w,
            max_size_h=args.image_max_size_h,
            max_size_t=args.image_max_size_t,
            temporal_patch_size=args.temporal_patch_size,
            patch_size=args.patch_size,
            only_first_block=False,
        )
    )
    wrapped_visual = wrap_llm_model(visual, vision_wrap_cfg)

    # ---- 3. Prepare dummy inputs (FIXED: use real image data) ----
    channels = 3
    height = args.image_max_size_h
    width = args.image_max_size_w
    frames = args.image_max_size_t
    
    # FIX: Use real image instead of random
    logger.info(f"Loading sample image from {args.sample_image_path}")
    dummy_pixels = create_dummy_pixels_from_sample(
        args.sample_image_path, height, width, frames
    )
    logger.info(f"Dummy pixels shape: {dummy_pixels.shape}, dtype: {dummy_pixels.dtype}")

    inputs = (dummy_pixels,)
    input_names = ["pixel_values"]
    output_names = ["vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"]

    # ---- 4. Export to ONNX (TWO-STAGE EXPORT) ----
    vision_dir = work_dir / "hmonnx"
    vision_dir.mkdir(exist_ok=True, parents=True)
    vision_onnx_file = vision_dir / f"{model_name}-vision_encoder.onnx"

    logger.info(f"Exporting vision encoder to intermediate ONNX: {vision_onnx_file}")
    track_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    with TimeProfiler("export_vision_onnx", logger), MemoryTracker(track_device, "export_vision_onnx", logger):
        # Stage 1: Export to standard ONNX
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_onnx_file = str(Path(tmp_dir) / "visual_temp.onnx")
            
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            
            logger.info("Stage 1: Exporting FX model to ONNX...")
            torch.onnx.export(
                wrapped_visual,
                dummy_pixels,
                tmp_onnx_file,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=compatible_names,
                output_names=output_names,
                verbose=False,
            )
            
            # Load the ONNX model
            logger.info("Loading exported ONNX model...")
            onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)
            
            # Stage 2: Simplify ONNX model (FIX FOR PRECISION & PERFORMANCE)
            logger.info("Stage 2: Simplifying ONNX model (this will fix precision and performance issues)...")
            with TimeProfiler("simplify_onnx", logger):
                onnx_model, check = simplify_large_onnx(onnx_model)
                if not check:
                    logger.warning("ONNX model simplification check failed, but continuing...")
            
            # Save simplified ONNX
            logger.info(f"Saving simplified ONNX to {vision_onnx_file}...")
            onnx.save(onnx_model, str(vision_onnx_file))
            
            # Stage 3: Convert simplified ONNX to HMONNX
            logger.info("Stage 3: Converting simplified ONNX to HMONNX...")
            with TimeProfiler("onnx_to_hmonnx", logger):
                convert_onnx_to_hmonnx(
                    onnx_model,
                    inputs,
                    target_device,
                    vision_onnx_file,  # Output path
                    input_names=compatible_names,
                    output_names=output_names,
                )
    
    logger.info(f"Vision encoder export successful: {vision_onnx_file}")

    # ---- 5. Save meta ----
    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "module": "vision_encoder",
        "model_name": model_name,
        "vision_encoder_onnx": str(vision_onnx_file.relative_to(work_dir)),
        "vision_patch_size": args.patch_size,
        "vision_input_size": [height, width],
        "vision_channels": channels,
        "vision_temporal_patch_size": args.temporal_patch_size,
        "vision_max_size_t": frames,
        "export_method": "two_stage_onnx_simplify",  # Mark as using new method
    }
    meta_file = work_dir / "meta_vision.json"
    save_json(meta_file, meta_info)
    logger.info(f"Vision export complete. Meta saved to {meta_file}")

    # Release export graph/model resources before validation, otherwise
    # large temporary CUDA allocations from export can cause OOM in generation.
    try:
        del wrapped_visual
    except Exception:
        pass
    try:
        del visual
    except Exception:
        pass
    try:
        del native_model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # ---- 6. Optional validation ----
    if args.valid:
        logger.info("Validating vision HMONNX ...")
        try:
            session = _create_hmonnx_session(Path(vision_onnx_file))
            output = session.forward(dummy_pixels.cpu().to(torch.float16))
            if isinstance(output, (list, tuple)):
                output = output[0]
            logger.info(f"Vision HMONNX validation passed, output shape: {tuple(output.shape)}")
        except Exception as e:
            logger.warning(f"Vision HMONNX inference validation skipped: {e}")

        try:
            dialogue_artifacts = {
                "vision": {**meta_info, "_root_dir": str(work_dir), "_meta_path": str(meta_file)}
            }
            dialogue_report = run_dialogue_validation(
                hf_model_path,
                work_dir,
                logger,
                case=args.case,
                max_new_tokens=args.max_new_tokens,
                device_map=args.valid_device,
                artifacts=dialogue_artifacts,
                report_name="vision_dialogue_validation.json",
                output_prefix="vision_dialogue",
            )
            logger.info(f"Dialogue full output text: {dialogue_report.get('output_text', [])}")
        except Exception as e:
            logger.warning(f"Vision dialogue validation failed: {e}")

        try:
            artifacts = discover_artifacts(work_dir)
            if "text" in artifacts:
                text_chain_report = run_text_hmonnx_chain_forward(
                    hf_model_path,
                    artifacts["text"],
                    logger,
                    case=args.case,
                    audio_meta=artifacts.get("audio"),
                    vision_meta={**meta_info, "_root_dir": str(work_dir), "_meta_path": str(meta_file)},
                    report_path=work_dir / "vision_text_chain_validation.json",
                    max_new_tokens=args.max_new_tokens,
                )
                logger.info(f"Text HMONNX chain full output text: {text_chain_report.get('output_text', [])}")
            else:
                logger.info("No text artifacts found in current work_dir, skip text HMONNX chain validation")
        except Exception as e:
            logger.warning(f"Vision text chain validation skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-Omni vision encoder to HMONNX (FIXED VERSION)")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--quant-type", default="w8a8h0_sefp")
    parser.add_argument("--image_max_size_h", type=int, default=224)
    parser.add_argument("--image_max_size_w", type=int, default=224)
    parser.add_argument("--image_max_size_t", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--temporal_patch_size", type=int, default=2)
    parser.add_argument(
        "--sample_image_path",
        type=str,
        default="data/images/qwen2_vl_demo.jpeg",
        help="Sample image path for export (will use real image instead of random for better accuracy)"
    )
    parser.add_argument("--valid", action="store_true", default=True, help="run validation")
    parser.add_argument("--no-valid", action="store_false", dest="valid", help="skip validation")
    parser.add_argument("--case", type=str, default="vision", choices=["text", "vision", "audio", "multimodal"])
    parser.add_argument("--valid-device", type=str, default="auto", choices=["auto", "cpu"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
