"""Gemma4 MoE visual ONNX/HMONNX export in Merak style.

This entry keeps the source xhquant_llm visual export contract while using the
Merak model/config layout. It exports the Gemma4 vision tower plus
``embed_vision`` as a standalone visual encoder and writes ``export_meta_info``
with the fields consumed by downstream VLM generation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


def _bootstrap_gemma4_transformers() -> None:
    """Prefer the workspace vendored transformers 5.5 Gemma4 package.

    The xhquant conda env has transformers 5.0, which does not register
    ``gemma4``. The workspace vendor tree contains a full transformers package,
    but also contains a partial ``regex`` package. A symlink-only path exposes
    only ``transformers`` so dependencies still come from the conda env.
    """

    this_file = Path(__file__).resolve()
    workspace_root = this_file.parents[4]
    vendor_transformers = workspace_root / ".vendor" / "python" / "transformers"
    if not vendor_transformers.exists():
        return
    link_root = Path(tempfile.gettempdir()) / "xh2a_vendor_transformers_only"
    link_root.mkdir(parents=True, exist_ok=True)
    link_path = link_root / "transformers"
    if link_path.exists() or link_path.is_symlink():
        if not link_path.is_symlink() or link_path.resolve() != vendor_transformers:
            if link_path.is_dir() and not link_path.is_symlink():
                shutil.rmtree(link_path)
            else:
                link_path.unlink()
    if not link_path.exists():
        link_path.symlink_to(vendor_transformers, target_is_directory=True)
    sys.path.insert(0, str(link_root))


_bootstrap_gemma4_transformers()

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

from examples_merak.llm.gemma4_moe.gemma4_moe_visual_preprocess import (
    configure_gemma4_visual_processor,
    extract_valid_patch_tokens,
    prepare_visual_input_image,
    resolve_visual_output_dir,
)
from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
from xhmodel_merak.xh_llm.models.gemma4_moe.gemma4_moe_visual_model import (
    Gemma4MoeVisionWrapper,
    XHGemma4MoeVisualModel,
)
from xhquant.api import Config, get_xhquant_logger, xhquant_init


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@lru_cache(maxsize=1)
def build_xh2a_custom_translation_table() -> dict[Any, Any]:
    from torch.onnx._internal.exporter._registration import _get_overload

    from xhquant.export.onnx.xh2a_onnx_registry import xh2a_default_registry

    custom_translation_table = {}
    for qualified_name, aten_overload_func in xh2a_default_registry.items():
        target = _get_overload(qualified_name)
        if target is None:
            continue
        for overload_func in aten_overload_func.overloads:
            custom_translation_table[target] = overload_func
    return custom_translation_table


def export_onnx_with_xh2a_custom_ops(
    model: nn.Module,
    export_args: tuple[torch.Tensor, ...],
    onnx_file: str,
    input_names: list[str],
    output_names: list[str],
) -> None:
    onnx_program = torch.onnx.export(
        model,
        export_args,
        None,
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        do_constant_folding=True,
        custom_translation_table=build_xh2a_custom_translation_table(),
        dynamo=True,
        optimize=True,
    )
    onnx_program.save(onnx_file)


def export_onnx_plain(
    model: nn.Module,
    export_args: tuple[torch.Tensor, ...],
    onnx_file: str,
    input_names: list[str],
    output_names: list[str],
) -> None:
    torch.onnx.export(
        model,
        export_args,
        onnx_file,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
    )


def patch_gemma4_rmsnorm_for_export(fuse_norm: bool) -> None:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
    from xhquant.backend.xh2a import torch_ops_xh2a_rmsnorm

    def _safe_norm(self, hidden_states: torch.Tensor):
        max_val = hidden_states.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
        scaled = hidden_states / max_val
        variance = (scaled * scaled).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.reciprocal(max_val * torch.sqrt(variance))

    def _get_export_weight(self, hidden_states: torch.Tensor):
        if self.with_scale:
            return self.weight.float()
        export_weight = getattr(self, "_xh2a_export_unit_weight", None)
        hidden_size = hidden_states.shape[-1]
        if (
            export_weight is None
            or export_weight.shape != (hidden_size,)
            or export_weight.dtype != hidden_states.dtype
            or export_weight.device != hidden_states.device
        ):
            export_weight = torch.ones(hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
            if "_xh2a_export_unit_weight" in self._buffers:
                self._buffers["_xh2a_export_unit_weight"] = export_weight
            else:
                self.register_buffer("_xh2a_export_unit_weight", export_weight, persistent=False)
        return export_weight

    def _exportable_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_fp32 = hidden_states.float()
        if torch.onnx.is_in_onnx_export():
            weight = _get_export_weight(self, hidden_states_fp32)
            normed_output = torch_ops_xh2a_rmsnorm(hidden_states_fp32, weight, self.eps, -1, "normal", 0, True)
        else:
            if not self.with_scale:
                _get_export_weight(self, hidden_states_fp32)
            normed_output = _safe_norm(self, hidden_states_fp32)
            if self.with_scale:
                normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)

    if fuse_norm:
        Gemma4RMSNorm.forward = _exportable_forward
    else:
        Gemma4RMSNorm._norm = _safe_norm


def _create_default_image() -> Path:
    image_path = Path(tempfile.gettempdir()) / "gemma4_moe_visual_default.png"
    if not image_path.exists():
        Image.new("RGB", (448, 448), color=(255, 255, 255)).save(image_path)
    return image_path


def _load_cfg(args: argparse.Namespace) -> tuple[str, Config]:
    if args.config:
        cfg = Config.fromfile(args.config)
        if args.upsample_token is not None:
            cfg.model.upsample_token = args.upsample_token
        if args.fuse_norm is not None:
            cfg.model.fuse_norm = args.fuse_norm
        return Path(args.config).stem, cfg
    if not args.model:
        raise ValueError("Either --config or --model must be specified.")
    hf_model_path = str(Path(args.model).resolve())
    cfg_name = f"gemma4_moe_visual_{Path(hf_model_path).name}_xh2a_448x448"
    cfg = Config(
        dict(
            model=dict(
                model_type="Gemma4ForConditionalGeneration_visual",
                hf_model=hf_model_path,
                model_name="xh2_gemma4_moe_visual_w8a8",
                quant_scheme=dict(quant_type="w8a8h1_sefp", ops={}),
                max_size_w=args.image_size_w,
                max_size_h=args.image_size_h,
                upsample_token=False if args.upsample_token is None else args.upsample_token,
                fuse_norm=True if args.fuse_norm is None else args.fuse_norm,
            )
        )
    )
    return cfg_name, cfg


def export_visual(args: argparse.Namespace) -> Path:
    cfg_name, cfg = _load_cfg(args)
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    xh_visual_model: XHGemma4MoeVisualModel = AutoLLMModel.from_pretrained(model_cfg)

    output_base = Path(args.output) if args.output else Path("work_dirs") / cfg_name
    output_dir = resolve_visual_output_dir(
        output_base,
        xh_visual_model.config.upsample_token,
        (xh_visual_model.config.max_size_w, xh_visual_model.config.max_size_h),
    )
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    onnx_dir = output_dir / "vision_onnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)

    xhquant_init(str(output_dir / "visual_export.log"), args.debug)
    logger = get_xhquant_logger()
    logger.info(f"Vision export dir: {output_dir}")

    dtype = torch.bfloat16
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    if args.metadata_only:
        processor = AutoProcessor.from_pretrained(xh_visual_model.hf_model_dir, trust_remote_code=True)
        processor_pooling_kernel_size = configure_gemma4_visual_processor(
            processor,
            xh_visual_model.config.upsample_token,
        )
        image_path = Path(args.image) if args.image else _create_default_image()
        processed_image, preprocess_meta = prepare_visual_input_image(
            image_path,
            upsample_token=xh_visual_model.config.upsample_token,
            target_image_size=(xh_visual_model.config.max_size_w, xh_visual_model.config.max_size_h),
        )
        inputs = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_image},
                        {"type": "text", "text": "Describe."},
                    ],
                }
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        _, _, valid_mask = extract_valid_patch_tokens(inputs["pixel_values"], inputs["image_position_ids"])
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "hf_model": xh_visual_model.hf_model_dir,
            "vision_onnx": str(Path("vision_onnx") / "vision_encoder.onnx"),
            "vision_variant": preprocess_meta["vision_variant"],
            "upsample_token": preprocess_meta["upsample_token"],
            "fuse_norm": xh_visual_model.config.fuse_norm,
            "image_preprocess": preprocess_meta,
            "vision_config": {
                "patch_size": processor.image_processor.patch_size,
                "processor_pooling_kernel_size": processor_pooling_kernel_size,
            },
            "image_embeds_shape": [],
            "valid_mask": valid_mask.tolist(),
            "n_valid_patches": int(valid_mask.sum().item()),
            "n_total_patches": int(inputs["pixel_values"].shape[1]),
            "n_input_tokens": int(valid_mask.sum().item()),
            "output_length": int(valid_mask.sum().item()) // processor.image_processor.pooling_kernel_size**2,
            "metadata_only": True,
        }
        with open(output_dir / "export_meta_info.json", "w") as f:
            json.dump(meta, f, indent=4)
        return output_dir

    logger.info(f"Loading Gemma4 model from {xh_visual_model.hf_model_dir} ...")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        xh_visual_model.hf_model_dir,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()

    processor = AutoProcessor.from_pretrained(xh_visual_model.hf_model_dir, trust_remote_code=True)
    processor_pooling_kernel_size = configure_gemma4_visual_processor(
        processor,
        xh_visual_model.config.upsample_token,
    )
    vision_tower = model.model.vision_tower
    embed_vision = model.model.embed_vision
    vision_config = model.config.vision_config
    vision_config._attn_implementation = "eager"
    vision_config.pooling_kernel_size = processor_pooling_kernel_size
    patch_gemma4_rmsnorm_for_export(xh_visual_model.config.fuse_norm)

    vision_wrapper = Gemma4MoeVisionWrapper(vision_tower, embed_vision, vision_config).eval().to(device).to(dtype)
    image_path = Path(args.image) if args.image else _create_default_image()
    processed_image, preprocess_meta = prepare_visual_input_image(
        image_path,
        upsample_token=xh_visual_model.config.upsample_token,
        target_image_size=(xh_visual_model.config.max_size_w, xh_visual_model.config.max_size_h),
    )
    inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_image},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_position_ids = inputs["image_position_ids"].to(device=device)
    with torch.no_grad():
        vision_wrapper.precompute_constants(pixel_values, image_position_ids)
    pixel_values_valid, _, valid_mask = extract_valid_patch_tokens(pixel_values, image_position_ids)
    with torch.no_grad():
        image_embeds = vision_wrapper(pixel_values_valid)

    onnx_file = str(onnx_dir / "vision_encoder.onnx")
    vision_wrapper_fp32 = vision_wrapper.float().cpu()
    pixel_values_cpu = pixel_values_valid.float().cpu()
    input_names = ["pixel_values"]
    output_names = ["image_embeds"]
    export_args = (pixel_values_cpu,)
    if xh_visual_model.config.fuse_norm:
        export_onnx_with_xh2a_custom_ops(vision_wrapper_fp32, export_args, onnx_file, input_names, output_names)
    else:
        export_onnx_plain(vision_wrapper_fp32, export_args, onnx_file, input_names, output_names)

    hf_config_dir = output_dir / "hf_config"
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    for name in ["config.json", "processor_config.json", "tokenizer_config.json", "tokenizer.json"]:
        src = Path(xh_visual_model.hf_model_dir) / name
        if src.exists():
            shutil.copyfile(src, hf_config_dir / name)

    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hf_model": xh_visual_model.hf_model_dir,
        "vision_onnx": str(Path(onnx_file).relative_to(output_dir)),
        "vision_variant": preprocess_meta["vision_variant"],
        "upsample_token": preprocess_meta["upsample_token"],
        "fuse_norm": xh_visual_model.config.fuse_norm,
        "image_preprocess": preprocess_meta,
        "vision_config": {
            "hidden_size": vision_config.hidden_size,
            "num_hidden_layers": vision_config.num_hidden_layers,
            "patch_size": vision_config.patch_size,
            "default_output_length": vision_config.default_output_length,
            "processor_pooling_kernel_size": processor_pooling_kernel_size,
        },
        "image_embeds_shape": list(image_embeds.shape),
        "valid_mask": valid_mask.tolist(),
        "n_valid_patches": int(valid_mask.sum().item()),
        "n_total_patches": int(pixel_values.shape[1]),
        "n_input_tokens": int(pixel_values_valid.shape[1]),
        "output_length": vision_wrapper._output_length,
    }
    with open(output_dir / "export_meta_info.json", "w") as f:
        json.dump(meta, f, indent=4)

    if args.golden:
        from xhquant.api import DeviceType, HMONNXGoldenInference, convert_onnx_to_hmonnx

        hmonnx_dir = output_dir / "vision_hmonnx"
        hmonnx_dir.mkdir(exist_ok=True, parents=True)
        hmonnx_file = hmonnx_dir / "vision_encoder.onnx"
        convert_onnx_to_hmonnx(onnx_file, [pixel_values_cpu], DeviceType.XH2a, str(hmonnx_file), None)
        meta["vision_hmonnx"] = str(hmonnx_file.relative_to(output_dir))
        with open(output_dir / "export_meta_info.json", "w") as f:
            json.dump(meta, f, indent=4)
        golden_dir = output_dir / "golden" / "vision"
        golden_dir.mkdir(exist_ok=True, parents=True)
        hm_model = HMONNXGoldenInference(hmonnx_file)
        hm_model.save_golden = True
        hm_model.exec_device = device
        hm_model.golden_dir = str(golden_dir)
        with torch.no_grad():
            hm_model.forward(pixel_values_valid.half().to(device))
    logger.info("Vision export done.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs_merak/xh2a/llm_models/gemma4_moe/26b_a4b_it/gemma4_moe_visual_26b_a4b_it_xh2a_448x448.py")
    parser.add_argument("--model", type=str, default="/data01/datasets/gemma-4-26B-A4B-it")
    parser.add_argument("--output", type=str, default="work_dirs/gemma4_moe_26b_a4b_it_vision_xh2a/")
    parser.add_argument("--image", type=str, default="data/images/bee.jpg")
    parser.add_argument("--image-size-w", type=int, default=448)
    parser.add_argument("--image-size-h", type=int, default=448)
    parser.add_argument("--upsample-token", type=str2bool, default=None)
    parser.add_argument("--fuse-norm", type=str2bool, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--metadata-only", action="store_true", help="Smoke mode: write metadata without loading weights.")
    args = parser.parse_args()
    output_dir = export_visual(args)
    print(f"Vision export output: {output_dir}")


if __name__ == "__main__":
    main()