#!/usr/bin/env python3
"""Unified Gemma4 series HMONNX export script.

Covers both variants:
  - gemma4_26b_a4b (MoE, ``Gemma4ForConditionalGeneration_with_mask``)
  - gemma4_31b_it   (dense, ``Gemma4ForConditionalGeneration``)

Supports three export modes:
  - ``--mode llm``    : LLM backbone export (default)
  - ``--mode vision`` : standalone visual encoder export (MoE variant only)
  - ``--mode all``    : vision → LLM sequential export

Usage::

    # MoE – full export with golden
    python export_hmonnx.py --variant moe --mode all --model /data01/datasets/gemma-4-26B-A4B-it --golden

    # Dense – LLM-only export (vision is embedded in the LLM pipeline)
    python export_hmonnx.py --variant dense --mode llm --model /data01/models/gemma-4-31B-it --golden

    # Use existing config file
    python export_hmonnx.py --config configs_merak/xh2a/llm_models/gemma4_moe/26b_a4b_it/gemma4_moe_with_mask_26b_a4b_it_xh2a_w8a8_256_2k.py --golden
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # xh2modelzoo/
WORKSPACE_ROOT = SCRIPT_DIR.parents[3]  # project root (for .vendor)
sys.path.insert(0, str(REPO_ROOT))

# Ensure gptqmodel is importable for weight parsing
_gptqmodel_path = "/data01/home/chenzx/project/gerrit/gptqmodel"
if Path(_gptqmodel_path).exists() and _gptqmodel_path not in sys.path:
    sys.path.insert(0, _gptqmodel_path)


# ---------------------------------------------------------------------------
# Bootstrap vendored transformers for gemma4 support
# ---------------------------------------------------------------------------
def _bootstrap_transformers() -> None:
    vendor_transformers = WORKSPACE_ROOT / ".vendor" / "python" / "transformers"
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


_bootstrap_transformers()

import contextlib
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

from examples_merak.llm.gemma4_moe.gemma4_moe_visual_preprocess import (
    configure_gemma4_visual_processor,
    extract_valid_patch_tokens,
    prepare_visual_input_image,
)
from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name, support_llm_model_types
from xhmodel_merak.xh_llm.models.gemma4_moe.gemma4_moe_visual_model import (
    Gemma4MoeVisionWrapper,
)
# Importing gemma4_moe above transitively loads the legacy gemma4 module, which
# registers ``Gemma4ForConditionalGeneration`` first.  Explicitly import gemma4e
# afterwards so it overrides the registry; otherwise AutoLLMConfig/AutoLLMModel
# will silently pick the legacy class and produce a graph signature that does
# not match the gemma4e runtime used by generate.py.
import xhmodel_merak.xh_llm.models.gemma4e  # noqa: F401
from xhquant.api import Config, DeviceType, HMONNXGoldenInference, convert_onnx_to_hmonnx
from xhquant.api import get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import MemoryTracker, TimeProfiler

# ---------------------------------------------------------------------------
# Monkey-patch: prevent CUDA OOM in ``graph_module_guard`` during LLM quant.
#
# ``quant_weight`` buffers (int8 quantized weights) are intentionally kept
# on CPU by ``_detach_quant_weights`` / ``_reattach_quant_weights`` in
# ``gemma4_llm_model.py``.  When ``graph_module_guard`` later calls
# ``graph_module.to(device)``, the ``_apply()`` loop tries to move those
# CPU buffers to GPU, allocating temporary memory that OOMs at 78/79 GiB.
#
# We patch ``graph_module_guard`` to temporarily detach quant_weight buffers
# before the ``.to()`` call and reattach them afterwards, mirroring the
# pattern already used in ``gemma4_llm_model._to_quanted``.
# ---------------------------------------------------------------------------
_OOM_PATCH_INSTALLED = False


def _install_oom_patch() -> None:
    global _OOM_PATCH_INSTALLED
    if _OOM_PATCH_INSTALLED:
        return
    _OOM_PATCH_INSTALLED = True
    import xhquant.common.context as ctx_module

    _orig_guard = ctx_module.graph_module_guard

    @contextlib.contextmanager
    def _patched_guard(graph_module):
        # Extract device info (same as original)
        try:
            device = next(iter(graph_module.parameters())).device
            if device == torch.device("meta"):
                device = None
        except Exception:
            device = None
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        graph_module.recompile()
        yield
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        graph_module.recompile()
        if device:
            # --- Detach quant_weight buffers so .to() won't try to move them ---
            # NOTE: must walk ALL nested submodules (not just direct children of
            # call_module targets), because in dense Gemma4 the nn.Linear holding
            # quant_weight is nested deeper inside wrapper modules. Mirrors the
            # logic of ``_detach_quant_weights`` in gemma4/gemma4_llm_model.py.
            qw_cache: list[tuple[nn.Module, torch.Tensor]] = []
            for _name, sub in graph_module.named_modules():
                buf = sub._buffers.get("quant_weight", None)
                if buf is not None:
                    qw_cache.append((sub, sub._buffers.pop("quant_weight")))
            # --- Move graph to device (quant_weight stays on CPU, no temp alloc) ---
            graph_module.to(device)
            # --- Reattach quant_weight buffers (stays on original device, i.e. CPU) ---
            for sub, buf in qw_cache:
                sub.register_buffer("quant_weight", buf)

    ctx_module.graph_module_guard = _patched_guard  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Companion patch: keeping ``quant_weight`` on CPU above forces the
    # ptq weight-quant kernel (``QLinear.weight_static_quant`` →
    # ``convert_to_ssfp``) to fail with a cuda/cpu device mismatch,
    # because ``self.weight`` was moved to GPU by ``_to_quanted`` while
    # ``self.quant_weight`` stays on CPU. Fix this by lazily migrating
    # ``quant_weight`` to ``self.weight.device`` just before the kernel
    # runs. The qmodule consumes ``quant_weight`` once per Linear and
    # then ``del``s it (see ``_weight_static_quant_aligned``), so peak
    # GPU memory grows by at most one Linear's int8 weight at a time.
    # ------------------------------------------------------------------
    import xhquant.quantization.xh2a.qmodules.linear as _qlin_module

    _orig_weight_static_quant = _qlin_module.QLinear.weight_static_quant

    def _patched_weight_static_quant(self):
        qw = getattr(self, "quant_weight", None)
        if qw is not None and self.weight is not None and qw.device != self.weight.device:
            self.quant_weight = qw.to(self.weight.device)
        return _orig_weight_static_quant(self)

    _qlin_module.QLinear.weight_static_quant = _patched_weight_static_quant  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_VARIANTS = ("moe", "dense")
MODEL_TYPE_MOE = "Gemma4ForConditionalGeneration_with_mask"
MODEL_TYPE_DENSE = "Gemma4ForConditionalGeneration"
MODEL_TYPE_VISUAL = "Gemma4ForConditionalGeneration_visual"

# Default image size for visual encoder export
DEFAULT_IMAGE_SIZE = (448, 448)


# ---------------------------------------------------------------------------
# Vision constants (always use official Gemma4 9× upsampling pipeline)
# ---------------------------------------------------------------------------
_VISION_UPSAMPLE_TOKEN = True   # pooling_kernel_size=3, official upsampling
_VISION_FUSE_NORM = True        # fuse RMSNorm for XH2a ONNX export


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# ONNX export utilities (shared with gemma4_moe visual export)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _build_xh2a_custom_translation_table() -> dict:
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


def _export_onnx_with_xh2a_custom_ops(
    model: nn.Module,
    export_args: tuple,
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
        custom_translation_table=_build_xh2a_custom_translation_table(),
        dynamo=True,
        optimize=True,
    )
    onnx_program.save(onnx_file)


def _export_onnx_plain(
    model: nn.Module,
    export_args: tuple,
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


# ---------------------------------------------------------------------------
# RMSNorm patching (shared with gemma4_moe visual export)
# ---------------------------------------------------------------------------
def _patch_gemma4_rmsnorm_for_export(fuse_norm: bool) -> None:
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


# ---------------------------------------------------------------------------
# Vision export (MoE variant only)
# ---------------------------------------------------------------------------
def _create_default_image() -> Path:
    image_path = Path(tempfile.gettempdir()) / "gemma4_series_visual_default.png"
    if not image_path.exists():
        Image.new("RGB", DEFAULT_IMAGE_SIZE, color=(255, 255, 255)).save(image_path)
    return image_path


def _export_vision_moe(args: argparse.Namespace, logger, temp_dir: Path) -> dict:
    """Export Gemma4 MoE visual encoder (vision_tower + embed_vision).

    Exports ONNX → HMONNX into *temp_dir*, then returns the HMONNX path and
    vision meta so the caller can move them into the final hmquant_* directory.

    Only applicable to the MoE variant. The dense variant handles vision
    inside the LLM export pipeline via ``visual_config``.
    """
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name

    upsample_token = _VISION_UPSAMPLE_TOKEN
    fuse_norm = _VISION_FUSE_NORM

    cfg = Config(
        dict(
            model=dict(
                model_type=MODEL_TYPE_VISUAL,
                hf_model=hf_model_path,
                model_name=f"xh2_gemma4_series_{model_name}_visual_w8a8",
                quant_scheme=dict(quant_type="w8a8h1_sefp", ops={}),
                max_size_w=args.image_size_w,
                max_size_h=args.image_size_h,
                upsample_token=upsample_token,
                fuse_norm=fuse_norm,
            )
        )
    )

    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    from xhmodel_merak.xh_llm.models.gemma4_moe.gemma4_moe_visual_model import (
        XHGemma4MoeVisualModel,
    )
    xh_visual_model: XHGemma4MoeVisualModel = AutoLLMModel.from_pretrained(model_cfg)

    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Vision temp dir: {temp_dir}")
    logger.info(f"Vision config: upsample_token={upsample_token}, fuse_norm={fuse_norm}")

    dtype = torch.bfloat16
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    logger.info(f"Loading Gemma4 model from {xh_visual_model.hf_model_dir} ...")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        xh_visual_model.hf_model_dir,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()

    processor = AutoProcessor.from_pretrained(xh_visual_model.hf_model_dir, trust_remote_code=True)
    processor_pooling_kernel_size = configure_gemma4_visual_processor(processor, upsample_token)
    vision_tower = model.model.vision_tower
    embed_vision = model.model.embed_vision
    vision_config = model.config.vision_config
    vision_config._attn_implementation = "eager"
    vision_config.pooling_kernel_size = processor_pooling_kernel_size
    _patch_gemma4_rmsnorm_for_export(fuse_norm)

    vision_wrapper = Gemma4MoeVisionWrapper(vision_tower, embed_vision, vision_config).eval().to(device).to(dtype)

    image_path = Path(args.image) if args.image else _create_default_image()
    processed_image, preprocess_meta = prepare_visual_input_image(
        image_path,
        upsample_token=upsample_token,
        target_image_size=(xh_visual_model.config.max_size_w, xh_visual_model.config.max_size_h),
    )
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": processed_image}, {"type": "text", "text": "Describe."}]}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_position_ids = inputs["image_position_ids"].to(device=device)

    with torch.no_grad():
        vision_wrapper.precompute_constants(pixel_values, image_position_ids)
    pixel_values_valid, _, valid_mask = extract_valid_patch_tokens(pixel_values, image_position_ids)

    with torch.no_grad():
        image_embeds = vision_wrapper(pixel_values_valid)

    # Export ONNX (temp), then convert to HMONNX directly; discard ONNX.
    onnx_tmp = str(temp_dir / "vision_encoder_onnx_tmp.onnx")
    vision_wrapper_fp32 = vision_wrapper.float().cpu()
    pixel_values_cpu = pixel_values_valid.float().cpu()
    _export_onnx_with_xh2a_custom_ops(vision_wrapper_fp32, (pixel_values_cpu,), onnx_tmp, ["pixel_values"], ["image_embeds"])
    logger.info(f"Vision ONNX (temp) exported to: {onnx_tmp}")

    hmonnx_file = str(temp_dir / "vision_encoder.onnx")
    convert_onnx_to_hmonnx(onnx_tmp, [pixel_values_cpu], DeviceType.XH2a, hmonnx_file, None)
    logger.info(f"Vision HMONNX exported to: {hmonnx_file}")
    # Discard intermediate ONNX
    try:
        Path(onnx_tmp).unlink()
    except OSError:
        pass

    vision_meta = {
        "hf_model": xh_visual_model.hf_model_dir,
        "vision_hmonnx": "vision/vision_encoder.onnx",
        "upsample_token": upsample_token,
        "fuse_norm": fuse_norm,
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

    return {"hmonnx_file": hmonnx_file, "vision_meta": vision_meta, "temp_dir": temp_dir}


# ---------------------------------------------------------------------------
# Golden generation
# ---------------------------------------------------------------------------
def _generate_golden_for_hmonnx(
    hmonnx_file: str,
    golden_dir: str,
    device: str = "cuda",
    logger=None,
) -> None:
    from xhquant.core import CacheTensor

    if logger is None:
        logger = get_xhquant_logger()
    if not Path(hmonnx_file).exists():
        logger.warning(f"HMONNX file not found, skipping golden: {hmonnx_file}")
        return
    logger.info(f"Generating golden for: {hmonnx_file}")
    session = HMONNXGoldenInference(hmonnx_file)
    session.to(torch.device(device))
    session.save_golden = True
    session.golden_dir = golden_dir
    session.initialize()
    input_names = session.get_input_names()
    net_inputs = []
    for input_name in input_names:
        input_tensor_info = session.get_input(input_name)
        if input_tensor_info.dtype in [torch.float32, torch.float16, torch.float64]:
            inp = torch.randn(input_tensor_info.shape, dtype=input_tensor_info.dtype, device=device)
        elif input_tensor_info.dtype in [torch.int32, torch.int64, torch.int16]:
            inp = torch.randint(0, 10, input_tensor_info.shape, dtype=input_tensor_info.dtype, device=device)
        elif input_tensor_info.dtype == torch.bool:
            inp = torch.randint(0, 2, input_tensor_info.shape, dtype=input_tensor_info.dtype, device=device)
        else:
            raise NotImplementedError(f"dtype {input_tensor_info.dtype} not supported for golden generation")
        if "past_key_cache" in input_name or "past_value_cache" in input_name:
            inp = CacheTensor(inp)
        net_inputs.append(inp)
    session(*net_inputs)
    logger.info(f"Golden saved to: {golden_dir}")


def _generate_all_golden(exported_dir: str, device: str = "cuda", logger=None) -> None:
    if logger is None:
        logger = get_xhquant_logger()
    exported_path = Path(exported_dir).resolve()
    # Only final HMONNX outputs (``*_with_act.onnx``). Any ``onnx/`` /
    # ``vision_onnx/`` subdir contains pre-conversion intermediates whose
    # external_data references are not resolvable and would break onnx.load.
    candidates = sorted({p.resolve() for p in exported_path.rglob("*_with_act.onnx")})
    hmonnx_files = [
        f for f in candidates
        if "/onnx/" not in str(f).replace("\\", "/")
        and "/vision_onnx/" not in str(f).replace("\\", "/")
    ]
    if not hmonnx_files:
        logger.warning(f"No hmonnx files found in {exported_dir}")
        return
    for hmonnx_file in hmonnx_files:
        golden_dir = str(hmonnx_file.parent / f"{hmonnx_file.stem}_golden")
        _generate_golden_for_hmonnx(str(hmonnx_file), golden_dir, device, logger)


def _merge_vision_into_golden_meta(hmquant_dir: Path, vision_meta: dict, logger) -> None:
    """Merge vision metadata into the LLM golden_meta_info.json."""
    golden_meta_path = hmquant_dir / "golden_meta_info.json"
    if not golden_meta_path.exists():
        logger.warning(f"golden_meta_info.json not found at {golden_meta_path}, skipping vision merge")
        return
    meta = _load_json(golden_meta_path)
    meta["vision"] = vision_meta
    with open(golden_meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    logger.info(f"Vision info merged into: {golden_meta_path}")

def _copy_processor_config_to_export(hmquant_dir: Path, hf_model_path: str, logger) -> None:
    """Copy processor_config.json from the full HF model to the exported hf_config/ directory.

    This ensures ``AutoProcessor.from_pretrained(hf_config/)`` works correctly
    for vision-language processing during inference, even when the quantized
    model directory omits this file.
    """
    src = Path(hf_model_path) / "processor_config.json"
    if not src.exists():
        logger.warning(f"processor_config.json not found at {src}, skipping copy")
        return
    dst_dir = hmquant_dir / "hf_config"
    dst = dst_dir / "processor_config.json"
    if dst.exists():
        logger.info(f"processor_config.json already exists at {dst}")
        return
    shutil.copy2(str(src), str(dst))
    logger.info(f"Copied processor_config.json from {src} → {dst}")

# ---------------------------------------------------------------------------
# LLM export
# ---------------------------------------------------------------------------
def _build_llm_cfg_from_model(args, logger) -> Tuple[str, Config]:
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = args.chip_arch
    quant_type = args.quant_type
    prefill_chunk_length = args.prefill_chunk_length
    context_length = args.context_length

    variant = args.variant
    if variant == "moe":
        model_type = MODEL_TYPE_MOE
        cfg_name = (
            f"{target_device}_{model_name}_with_mask_{quant_type}_"
            f"{prefill_chunk_length}_{context_length // 1024}k_cli"
        )
        cfg = dict(
            chip_arch=target_device,
            model=dict(
                model_type=model_type,
                hf_model=hf_model_path,
                fallback_hf_model=hf_model_path,
                model_name=model_name,
                context_max_length=context_length,
                prefill_chunk_length=prefill_chunk_length,
                use_cache=True,
                num_logits_to_keep=1,
                quant_scheme=dict(quant_type=quant_type),
                only_first_block=False,
                quant_weight=args.quant_weight,
            ),
        )
    else:
        model_type = MODEL_TYPE_DENSE
        cfg_name = (
            f"{target_device}_{model_name}_{quant_type}_"
            f"{prefill_chunk_length}_{context_length // 1024}k_cli"
        )
        cfg = dict(
            chip_arch=target_device,
            model=dict(
                model_type=model_type,
                hf_model=hf_model_path,
                model_name=model_name,
                context_max_length=context_length,
                prefill_chunk_length=prefill_chunk_length,
                max_pe_length=max(context_length, 32768),
                use_cache=True,
                num_logits_to_keep=1,
                quant_scheme=dict(quant_type=quant_type),
                quant_weight=args.quant_weight,
                only_first_block=False,
                visual_config=dict(
                    image_seq_length=280,
                    patch_size=16,
                    pooling_kernel_size=3,
                    quant_scheme=dict(quant_type=quant_type),
                ),
            ),
        )
    cfg = format_model_name(cfg)
    return cfg_name.lower(), Config(cfg)


def _export_llm(args, logger) -> Path:
    """Export LLM backbone via the Merak pipeline.

    Works for both MoE and dense variants – only the config building differs.
    """
    if args.config:
        cfg_name = Path(args.config).stem
        cfg = Config.fromfile(args.config)
    elif args.model:
        cfg_name, cfg = _build_llm_cfg_from_model(args, logger)
    else:
        raise ValueError("Either --config or --model must be specified.")

    if args.debug:
        cfg_name += "_debug"
    if args.valid:
        cfg_name += "_valid"
        cfg.model.only_first_block = True

    work_dir = args.work_dir or str(Path("./work_dirs") / cfg_name)
    work_dir = Path(work_dir)
    if work_dir.exists():
        if args.force:
            shutil.rmtree(work_dir, ignore_errors=True)
        elif list(work_dir.glob("hmquant_*")):
            logger.info(f"Exported model already exists at {work_dir}, use --force to overwrite.")
            return work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    seed = 1024
    set_random_seed(seed)
    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    dumped_config_file = work_dir / f"{cfg_name}.py"
    cfg.dump(dumped_config_file)

    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")

    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    variant = args.variant
    if variant == "moe":
        expected_cfg = "XHGemma4MoeWithMaskConfig"
    else:
        expected_cfg = "XHGemma4ModelConfig"
    actual_cfg = type(model_cfg).__name__
    assert actual_cfg == expected_cfg, (
        f"Expected config type {expected_cfg}, got {actual_cfg}. "
        f"Is --variant correct? (current: {variant})"
    )
    logger.info(f"Model Config: {actual_cfg}")

    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    if variant == "moe":
        expected_model = "XHGemma4MoeWithMaskModel"
        # Strip visual for MoE (exported separately)
        if hasattr(xh_model, "visual"):
            try:
                delattr(xh_model, "visual")
            except AttributeError:
                pass
        if hasattr(xh_model, "_models") and isinstance(xh_model._models, dict):
            xh_model._models.pop("visual", None)
    else:
        expected_model = "XHGemma4Model"
    actual_model = type(xh_model).__name__
    assert actual_model == expected_model, (
        f"Expected model type {expected_model}, got {actual_model}"
    )

    with TimeProfiler("convert", logger), MemoryTracker(device, "convert2hmonnx", logger):
        xh_model.export_hmonnx(str(work_dir))

    logger.info(f"LLM HMONNX exported to: {work_dir}")

    # Golden – for MoE mode=all, golden is generated in Phase 3 of ``main()``
    # together with the vision golden + meta merge, so skip here to avoid
    # doubling the work.
    skip_llm_golden = (args.mode == "all" and args.variant == "moe")
    if args.golden and not skip_llm_golden:
        exported_dirs = sorted(work_dir.glob("hmquant_*"))
        exported_dir = str(exported_dirs[-1]) if exported_dirs else str(work_dir)
        _generate_all_golden(exported_dir, args.device, logger)

    return work_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Gemma4 Series HMONNX Export (MoE + Dense)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Variant / source
    parser.add_argument(
        "--variant", type=str, default="moe", choices=VALID_VARIANTS,
        help="Model variant: moe (gemma4_26b_a4b) or dense (gemma4_31b_it).",
    )
    parser.add_argument(
        "--mode", type=str, default="llm", choices=["llm", "vision", "all"],
        help="Export mode: llm (LLM only), vision (visual encoder only, MoE variant), all (vision → LLM).",
    )
    parser.add_argument("--model", type=str, default="", help="HF model directory.")
    parser.add_argument(
        "--config", type=str, default="",
        help="Existing config file path (alternative to --model).",
    )

    # LLM export options
    parser.add_argument("--model-type", type=str, default=None, choices=support_llm_model_types)
    parser.add_argument("--chip-arch", type=str, default="XH2a", choices=["XH2a", "YueHui"])
    parser.add_argument("--context-length", type=int, default=2048, help="Max context sequence length.")
    parser.add_argument("--prefill-chunk-length", type=int, default=256, help="Prefill chunk length.")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="Quantization type.")
    parser.add_argument("--quant-weight", type=str, default=None, help="Optional quant weight path (gptq/quarot).")

    # Vision export options
    parser.add_argument("--image-size-w", type=int, default=448, help="Vision export image width.")
    parser.add_argument("--image-size-h", type=int, default=448, help="Vision export image height.")
    parser.add_argument("--image", type=str, default=None, help="Path to reference image for vision export.")

    # Common
    parser.add_argument("--work-dir", type=str, default="", help="Override output work directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (cuda / cpu / cuda:N).")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing export.")
    parser.add_argument("--valid", action="store_true", help="Export only first decoder block (fast smoke).")
    parser.add_argument("--golden", action="store_true", help="Generate golden data after export.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    # When --config is given, auto-detect --variant from the config's model_type.
    _cfg_from_file = None
    if args.config:
        _cfg_from_file = Config.fromfile(args.config)
        _cfg_model = _cfg_from_file.get("model", {})
        _model_type = _cfg_model.get("model_type", "")
        if "with_mask" in _model_type or "moe" in _model_type.lower():
            args.variant = "moe"
        else:
            args.variant = "dense"

    # Set default model paths if not provided.
    if not args.model and not args.config:
        if args.variant == "moe":
            args.model = "/data01/datasets/gemma-4-26B-A4B-it"
        else:
            args.model = "./weights/gemma-4-31B-it"

    # When --config is given and vision export is needed for MoE, extract the
    # float HF model path (fallback_hf_model > hf_model).
    if _cfg_from_file is not None and args.mode in ("vision", "all") and args.variant == "moe":
        llm_cfg = _cfg_from_file.get("model", {})
        float_hf = llm_cfg.get("fallback_hf_model") or llm_cfg.get("hf_model", "")
        if float_hf:
            if not Path(float_hf).is_absolute():
                float_hf = str(Path(float_hf).resolve())
            args.model = float_hf

    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()

    # MoE variant relies on the OOM patch (see _install_oom_patch comment).
    # Dense variant must NOT use it — the patch keeps quant_weight on CPU,
    # which breaks the dense ptq weight-quant kernel that requires
    # quant_weight on the same device as the dequantized weight.
    # Install the OOM patch unconditionally: MoE needs it for the original
    # 78GiB graph_module.to(device) failure; dense (autoround w4a8) also
    # needs it because the dequantized 31B bf16 prefill model plus all int8
    # quant_weight buffers (~93GiB total) cannot coexist on a single 80GiB
    # GPU. The companion ``QLinear.weight_static_quant`` patch (installed
    # together) migrates ``quant_weight`` to GPU lazily per Linear.
    _install_oom_patch()

    if args.device != "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.replace("cuda:", "") if ":" in args.device else args.device)

    logger.info(f"Gemma4 Series Export – variant={args.variant}, mode={args.mode}")

    # ── Phase 1: Vision export (MoE only) → temp dir ──
    # Keep vision temp OUTSIDE ``work_dir`` so Phase 2's ``--force`` rmtree of
    # ``work_dir`` does not wipe the just-exported vision assets.
    vision_result = None
    if args.mode in ("vision", "all") and args.variant == "moe":
        vision_temp = Path(tempfile.mkdtemp(prefix="gemma4_vision_"))
        vision_result = _export_vision_moe(args, logger, vision_temp)
        logger.info("Vision export (temp) completed.")

    # ── Phase 2: LLM export → creates work_dir/hmquant_* ──
    llm_dir = None
    if args.mode in ("llm", "all"):
        llm_dir = _export_llm(args, logger)
        logger.info(f"LLM export completed: {llm_dir}")

        # Copy processor_config.json into the exported hf_config/ so that
        # AutoProcessor.from_pretrained() works for VLM inference.
        hmquant_dirs = sorted(llm_dir.glob("hmquant_*"))
        if hmquant_dirs:
            _copy_processor_config_to_export(hmquant_dirs[-1], args.model, logger)
        else:
            _copy_processor_config_to_export(llm_dir, args.model, logger)

    # ── Phase 3: Move vision into hmquant_*/vision/ and integrate golden ──
    if vision_result is not None:
        if llm_dir is None:
            # Vision-only mode: golden stays in temp dir
            if args.golden:
                _generate_golden_for_hmonnx(
                    vision_result["hmonnx_file"],
                    str(Path(vision_result["hmonnx_file"]).parent / "vision_encoder_golden"),
                    args.device, logger,
                )
            logger.info("Vision export completed (standalone).")
        else:
            hmquant_dirs = sorted(llm_dir.glob("hmquant_*"))
            if not hmquant_dirs:
                logger.error("No hmquant_* directory found after LLM export, cannot integrate vision.")
            else:
                hmquant_dir = hmquant_dirs[-1]
                vision_dst = hmquant_dir / "vision"
                vision_dst.mkdir(exist_ok=True, parents=True)

# Move vision temp contents (onnx + external data) into vision_dst
            for item in vision_result["temp_dir"].iterdir():
                dst = vision_dst / item.name
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst, ignore_errors=True)
                    else:
                        dst.unlink()
                shutil.move(str(item), str(dst))
            logger.info(f"Vision assets moved to: {vision_dst}")
            try:
                shutil.rmtree(vision_result["temp_dir"], ignore_errors=True)
            except OSError:
                pass

            # Golden — vision + LLM (prefill/decode) all under hmquant_dir/
            if args.golden:
                dst_onnx = vision_dst / "vision_encoder.onnx"
                # Vision golden
                _generate_golden_for_hmonnx(str(dst_onnx), str(vision_dst / "vision_encoder_golden"), args.device, logger)
                # LLM golden (prefill / decode)
                _generate_all_golden(str(hmquant_dir), args.device, logger)
                # Merge vision meta into golden_meta_info.json
                _merge_vision_into_golden_meta(hmquant_dir, vision_result["vision_meta"], logger)

    logger.info("All exports completed successfully.")


if __name__ == "__main__":
    main()
