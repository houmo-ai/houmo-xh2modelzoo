#!/usr/bin/env python3
"""Unified Gemma4 Series HMONNX / HF generate script.

Works for both variants:
  - gemma4_26b_a4b (MoE)
  - gemma4_31b_it   (dense)

Supports text-only and vision-language (VLM) generation.
Auto-detects the variant from the runtime config.

Usage::

    # Text-only HMONNX
    python generate.py --model-config work_dirs/.../golden_meta_info.json --prompt "你好"

    # VLM (with image)
    python generate.py --model-config work_dirs/.../golden_meta_info.json --image-path cat.jpg --prompt "描述这张图片"

    # HF baseline
    python generate.py --backend hf --hf-model-dir /path/to/model --prompt "你好"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # xh2modelzoo/
sys.path.insert(0, str(REPO_ROOT))

# Ensure gptqmodel is importable for weight parsing
_gptqmodel_path = "/data01/home/chenzx/project/gerrit/gptqmodel"
if Path(_gptqmodel_path).exists() and _gptqmodel_path not in sys.path:
    sys.path.insert(0, _gptqmodel_path)


def bootstrap_runtime() -> None:
    workspace_root = REPO_ROOT.parent  # /data01/home/chenzx/project/
    vendor_transformers = workspace_root / ".vendor" / "python" / "transformers"
    if vendor_transformers.exists():
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


bootstrap_runtime()

import torch
from PIL import Image
from transformers import AutoProcessor, TextStreamer
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


MAX_PROMPT_TOKENS = 40000


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_meta_path(meta_path: str | Path, referenced_path: str | Path) -> Path:
    resolved_path = Path(referenced_path)
    if resolved_path.is_absolute():
        return resolved_path
    return Path(meta_path).resolve().parent / resolved_path


def _load_hf_config_from_runtime_meta(runtime_meta_path: str | Path, runtime_meta: dict) -> dict:
    hf_config = runtime_meta.get("hf_config")
    if not hf_config:
        return {}
    hf_config_path = _resolve_meta_path(runtime_meta_path, hf_config)
    if hf_config_path.is_dir():
        hf_config_path = hf_config_path / "config.json"
    if not hf_config_path.exists():
        return {}
    return _load_json(hf_config_path)


def _load_hf_generation_config_from_runtime_meta(runtime_meta_path: str | Path, runtime_meta: dict) -> dict:
    hf_config = runtime_meta.get("hf_config")
    if not hf_config:
        return {}
    hf_config_path = _resolve_meta_path(runtime_meta_path, hf_config)
    if not hf_config_path.is_dir():
        hf_config_path = hf_config_path.parent
    generation_config_path = hf_config_path / "generation_config.json"
    if not generation_config_path.exists():
        return {}
    return _load_json(generation_config_path)


def _load_hf_config_from_model_dir(hf_model_dir: str | Path) -> dict:
    config_path = Path(hf_model_dir) / "config.json"
    if not config_path.exists():
        return {}
    return _load_json(config_path)


def _load_hf_generation_config_from_model_dir(hf_model_dir: str | Path) -> dict:
    generation_config_path = Path(hf_model_dir) / "generation_config.json"
    if not generation_config_path.exists():
        return {}
    return _load_json(generation_config_path)


def _append_token_ids(token_ids: list[int], value) -> None:
    if value is None:
        return
    if isinstance(value, int):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        return
    for token_id in candidates:
        token_id = int(token_id)
        if token_id not in token_ids:
            token_ids.append(token_id)


def _resolve_eos_token_id(tokenizer, *configs: dict) -> int | list[int] | None:
    token_ids: list[int] = []
    for config in configs:
        _append_token_ids(token_ids, config.get("eos_token_id"))
    _append_token_ids(token_ids, tokenizer.eos_token_id)

    for token in (getattr(tokenizer, "eot_token", None), "<turn|>"):
        if token is None:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id and token_id not in token_ids:
            token_ids.append(int(token_id))

    if not token_ids:
        return None
    if len(token_ids) == 1:
        return token_ids[0]
    return token_ids


def _resolve_runtime_meta_path(meta_path: str) -> Path:
    """Resolve to golden_meta_info.json from an export_meta_info.json or directly."""
    resolved_meta_path = Path(meta_path).resolve()
    meta = _load_json(resolved_meta_path)
    if meta.get("model_config", {}).get("model_type"):
        return resolved_meta_path

    exported_dir = meta.get("exported_dir")
    if exported_dir:
        golden_meta_path = (resolved_meta_path.parent / exported_dir / "golden_meta_info.json").resolve()
        if golden_meta_path.exists():
            return golden_meta_path

    raise ValueError(
        "model-config must point to golden_meta_info.json, or to an export_meta_info.json that contains exported_dir"
    )


def _resolve_pad_token_id(tokenizer) -> int:
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        return int(eos_token_id[0])
    if eos_token_id is not None:
        return int(eos_token_id)
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    raise ValueError("Unable to resolve pad_token_id from tokenizer")


def _tokenize_with_prompt_limit(tokenizer, text: str, max_prompt_tokens: int | None = None):
    if max_prompt_tokens is None or max_prompt_tokens <= 0:
        return tokenizer([text], return_tensors="pt")

    truncation_side = getattr(tokenizer, "truncation_side", "right")
    try:
        tokenizer.truncation_side = "left"
        return tokenizer([text], return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
    finally:
        tokenizer.truncation_side = truncation_side


def _load_text_prompt_inputs(
    tokenizer,
    prompt: str,
    enable_thinking: bool = False,
    max_prompt_tokens: int | None = None,
):
    messages = [{"role": "user", "content": prompt}]
    chat_template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking:
        chat_template_kwargs["enable_thinking"] = True
    try:
        text = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
    except TypeError:
        chat_template_kwargs.pop("enable_thinking", None)
        text = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
    return tokenizer, _tokenize_with_prompt_limit(tokenizer, text, max_prompt_tokens=max_prompt_tokens)


def _validate_generation_window(input_ids: torch.Tensor, max_new_tokens: int, context_max_length: int) -> None:
    prompt_length = int(input_ids.shape[1])
    if prompt_length > context_max_length:
        raise ValueError(
            f"Prompt token length {prompt_length} exceeds context_max_length={context_max_length}. "
            "Use a shorter prompt or export a model with a larger context length."
        )
    if prompt_length + max_new_tokens > context_max_length:
        remaining_tokens = max(context_max_length - prompt_length, 0)
        raise ValueError(
            f"Prompt token length {prompt_length} plus max_new_tokens={max_new_tokens} exceeds "
            f"context_max_length={context_max_length}. Reduce --max-decode-steps to at most {remaining_tokens}."
        )


def _detect_variant(runtime_meta: dict) -> str:
    """Detect model variant from runtime meta."""
    model_type = runtime_meta.get("model_config", {}).get("model_type", "")
    if "with_mask" in model_type or "moe" in model_type.lower():
        return "moe"
    return "dense"


# ---------------------------------------------------------------------------
# HF backend
# ---------------------------------------------------------------------------
def run_hf(args) -> None:
    logger = get_xhquant_logger()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)
    _, inputs = _load_text_prompt_inputs(
        tokenizer, args.prompt, args.enable_thinking, max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    if args.smoke_test:
        print("Task09 HF smoke OK", inputs["input_ids"].shape)
        return

    device = args.device if args.device != "cpu" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dtype_map = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(args.hf_dtype, torch.bfloat16)

    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.hf_model_dir,
        torch_dtype=dtype,
        device_map=args.hf_device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval().to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    eos_token_id = _resolve_eos_token_id(
        tokenizer,
        _load_hf_generation_config_from_model_dir(args.hf_model_dir),
        _load_hf_config_from_model_dir(args.hf_model_dir),
    )
    generation_kwargs = {
        **inputs,
        "max_new_tokens": args.max_decode_steps,
        "do_sample": args.do_sample,
    }
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id

    with torch.no_grad():
        out = model.generate(**generation_kwargs)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


# ---------------------------------------------------------------------------
# HMONNX backend – text only
# ---------------------------------------------------------------------------
def run_hmonnx_text(args) -> None:
    logger = get_xhquant_logger()
    runtime_meta_path = _resolve_runtime_meta_path(args.model_config)
    runtime_meta = _load_json(runtime_meta_path)
    variant = _detect_variant(runtime_meta)
    logger.info(f"Detected variant: {variant}")

    hmonnx_model = AutoLLMHONNXModel.from_pretrained(str(runtime_meta_path))
    context_max_length = int(runtime_meta["model_config"]["context_max_length"])

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU.")
        device = "cpu"

    tokenizer = hmonnx_model.get_tokenizer(trust_remote_code=True)
    _, model_inputs = _load_text_prompt_inputs(
        tokenizer, args.prompt, args.enable_thinking, max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    if args.smoke_test:
        print(f"Task09 HMONNX smoke OK variant={variant} path={runtime_meta_path} shape={model_inputs['input_ids'].shape}")
        return

    _validate_generation_window(model_inputs["input_ids"], args.max_decode_steps, context_max_length)

    streamer = TextStreamer(tokenizer, skip_prompt=True) if args.streaming_out else None
    hmonnx_model.to(device)

    if args.golden:
        hmonnx_model.enable_golden = True
        if args.fast:
            logger.warning("Golden outputs should be generated in aligned precision; ignoring --fast.")
    elif args.fast:
        hmonnx_model.to_fast()

    generation_kwargs = {
        **{key: value.to(device) for key, value in model_inputs.items()},
        "max_new_tokens": args.max_decode_steps,
        "do_sample": args.do_sample,
        "pad_token_id": _resolve_pad_token_id(tokenizer),
    }
    if streamer is not None:
        generation_kwargs["streamer"] = streamer

    eos_token_id = _resolve_eos_token_id(
        tokenizer,
        _load_hf_generation_config_from_runtime_meta(runtime_meta_path, runtime_meta),
        _load_hf_config_from_runtime_meta(runtime_meta_path, runtime_meta),
    )
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
        logger.info("Using eos_token_id for generation: %s", eos_token_id)

    tag = "gemma4_series_hmonnx_generate"
    contexts = [
        TimeProfiler(tag, logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(hmonnx_model),
    ]
    with ContextManagers(contexts):
        generated_ids = hmonnx_model.generate(**generation_kwargs)

    prompt_len = len(model_inputs["input_ids"][0])
    output_ids = generated_ids[0][prompt_len:].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    logger.info(f"{'-' * 20} HMONNX content {'-' * 20}")
    logger.info(output_text)
    print(output_text)


# ---------------------------------------------------------------------------
# HMONNX backend – VLM (with image)
# ---------------------------------------------------------------------------
def _run_vlm_dense(args, hmonnx_model, runtime_meta: dict, runtime_meta_path, device: str) -> None:
    """VLM generation for dense Gemma4 variant."""
    logger = get_xhquant_logger()
    # Resize input image to match the export image size
    vis_meta = getattr(hmonnx_model, "visual_meta", None)
    if vis_meta is not None:
        export_w = getattr(vis_meta, "image_size_w", 224)
        export_h = getattr(vis_meta, "image_size_h", 224)
    else:
        export_w, export_h = 448, 448
        vision_meta_path = args.vision_meta
        if vision_meta_path:
            vision_meta = _load_json(vision_meta_path)
            preproc = vision_meta.get("image_preprocess", {})
            target_size = preproc.get("target_image_size", [448, 448])
            export_w, export_h = target_size[0], target_size[1]
        logger.warning(f"No visual_meta on HMONNX model, using fallback size {export_w}x{export_h}")

    image = Image.open(args.image_path).convert("RGB").resize((export_w, export_h))
    logger.info(f"Resized input image to {export_w}x{export_h}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]

    try:
        processor = hmonnx_model.get_tf_processor()
    except Exception:
        processor = AutoProcessor.from_pretrained(
            runtime_meta.get("hf_config", args.hf_model_dir or "."),
            trust_remote_code=True,
        )
    tokenizer = processor.tokenizer
    model_inputs = processor.apply_chat_template(messages).to(device)

    streamer = TextStreamer(tokenizer, skip_prompt=True) if args.streaming_out else None
    hmonnx_model.to(device)

    if args.golden:
        hmonnx_model.enable_golden = True
    elif args.fast:
        hmonnx_model.to_fast()

    generation_kwargs = {
        **{key: value.to(device) if isinstance(value, torch.Tensor) else value
           for key, value in model_inputs.items()},
        "max_new_tokens": args.max_decode_steps,
        "do_sample": args.do_sample,
        "pad_token_id": _resolve_pad_token_id(tokenizer),
    }
    if streamer is not None:
        generation_kwargs["streamer"] = streamer

    tag = "gemma4_series_hmonnx_vlm_dense_generate"
    contexts = [
        TimeProfiler(tag, logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(hmonnx_model),
    ]
    with ContextManagers(contexts):
        generated_ids = hmonnx_model.generate(**generation_kwargs)

    prompt_len = len(model_inputs.input_ids[0])
    output_ids = generated_ids[0][prompt_len:].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    logger.info(f"{'-' * 20} VLM content {'-' * 20}")
    logger.info(content)
    print(content)


def _build_pixel_values_for_moe(
    args, hf_config_dir: str, runtime_meta: dict, logger
) -> tuple[torch.Tensor, int]:
    """Build pixel_values for vision HMONNX encoder.

    Uses ``XHGemma4MoeVisualProcessor`` via the exported ``hf_config/``
    directory, which must contain ``processor_config.json``.
    """
    from examples_merak.llm.gemma4_moe.gemma4_moe_visual_preprocess import (
        extract_valid_patch_tokens,
        prepare_visual_input_image,
    )
    from xhmodel_merak.xh_llm.models.gemma4_moe.gemma4_moe_visual_model import (
        XHGemma4MoeVisualProcessor,
    )

    vis_cfg = runtime_meta.get("model_config", {}).get("visual_config", {})
    # ``_merge_vision_into_golden_meta`` writes the authoritative export-time
    # vision settings under ``runtime_meta["vision"]``. Prefer those because
    # ``model_config.visual_config`` is often missing on MoE LLM exports.
    vis_export = runtime_meta.get("vision", {})
    upsample_token = bool(vis_export.get("upsample_token", vis_cfg.get("upsample_token", False)))
    vis_preproc = vis_export.get("image_preprocess", {})
    target_image_size = vis_preproc.get("target_image_size") or [
        vis_cfg.get("max_size_w", 448),
        vis_cfg.get("max_size_h", 448),
    ]
    max_size_w = int(target_image_size[0])
    max_size_h = int(target_image_size[1])

    processed_image, _ = prepare_visual_input_image(
        args.image_path,
        upsample_token=upsample_token,
        target_image_size=(max_size_w, max_size_h),
    )
    vis_processor = XHGemma4MoeVisualProcessor.from_pretrained(
        hf_config_dir, upsample_token=upsample_token,
    )
    vision_inputs = vis_processor(images=processed_image, return_tensors="pt")
    pixel_values = vision_inputs["pixel_values"]
    image_position_ids = vision_inputs["image_position_ids"]
    logger.info(
        "MoE VLM vision inputs: pixel_values=%s, image_position_ids=%s",
        tuple(pixel_values.shape),
        tuple(image_position_ids.shape),
    )
    pixel_values_valid, _, _ = extract_valid_patch_tokens(pixel_values, image_position_ids)
    n_patches = int(pixel_values_valid.shape[1])
    return pixel_values_valid, n_patches


def _run_vlm_moe(args, hmonnx_model, runtime_meta: dict, runtime_meta_path, device: str) -> None:
    """VLM generation for Gemma4 MoE variant.

    Uses vision HMONNX model to encode image, then passes pre-computed
    ``image_embeds`` directly to the LLM HMONNX generate().
    """
    logger = get_xhquant_logger()

    from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel

    # -- 1. Resolve vision config --
    image_token_id = int(runtime_meta.get("model_config", {}).get("image_token_id", 258880))
    logger.info(f"MoE VLM: image_token_id={image_token_id}")

    # -- 2. Tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(
        str(_resolve_meta_path(runtime_meta_path, runtime_meta["hf_config"])),
        trust_remote_code=True,
    )

    # -- 3. Find vision HMONNX model --
    vision_hmonnx_path = Path(runtime_meta_path).resolve().parent / "vision" / "vision_encoder.onnx"
    if not vision_hmonnx_path.exists():
        raise FileNotFoundError(
            f"Vision HMONNX model not found at {vision_hmonnx_path}. "
            "Please ensure the vision encoder was exported."
        )
    logger.info(f"MoE VLM: using vision HMONNX {vision_hmonnx_path}")

    # -- 4. Preprocess image to match vision HMONNX expected shape --
    hf_config_dir = str(_resolve_meta_path(runtime_meta_path, runtime_meta["hf_config"]))
    pixel_values, _n_patches = _build_pixel_values_for_moe(args, hf_config_dir, runtime_meta, logger)

    # -- 5. Run vision HMONNX encoder --
    vision_encoder = HMONNXModel(str(vision_hmonnx_path))
    vision_encoder.to(device)
    image_embeds = vision_encoder(pixel_values.to(device=device, dtype=torch.float16))
    if isinstance(image_embeds, (tuple, list)):
        image_embeds = image_embeds[0]
    if image_embeds.dim() == 3 and image_embeds.shape[0] == 1:
        image_embeds = image_embeds[0]
    n_image_tokens = int(image_embeds.shape[0])
    logger.info(f"MoE VLM vision output: image_embeds={tuple(image_embeds.shape)}, n_tokens={n_image_tokens}")

    # -- 6. Build chat prompt with <|image|> placeholder --
    content = f"\n\n<|image|>\n\n{args.prompt}"
    messages = [{"role": "user", "content": content}]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if args.enable_thinking:
        chat_kwargs["enable_thinking"] = True
    try:
        prompt_text = tokenizer.apply_chat_template(messages, **chat_kwargs)
    except TypeError:
        chat_kwargs.pop("enable_thinking", None)
        prompt_text = tokenizer.apply_chat_template(messages, **chat_kwargs)

    prompt_input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids

    # -- 7. Expand image placeholders → input_ids + mm_token_type_ids --
    expanded_ids = []
    mm_token_list = []
    for tid in prompt_input_ids[0].tolist():
        if tid == image_token_id:
            expanded_ids.extend([image_token_id] * n_image_tokens)
            mm_token_list.extend([1] * n_image_tokens)
        else:
            expanded_ids.append(tid)
            mm_token_list.append(0)
    input_ids = torch.tensor([expanded_ids], dtype=torch.long)
    mm_token_type_ids = torch.tensor([mm_token_list], dtype=torch.long)
    logger.info(f"MoE VLM prompt: input_ids={tuple(input_ids.shape)}, "
                f"mm_tokens={int(mm_token_type_ids.sum().item())}")

    if args.smoke_test:
        print(
            f"Task09 HMONNX VLM smoke OK variant=moe path={runtime_meta_path} "
            f"prompt_tokens={int(input_ids.shape[-1])} image_tokens={n_image_tokens}"
        )
        return

    # -- 8. Generate --
    streamer = TextStreamer(tokenizer, skip_prompt=True) if args.streaming_out else None
    hmonnx_model.to(device)
    if args.golden:
        hmonnx_model.enable_golden = True
        if args.fast:
            logger.warning("Golden outputs should be generated in aligned precision; ignoring --fast.")
    elif args.fast:
        hmonnx_model.to_fast()

    generation_kwargs = {
        "input_ids": input_ids.to(device),
        "attention_mask": torch.ones_like(input_ids).to(device),
        "mm_token_type_ids": mm_token_type_ids.to(device),
        "image_embeds": image_embeds.to(device=device, dtype=hmonnx_model.dtype),
        "max_new_tokens": args.max_decode_steps,
        "do_sample": args.do_sample,
        "pad_token_id": _resolve_pad_token_id(tokenizer),
    }
    if streamer is not None:
        generation_kwargs["streamer"] = streamer

    eos_token_id = _resolve_eos_token_id(
        tokenizer,
        _load_hf_generation_config_from_runtime_meta(runtime_meta_path, runtime_meta),
        _load_hf_config_from_runtime_meta(runtime_meta_path, runtime_meta),
    )
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
        logger.info("Using eos_token_id for generation: %s", eos_token_id)

    tag = "gemma4_series_hmonnx_vlm_moe_generate"
    contexts = [
        TimeProfiler(tag, logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(hmonnx_model),
    ]
    with ContextManagers(contexts):
        generated_ids = hmonnx_model.generate(**generation_kwargs)

    prompt_len = len(input_ids[0])
    output_ids = generated_ids[0][prompt_len:].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    logger.info(f"{'-' * 20} VLM content {'-' * 20}")
    logger.info(content)
    print(content)


def run_hmonnx_vlm(args) -> None:
    logger = get_xhquant_logger()
    runtime_meta_path = _resolve_runtime_meta_path(args.model_config)
    runtime_meta = _load_json(runtime_meta_path)
    variant = _detect_variant(runtime_meta)
    logger.info(f"Detected variant: {variant}")

    hmonnx_model = AutoLLMHONNXModel.from_pretrained(str(runtime_meta_path))
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU.")
        device = "cpu"

    if variant == "moe":
        _run_vlm_moe(args, hmonnx_model, runtime_meta, runtime_meta_path, device)
    else:
        _run_vlm_dense(args, hmonnx_model, runtime_meta, runtime_meta_path, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Gemma4 Series Generate (MoE + Dense, text + VLM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Backend
    parser.add_argument("--backend", type=str, default="hmonnx", choices=["hmonnx", "hf"])
    parser.add_argument(
        "--model-config", type=str,
        default="work_dirs/gemma4_moe_with_mask_26b_a4b_it_xh2a_w8a8_256_2k/export_meta_info.json",
        help="HMONNX runtime meta: golden_meta_info.json or export_meta_info.json with exported_dir.",
    )
    parser.add_argument(
        "--hf-model-dir", type=str, default="/data01/datasets/gemma-4-26B-A4B-it",
        help="HF float model directory for --backend hf.",
    )
    parser.add_argument("--hf-dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--hf-device-map", type=str, default="auto")

    # Prompt
    parser.add_argument("--prompt", type=str, default="你好，请详细介绍一下大语言模型的原理。")
    parser.add_argument("--image-path", type=str, default=None, help="Path to image for VLM mode.")
    parser.add_argument("--vision-meta", type=str, default=None, help="Vision export_meta_info.json for image size info.")

    # Generation
    parser.add_argument("--max-decode-steps", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode.")
    parser.add_argument("--streaming-out", action="store_true", help="Stream output tokens.")

    # Runtime
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument("--fast", action="store_true", help="Run HMONNX in fast mode.")
    parser.add_argument("--golden", action="store_true", help="Save golden outputs.")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()
    xhquant_init(None, args.debug)
    torch.manual_seed(args.seed)

    if args.backend == "hf":
        run_hf(args)
    else:
        if args.image_path:
            run_hmonnx_vlm(args)
        else:
            run_hmonnx_text(args)


if __name__ == "__main__":
    main()
