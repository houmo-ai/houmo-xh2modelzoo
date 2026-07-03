#!/usr/bin/env python3
"""Unified Gemma4 Series HMONNX / HF generate script.

Works for the unified Gemma4 Series presets: E4B, 31B dense, and 26B-A4B MoE.

Supports text-only, image, video and audio generation. Auto-detects the variant
from the runtime config.

Usage::

    # Text-only HMONNX
    python generate.py --model-config work_dirs/.../golden_meta_info.json --prompt "你好"

    # Image QA
    python generate.py --model-config work_dirs/.../golden_meta_info.json --image-path cat.jpg --prompt "描述这张图片"

    # Video QA (directory of frames or a processor-supported video path)
    python generate.py --model-config work_dirs/.../golden_meta_info.json --video-path frames/ --prompt "描述视频"

    # Audio QA (local wav is loaded by XHGemma4Processor)
    python generate.py --model-config work_dirs/.../golden_meta_info.json --audio-path speech.wav --prompt "音频里说了什么"

    # HF baseline
    python generate.py --backend hf --hf-model-dir /path/to/model --prompt "你好"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # xh2modelzoo/
sys.path.insert(0, str(REPO_ROOT))

# Ensure gptqmodel is importable for weight parsing when users point to a
# local checkout.  Do not bake personal workspace paths into the example.
for _gptqmodel_path in filter(
    None,
    (
        os.environ.get("GPTQMODEL_REPO"),
        str(REPO_ROOT.parent / "gptqmodel"),
    ),
):
    if Path(_gptqmodel_path).exists() and _gptqmodel_path not in sys.path:
        sys.path.insert(0, _gptqmodel_path)
        break


def bootstrap_runtime() -> None:
    workspace_root = REPO_ROOT.parent
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
SUPPORTED_FRAME_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


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
    model_config = runtime_meta.get("model_config", {})
    for value in (runtime_meta.get("variant"), model_config.get("variant")):
        if isinstance(value, str) and value:
            return value
    model_type = model_config.get("model_type", "")
    if "with_mask" in model_type or "moe" in model_type.lower():
        return "moe"
    return "dense"


def _positive_int_or_none(value) -> int | None:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _resolve_dense_visual_resize(args, hmonnx_model) -> tuple[int, int] | None:
    """Return an explicit resize size only when metadata carries positive dimensions.

    Gemma4 padded ViT exports intentionally write ``image_size_h/w=0`` in
    ``visual_meta`` because the processor pads to ``max_patches`` rather than a
    single fixed image resolution.  In that case the VLM demo must keep the
    original image and let ``XHGemma4Processor`` build padded visual inputs.
    """
    vis_meta = getattr(hmonnx_model, "visual_meta", None)
    if vis_meta is not None:
        export_w = _positive_int_or_none(getattr(vis_meta, "image_size_w", None))
        export_h = _positive_int_or_none(getattr(vis_meta, "image_size_h", None))
        if export_w is not None and export_h is not None:
            return export_w, export_h

    vision_meta_path = getattr(args, "vision_meta", None)
    if vision_meta_path:
        vision_meta = _load_json(vision_meta_path)
        preproc = vision_meta.get("image_preprocess", {})
        target_size = preproc.get("target_image_size", [])
        if len(target_size) >= 2:
            export_w = _positive_int_or_none(target_size[0])
            export_h = _positive_int_or_none(target_size[1])
            if export_w is not None and export_h is not None:
                return export_w, export_h

    return None


def _load_video_media(video_path: str | Path):
    """Return video media accepted by Gemma4 processor.

    A directory is treated as an ordered frame sequence and loaded as PIL
    images.  A regular file is passed through as a local path so processor
    versions that can decode videos keep that behavior.
    """
    path = Path(video_path)
    if path.is_dir():
        frame_paths = sorted(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_FRAME_SUFFIXES)
        if not frame_paths:
            raise FileNotFoundError(f"No video frames found under {path}")
        return [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]
    if not path.exists():
        raise FileNotFoundError(f"Video path not found: {path}")
    if path.suffix.lower() in SUPPORTED_FRAME_SUFFIXES:
        return [Image.open(path).convert("RGB")]
    return str(path)


def _resolve_requested_modality(args) -> str | None:
    requested = [
        ("image", args.image_path),
        ("video", getattr(args, "video_path", None)),
        ("audio", getattr(args, "audio_path", None)),
    ]
    active = [name for name, value in requested if value]
    if len(active) > 1:
        raise ValueError(f"Only one media input is supported per generate call, got: {active}")
    return active[0] if active else None


def _build_multimodal_messages(args, modality: str, hmonnx_model):
    logger = get_xhquant_logger()
    content: list[dict] = []
    if modality == "image":
        image = Image.open(args.image_path).convert("RGB")
        resize_size = _resolve_dense_visual_resize(args, hmonnx_model)
        if resize_size is not None:
            export_w, export_h = resize_size
            image = image.resize((export_w, export_h))
            logger.info(f"Resized input image to {export_w}x{export_h}")
        else:
            logger.info(f"Using original image size {image.size[0]}x{image.size[1]} for padded visual preprocessing")
        content.append({"type": "image", "image": image})
    elif modality == "video":
        video = _load_video_media(args.video_path)
        if isinstance(video, list):
            logger.info(f"Using {len(video)} local video frames for padded video preprocessing")
        else:
            logger.info(f"Using video path for processor preprocessing: {video}")
        content.append({"type": "video", "video": video})
    elif modality == "audio":
        if getattr(hmonnx_model, "audio", None) is None:
            raise ValueError("Audio generate requested, but exported runtime meta has no audio HMONNX submodel.")
        logger.info(f"Using audio path for processor preprocessing: {args.audio_path}")
        content.append({"type": "audio", "audio": args.audio_path})
    else:
        raise ValueError(f"Unsupported multimodal modality: {modality}")
    content.append({"type": "text", "text": args.prompt})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# HF backend
# ---------------------------------------------------------------------------
def run_hf(args) -> None:
    logger = get_xhquant_logger()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)
    _, inputs = _load_text_prompt_inputs(
        tokenizer, args.prompt, args.enable_thinking, max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
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
# HMONNX backend – multimodal (image/video/audio)
# ---------------------------------------------------------------------------
def _run_multimodal_dense(args, hmonnx_model, runtime_meta: dict, runtime_meta_path, device: str, modality: str) -> None:
    """Multimodal generation for unified dense-compatible Gemma4 runtime."""
    logger = get_xhquant_logger()

    messages = _build_multimodal_messages(args, modality, hmonnx_model)

    try:
        processor = hmonnx_model.get_tf_processor()
    except Exception:
        processor = AutoProcessor.from_pretrained(
            runtime_meta.get("hf_config", args.hf_model_dir or "."),
            trust_remote_code=True,
        )
    tokenizer = processor.tokenizer
    processor_kwargs = {}
    if modality == "video" and getattr(args, "video_num_frames", None):
        processor_kwargs["processor_kwargs"] = {"videos_kwargs": {"num_frames": int(args.video_num_frames)}}
    model_inputs = processor.apply_chat_template(messages, **processor_kwargs).to(device)
    context_max_length = int(runtime_meta["model_config"]["context_max_length"])
    _validate_generation_window(model_inputs.input_ids, args.max_decode_steps, context_max_length)

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

    eos_token_id = _resolve_eos_token_id(
        tokenizer,
        _load_hf_generation_config_from_runtime_meta(runtime_meta_path, runtime_meta),
        _load_hf_config_from_runtime_meta(runtime_meta_path, runtime_meta),
    )
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
        logger.info("Using eos_token_id for generation: %s", eos_token_id)

    tag = f"gemma4_series_hmonnx_{modality}_generate"
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
    logger.info(f"{'-' * 20} {modality.upper()} content {'-' * 20}")
    logger.info(content)
    print(content)


def _run_vlm_dense(args, hmonnx_model, runtime_meta: dict, runtime_meta_path, device: str) -> None:
    """Backward-compatible image VLM entry point."""
    _run_multimodal_dense(args, hmonnx_model, runtime_meta, runtime_meta_path, device, "image")



def run_hmonnx_multimodal(args) -> None:
    logger = get_xhquant_logger()
    runtime_meta_path = _resolve_runtime_meta_path(args.model_config)
    runtime_meta = _load_json(runtime_meta_path)
    variant = _detect_variant(runtime_meta)
    logger.info(f"Detected variant: {variant}")
    modality = _resolve_requested_modality(args)
    if modality is None:
        raise ValueError("run_hmonnx_multimodal requires --image-path, --video-path, or --audio-path")

    hmonnx_model = AutoLLMHONNXModel.from_pretrained(str(runtime_meta_path))
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU.")
        device = "cpu"

    _run_multimodal_dense(args, hmonnx_model, runtime_meta, runtime_meta_path, device, modality)


def run_hmonnx_vlm(args) -> None:
    """Backward-compatible public name for image VLM generation."""
    run_hmonnx_multimodal(args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Gemma4 Series Generate (MoE + Dense, text/image/video/audio)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Backend
    parser.add_argument("--backend", type=str, default="hmonnx", choices=["hmonnx", "hf"])
    parser.add_argument(
        "--model-config", type=str,
        default="work_dirs/gemma4_series_export/export_meta_info.json",
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
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to a video file, a single frame image, or a directory of ordered video frames.",
    )
    parser.add_argument(
        "--video-num-frames",
        type=int,
        default=4,
        help="Number of frames to sample for video generation. Keep small enough for the 2048 context budget.",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Path to local audio wav for E4B audio generation.",
    )
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

    args = parser.parse_args()
    if args.video_num_frames < 2:
        parser.error("--video-num-frames must be >= 2 for a real multi-frame video generate")
    xhquant_init(None, args.debug)
    torch.manual_seed(args.seed)

    if args.backend == "hf":
        if _resolve_requested_modality(args) is not None:
            raise ValueError("--backend hf currently supports text-only in this example; use --backend hmonnx for media.")
        run_hf(args)
    else:
        if _resolve_requested_modality(args) is not None:
            run_hmonnx_multimodal(args)
        else:
            run_hmonnx_text(args)


if __name__ == "__main__":
    main()
