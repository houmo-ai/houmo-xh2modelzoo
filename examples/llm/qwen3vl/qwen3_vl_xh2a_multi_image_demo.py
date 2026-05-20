# Copyright 2025 HOUMO AI
#
# File: qwen3_vl_xh2a_multi_image_demo.py
# Description:
#   Generic Qwen3-VL multi-image demo using HMONNX graph runtime with
#   streaming output. Supports three mutually exclusive input modes:
#     * --image-paths PATH[,PATH...]   explicit comma-separated list
#     * --image-dir DIR [--image-glob G]   scan a directory (sorted)
#     * --task-json JSON --task-id ID  custom record-based loading
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from transformers import AutoConfig

import xhquant.utils.suppress_printing
from xh_model_zoo.api import ConfigDict, get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.qwen3_vl import Qwen3VLONNXModel, Qwen3VLProcessor


Image.MAX_IMAGE_PIXELS = 900000000

DTYPE_NAME_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL multi-image graph-runtime demo with streaming output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- runtime / model wiring -----------------------------------------
    parser.add_argument(
        "--model-config",
        "--model_config",
        type=str,
        default=None,
        help="Path to export_meta_info.json; required unless --prepare-only",
    )
    parser.add_argument(
        "--model-dir",
        "--model_dir",
        type=str,
        default=None,
        help="Compatibility mode: exported work dir containing hf_config/token_embedding.pt",
    )
    parser.add_argument("--visual-onnx", "--visual_onnx", type=str, default=None)
    parser.add_argument("--prefill-onnx", "--prefill_onnx", type=str, default=None)
    parser.add_argument("--decode-onnx", "--decode_onnx", type=str, default=None)
    parser.add_argument("--hf-model-dir", "--hf_model_dir", type=str, default=None)

    # --- image input (mutually exclusive group) ------------------------
    img_group = parser.add_argument_group("image input (choose exactly one)")
    img_group.add_argument(
        "--image-paths",
        "--image_paths",
        type=str,
        default=None,
        help="Comma-separated list of image files, e.g. img1.jpg,img2.jpg",
    )
    img_group.add_argument(
        "--image-dir",
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing images (scanned and sorted by file name)",
    )
    img_group.add_argument(
        "--image-glob",
        "--image_glob",
        type=str,
        default=None,
        help="Optional glob pattern used together with --image-dir, e.g. '*.jpg'",
    )
    img_group.add_argument(
        "--task-json",
        "--task_json",
        type=str,
        default=None,
        help="Optional task description JSON (backward-compat custom format)",
    )
    img_group.add_argument(
        "--task-id",
        "--task_id",
        type=str,
        default=None,
        help="Task id used together with --task-json to locate the record",
    )

    # --- prompt (mutually exclusive, one required) ---------------------
    prompt_group = parser.add_argument_group("prompt (choose exactly one)")
    prompt_group.add_argument("--prompt", type=str, default=None, help="Prompt string")
    prompt_group.add_argument(
        "--prompt-file",
        "--prompt_file",
        type=str,
        default=None,
        help="Path to a UTF-8 text file holding the prompt",
    )

    # --- scene label / batching ----------------------------------------
    parser.add_argument(
        "--scene-name",
        "--scene_name",
        type=str,
        default="multi_image",
        help="Free-form scene label used for log/work_dir naming only",
    )
    parser.add_argument(
        "--images-per-batch",
        "--images_per_batch",
        type=int,
        default=20,
        help="Images per batch (page-style grouping)",
    )
    parser.add_argument("--max-batches", "--max_batches", type=int, default=None)
    parser.add_argument("--max-images", "--max_images", type=int, default=None)

    # --- visual sizing -------------------------------------------------
    parser.add_argument("--image-size-w", "--image_size_w", type=int, default=None)
    parser.add_argument("--image-size-h", "--image_size_h", type=int, default=None)
    parser.add_argument("--max-size-t", "--max_size_t", type=int, default=None)
    parser.add_argument("--patch-size", "--patch_size", type=int, default=None)
    parser.add_argument(
        "--temporal-patch-size",
        "--temporal_patch_size",
        type=int,
        default=None,
    )

    # --- generation ----------------------------------------------------
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--max-context-tokens",
        "--max_context_tokens",
        "--cache_len",
        dest="max_context_tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--model-type",
        "--model_type",
        type=str,
        default=None,
        help="Compatibility arg; ignored by graph runtime",
    )
    parser.add_argument("--dtype", type=str, default="fp16", choices=sorted(DTYPE_NAME_MAP.keys()))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exec-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-auto-offload", action="store_true")
    parser.add_argument(
        "--auto-offload-max-memory",
        type=str,
        default=None,
        help='JSON string, e.g. {"0":"35GB","cpu":"120GB"}',
    )
    parser.add_argument("--prefill-auto-offload-max-memory", type=str, default=None)
    parser.add_argument("--decode-auto-offload-max-memory", type=str, default=None)
    parser.add_argument("--vision-auto-offload-max-memory", type=str, default=None)
    parser.add_argument("--resource-tight-mode", action="store_true")
    parser.add_argument("--enable-cuda-graph", action="store_true")
    parser.add_argument(
        "--cuda-graph-modules",
        nargs="*",
        default=None,
        help="Subset of sessions to capture, e.g. prefill decode vision",
    )
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=False)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--stream-output", dest="stream_output", action="store_true", default=True)
    parser.add_argument("--no-stream-output", dest="stream_output", action="store_false")
    parser.add_argument("--prepare-only", "--prepare_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


# ----------------------------------------------------------------------
# Validation helpers (pure functions, easy to unit-test)
# ----------------------------------------------------------------------


def select_image_mode(args) -> str:
    """Return one of {"paths", "dir", "task"} based on which input was given.

    Exactly one of --image-paths / --image-dir / --task-json must be set.
    Raises ValueError on misuse so callers can map to parser.error.
    """
    flags = {
        "paths": bool(args.image_paths),
        "dir": bool(args.image_dir),
        "task": bool(args.task_json),
    }
    chosen = [name for name, set_ in flags.items() if set_]
    if not chosen:
        raise ValueError(
            "one of --image-paths / --image-dir / --task-json is required"
        )
    if len(chosen) > 1:
        raise ValueError(
            "--image-paths / --image-dir / --task-json are mutually exclusive; "
            f"got: {', '.join('--' + c.replace('task', 'task-json').replace('paths', 'image-paths').replace('dir', 'image-dir') for c in chosen)}"
        )
    mode = chosen[0]
    if mode == "task" and not args.task_id:
        raise ValueError("--task-id is required when --task-json is set")
    if mode != "dir" and args.image_glob:
        raise ValueError("--image-glob is only valid together with --image-dir")
    return mode


def resolve_prompt(args) -> str:
    """Return the prompt string from --prompt or --prompt-file (mutually exclusive)."""
    if args.prompt and args.prompt_file:
        raise ValueError("--prompt and --prompt-file are mutually exclusive")
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        path = Path(args.prompt_file)
        if not path.is_file():
            raise FileNotFoundError(f"--prompt-file not found: {path}")
        return path.read_text(encoding="utf-8").strip()
    raise ValueError("one of --prompt / --prompt-file is required")


def parse_image_paths_list(spec: str) -> list[Path]:
    """Parse a comma-separated path list and verify each path exists."""
    if not spec or not spec.strip():
        raise ValueError("--image-paths is empty")
    raw_items = [item.strip() for item in spec.split(",")]
    items = [item for item in raw_items if item]
    if not items:
        raise ValueError("--image-paths is empty")
    paths = []
    for item in items:
        path = Path(item).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        paths.append(path)
    return paths


def scan_image_dir(image_dir: Path, glob_pattern: Optional[str]) -> list[Path]:
    """List images from a directory, sorted by file name.

    When ``glob_pattern`` is None, scan top-level files filtered by
    ``DEFAULT_IMAGE_EXTENSIONS`` (case-insensitive). Otherwise apply
    ``Path.glob(glob_pattern)`` to keep behaviour predictable.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(f"--image-dir not found or not a directory: {image_dir}")
    if glob_pattern:
        candidates = [p for p in image_dir.glob(glob_pattern) if p.is_file()]
    else:
        candidates = [
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS
        ]
    if not candidates:
        raise FileNotFoundError(
            f"no images found in {image_dir} (glob={glob_pattern or 'default-extensions'})"
        )
    return sorted(candidates, key=lambda p: p.name)


def load_task_record(json_file: Path, task_id: str) -> dict:
    """Backward-compat custom JSON loader.

    The JSON is expected to be either:
      * a list of records each containing ``taskId`` and ``group``
      * a single record dict with ``group``
    Each ``group`` entry holds ``materials`` with ``filePath`` strings, which
    are basename-matched against files in ``--image-dir`` (sibling of JSON
    by default).
    """
    with json_file.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        for record in data:
            if isinstance(record, dict) and record.get("taskId") == task_id:
                return record
        raise ValueError(f"task_id={task_id} not found in {json_file}")
    if isinstance(data, dict):
        if data.get("taskId") in (task_id, None):
            return data
        raise ValueError(f"task_id={task_id} mismatch (json has {data.get('taskId')!r})")
    raise ValueError(f"unsupported task JSON shape in {json_file}: {type(data).__name__}")


def collect_task_image_paths(
    task_record: dict,
    image_dir: Path,
    max_images: Optional[int] = None,
):
    """Match ``filePath`` basenames in the task record against files in image_dir."""
    available_files = {path.name: path for path in image_dir.iterdir() if path.is_file()}
    matched_paths: list[Path] = []
    missing_names: list[str] = []
    total_materials = 0
    for group in task_record.get("group", []):
        for material in group.get("materials", []):
            file_path = material.get("filePath")
            if not file_path:
                continue
            total_materials += 1
            basename = Path(file_path).name
            local_path = available_files.get(basename)
            if local_path is None:
                missing_names.append(basename)
                continue
            matched_paths.append(local_path)
            if max_images is not None and len(matched_paths) >= max_images:
                return matched_paths, total_materials, missing_names
    return matched_paths, total_materials, missing_names


def resolve_image_paths(args, logger) -> list[Path]:
    """Dispatch to the chosen input mode and return a list of image paths."""
    mode = select_image_mode(args)
    if mode == "paths":
        paths = parse_image_paths_list(args.image_paths)
    elif mode == "dir":
        paths = scan_image_dir(Path(args.image_dir).expanduser(), args.image_glob)
    else:  # task
        json_file = Path(args.task_json).expanduser()
        if not json_file.is_file():
            raise FileNotFoundError(f"--task-json not found: {json_file}")
        # When --image-dir is not provided in task mode, default to json sibling dir.
        image_dir = Path(args.image_dir).expanduser() if args.image_dir else json_file.parent
        if not image_dir.is_dir():
            raise FileNotFoundError(f"image_dir for task mode not found: {image_dir}")
        record = load_task_record(json_file, args.task_id)
        paths, total, missing = collect_task_image_paths(
            record, image_dir, max_images=args.max_images
        )
        logger.info(
            "Task %s image stats: total=%d matched=%d skipped=%d image_dir=%s",
            args.task_id,
            total,
            len(paths),
            total - len(paths),
            image_dir,
        )
        if missing:
            logger.info("First missing images: %s", ", ".join(missing[:5]))
        if not paths:
            raise FileNotFoundError(f"no task images matched under {image_dir}")
        return paths

    if args.max_images is not None and len(paths) > args.max_images:
        paths = paths[: args.max_images]
    logger.info("Resolved %d image(s) via mode=%s", len(paths), mode)
    return paths


# ----------------------------------------------------------------------
# Misc helpers
# ----------------------------------------------------------------------


def parse_dtype(dtype_name: str) -> torch.dtype:
    key = dtype_name.strip().lower()
    if key not in DTYPE_NAME_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return DTYPE_NAME_MAP[key]


def parse_auto_offload_max_memory(max_memory_json: Optional[str]):
    if max_memory_json is None or max_memory_json.strip() == "":
        return None
    parsed = json.loads(max_memory_json)
    if not isinstance(parsed, dict):
        raise ValueError("auto_offload_max_memory must be a JSON object")
    fixed = {}
    for key, value in parsed.items():
        try:
            fixed[int(key)] = value
        except Exception:
            fixed[key] = value
    return fixed


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_token_embedding(embed_path: Path) -> nn.Module:
    torch.serialization.add_safe_globals([nn.Embedding])
    try:
        try:
            obj = torch.load(str(embed_path), map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(str(embed_path), map_location="cpu")
    finally:
        torch.serialization.clear_safe_globals()

    if isinstance(obj, nn.Module):
        obj.eval()
        return obj
    if isinstance(obj, dict):
        if "weight" not in obj:
            raise ValueError(f"Unsupported token embedding state dict format: {embed_path}")
        emb = nn.Embedding(obj["weight"].shape[0], obj["weight"].shape[1])
        emb.load_state_dict(obj)
        emb.eval()
        return emb
    raise TypeError(f"Unsupported token embedding object type: {type(obj)}")


def resolve_processor_source(hf_model_config_dir: Path, hf_model_dir: Optional[str]) -> Path:
    if (hf_model_config_dir / "preprocessor_config.json").exists():
        return hf_model_config_dir
    if hf_model_dir is None:
        return hf_model_config_dir
    candidate = Path(hf_model_dir).resolve()
    if (candidate / "preprocessor_config.json").exists():
        return candidate
    return hf_model_config_dir


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    if height <= 0 or width <= 0:
        raise ValueError(f"smart_resize received invalid image size: width={width}, height={height}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def resize_for_visual_encoder(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"resize_for_visual_encoder received invalid image size: width={orig_w}, height={orig_h}")
    if (orig_w, orig_h) == (target_w, target_h):
        return image
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    return ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=(114, 114, 114))


def prepare_image_entry(image_path: Path, page_index: int, target_w: int, target_h: int) -> dict:
    with Image.open(image_path) as raw_image:
        base_image = raw_image.convert("RGB")
    original_width, original_height = base_image.size
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"Invalid image size for {image_path}: width={original_width}, height={original_height}")
    processed_image = resize_for_visual_encoder(base_image, target_w=target_w, target_h=target_h)
    return {
        "page_index": page_index,
        "image_path": image_path,
        "processed_image": processed_image,
        "resized_height": target_h,
        "resized_width": target_w,
        "original_height": original_height,
        "original_width": original_width,
    }


def build_messages(prompt: str, image_entries: list[dict]) -> list[dict]:
    content = [{"type": "text", "text": prompt}]
    for entry in image_entries:
        content.append({"type": "text", "text": f"page{entry['page_index']}"})
        content.append(
            {
                "type": "image",
                "image": entry["processed_image"],
                "resized_height": entry["resized_height"],
                "resized_width": entry["resized_width"],
            }
        )
    return [{"role": "user", "content": content}]


def get_image_token_counts(image_grid_thw: torch.Tensor, merge_size: int) -> list[int]:
    if image_grid_thw is None:
        raise ValueError("image_grid_thw is required for multi-image inference")
    return (image_grid_thw.prod(-1) // (merge_size**2)).tolist()


def validate_multi_image_inputs(inputs, processor, logger):
    image_grid_thw = inputs.get("image_grid_thw")
    hm_pixel_values = inputs.get("hm_pixel_values")
    if image_grid_thw is None or hm_pixel_values is None:
        raise ValueError(
            "processor output must contain both image_grid_thw and hm_pixel_values for multi-image inference"
        )

    expected_token_counts = get_image_token_counts(image_grid_thw, processor.image_processor.merge_size)
    image_token_count = int((inputs["input_ids"] == processor.image_token_id).sum().item())
    expected_total_tokens = sum(expected_token_counts)

    if len(hm_pixel_values) != image_grid_thw.shape[0]:
        raise ValueError(
            f"hm_pixel_values/image_grid_thw image count mismatch: {len(hm_pixel_values)} vs {image_grid_thw.shape[0]}"
        )
    if image_token_count != expected_total_tokens:
        raise ValueError(
            "image token count mismatch: "
            f"processor emitted {image_token_count} image tokens but "
            f"grid_thw expects {expected_total_tokens}"
        )

    logger.info(
        "Validated multi-image inputs: images=%d token_counts=%s total_image_tokens=%d",
        image_grid_thw.shape[0],
        expected_token_counts,
        expected_total_tokens,
    )
    return expected_token_counts


def prepare_batches(
    image_paths: list[Path],
    prompt: str,
    images_per_batch: int,
    max_batches: Optional[int],
    logger,
    target_w: int,
    target_h: int,
):
    if not image_paths:
        raise ValueError("image_paths is empty; nothing to do")

    prepared_entries = [
        prepare_image_entry(image_path, page_index=index, target_w=target_w, target_h=target_h)
        for index, image_path in enumerate(image_paths, start=1)
    ]

    batches = []
    for batch_index, start in enumerate(range(0, len(prepared_entries), images_per_batch), start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch_entries = prepared_entries[start : start + images_per_batch]
        batches.append(
            {
                "batch_index": batch_index,
                "messages": build_messages(prompt, batch_entries),
                "image_entries": batch_entries,
            }
        )

    if not batches:
        raise ValueError("No batches were prepared; adjust --images-per-batch/--max-batches/--max-images")

    logger.info(
        "Prepared %d batch(es) with %d image(s)",
        len(batches),
        sum(len(batch["image_entries"]) for batch in batches),
    )
    return batches


def validate_runtime_args(args, parser):
    if args.images_per_batch <= 0:
        parser.error("--images-per-batch must be a positive integer")
    try:
        select_image_mode(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        resolve_prompt(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.prepare_only:
        return
    if not args.model_config and not args.model_dir:
        parser.error("--model-config or --model-dir is required unless --prepare-only is set")
    if not args.model_config:
        required = {
            "--visual_onnx": args.visual_onnx,
            "--prefill_onnx": args.prefill_onnx,
            "--decode_onnx": args.decode_onnx,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"{', '.join(missing)} required when using compatibility --model_dir mode")


def load_runtime(args, logger):
    if args.model_config:
        model_meta_file = Path(args.model_config).resolve()
        if not model_meta_file.is_file():
            raise FileNotFoundError(f"model_config not found: {model_meta_file}")

        model_dir = model_meta_file.parent
        meta_info = json.load(open(model_meta_file, "r", encoding="utf-8"))
        wrap_cfg = meta_info.get("wrap_cfg", {})
        visual_cfg = wrap_cfg.get("visual", {})

        prefill_onnx = resolve_path(
            model_dir,
            args.prefill_onnx or meta_info.get("prefill_onnx") or meta_info["prefill_onnx_file"],
        )
        decode_onnx = resolve_path(
            model_dir,
            args.decode_onnx or meta_info.get("decode_onnx") or meta_info["decode_onnx_file"],
        )
        vision_onnx = resolve_path(
            model_dir,
            args.visual_onnx or meta_info.get("vision_onnx") or meta_info.get("visual_onnx"),
        )
        hf_model_config_dir = resolve_path(model_dir, meta_info["hf_config"])
        token_embedding_file = resolve_path(model_dir, meta_info["token_embedding_file"])
    else:
        model_dir = Path(args.model_dir).resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"model_dir not found: {model_dir}")
        meta_info = {}
        wrap_cfg = {}
        visual_cfg = {}
        prefill_onnx = Path(args.prefill_onnx).resolve()
        decode_onnx = Path(args.decode_onnx).resolve()
        vision_onnx = Path(args.visual_onnx).resolve()
        hf_model_config_dir = model_dir / "hf_config"
        if not hf_model_config_dir.is_dir():
            hf_model_config_dir = model_dir
        token_embedding_file = model_dir / "token_embedding.pt"
    processor_source = resolve_processor_source(
        hf_model_config_dir,
        args.hf_model_dir or meta_info.get("hf_model"),
    )

    max_context_tokens = (
        args.max_context_tokens
        or meta_info.get("max_context_tokens")
        or wrap_cfg.get("max_pe_length")
        or wrap_cfg.get("max_sequence_length")
    )
    prefill_input_sequence_length = wrap_cfg.get("input_sequence_length")
    image_size_h = args.image_size_h or visual_cfg.get("image_max_size_h") or 448
    image_size_w = args.image_size_w or visual_cfg.get("image_max_size_w") or 448
    max_size_t = args.max_size_t or visual_cfg.get("image_max_size_t") or 2
    patch_size = args.patch_size or visual_cfg.get("patch_size") or 16
    temporal_patch_size = args.temporal_patch_size or visual_cfg.get("temporal_patch_size") or 2

    dtype = parse_dtype(args.dtype)
    auto_offload_max_memory = parse_auto_offload_max_memory(args.auto_offload_max_memory)
    prefill_auto_offload_max_memory = parse_auto_offload_max_memory(args.prefill_auto_offload_max_memory)
    decode_auto_offload_max_memory = parse_auto_offload_max_memory(args.decode_auto_offload_max_memory)
    vision_auto_offload_max_memory = parse_auto_offload_max_memory(args.vision_auto_offload_max_memory)

    processor = Qwen3VLProcessor.from_pretrained(str(processor_source), trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(str(hf_model_config_dir), trust_remote_code=True)
    token_embedding = load_token_embedding(token_embedding_file).to(dtype=dtype)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    xh_model = Qwen3VLONNXModel(
        image_feature=ConfigDict(
            onnx=str(vision_onnx),
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        ),
        prefill=ConfigDict(
            onnx=str(prefill_onnx),
            input_sequence_length=prefill_input_sequence_length,
        ),
        decode=ConfigDict(onnx=str(decode_onnx)),
        max_context_tokens=max_context_tokens,
        image_size_w=image_size_w,
        image_size_h=image_size_h,
        max_size_t=max_size_t,
        presence_penalty=args.presence_penalty,
        auto_offload=not args.disable_auto_offload,
        auto_offload_max_memory=auto_offload_max_memory,
        prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
        decode_auto_offload_max_memory=decode_auto_offload_max_memory,
        vision_auto_offload_max_memory=vision_auto_offload_max_memory,
        resource_tight_mode=args.resource_tight_mode,
        pad_token_id=pad_token_id,
        enable_cuda_graph=args.enable_cuda_graph,
        cuda_graph_modules=args.cuda_graph_modules,
    )
    xh_model.image_token_id = int(
        getattr(processor, "image_token_id", getattr(model_config, "image_token_id", xh_model.image_token_id))
    )
    xh_model.video_token_id = int(
        getattr(processor, "video_token_id", getattr(model_config, "video_token_id", xh_model.video_token_id))
    )
    xh_model.vision_start_token_id = int(
        getattr(
            processor,
            "vision_start_token_id",
            getattr(model_config, "vision_start_token_id", xh_model.vision_start_token_id),
        )
    )
    xh_model.spatial_merge_size = int(
        getattr(getattr(model_config, "vision_config", None), "spatial_merge_size", xh_model.spatial_merge_size)
    )
    xh_model.set_input_embeddings(token_embedding)
    xh_model.to(torch.device(args.device))
    xh_model.set_exec_device(torch.device(args.exec_device))
    xh_model.to(dtype)

    logger.info("vision hmonnx: %s", vision_onnx)
    logger.info("prefill hmonnx: %s", prefill_onnx)
    logger.info("decode  hmonnx: %s", decode_onnx)
    logger.info("hf config: %s", hf_model_config_dir)
    logger.info("processor source: %s", processor_source)
    logger.info("token embedding: %s", token_embedding_file)
    logger.info("max_context_tokens=%s max_new_tokens=%s", max_context_tokens, args.max_new_tokens)
    logger.info("cuda_graph=%s modules=%s", args.enable_cuda_graph, args.cuda_graph_modules)
    logger.info("runtime status: %s", xh_model.get_cuda_graph_status())
    return xh_model, processor, meta_info


def run_batch(xh_model, processor, batch: dict, args, logger):
    print(
        f"\n[Batch {batch['batch_index']}] pages "
        f"{batch['image_entries'][0]['page_index']}-{batch['image_entries'][-1]['page_index']}",
        flush=True,
    )
    inputs = xh_model.preprocess(
        batch["messages"],
        media_path="multi-image",
        processor=processor,
        cus_temp=True,
        media_type="image",
    )
    validate_multi_image_inputs(inputs, processor, logger)
    expected_token_counts = get_image_token_counts(
        inputs["image_grid_thw"], processor.image_processor.merge_size
    )
    total_expected_tokens = sum(expected_token_counts)

    extracted_features = xh_model.extract_all_image_features(inputs["hm_pixel_values"])
    if not isinstance(extracted_features, tuple) or len(extracted_features) < 4:
        raise ValueError("Vision runtime must return image_embeds + 3 deepstack tensors")

    image_embeds = extracted_features[0]
    deepstack_image_embeds = tuple(extracted_features[1:4])
    if image_embeds.shape[0] != total_expected_tokens:
        raise ValueError(
            f"image_embeds token count mismatch: expected {total_expected_tokens}, got {image_embeds.shape[0]}"
        )
    for idx, deepstack_embed in enumerate(deepstack_image_embeds):
        if deepstack_embed.shape[0] != total_expected_tokens:
            raise ValueError(
                f"deepstack_image_embed_{idx} token count mismatch: "
                f"expected {total_expected_tokens}, got {deepstack_embed.shape[0]}"
            )

    output_text = xh_model.generate_multimodal(
        input_ids=inputs["input_ids"].to(xh_model.device),
        attention_mask=inputs.get("attention_mask"),
        image_embeds=image_embeds,
        deepstack_image_embeds=deepstack_image_embeds,
        image_grid_thw=inputs.get("image_grid_thw"),
        tokenizer=processor.tokenizer,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        stream_output=args.stream_output,
    )
    if args.stream_output:
        print("", flush=True)
    logger.info("Batch %d output: %s", batch["batch_index"], output_text)
    return output_text


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_runtime_args(args, parser)

    prompt = resolve_prompt(args)

    safe_scene = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in args.scene_name) or "multi_image"
    cfg_name = f"qwen3_vl_multi_image_demo_{safe_scene}"
    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / f"{cfg_name}.log"
    xhquant_llm_init(log_file, args.debug)
    logger = get_root_logger()
    xhquant.utils.suppress_printing.disable_printing = True

    image_paths = resolve_image_paths(args, logger)

    image_size_w = args.image_size_w or 448
    image_size_h = args.image_size_h or 448

    if not args.prepare_only:
        xh_model, processor, meta_info = load_runtime(args, logger)
        visual_cfg = meta_info.get("wrap_cfg", {}).get("visual", {})
        image_size_w = args.image_size_w or visual_cfg.get("image_max_size_w") or image_size_w
        image_size_h = args.image_size_h or visual_cfg.get("image_max_size_h") or image_size_h
    else:
        xh_model = None
        processor = None

    batches = prepare_batches(
        image_paths=image_paths,
        prompt=prompt,
        images_per_batch=args.images_per_batch,
        max_batches=args.max_batches,
        logger=logger,
        target_w=image_size_w,
        target_h=image_size_h,
    )

    preview_batch = batches[0]
    logger.info(
        "Preview batch1 (scene=%s): images=%d first_image=%s",
        args.scene_name,
        len(preview_batch["image_entries"]),
        preview_batch["image_entries"][0]["image_path"].name,
    )

    if args.prepare_only:
        return

    try:
        for batch in batches:
            run_batch(xh_model=xh_model, processor=processor, batch=batch, args=args, logger=logger)
        logger.info("runtime status after generation: %s", xh_model.get_cuda_graph_status())
    finally:
        xh_model.release_image_feature()
        xh_model.release_prefill_session()
        xh_model.release_decode_session()


if __name__ == "__main__":
    main()
