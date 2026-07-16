#!/usr/bin/env python3
"""Strict Gemma4 Series e2e validation harness.

The script is intentionally stricter than the lightweight dry-run helpers:

- every checked runtime meta must be exported with context=2048 and
  prefill/input sequence length=320;
- text prompts must tokenize to >1024 tokens and still fit the 2048 context
  with ``max_new_tokens``;
- every preset must cover text + image generate;
- E4B additionally covers video + audio generate;
- ``--run`` executes the real ``generate.py`` commands sequentially.  Without
  ``--run`` the script validates metadata/prompt/media and prints summarized
  commands, but does not claim e2e completion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw
from transformers import AutoTokenizer


try:
    import onnx
except ImportError:  # pragma: no cover - surfaced only when MTP ONNX validation runs.
    onnx = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
GENERATE_SCRIPT = SCRIPT_DIR / "generate.py"

PRESETS = ("e2b", "e4b", "31b", "26b-a4b")
REQUIRED_CONTEXT = 2048
REQUIRED_PREFILL = 320
MIN_PROMPT_TOKENS = 1025
DEFAULT_MAX_NEW_TOKENS = 32


@dataclass(frozen=True)
class Case:
    preset: str
    modality: str
    meta_path: Path
    prompt: str
    media_path: Path | None = None


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _resolve_meta_path(path: str | Path) -> Path:
    meta_path = Path(path).resolve()
    if not meta_path.exists():
        raise FileNotFoundError(f"Runtime meta does not exist: {meta_path}")
    meta = _load_json(meta_path)
    if meta.get("model_config", {}).get("model_type"):
        return meta_path
    exported_dir = meta.get("exported_dir")
    if exported_dir:
        candidate = (meta_path.parent / exported_dir / "golden_meta_info.json").resolve()
        if candidate.exists():
            return candidate
    raise ValueError(f"Meta path must be golden_meta_info.json or export_meta_info.json: {meta_path}")


def _resolve_hf_config_dir(meta_path: Path, meta: dict) -> Path:
    hf_config = meta.get("hf_config")
    if not hf_config:
        raise ValueError(f"Runtime meta has no hf_config: {meta_path}")
    path = Path(hf_config)
    if not path.is_absolute():
        path = meta_path.parent / path
    if path.is_file():
        path = path.parent
    if not path.exists():
        raise FileNotFoundError(f"hf_config path does not exist: {path}")
    return path


def _read_prefill_length(meta: dict) -> int | None:
    model_cfg = meta.get("model_config", {})
    for key in ("prefill_chunk_length", "input_sequence_length"):
        value = model_cfg.get(key)
        if value is not None:
            return int(value)
    prefill_graph = (meta.get("prefill_graphs") or {}).get("prefill")
    if isinstance(prefill_graph, dict):
        value = prefill_graph.get("input_sequence_length") or prefill_graph.get("prefill_chunk_length")
        if value is not None:
            return int(value)
    return None


def _aligned(size: int, alignment: int = 16) -> int:
    return ((size + alignment - 1) // alignment) * alignment


def _required_sliding_cache_length(prefill: int, sliding_window: int) -> int:
    # Gemma4 sliding layers use the slice-window cache contract: prefill feeds
    # the full current input chunk, while the cache output keeps only
    # ``sliding_window + input_sequence_length`` tokens.  Full-attention layers
    # still keep the exported context length.  This mirrors runtime metadata
    # where sliding E4B layers are 512 + 320 = 832 and global layers are 2048.
    return _aligned(sliding_window + prefill, 16)


def _validate_sliding_cache_shapes(meta_path: Path, meta: dict, context: int, prefill: int) -> None:
    layer_types = meta.get("layer_types") or []
    if "sliding_attention" not in layer_types:
        return
    model_cfg = meta.get("model_config", {})
    sliding_window = int(meta.get("sliding_window") or model_cfg.get("sliding_window") or 0)
    if sliding_window <= 0:
        raise ValueError(f"{meta_path}: sliding_attention layers require a positive sliding_window")
    required_sliding = _required_sliding_cache_length(prefill, sliding_window)
    layer_kv_shapes = meta.get("layer_kv_shapes") or meta.get("kv_cache_shapes_per_layer") or []
    if not layer_kv_shapes:
        raise ValueError(f"{meta_path}: missing per-layer KV cache shapes for Gemma4 sliding cache")
    for idx, shape in enumerate(layer_kv_shapes):
        if len(shape) <= 2:
            raise ValueError(f"{meta_path}: layer {idx} has invalid KV shape: {shape}")
        cache_len = int(shape[2])
        # E4B has shared-KV layers: not every transformer layer owns an
        # independent cache tensor.  Metadata therefore stores cache tensors
        # only for non-shared KV owners, and individual cache lengths must be
        # valid for either a sliding owner or a full/global owner.
        if cache_len < required_sliding:
            raise ValueError(
                f"{meta_path}: cache tensor {idx} length={cache_len}, "
                f"expected >= sliding window contract {required_sliding}"
            )
    if any(layer_type != "sliding_attention" for layer_type in layer_types):
        if not any(int(shape[2]) >= context for shape in layer_kv_shapes if len(shape) > 2):
            raise ValueError(f"{meta_path}: full/global attention layers require at least one context-length cache")


def _tensor_dims(value_info) -> list[int | str]:
    dims: list[int | str] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        else:
            dims.append(dim.dim_param or "?")
    return dims


def _onnx_tensor_dims(model, name: str, *, include_outputs: bool = False) -> list[int | str] | None:
    values = list(model.graph.input)
    if include_outputs:
        values += list(model.graph.output)
    for value in values:
        if value.name == name:
            return _tensor_dims(value)
    return None


def _kv_cache_attr_values(model, attr_name: str) -> list[int]:
    values: list[int] = []
    for node in model.graph.node:
        if node.op_type != "KVcache":
            continue
        for attr in node.attribute:
            if attr.name == attr_name:
                values.append(int(onnx.helper.get_attribute_value(attr)))
    return values


def _kv_cache_only_handle_attrs(model) -> list[int]:
    values: list[int] = []
    for node in model.graph.node:
        if node.op_type != "KVcache":
            continue
        for attr in node.attribute:
            if attr.name in {"only_handle_old_cache", "only-handle-old-cache"}:
                values.append(int(onnx.helper.get_attribute_value(attr)))
    return values


def _load_onnx_graph(meta_path: Path, meta: dict, key: str):
    rel_path = meta.get(key)
    if not rel_path:
        raise ValueError(f"{meta_path}: MTP meta missing {key}")
    path = Path(rel_path)
    if not path.is_absolute():
        path = meta_path.parent / path
    if not path.exists():
        raise FileNotFoundError(f"{meta_path}: {key} does not exist: {path}")
    return path, onnx.load(str(path), load_external_data=False)


def _validate_mtp_hmonnx_contract(meta_path: Path, meta: dict, context: int, prefill: int) -> None:
    """Validate exported MTP prefill/decode graphs, not only metadata.

    Stale exports can keep correct ``layer_kv_shapes`` in JSON while the ONNX
    decode graph still has the old single-token contract.  The acceptance path
    needs decode verify length = 1 + draft tokens.  The target sliding mask
    follows LLMCache's compact output width
    ``aligned(sliding_window + verify_length - 1, 16)`` while the draft shared
    KV inputs keep the prefill-owned physical slice-window length.
    """

    if int(meta.get("attention_contract_version", 1)) >= 2:
        return
    model_cfg = meta.get("model_config", {})
    spec_decode = meta.get("spec_decode") or {}
    is_mtp = meta.get("spec_decode_mode") == "mtp" or bool(model_cfg.get("enable_mtp_outputs"))
    if not is_mtp:
        return
    if onnx is None:
        raise RuntimeError("onnx is required to validate Gemma4 MTP HMONNX contracts")

    sliding_window = int(meta.get("sliding_window") or model_cfg.get("sliding_window") or 0)
    if sliding_window <= 0:
        raise ValueError(f"{meta_path}: MTP export requires positive sliding_window")
    expected_sliding = _required_sliding_cache_length(prefill, sliding_window)
    configured_sliding = int(spec_decode.get("shared_sliding_cache_length") or expected_sliding)
    if configured_sliding != expected_sliding:
        raise ValueError(
            f"{meta_path}: spec_decode shared sliding cache length={configured_sliding}, "
            f"expected {expected_sliding} (= sliding_window + prefill)"
        )

    verify_length = int(
        spec_decode.get("verify_length")
        or (int(spec_decode.get("block_size", model_cfg.get("num_draft_tokens", 4))) + 1)
    )
    if verify_length <= 1:
        raise ValueError(f"{meta_path}: MTP verify_length must be >1, got {verify_length}")

    prefill_path, prefill_model = _load_onnx_graph(meta_path, meta, "prefill_hmonnx")
    decode_path, decode_model = _load_onnx_graph(meta_path, meta, "decode_hmonnx")

    expected_decode_sliding = _aligned(sliding_window + verify_length - 1, 16)
    expected_prefill_mask = [1, 1, prefill, expected_sliding]
    expected_decode_mask = [1, 1, verify_length, expected_decode_sliding]
    prefill_mask = _onnx_tensor_dims(prefill_model, "sliding_attention_mask")
    decode_mask = _onnx_tensor_dims(decode_model, "sliding_attention_mask")
    if prefill_mask != expected_prefill_mask:
        raise ValueError(
            f"{prefill_path}: sliding_attention_mask shape={prefill_mask}, expected {expected_prefill_mask}"
        )
    if decode_mask != expected_decode_mask:
        raise ValueError(
            f"{decode_path}: sliding_attention_mask shape={decode_mask}, "
            f"expected {expected_decode_mask}; stale decode exports often show sequence length 1"
        )

    for graph_path, model in ((prefill_path, prefill_model), (decode_path, decode_model)):
        outputs = [out.name for out in model.graph.output]
        if outputs != ["logits", "target_hidden_state"]:
            raise ValueError(f"{graph_path}: MTP target outputs={outputs}, expected logits + target_hidden_state only")

    expected_prefill_amax = sliding_window
    expected_decode_amax = sliding_window
    prefill_amax = set(_kv_cache_attr_values(prefill_model, "attention_max_length"))
    decode_amax = set(_kv_cache_attr_values(decode_model, "attention_max_length"))
    if expected_prefill_amax not in prefill_amax:
        raise ValueError(
            f"{prefill_path}: missing sliding KVcache attention_max_length={expected_prefill_amax}; "
            f"found {sorted(prefill_amax)}"
        )
    if expected_decode_amax not in decode_amax:
        raise ValueError(
            f"{decode_path}: missing sliding KVcache attention_max_length={expected_decode_amax}; "
            f"found {sorted(decode_amax)}"
        )

    for graph_path, model in ((prefill_path, prefill_model), (decode_path, decode_model)):
        default_attrs = [value for value in _kv_cache_only_handle_attrs(model) if value == 0]
        if default_attrs:
            raise ValueError(
                f"{graph_path}: default-false only_handle_old_cache attrs must be omitted, "
                f"found {len(default_attrs)} explicit zeros"
            )


def _validate_flash_hmonnx_contract(meta_path: Path, meta: dict) -> dict[str, dict[str, int]]:
    """Validate contract-v2 compact inputs and emitted FlashAttention nodes."""

    if int(meta.get("attention_contract_version", 1)) < 2:
        return {}
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        validate_gemma4_flash_attention_graph,
    )

    facts: dict[str, dict[str, int]] = {}
    for graph_key in ("prefill_hmonnx", "decode_hmonnx"):
        graph_value = meta.get(graph_key)
        if not graph_value:
            raise ValueError(f"{meta_path}: contract-v2 meta missing {graph_key}")
        graph_path = Path(graph_value)
        if not graph_path.is_absolute():
            graph_path = meta_path.parent / graph_path
        if not graph_path.exists():
            raise FileNotFoundError(f"{meta_path}: {graph_key} does not exist: {graph_path}")
        facts[graph_key] = validate_gemma4_flash_attention_graph(graph_path, meta)
    return facts


def validate_meta(meta_path: Path, expected_preset: str) -> dict:
    meta = _load_json(meta_path)
    model_cfg = meta.get("model_config", {})
    context = int(model_cfg.get("context_max_length", 0))
    if context != REQUIRED_CONTEXT:
        raise ValueError(f"{meta_path}: context_max_length={context}, expected {REQUIRED_CONTEXT}")
    prefill = _read_prefill_length(meta)
    if prefill != REQUIRED_PREFILL:
        raise ValueError(f"{meta_path}: prefill/input length={prefill}, expected {REQUIRED_PREFILL}")
    _validate_sliding_cache_shapes(meta_path, meta, context, prefill)
    _validate_mtp_hmonnx_contract(meta_path, meta, context, prefill)
    _validate_flash_hmonnx_contract(meta_path, meta)
    model_type = model_cfg.get("model_type")
    if model_type != "Gemma4ForConditionalGeneration":
        raise ValueError(f"{meta_path}: model_type={model_type!r}, expected Gemma4ForConditionalGeneration")

    if expected_preset in {"e2b", "e4b"}:
        if not meta.get("audio_config"):
            raise ValueError(f"{meta_path}: {expected_preset} e2e requires audio_config in runtime meta")
        if not meta.get("per_layer_input_embedding"):
            raise ValueError(f"{meta_path}: {expected_preset} e2e requires per_layer_input_embedding artifact")
    else:
        if meta.get("audio_config"):
            raise ValueError(f"{meta_path}: {expected_preset} should not expose audio_config")

    if not meta.get("visual_config"):
        raise ValueError(f"{meta_path}: missing image visual_config")
    if not meta.get("video_visual_config"):
        raise ValueError(f"{meta_path}: missing separate video_visual_config")
    return meta


def _long_prompt(preset: str, modality: str) -> str:
    # Multimodal processors add media placeholder tokens after this text-only
    # check. Keep prompts long enough to expose >1024-token prefill behavior,
    # but leave fixed 2048-context headroom for image/video/audio tokens and
    # decode. Text-only can use the longest prompt.
    if modality == "text":
        fact_count = 23
    elif modality == "image":
        fact_count = 21
    else:
        fact_count = 15
    facts = []
    for idx in range(1, fact_count + 1):
        facts.append(
            f"资料条目{idx:03d}: Gemma4 Series 统一 API 要求 preset={preset} "
            f"在 modality={modality} 验证中保持 context=2048、input_sequence_length=320，"
            f"并且不能把旧 gemma4/gemma4e/gemma4_moe 路径作为新实现入口。"
        )
    media_questions = {
        "text": "",
        "image": (" 同时请观察随附图片，回答图片里有哪些文字、颜色和几何形状，不要忽略图像内容。"),
        "video": (" 同时请观察随附多帧视频，回答绿色矩形如何随帧移动以及画面中有哪些文字，不要把视频当成单张图片。"),
        "audio": (" 同时请聆听随附音频，概括你听到的人声/语音内容；如果不是语音，请说明可听到的声音特征。"),
    }
    question = (
        f"请只根据以上资料回答：当前 preset={preset} 的 {modality} 验收为什么必须使用"
        "统一 gemma4_series workflow，并指出 context 和 input_sequence_length 的数值。"
        + media_questions.get(modality, "")
    )
    return "\n".join(facts + [question])


def _prompt_token_count(meta_path: Path, meta: dict, prompt: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(_resolve_hf_config_dir(meta_path, meta), trust_remote_code=True)
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return int(tokenizer([text], return_tensors="pt")["input_ids"].shape[1])


def validate_prompt(meta_path: Path, meta: dict, prompt: str, max_new_tokens: int) -> int:
    token_count = _prompt_token_count(meta_path, meta, prompt)
    if token_count < MIN_PROMPT_TOKENS:
        raise ValueError(f"Prompt has {token_count} tokens, expected >= {MIN_PROMPT_TOKENS}")
    if token_count + max_new_tokens > REQUIRED_CONTEXT:
        raise ValueError(f"Prompt tokens {token_count} + max_new_tokens {max_new_tokens} exceeds {REQUIRED_CONTEXT}")
    return token_count


def _validate_media(case: Case) -> None:
    if case.modality == "text":
        return
    if case.media_path is None:
        raise ValueError(f"{case.preset}/{case.modality}: media path is required")
    if not case.media_path.exists():
        raise FileNotFoundError(f"{case.preset}/{case.modality}: media does not exist: {case.media_path}")
    if case.modality == "video" and case.media_path.is_dir():
        frames = [p for p in case.media_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if len(frames) < 2:
            raise ValueError(f"video directory must contain at least 2 frames: {case.media_path}")


def _create_image_fixture(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 280, 220), fill="red")
    draw.ellipse((340, 60, 560, 240), fill="blue")
    draw.text((60, 300), "Gemma4 Series QTL-384", fill="black")
    draw.text((60, 340), "red rectangle + blue circle", fill="black")
    image.save(path)
    return path


def _create_video_fixture(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for idx in range(8):
        image = Image.new("RGB", (640, 480), "white")
        draw = ImageDraw.Draw(image)
        x0 = 40 + idx * 30
        draw.rectangle((x0, 80, x0 + 120, 200), fill="green")
        draw.text((60, 300), f"Gemma4 video frame {idx}", fill="black")
        image.save(path / f"frame_{idx:03d}.png")
    return path


def maybe_create_fixtures(work_dir: Path) -> tuple[Path, Path]:
    fixture_dir = work_dir / "fixtures"
    return (
        _create_image_fixture(fixture_dir / "gemma4_series_image.png"),
        _create_video_fixture(fixture_dir / "video_frames"),
    )


def build_cases(args: argparse.Namespace) -> list[Case]:
    meta_by_preset = {
        "e2b": args.e2b_meta,
        "e4b": args.e4b_meta,
        "31b": args.meta_31b,
        "26b-a4b": args.meta_26b_a4b,
    }
    selected = PRESETS if args.preset == "all" else (args.preset,)
    image_path = Path(args.image_path).resolve() if args.image_path else None
    video_path = Path(args.video_path).resolve() if args.video_path else None
    if args.create_visual_fixtures:
        image_path, video_path = maybe_create_fixtures(Path(args.work_dir).resolve())
    audio_path = Path(args.audio_path).resolve() if args.audio_path else None

    cases: list[Case] = []
    for preset in selected:
        meta_arg = meta_by_preset[preset]
        if not meta_arg:
            raise ValueError(f"--{preset.replace('-', '_')}-meta is required for preset {preset}")
        meta_path = _resolve_meta_path(meta_arg)
        modalities = ["text", "image"]
        if preset == "e4b":
            modalities += ["video", "audio"]
        for modality in modalities:
            media = None
            if modality == "image":
                media = image_path
            elif modality == "video":
                media = video_path
            elif modality == "audio":
                media = audio_path
            cases.append(Case(preset, modality, meta_path, _long_prompt(preset, modality), media))
    return cases


def command_for_case(case: Case, device: str, max_new_tokens: int) -> list[str]:
    cmd = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "--backend",
        "hmonnx",
        "--model-config",
        str(case.meta_path),
        "--prompt",
        case.prompt,
        "--max-decode-steps",
        str(max_new_tokens),
        "--device",
        device,
    ]
    if case.modality == "image":
        cmd += ["--image-path", str(case.media_path)]
    elif case.modality == "video":
        cmd += ["--video-path", str(case.media_path), "--video-num-frames", "8"]
    elif case.modality == "audio":
        cmd += ["--audio-path", str(case.media_path)]
    return cmd


def iter_cases(args: argparse.Namespace) -> Iterable[tuple[Case, dict, int, list[str]]]:
    for case in build_cases(args):
        meta = validate_meta(case.meta_path, case.preset)
        _validate_media(case)
        token_count = validate_prompt(case.meta_path, meta, case.prompt, args.max_new_tokens)
        yield case, meta, token_count, command_for_case(case, args.device, args.max_new_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--preset", choices=("all", *PRESETS), default="all")
    parser.add_argument("--e2b-meta")
    parser.add_argument("--e4b-meta")
    parser.add_argument("--31b-meta", dest="meta_31b")
    parser.add_argument("--26b-a4b-meta", dest="meta_26b_a4b")
    parser.add_argument("--image-path")
    parser.add_argument("--video-path")
    parser.add_argument("--audio-path")
    parser.add_argument("--create-visual-fixtures", action="store_true")
    parser.add_argument("--work-dir", default="./work_dirs/gemma4_series_e2e_validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--run", action="store_true", help="Execute real generate.py commands sequentially.")
    parser.add_argument(
        "--print-full-commands", action="store_true", help="Print full commands including long prompts."
    )
    args = parser.parse_args()

    planned = list(iter_cases(args))
    for case, _meta, token_count, cmd in planned:
        media = f" media={case.media_path}" if case.media_path else ""
        print(f"[{case.preset}/{case.modality}] prompt_tokens={token_count}{media}")
        if args.print_full_commands:
            print("  " + " ".join(_shell_quote(part) for part in cmd))
        else:
            print("  " + _summarize_command(cmd, case))

    if not args.run:
        print("DRY-RUN ONLY: metadata, media, and long-prompt checks passed; generate was not executed.")
        return

    for case, _meta, _token_count, cmd in planned:
        print(f"RUN {case.preset}/{case.modality}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _summarize_command(cmd: list[str], case: Case) -> str:
    parts: list[str] = []
    skip_next = False
    for part in cmd:
        if skip_next:
            skip_next = False
            continue
        if part == "--prompt":
            parts.extend([part, f"<long-{case.preset}-{case.modality}-prompt>"])
            skip_next = True
        else:
            parts.append(part)
    return " ".join(_shell_quote(part) for part in parts)


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:@%")
    if all(ch in safe for ch in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    main()
