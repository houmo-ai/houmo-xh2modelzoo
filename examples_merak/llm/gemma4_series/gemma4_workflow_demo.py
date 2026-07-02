#!/usr/bin/env python3
"""Gemma4 unified workflow API demo.

This example intentionally keeps all Gemma4 variants behind the same public
workflow/API shape. E2B/E4B, 31B Dense, and 26B-A4B are selected only as presets
(model path + topology label + output naming); the script does not expose the
legacy gemma4_moe/_with_mask public entry points.

Default action is ``base-export`` and applies ``config_overrides={"quant": None}``
so it does not run heavy quantization unless ``--action quant-export`` is chosen.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DEVICE = "cuda"
DEFAULT_SEED = 1024
DEFAULT_PROMPT = "用中文简要介绍一下你自己。"
MIN_VIDEO_FRAMES = 16
PUBLIC_MODEL_ENTRY = "Gemma4ForConditionalGeneration"

Action = Literal["base-export", "existing-hf", "quant-export"]


@dataclass(frozen=True)
class Gemma4Preset:
    """Public demo preset for the unified Gemma4 workflow."""

    name: str
    hf_model_dir: str
    output_slug: str
    topology: str
    config_path: str
    mtp_config_path: str
    assistant_model_dir: str
    public_model_entry: str = PUBLIC_MODEL_ENTRY


# Keep one table for all public variants.  E2B/E4B/31B are dense-size presets in the
# same config family; 26B-A4B is only a topology/model-path preset here, not a
# separate gemma4_moe public API.
PRESETS: dict[str, Gemma4Preset] = {
    "e2b": Gemma4Preset(
        name="e2b",
        hf_model_dir="weights/gemma-4-E2B-it",
        output_slug="gemma4_e2b_unified",
        topology="dense-e2b",
        config_path="./configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml",
        mtp_config_path="./configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full_mtp.yaml",
        assistant_model_dir="weights/gemma-4-E2B-it-assistant",
    ),
    "e4b": Gemma4Preset(
        name="e4b",
        hf_model_dir="weights/gemma-4-E4B-it",
        output_slug="gemma4_e4b_unified",
        topology="dense-e4b",
        config_path="./configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml",
        mtp_config_path="./configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full_mtp.yaml",
        assistant_model_dir="weights/gemma-4-E4B-it-assistant",
    ),
    "31b": Gemma4Preset(
        name="31b",
        hf_model_dir="weights/gemma-4-31B-it",
        output_slug="gemma4_31b_unified",
        topology="dense-31b",
        config_path="./configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full.yaml",
        mtp_config_path="./configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full_mtp.yaml",
        assistant_model_dir="weights/gemma-4-31B-it-assistant",
    ),
    "26b-a4b": Gemma4Preset(
        name="26b-a4b",
        hf_model_dir="weights/gemma-4-26B-A4B-it",
        output_slug="gemma4_26b_a4b_unified",
        topology="moe-26b-a4b",
        config_path="./configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml",
        mtp_config_path="./configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full_mtp.yaml",
        assistant_model_dir="weights/gemma-4-26B-A4B-it-assistant",
    ),
}


def _remove_output_dir_if_needed(output_dir: Path, force: bool) -> None:
    if force and output_dir.exists():
        shutil.rmtree(output_dir)


def _preset_with_cli_overrides(args: argparse.Namespace) -> Gemma4Preset:
    preset = PRESETS[args.preset]
    config_path = args.config_path or (preset.mtp_config_path if args.mtp_config else preset.config_path)
    return Gemma4Preset(
        name=preset.name,
        hf_model_dir=args.hf_model_dir or preset.hf_model_dir,
        output_slug=preset.output_slug,
        topology=preset.topology,
        config_path=config_path,
        mtp_config_path=preset.mtp_config_path,
        assistant_model_dir=args.assistant_model_dir or preset.assistant_model_dir,
        public_model_entry=preset.public_model_entry,
    )


def _build_output_dirs(work_dir: Path, preset: Gemma4Preset, action: Action) -> tuple[Path, Path]:
    root = work_dir / preset.output_slug / action
    return root / "quant", root / "export"


def _looks_like_gguf_artifact(path: str | None) -> bool:
    if not path:
        return False
    artifact = Path(path)
    if artifact.is_file():
        return artifact.suffix.lower() == ".gguf"
    if artifact.is_dir():
        return any(child.suffix.lower() == ".gguf" for child in artifact.iterdir())
    return str(path).lower().endswith(".gguf")


def _existing_artifact_format(args: argparse.Namespace) -> str:
    if args.existing_artifact_format != "auto":
        return args.existing_artifact_format
    return "gguf_qat" if _looks_like_gguf_artifact(args.existing_hf_model_dir) else "gptqmodel_hf"


def _quant_overrides(args: argparse.Namespace, action: Action) -> dict[str, Any] | None:
    if action == "base-export":
        # Explicit base export: export from the original HF checkpoint and do not
        # run heavy quantization by default.
        return {"quant": None}
    if action == "existing-hf":
        if not args.existing_hf_model_dir:
            raise ValueError("--existing-hf-model-dir is required for --action existing-hf")
        # Intended for unified workflow configs that support an already-quantized
        # HF checkpoint as the quant stage input.  The concrete workflow owns the
        # interpretation and validation of these quant keys.
        artifact_format = _existing_artifact_format(args)
        return {
            "quant": {
                "algorithm": "existing_hf",
                "artifact_format": artifact_format,
                "output_format": artifact_format,
                "existing_hf_model_dir": args.existing_hf_model_dir,
                "bits": 4,
                "group_size": 64,
            }
        }
    # Heavy quantization is opt-in only.  Use the quant block from the workflow
    # config unless callers pass their own config file/implementation defaults.
    return None


def _export_overrides(args: argparse.Namespace, preset: Gemma4Preset, action: Action) -> dict[str, Any] | None:
    overrides: dict[str, Any] = {}
    if args.context_max_length is not None:
        overrides["export.model.context_max_length"] = args.context_max_length
    if args.prefill_chunk_length is not None:
        overrides["export.model.prefill_chunk_length"] = args.prefill_chunk_length
    if action == "existing-hf" and _existing_artifact_format(args) == "gguf_qat":
        overrides["export.model.quant_scheme.quant_type"] = "w4a8h0_ssfp"
        overrides["export.model.visual_config.quant_scheme.quant_type"] = "w4a8h0_ssfp"
        overrides["export.model.video_visual_config.quant_scheme.quant_type"] = "w4a8h0_ssfp"
    if args.mtp_config:
        overrides["export.model.spec_decode_mode"] = "mtp"
    if args.mtp_config:
        overrides["export.model.mtp_config.assistant_hf_model"] = preset.assistant_model_dir
        target_hf_model = args.existing_hf_model_dir if action == "existing-hf" else preset.hf_model_dir
        overrides["export.model.mtp_config.target_hf_model"] = target_hf_model
    elif args.assistant_model_dir:
        overrides["export.model.mtp_config.assistant_hf_model"] = args.assistant_model_dir
    if args.sliding_kv_cache_input_mode is not None:
        overrides["export.model.sliding_kv_cache_input_mode"] = args.sliding_kv_cache_input_mode
    return overrides or None


def _input_messages(args: argparse.Namespace, *, dry_run: bool = False) -> dict[str, str | int]:
    messages: dict[str, str | int] = {"text": args.prompt}
    if args.modality == "image":
        if args.image_path:
            messages["image"] = args.image_path
        elif dry_run:
            messages["image"] = "<synthetic_image>"
    elif args.modality == "video":
        if args.video_path:
            messages["video"] = args.video_path
        elif dry_run:
            messages["video_frames"] = max(args.video_num_frames, MIN_VIDEO_FRAMES)
    elif args.modality == "audio":
        if args.audio_path:
            messages["audio"] = args.audio_path
        elif dry_run:
            messages["audio"] = "<synthetic_audio_wav>"
    return messages


def _audio_support_status(preset: Gemma4Preset) -> str:
    if preset.name not in {"e2b", "e4b"}:
        return "unsupported_by_model"
    config_path = Path(preset.hf_model_dir) / "config.json"
    if not config_path.exists():
        return "unknown_missing_hf_config"
    import json

    with config_path.open(encoding="utf-8") as f:
        hf_config = json.load(f)
    return "supported" if isinstance(hf_config.get("audio_config"), dict) else "unsupported_by_model"


def run(args: argparse.Namespace) -> None:
    action: Action = args.action
    preset = _preset_with_cli_overrides(args)
    quant_output_dir, export_output_dir = _build_output_dirs(Path(args.work_dir), preset, action)
    quant_overrides = _quant_overrides(args, action)
    export_overrides = _export_overrides(args, preset, action)

    if args.dry_run:
        print(f"preset: {preset.name} ({preset.topology})")
        print(f"hf_model_dir: {preset.hf_model_dir}")
        print(f"assistant_model_dir: {preset.assistant_model_dir}")
        print(f"config_path: {preset.config_path}")
        print(f"public_model_entry: {preset.public_model_entry}")
        print(f"action: {action}")
        print(f"quant_output_dir: {quant_output_dir}")
        print(f"export_output_dir: {export_output_dir}")
        print(f"quant_overrides: {quant_overrides}")
        print(f"export_overrides: {export_overrides}")
        print(f"input_messages: {_input_messages(args, dry_run=True)}")
        if args.modality == "audio":
            print(f"audio_support: {_audio_support_status(preset)}")
        print(f"dump_golden: {args.golden}")
        return

    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    _remove_output_dir_if_needed(quant_output_dir, args.force)
    _remove_output_dir_if_needed(export_output_dir, args.force)

    workflow = AutoLLMWorkflow.from_config(
        hf_model_dir=preset.hf_model_dir,
        config_path=preset.config_path,
        seed=args.seed,
        debug=args.debug,
    )

    quant_result = workflow.quant(
        output_dir=str(quant_output_dir),
        device=args.device,
        config_overrides=quant_overrides,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=str(export_output_dir),
        device=args.device,
        config_overrides=export_overrides,
    )

    if args.golden and args.modality == "audio" and _audio_support_status(preset) != "supported":
        raise ValueError(
            f"audio is unsupported by preset {preset.name}; use e2b/e4b with a HF config containing audio_config"
        )
    if args.golden and args.modality == "audio" and not args.audio_path:
        raise ValueError("audio golden dumping requires --audio-path; synthetic audio is dry-run only")
    if args.golden and args.modality == "video" and not args.video_path:
        raise ValueError("video golden dumping requires --video-path; synthetic frame count is dry-run only")

    if args.golden:
        workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages=_input_messages(args),
        )

    print(f"preset: {preset.name} ({preset.topology})")
    print(f"quant_result: {quant_result}")
    print(f"export_result: {export_result}")
    return {"quant_result": quant_result, "export_result": export_result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="e4b")
    parser.add_argument(
        "--action",
        choices=("base-export", "existing-hf", "quant-export"),
        default="base-export",
        help=(
            "base-export skips quant with config_overrides={'quant': None}; "
            "existing-hf demonstrates an already-quantized HF override; "
            "quant-export runs the config quant block and is intentionally opt-in."
        ),
    )
    parser.add_argument("--hf-model-dir", help="Override the preset target/base HF model path.")
    parser.add_argument("--assistant-model-dir", help="Override the preset Gemma4 assistant/MTP draft HF model path.")
    parser.add_argument(
        "--existing-hf-model-dir",
        help=(
            "Already-quantized artifact for --action existing-hf. Supports GPTQModel-compatible HF directories "
            "and official Gemma4 QAT GGUF directories/files."
        ),
    )
    parser.add_argument(
        "--existing-artifact-format",
        choices=("auto", "gptqmodel_hf", "gguf_qat"),
        default="auto",
        help="Artifact format for --existing-hf-model-dir; auto detects .gguf paths as gguf_qat.",
    )
    parser.add_argument("--config-path", help="Override the shared unified Gemma4 workflow YAML path.")
    parser.add_argument("--mtp-config", action="store_true", help="Use the preset full_mtp YAML and enable base MTP outputs.")
    parser.add_argument("--work-dir", default="./work_dirs/gemma4_unified_workflow_demo")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--context-max-length", type=int)
    parser.add_argument("--prefill-chunk-length", type=int)
    parser.add_argument(
        "--sliding-kv-cache-input-mode",
        choices=("slice_window", "legacy_full"),
        help=(
            "Static KV input shape for sliding-attention layers. "
            "slice_window exports compact sliding_window+input_sequence caches; "
            "legacy_full keeps full-context-sized inputs like full-attention layers."
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--modality", choices=("text", "image", "video", "audio"), default="text")
    parser.add_argument("--image-path", default=None, help="Optional image path for image golden dumping.")
    parser.add_argument(
        "--video-path",
        default=None,
        help="Optional local video path for video dry-run/golden metadata.",
    )
    parser.add_argument(
        "--video-num-frames",
        type=int,
        default=MIN_VIDEO_FRAMES,
        help="Synthetic video frame count for dry-run metadata; clamped to >=16.",
    )
    parser.add_argument(
        "--audio-path",
        default=None,
        help="Optional local audio path for audio dry-run/golden metadata; supported only by E4B.",
    )
    parser.add_argument("--golden", action="store_true", help="Call dump_golden after export.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true", help="Remove existing demo output directories before running.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved preset/actions without importing or running workflow.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
