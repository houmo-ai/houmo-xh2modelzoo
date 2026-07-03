#!/usr/bin/env python3
"""Compatibility export wrapper for the unified Gemma4 Series workflow.

This script keeps the old ``examples_merak/llm/gemma4_series/export_hmonnx.py``
entry point, but it no longer imports or dispatches to ``gemma4``, ``gemma4e``,
or ``gemma4_moe`` implementations.  All real work is delegated to
``gemma4_workflow_demo.py`` so E4B, 31B dense, and 26B-A4B MoE use the same
public workflow/model API.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples_merak.llm.gemma4_series import gemma4_workflow_demo


_VARIANT_TO_PRESET = {
    "dense": "31b",
    "e2b": "e2b",
    "e4b": "e4b",
    "moe": "26b-a4b",
    "26b-a4b": "26b-a4b",
    "31b": "31b",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gemma4 Series export compatibility wrapper. Prefer calling "
            "gemma4_workflow_demo.py directly for new automation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        choices=sorted(_VARIANT_TO_PRESET),
        default="dense",
        help="Legacy variant selector mapped to a unified Gemma4 Series preset.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(gemma4_workflow_demo.PRESETS),
        help="Explicit unified preset. Overrides --variant when provided.",
    )
    parser.add_argument(
        "--mode",
        choices=("llm", "vision", "all"),
        default="llm",
        help="Legacy option kept for CLI compatibility. Unified export is model-owned.",
    )
    parser.add_argument(
        "--action",
        choices=("base-export", "existing-hf", "quant-export"),
        default="base-export",
        help="Unified workflow action.",
    )
    parser.add_argument("--model", dest="hf_model_dir", help="HF model directory override.")
    parser.add_argument("--config", dest="config_path", help="Workflow YAML override.")
    parser.add_argument("--existing-hf-model-dir", help="Already-quantized HF checkpoint for existing-hf.")
    parser.add_argument("--work-dir", default="./work_dirs/gemma4_series_export")
    parser.add_argument("--device", default=gemma4_workflow_demo.DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=gemma4_workflow_demo.DEFAULT_SEED)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--prefill-chunk-length", type=int, default=320)
    parser.add_argument("--prompt", default=gemma4_workflow_demo.DEFAULT_PROMPT)
    parser.add_argument("--modality", choices=("text", "image", "video", "audio"), default="text")
    parser.add_argument("--image-path")
    parser.add_argument("--video-path")
    parser.add_argument("--video-num-frames", type=int, default=gemma4_workflow_demo.MIN_VIDEO_FRAMES)
    parser.add_argument("--audio-path")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    # Deprecated no-op options accepted so existing shell snippets fail less
    # abruptly while still using the unified workflow path.
    parser.add_argument("--model-type", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--chip-arch", default="XH2a", help=argparse.SUPPRESS)
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help=argparse.SUPPRESS)
    parser.add_argument("--quant-weight", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--image-size-w", type=int, default=448, help=argparse.SUPPRESS)
    parser.add_argument("--image-size-h", type=int, default=448, help=argparse.SUPPRESS)
    parser.add_argument("--image", dest="legacy_image", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--valid", action="store_true", help=argparse.SUPPRESS)
    return parser


def _to_workflow_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "vision":
        raise ValueError(
            "Gemma4 Series no longer exposes a standalone legacy vision export "
            "entry here. Use the unified model export; image/video subgraphs are "
            "exported by XHGemma4SeriesModel."
        )
    if args.valid:
        raise ValueError(
            "--valid was a legacy smoke-test shortcut. Gemma4 Series validation "
            "must use full context=2048 and prefill/input length=320."
        )
    if args.quant_weight:
        raise ValueError(
            "--quant-weight belongs to the legacy export path. Use "
            "--action existing-hf --existing-hf-model-dir for an already-quantized "
            "HF artifact, or --action quant-export for workflow-owned quant."
        )

    preset = args.preset or _VARIANT_TO_PRESET[args.variant]
    image_path = args.image_path or args.legacy_image
    return argparse.Namespace(
        preset=preset,
        action=args.action,
        hf_model_dir=args.hf_model_dir,
        existing_hf_model_dir=args.existing_hf_model_dir,
        config_path=args.config_path,
        work_dir=args.work_dir,
        device=args.device,
        seed=args.seed,
        debug=args.debug,
        context_max_length=args.context_length,
        prefill_chunk_length=args.prefill_chunk_length,
        sliding_kv_cache_input_mode=None,
        prompt=args.prompt,
        modality=args.modality,
        image_path=image_path,
        video_path=args.video_path,
        video_num_frames=args.video_num_frames,
        audio_path=args.audio_path,
        golden=args.golden,
        force=args.force,
        dry_run=args.dry_run,
    )


def main() -> None:
    args = _build_parser().parse_args()
    workflow_args = _to_workflow_args(args)
    gemma4_workflow_demo.run(workflow_args)


if __name__ == "__main__":
    main()
