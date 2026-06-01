#!/usr/bin/env python3
"""Check whether a migrated model has the expected artifacts in xh2modelzoo.

Usage:
  python scripts/check_migration_artifacts.py --model qwen3_legacy --architecture Qwen3ForCausalLM_legacy
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable


def normalize_name(name: str) -> str:
    return re.sub(r"[-.]", "_", name)


def has_pattern(files: list[str], pattern: str) -> bool:
    rgx = re.compile(pattern)
    return any(rgx.search(f) for f in files)


def infer_archetype(files: list[str]) -> str:
    has_moe = any("_moe_model" in f for f in files)
    has_vision = any("_vision_model" in f or "_vision_model_impl" in f for f in files)
    has_llm = any("_llm_model" in f or "_llm_model_impl" in f for f in files)
    multi_component_hits = sum(
        1
        for f in files
        if any(
            k in f
            for k in (
                "_audio_",
                "_tts",
                "talker",
                "code_predictor",
                "token2wav",
                "_speech_",
                "_acoustic_",
            )
        )
        and f.endswith(".py")
    )

    if has_moe:
        return "moe"
    if multi_component_hits >= 2 and has_llm:
        return "multi-component"
    if has_vision and has_llm:
        return "vlm-ocr"
    return "llm"


def check_required(files: list[str], checks: list[tuple[str, Callable[[list[str]], bool]]]) -> list[str]:
    missing: list[str] = []
    for label, fn in checks:
        if not fn(files):
            missing.append(label)
    return missing


def main() -> int:
    default_xh2 = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Check migration artifacts for one model family")
    parser.add_argument("--xh2modelzoo-root", type=Path, default=default_xh2)
    parser.add_argument("--model", required=True, help="Model directory name under xh_model_zoo/xh_llm/models")
    parser.add_argument(
        "--archetype",
        choices=["auto", "llm", "vlm-ocr", "moe", "multi-component"],
        default="auto",
        help="Artifact template to validate",
    )
    parser.add_argument("--architecture", default=None, help="Architecture string expected in llm_converter.py")
    parser.add_argument(
        "--require-example",
        action="store_true",
        help="Require at least one export script under examples/llm/<model-like-dir>",
    )
    args = parser.parse_args()

    model_dir = args.xh2modelzoo_root / "xh_model_zoo" / "xh_llm" / "models" / args.model
    if not model_dir.exists():
        print(f"[FAIL] model directory not found: {model_dir}")
        return 1

    files = sorted([p.name for p in model_dir.iterdir() if p.is_file()])
    archetype = infer_archetype(files) if args.archetype == "auto" else args.archetype
    print(f"[INFO] model={args.model}, archetype={archetype}")

    common_checks: list[tuple[str, Callable[[list[str]], bool]]] = [
        ("__init__.py", lambda fs: "__init__.py" in fs),
        ("*convert_config.py", lambda fs: has_pattern(fs, r"convert_config\.py$")),
        ("*converter.py or *_convert.py", lambda fs: has_pattern(fs, r"(_converter|_convert)\.py$")),
    ]

    archetype_checks: dict[str, list[tuple[str, Callable[[list[str]], bool]]]] = {
        "llm": [
            ("_model.py or _model_impl.py", lambda fs: has_pattern(fs, r"_model(_impl)?\.py$")),
        ],
        "vlm-ocr": [
            ("_llm_model_impl.py", lambda fs: "_llm_model_impl.py" in fs),
            ("_vision_model_impl.py", lambda fs: "_vision_model_impl.py" in fs),
            ("*_onnx_model.py", lambda fs: has_pattern(fs, r"onnx_model\.py$")),
        ],
        "moe": [
            ("_moe_model.py", lambda fs: "_moe_model.py" in fs),
        ],
        "multi-component": [
            (
                ">=2 component impl/model files",
                lambda fs: sum(1 for f in fs if has_pattern([f], r"_(audio|tts|talker|vision|llm).*model(_impl)?\.py$"))
                >= 2,
            ),
            (
                ">=1 *_convert.py",
                lambda fs: has_pattern(fs, r"_convert\.py$") or has_pattern(fs, r"_converter\.py$"),
            ),
        ],
    }

    missing = check_required(files, common_checks + archetype_checks[archetype])
    warnings: list[str] = []
    if not has_pattern(files, r"_hf_compatible\.py$"):
        warnings.append("missing *_hf_compatible.py (optional but recommended for generation/eval compatibility)")
    if not has_pattern(files, r"^inference\.py$"):
        warnings.append("missing inference.py (optional but recommended for hmonnx smoke tests)")

    # Check registration in llm_converter.py
    if args.architecture:
        converter_file = args.xh2modelzoo_root / "xh_model_zoo" / "xh_llm" / "llm_converter.py"
        text = converter_file.read_text(encoding="utf-8")
        marker = f'architecture == "{args.architecture}"'
        if marker in text:
            print(f"[PASS] architecture registration found: {args.architecture}")
        else:
            missing.append(f'llm_converter.py registration for architecture="{args.architecture}"')

    # Check examples
    if args.require_example:
        examples_root = args.xh2modelzoo_root / "examples" / "llm"
        example_candidates = [
            p for p in examples_root.iterdir() if p.is_dir() and normalize_name(p.name) == normalize_name(args.model)
        ]
        if not example_candidates:
            missing.append("examples/llm/<family> directory")
        else:
            ex_files = sorted([p.name for p in example_candidates[0].iterdir() if p.is_file()])
            if not any("export" in f for f in ex_files):
                missing.append("at least one export script in examples directory")

    if missing:
        print("[FAIL] missing artifacts:")
        for item in missing:
            print(f"  - {item}")
        return 1

    for item in warnings:
        print(f"[WARN] {item}")
    print("[PASS] all required artifacts are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
