#!/usr/bin/env python3
"""Discover overlap models/examples between xhquant_llm and xh2modelzoo.

Usage:
  python scripts/discover_shared_models.py
  python scripts/discover_shared_models.py --write references/shared-model-matrix.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable


def normalize_name(name: str) -> str:
    return re.sub(r"[-.]", "_", name)


def list_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir() and not p.name.startswith("__")], key=lambda p: p.name)


def list_files(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_file()])


def pick(files: Iterable[str], patterns: Iterable[str], max_items: int = 6) -> list[str]:
    matches: list[str] = []
    for pat in patterns:
        rgx = re.compile(pat)
        for name in files:
            if rgx.search(name) and name not in matches:
                matches.append(name)
    return matches[:max_items]


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


def render_markdown(
    xh2_root: Path,
    xhq_root: Path,
    model_rows: list[dict[str, str]],
    only_xhq: list[str],
    only_xh2: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Shared Model Matrix (xhquant_llm -> xh2modelzoo)")
    lines.append("")
    lines.append(f"- xh2modelzoo: `{xh2_root}`")
    lines.append(f"- xhquant_llm: `{xhq_root}`")
    lines.append("")
    lines.append(f"## Overlap Models ({len(model_rows)})")
    lines.append("")
    lines.append("| model | archetype | xhquant_llm key files | xh2modelzoo key files | src example | dst example |")
    lines.append("|---|---|---|---|---|---|")
    for row in model_rows:
        lines.append(
            "| {model} | {archetype} | {src_files} | {dst_files} | {src_example} | {dst_example} |".format(**row)
        )

    lines.append("")
    lines.append(f"## Only In xhquant_llm ({len(only_xhq)})")
    lines.append("")
    lines.append(", ".join(only_xhq) if only_xhq else "(none)")

    lines.append("")
    lines.append(f"## Only In xh2modelzoo ({len(only_xh2)})")
    lines.append("")
    lines.append(", ".join(only_xh2) if only_xh2 else "(none)")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    default_xh2 = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Build overlap matrix for xhquant_llm and xh2modelzoo")
    parser.add_argument("--xh2modelzoo-root", type=Path, default=default_xh2)
    parser.add_argument("--xhquant-llm-root", type=Path, default=Path("/home/jiangyong.yu/xh2_work/xhquant_llm"))
    parser.add_argument("--write", type=Path, default=None, help="Write markdown output to a file")
    args = parser.parse_args()

    xh2_models = args.xh2modelzoo_root / "xh_model_zoo" / "xh_llm" / "models"
    xhq_models = args.xhquant_llm_root / "xhquant_llm" / "models"
    xh2_examples = args.xh2modelzoo_root / "examples" / "llm"
    xhq_examples = args.xhquant_llm_root / "examples"

    xh2_model_dirs = {p.name: p for p in list_dirs(xh2_models)}
    xhq_model_dirs = {p.name: p for p in list_dirs(xhq_models)}
    shared_models = sorted(set(xh2_model_dirs) & set(xhq_model_dirs))
    only_xhq = sorted(set(xhq_model_dirs) - set(xh2_model_dirs))
    only_xh2 = sorted(set(xh2_model_dirs) - set(xhq_model_dirs))

    xh2_example_map = {normalize_name(p.name): p.name for p in list_dirs(xh2_examples)}
    xhq_example_map = {normalize_name(p.name): p.name for p in list_dirs(xhq_examples)}

    model_rows: list[dict[str, str]] = []
    for model in shared_models:
        src_files = list_files(xhq_model_dirs[model])
        dst_files = list_files(xh2_model_dirs[model])
        archetype = infer_archetype(src_files + dst_files)

        src_key = pick(
            src_files,
            [
                r"_model\.py$",
                r"_model_impl\.py$",
                r"_moe_model\.py$",
                r"_llm_model\.py$",
                r"_vision_model\.py$",
                r"_hf_compatible\.py$",
                r"onnx_model\.py$",
            ],
        )
        dst_key = pick(
            dst_files,
            [
                r"_model\.py$",
                r"_model_impl\.py$",
                r"_moe_model\.py$",
                r"_llm_model\.py$",
                r"_vision_model\.py$",
                r"_hf_compatible\.py$",
                r"convert_config\.py$",
                r"converter\.py$",
                r"onnx_model\.py$",
            ],
        )

        nkey = normalize_name(model)
        src_example = xhq_example_map.get(nkey, "-")
        dst_example = xh2_example_map.get(nkey, "-")

        model_rows.append(
            {
                "model": model,
                "archetype": archetype,
                "src_files": "<br>".join(src_key) if src_key else "-",
                "dst_files": "<br>".join(dst_key) if dst_key else "-",
                "src_example": src_example,
                "dst_example": dst_example,
            }
        )

    markdown = render_markdown(args.xh2modelzoo_root, args.xhquant_llm_root, model_rows, only_xhq, only_xh2)

    if args.write is not None:
        out = args.write if args.write.is_absolute() else (Path.cwd() / args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        print(f"Wrote matrix to {out}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
