#!/usr/bin/env python3
"""Run the official LeRobot LIBERO evaluator with a local PaliGemma tokenizer."""

from __future__ import annotations

import argparse
import sys
from functools import wraps
from pathlib import Path
from typing import Any


def _merge_local_tokenizer_override(kwargs: dict[str, Any], tokenizer_dir: Path) -> dict[str, Any]:
    merged = dict(kwargs)
    preprocessor_overrides = dict(merged.get("preprocessor_overrides") or {})
    tokenizer_overrides = dict(preprocessor_overrides.get("tokenizer_processor") or {})
    tokenizer_overrides["tokenizer_name"] = str(tokenizer_dir)
    preprocessor_overrides["tokenizer_processor"] = tokenizer_overrides
    merged["preprocessor_overrides"] = preprocessor_overrides
    return merged


def _parse_wrapper_args(argv: list[str]) -> tuple[Path | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tokenizer-dir", type=Path)
    args, lerobot_args = parser.parse_known_args(argv)
    return args.tokenizer_dir, lerobot_args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    tokenizer_dir, lerobot_args = _parse_wrapper_args(argv)

    from lerobot.scripts import lerobot_eval

    if tokenizer_dir is None:
        if any(arg in {"-h", "--help"} for arg in lerobot_args):
            original_argv = sys.argv
            try:
                sys.argv = [original_argv[0], *lerobot_args]
                lerobot_eval.main()
            finally:
                sys.argv = original_argv
            return 0
        raise SystemExit("--tokenizer-dir is required")

    tokenizer_dir = tokenizer_dir.expanduser().resolve()
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(tokenizer_dir)
    if not any((tokenizer_dir / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise FileNotFoundError(f"No tokenizer.json or tokenizer.model in {tokenizer_dir}")

    original_make_processors = lerobot_eval.make_pre_post_processors

    @wraps(original_make_processors)
    def make_processors_with_local_tokenizer(*args: Any, **kwargs: Any):
        return original_make_processors(
            *args,
            **_merge_local_tokenizer_override(kwargs, tokenizer_dir),
        )

    original_argv = sys.argv
    try:
        lerobot_eval.make_pre_post_processors = make_processors_with_local_tokenizer
        sys.argv = [original_argv[0], *lerobot_args]
        lerobot_eval.main()
    finally:
        lerobot_eval.make_pre_post_processors = original_make_processors
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
