#!/usr/bin/env python3
"""Run the official LeRobot LIBERO evaluator with PI0.5 HMONNX software graphs."""

from __future__ import annotations

import argparse
import sys
from functools import wraps
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from examples_merak.vla.pi05.pi05_droid_hmonnx_validation import (
    HMONNX_RUNTIME_CHOICES,
    CompactHMONNX,
)
from examples_merak.vla.pi05.pi05_libero_fp_eval import _merge_local_tokenizer_override


def _parse_hmonnx_wrapper_args(
    argv: list[str],
) -> tuple[Path | None, Path | None, str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument(
        "--hmonnx-runtime",
        choices=HMONNX_RUNTIME_CHOICES,
        default="v2-cuda-graph",
    )
    args, lerobot_args = parser.parse_known_args(argv)
    return args.tokenizer_dir, args.export_dir, args.hmonnx_runtime, lerobot_args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    tokenizer_dir, export_dir, runtime_mode, lerobot_args = _parse_hmonnx_wrapper_args(argv)

    from lerobot.scripts import lerobot_eval

    if tokenizer_dir is None or export_dir is None:
        if any(arg in {"-h", "--help"} for arg in lerobot_args):
            original_argv = sys.argv
            try:
                sys.argv = [original_argv[0], *lerobot_args]
                lerobot_eval.main()
            finally:
                sys.argv = original_argv
            return 0
        raise SystemExit("--tokenizer-dir and --export-dir are required")

    tokenizer_dir = tokenizer_dir.expanduser().resolve()
    export_dir = export_dir.expanduser().resolve()
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(tokenizer_dir)
    if not (export_dir / "export_meta_info.json").is_file():
        raise FileNotFoundError(export_dir / "export_meta_info.json")

    original_make_policy = lerobot_eval.make_policy
    original_make_processors = lerobot_eval.make_pre_post_processors

    @wraps(original_make_policy)
    def make_hmonnx_policy(*args: Any, **kwargs: Any):
        policy = original_make_policy(*args, **kwargs)
        device = next(policy.parameters()).device
        runtime = CompactHMONNX(policy, export_dir, device, runtime_mode=runtime_mode)

        @torch.no_grad()
        def predict_action_chunk_with_hmonnx(self, batch, **predict_kwargs):
            noise = predict_kwargs.get("noise")
            if noise is None:
                tokens = batch["observation.language.tokens"]
                noise = self.model.sample_noise(
                    (
                        int(tokens.shape[0]),
                        int(self.config.chunk_size),
                        int(self.config.max_action_dim),
                    ),
                    tokens.device,
                )
            return runtime.predict(batch, noise)[0]

        policy.predict_action_chunk = MethodType(predict_action_chunk_with_hmonnx, policy)
        return policy

    @wraps(original_make_processors)
    def make_processors_with_local_tokenizer(*args: Any, **kwargs: Any):
        return original_make_processors(
            *args,
            **_merge_local_tokenizer_override(kwargs, tokenizer_dir),
        )

    original_argv = sys.argv
    try:
        lerobot_eval.make_policy = make_hmonnx_policy
        lerobot_eval.make_pre_post_processors = make_processors_with_local_tokenizer
        sys.argv = [original_argv[0], *lerobot_args]
        lerobot_eval.main()
    finally:
        lerobot_eval.make_policy = original_make_policy
        lerobot_eval.make_pre_post_processors = original_make_processors
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
