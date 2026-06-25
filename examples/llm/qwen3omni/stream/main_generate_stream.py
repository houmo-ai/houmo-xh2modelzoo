# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for Qwen3-Omni streaming generation.

The first version of this script built an independent Thinker -> Talker ->
Code2Wav pipeline. That path could run to completion while mis-shaping
residual codec tokens, producing garbled speech. Keep this filename for
existing commands, but route execution through the conservative
``model.generate()``-driven streaming path that observes official Talker codec
frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen3_omni_hmonnx_generate_stream import build_arg_parser as _build_stream_arg_parser  # noqa: E402
from qwen3_omni_hmonnx_generate_stream import main as _stream_main  # noqa: E402


def build_arg_parser():
    parser = _build_stream_arg_parser()
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        choices=["text", "vision", "audio", "multimodal"],
        help="backward-compatible alias for --cases with a single case",
    )
    return parser


def main(args):
    if getattr(args, "case", None) and not getattr(args, "cases", None):
        args.cases = args.case
    return _stream_main(args)


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
