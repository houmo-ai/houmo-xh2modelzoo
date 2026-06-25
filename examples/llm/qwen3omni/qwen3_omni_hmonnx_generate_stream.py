# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible entry point for Qwen3-Omni HMONNX streaming generation."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stream import qwen3_omni_hmonnx_generate_stream as _impl
from stream.qwen3_omni_hmonnx_generate_stream import *  # noqa: F401,F403


def build_arg_parser():
    return _impl.build_arg_parser()


def main(args):
    _impl.generate_stream = globals().get("generate_stream", _impl.generate_stream)
    return _impl.main(args)


if __name__ == "__main__":
    main(build_arg_parser().parse_args())