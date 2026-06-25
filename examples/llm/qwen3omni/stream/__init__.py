# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni HMONNX end-to-end streaming pipeline.

This package implements a vLLM-Omni-style async-chunk stage pipeline for
Qwen3-Omni using HMONNX inference.  Three independent stages run concurrently:

* **Stage 0 – Thinker**: HMONNX text prefill/decode, emits per-step token
  embeddings and hidden states.
* **Stage 1 – Talker**: HMONNX talker + code-predictor, consumes Thinker
  output and emits residual audio codec codes.
* **Stage 2 – Code2Wav**: HMONNX code-to-waveform, decodes codec chunks
  into audio waveforms.

Connectors between stages handle data packaging and chunking, mirroring
vLLM-Omni's ``thinker2talker_async_chunk`` and ``talker2code2wav_async_chunk``.
"""
