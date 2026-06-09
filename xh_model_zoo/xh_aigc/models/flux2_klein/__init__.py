# Copyright 2025 HOUMO AI
#
# File: __init__.py
# Description:
#   Flux2 klein module initialization.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from .flux2_klein_converter import Flux2KleinConvertConfig, Flux2KleinConverter
from .components_hmonnx import (
    Flux2KleinTransformerInference,
    Flux2KleinVAEEncoderInference,
    Flux2KleinVAEInference,
    attach_hmonnx_transformer,
)
from .text_encoder_hmonnx import Flux2KleinTextEncoderInference, attach_hmonnx_text_encoder
from .pipeline_hmonnx import (
    Flux2KleinHMONNXPipeline,
    attach_flux2_klein_hmonnx_components,
    attach_hmonnx_vae,
    build_hmonnx_pipeline_cls,
    build_hmonnx_vae_pipeline_cls,
    ensure_hmonnx_pipeline,
)

__all__ = [
    "Flux2KleinConvertConfig",
    "Flux2KleinConverter",
    "Flux2KleinTransformerInference",
    "Flux2KleinVAEEncoderInference",
    "Flux2KleinVAEInference",
    "Flux2KleinTextEncoderInference",
    "Flux2KleinHMONNXPipeline",
    "attach_hmonnx_transformer",
    "attach_hmonnx_text_encoder",
    "attach_hmonnx_vae",
    "attach_flux2_klein_hmonnx_components",
    "build_hmonnx_pipeline_cls",
    "build_hmonnx_vae_pipeline_cls",
    "ensure_hmonnx_pipeline",
]