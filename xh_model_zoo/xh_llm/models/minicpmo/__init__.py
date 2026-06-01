# Copyright 2025 HOUMO AI
#
# File: __init__.py
# Description:
#   Minicpmo module initialization.
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

from .minicpmo_audio_model import XHMiniCPMOAudioModel
from .minicpmo_hf_compatible import MiniCPMO_HFCompatible
from .minicpmo_llm_model import XHMiniCPMOLLMModel
from .minicpmo_tts_model import XHMiniCPMOTTSModel
from .minicpmo_tts_dvae_model import XHMiniCPMOTTSDVAEModel
from .minicpmo_tts_vocos_model import XHMiniCPMOTTSVOCOSModel
from .minicpmo_vision_model import XHMiniCPMOVisionModel

__all__ = [
    "XHMiniCPMOVisionModel",
    "MiniCPMO_HFCompatible",
    "XHMiniCPMOLLMModel",
    "XHMiniCPMOAudioModel",
    "XHMiniCPMOTTSModel",
    "XHMiniCPMOTTSDVAEModel",
    "XHMiniCPMOTTSVOCOSModel",
]
