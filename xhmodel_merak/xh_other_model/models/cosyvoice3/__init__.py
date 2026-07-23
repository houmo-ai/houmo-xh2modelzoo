# Copyright 2025 HOUMO AI
#
# File: __init__.py
# Description:
#   CosyVoice3 model package exports for xh2modelzoo (merak).
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

# Importing ``_model`` registers the Qwen2 traceable modules into
# ``XHLLM_TRACEABLE_MODULES`` as a side effect.
# Import ``qwen_llm_model`` eagerly to fire ``@register_other_model("XHCosyVoice3LLM")``.
# Without this, ``AutoWorkflow.from_config()`` cannot discover the model type via
# ``is_model_type_supported()`` because lazy ``__getattr__`` only responds to
# explicit attribute access, not to ``importlib.import_module()`` driven by the
# framework's AST-based model-type scan.
from . import (
    _model,  # noqa: F401
    qwen_llm_model,  # noqa: F401
)


def __getattr__(name: str):
    """Lazy import to avoid loading heavy deps (torch, transformers, onnx)
    until actually needed."""
    _lazy_imports = {
        "CosyVoice3HMONNXInference": ".cosyvoice3_inference",
        "XHQwen2HMONNXModel": ".llm_hmonnx_model",
        "Qwen2_HFCompatible": ".qwen2_hf_compatible",
        "create_llm_wraped_cls": ".qwen2_hf_compatible",
        "get_empty_hf_model": ".qwen2_hf_compatible",
        "XHQwen2LegacyModel": ".qwen_llm_model",
    }
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "XHQwen2LegacyModel",
    "Qwen2_HFCompatible",
    "create_llm_wraped_cls",
    "get_empty_hf_model",
    "XHQwen2HMONNXModel",
    "CosyVoice3HMONNXInference",
]
