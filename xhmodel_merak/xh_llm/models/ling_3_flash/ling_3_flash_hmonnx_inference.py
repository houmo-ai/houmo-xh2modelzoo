from __future__ import annotations

from ..qwen3_next.qwen3_next_hmonnx_inference import XHQwen3NextHMONNXModel


class XHLing3FlashHMONNXModel(XHQwen3NextHMONNXModel):
    """Ling runtime using the common asymmetric hybrid-cache HMONNX ABI."""


__all__ = ["XHLing3FlashHMONNXModel"]
