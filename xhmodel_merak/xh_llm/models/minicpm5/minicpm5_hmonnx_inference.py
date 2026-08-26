"""HMONNX runtime adapter for MiniCPM5."""

from ...hmonnx import TextLLMHMONNXModel


class XHMiniCPM5HMONNXModel(TextLLMHMONNXModel):
    """Use the standard text-LLM HMONNX ABI for MiniCPM5."""


__all__ = ["XHMiniCPM5HMONNXModel"]
