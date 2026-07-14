from ...builder import register_other_model

from . import voxcpm2_llm_model_impl  # noqa: F401
from .voxcpm2_hmonnx_model import VoxCPM2HMONNXTTSPipeline
from .voxcpm2_hmonnx_sessions import (
    AudioVAEDecoderSession,
    AudioVAEEncoderSession,
    AudioVAEStatefulStreamingDecoderSession,
    BaseLMDecodeSession,
    BaseLMPrefillSession,
    LocDiTStepSession,
    LocEncStepSession,
    ResidualLMDecodeSession,
    ResidualLMPrefillSession,
)
from .voxcpm2_llm_model import XHVoxCPM2BaseLMModel, XHVoxCPM2ResidualLMModel


@register_other_model("XHVoxCPM2Model")
class XHVoxCPM2Model:
    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.voxcpm2.workflow:VoxCPM2Workflow"


__all__ = [
    "XHVoxCPM2Model",
    "XHVoxCPM2BaseLMModel",
    "XHVoxCPM2ResidualLMModel",
    "VoxCPM2HMONNXTTSPipeline",
    "BaseLMPrefillSession",
    "BaseLMDecodeSession",
    "ResidualLMPrefillSession",
    "ResidualLMDecodeSession",
    "LocEncStepSession",
    "LocDiTStepSession",
    "AudioVAEEncoderSession",
    "AudioVAEDecoderSession",
    "AudioVAEStatefulStreamingDecoderSession",
]
