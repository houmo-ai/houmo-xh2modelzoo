"""VoxCPM2 的 xh2modelzoo 导出与推理子模块。

目录布局(放在 xh_model_zoo/xh_llm/models/voxcpm2/ 下):
    __init__.py                         # 本文件
    _voxcpm2_llm_model_impl.py          # MiniCPM4 wrap 注册(核心)
    _voxcpm2_llm_model.py               # base_lm / residual_lm 的 LLMBaseModel 子类
    _voxcpm2_hmonnx_sessions.py         # 各子图 session 的封装
    _voxcpm2_hmonnx_model.py            # 顶层 TTS pipeline
    utils.py                            # 导出/推理通用工具

    export_lm.py                        # 导出 base_lm / residual_lm(prefill+decode)
    export_locenc.py                    # 导出 LocEnc-Step
    export_locdit.py                    # 导出 LocDiT-Step(含 CFG batch=2)
    export_audiovae_encoder.py          # 导出 AudioVAE Encoder
    export_audiovae_decoder.py          # 导出 AudioVAE Decoder(stream/full 两份)
    demo_infer.py                       # 端到端推理 demo

配置文件(放在 config/llm/):
    voxcpm2_lm_xh2a.py
"""

# wrap class 必须在 LM 包装类之前 import,确保注册表已填充
from . import voxcpm2_llm_model_impl  # noqa: F401
from .voxcpm2_llm_model import XHVoxCPM2BaseLMModel, XHVoxCPM2ResidualLMModel
from .voxcpm3_hmonnx_model import VoxCPM2HMONNXTTSPipeline
from .voxcpm2_hmonnx_sessions import (
    AudioVAEDecoderSession,
    AudioVAEEncoderSession,
    BaseLMDecodeSession,
    BaseLMPrefillSession,
    LocDiTStepSession,
    LocEncStepSession,
    ResidualLMDecodeSession,
    ResidualLMPrefillSession,
)

__all__ = [
    # 导出阶段用的 LLMBaseModel 子类
    "XHVoxCPM2BaseLMModel",
    "XHVoxCPM2ResidualLMModel",
    # 推理阶段用的 pipeline 和 session
    "VoxCPM2HMONNXTTSPipeline",
    "BaseLMPrefillSession",
    "BaseLMDecodeSession",
    "ResidualLMPrefillSession",
    "ResidualLMDecodeSession",
    "LocEncStepSession",
    "LocDiTStepSession",
    "AudioVAEEncoderSession",
    "AudioVAEDecoderSession",
]
