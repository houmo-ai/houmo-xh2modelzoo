from .gptqmodel_compat import register_laguna_gptqmodel
from .laguna_hf_compatible import build_laguna_hf_compatible_model
from .laguna_hmonnx_inference import XHLagunaHMONNXModel
from .laguna_model import XHLagunaModel, XHLagunaModelConfig
from .moeblock_parser_compat import install_moeblock_shared_initializer_compat


install_moeblock_shared_initializer_compat()


__all__ = [
    "XHLagunaModel",
    "XHLagunaModelConfig",
    "XHLagunaHMONNXModel",
    "build_laguna_hf_compatible_model",
    "register_laguna_gptqmodel",
]
