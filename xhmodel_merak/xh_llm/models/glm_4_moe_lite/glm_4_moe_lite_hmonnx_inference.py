from ...hmonnx import TextLLMHMONNXModel
from ...types import LLMModelMeta
from .cache import Glm4MoeLiteKVCacheMixin


class XHGlm4MoeLiteHMONNXModel(TextLLMHMONNXModel):
    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        super().__init__(meta_info, **kwargs)
        self._kvcache_mixin = Glm4MoeLiteKVCacheMixin(self.kvcache_config)
