from ...hmonnx import TextLLMHMONNXModel
from ...types import LLMModelMeta


class XHQwen3LegacyOptHMONNXModel(TextLLMHMONNXModel):
    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        super().__init__(meta_info, **kwargs)
