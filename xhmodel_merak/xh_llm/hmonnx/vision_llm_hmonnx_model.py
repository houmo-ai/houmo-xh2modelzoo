from .base_llm_hmonnx_model import BaseLLMHMONNXModel


class VisonLLMHMONNXModel(BaseLLMHMONNXModel):
    def __init__(self, config):
        super().__init__(config)
