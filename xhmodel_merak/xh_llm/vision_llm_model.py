from .text_llm_model import TextLLMModel, TextLLMModelConfig
from .types import VLLMModelMeta


class VisionLLMModelConfig(TextLLMModelConfig):
    pass


class VisionLLMModel(TextLLMModel):
    META_CLS = VLLMModelMeta

    def __init__(self, config: VisionLLMModelConfig):
        super().__init__(config)
