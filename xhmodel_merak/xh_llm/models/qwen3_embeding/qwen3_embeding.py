from ...builder import register_llm_model
from ..qwen3 import XHQwen3Model, XHQwen3ModelConfig


class XHQwen3EmbeddingModelConfig(XHQwen3ModelConfig):
    pass


@register_llm_model("Qwen3ForCausalLM_embedding")
class XHQwen3EmbeddingModel(XHQwen3Model):
    pass
