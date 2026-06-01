from ...builder import register_llm_model
from ..qwen3_legacy import XHQwen3LegacyModel, XHQwen3LegacyModelConfig


class XHQwen3EmbeddingModelConfig(XHQwen3LegacyModelConfig):
    pass


@register_llm_model("Qwen3ForCausalLM_embedding")
class XHQwen3EmbeddingModel(XHQwen3LegacyModel):
    pass
