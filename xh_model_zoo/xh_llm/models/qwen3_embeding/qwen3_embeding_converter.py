from transformers import AutoConfig, AutoModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from ..qwen3_legacy import Qwen3LegacyConverterXH2a


class Qwen3EmbeddingConverterXH2a(Qwen3LegacyConverterXH2a):
    def load_hf_model(self, hf_model_dir: str, **kwargs):
        config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
        assert not hasattr(config, "quantization_config")
        native_model = AutoModel.from_pretrained(hf_model_dir, **kwargs)
        assert not hasattr(native_model, "hf_quantizer")
        assert isinstance(native_model, Qwen3Model), f"The model is not Qwen3Model, but {type(native_model)}"
        native_model: Qwen3Model = native_model  # type: ignore

        if native_model.config.tie_word_embeddings:  # type: ignore
            old_torchscript = native_model.config.torchscript  # type: ignore
            native_model.config.torchscript = True  # type: ignore
            native_model.tie_weights()  # type: ignore
            native_model.config.tie_word_embeddings = False  # type: ignore
            native_model.config.torchscript = old_torchscript  # type: ignore

        self.hf_model_path = hf_model_dir
        return native_model
