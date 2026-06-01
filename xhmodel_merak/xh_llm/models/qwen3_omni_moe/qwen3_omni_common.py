from accelerate import init_empty_weights
from transformers import AutoConfig


try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    no_init_weights = init_empty_weights

from .modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration


def _untie_word_embeddings(model) -> None:
    if getattr(model.config, "tie_word_embeddings", False):
        old_torchscript = getattr(model.config, "torchscript", False)
        model.config.torchscript = True
        model.tie_weights()
        model.config.tie_word_embeddings = False
        model.config.torchscript = old_torchscript


def load_omni_root_model(hf_model_dir: str, dtype, **kwargs):
    if "dtype" not in kwargs:
        kwargs["dtype"] = dtype
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(hf_model_dir, **kwargs)
    _untie_word_embeddings(model)
    return model


def build_empty_omni_root_model(hf_model_dir: str):
    config = AutoConfig.from_pretrained(hf_model_dir)
    with no_init_weights(), init_empty_weights():
        model = Qwen3OmniMoeForConditionalGeneration(config)
        _untie_word_embeddings(model)
    return model
