from transformers import AutoConfig

from xhmodel_merak.configuration_utils import HFModelConfig

from ...text_llm_model import TextLLMModelConfig


def _get_vision_defaults(hf_model: str | None) -> dict:
    defaults = {
        "max_size_w": 576,
        "max_size_h": 576,
        "max_size_t": 2,
        "patch_size": 16,
        "temporal_patch_size": 2,
    }
    if hf_model is None:
        return defaults

    try:
        config = AutoConfig.from_pretrained(hf_model)
    except Exception:
        return defaults

    thinker_config = getattr(config, "thinker_config", None)
    vision_config = getattr(thinker_config, "vision_config", None)
    if vision_config is None:
        return defaults

    image_size = getattr(vision_config, "image_size", None)
    if image_size is not None:
        defaults["max_size_w"] = image_size
        defaults["max_size_h"] = image_size
    defaults["patch_size"] = getattr(vision_config, "patch_size", defaults["patch_size"])
    defaults["temporal_patch_size"] = getattr(
        vision_config,
        "temporal_patch_size",
        defaults["temporal_patch_size"],
    )
    defaults["max_size_t"] = max(defaults["temporal_patch_size"], defaults["max_size_t"])
    return defaults


class XHQwen3OmniTextModelConfig(TextLLMModelConfig):
    pass


class XHQwen3OmniTalkerModelConfig(TextLLMModelConfig):
    pass


class XHQwen3OmniTalkerPredictionModelConfig(TextLLMModelConfig):
    pass


class XHQwen3OmniVisualConfig(HFModelConfig):
    def __init__(
        self,
        *,
        max_size_w: int | None = None,
        max_size_h: int | None = None,
        max_size_t: int | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        **kwargs,
    ):
        defaults = _get_vision_defaults(kwargs.get("hf_model"))
        super().__init__(**kwargs)

        self.max_size_w = defaults["max_size_w"] if max_size_w is None else max_size_w
        self.max_size_h = defaults["max_size_h"] if max_size_h is None else max_size_h
        self.max_size_t = defaults["max_size_t"] if max_size_t is None else max_size_t
        self.patch_size = defaults["patch_size"] if patch_size is None else patch_size
        self.temporal_patch_size = (
            defaults["temporal_patch_size"] if temporal_patch_size is None else temporal_patch_size
        )


class XHQwen3OmniAudioConfig(HFModelConfig):
    pass
