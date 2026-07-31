import torch

from xhquant.utils.registry import _DMRegistryCls

from ...text_llm_hf_compatible import TextLLMHFCompatible


class _LagunaHFCompatible(TextLLMHFCompatible):
    def _setup(self, text_llm_model):
        model = super()._setup(text_llm_model)
        if model is not None:
            for attr in ("model", "lm_head"):
                if hasattr(model, attr):
                    delattr(model, attr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return model

    def set_experts_implementation(self, experts_implementation):
        self.config.experts_implementation = experts_implementation

    def get_correct_experts_implementation(self, experts_implementation):
        return experts_implementation

    def _grouped_mm_can_dispatch(self):
        return False


def build_laguna_hf_compatible_model(hf_model, xh_model):
    compatible_modules = _DMRegistryCls("XHCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in compatible_modules:
        compatible_modules.register_module(
            {hf_model_cls: hf_model_cls.__name__},
            _LagunaHFCompatible,
        )
    return compatible_modules.convert(hf_model, text_llm_model=xh_model)
