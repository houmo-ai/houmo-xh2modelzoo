from xhquant.utils.registry import DynamicModule, Registry, _DMRegistryCls


class DynamicRegister(DynamicModule):
    @classmethod
    def register(cls: type, hf_cls: type):
        if hf_cls not in XHLLM_TRACEABLE_MODULES:
            XHLLM_TRACEABLE_MODULES.register_module(
                {
                    hf_cls: hf_cls.__name__,
                },
                cls,
            )


XH_LLM_MODELS = Registry("XH_LLM_MODELS")  # 适用于XH2/YueHui
XHLLM_TRACEABLE_MODULES = _DMRegistryCls("XHTrace")
# for torch.compile
ONLY_EVAL_MODULES = _DMRegistryCls("fast_eval_dynamic_modules")
XHLLM_TRACEABLE_MODULES_TORCH_COMPILE = _DMRegistryCls("XHTrace_Torch_Compile")
CUSTOM_MODELS = []
