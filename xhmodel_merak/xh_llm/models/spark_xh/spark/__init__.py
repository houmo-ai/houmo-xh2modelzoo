from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, Qwen2Tokenizer

from .configuration_ipt import IPTConfig
from .modeling_ipt import IPTForCausalLM, IPTModel


# 注册配置类
AutoConfig.register("ipt", IPTConfig)  # support ipt model_type for xinghuo moe mla

# 注册模型类
AutoModel.register(IPTConfig, IPTModel)
AutoModelForCausalLM.register(IPTConfig, IPTForCausalLM)

AutoTokenizer.register(IPTConfig, Qwen2Tokenizer)  # 注册tokenizer以支持自动加载
# ################################################################
# # VLLM Register


# def adaptor_is_deepseek_mla(self) -> bool:
#     if not hasattr(self.hf_text_config, "model_type"):
#         return False
#     elif self.hf_text_config.model_type in ("deepseek_v2", "deepseek_v3", "deepseek_mtp"):
#         return self.hf_text_config.kv_lora_rank is not None
#     elif self.hf_text_config.model_type == "ipt":
#         return self.hf_text_config.apply_mla
#     elif self.hf_text_config.model_type == "eagle":
#         # if the model is an EAGLE module, check for the
#         # underlying architecture
#         return (
#             self.hf_text_config.model.model_type in ("deepseek_v2", "deepseek_v3")
#             and self.hf_text_config.kv_lora_rank is not None
#         )
#     return False


# from vllm import ModelRegistry

# from .ipt_vllm import IPTForCausalLM


# if "IPTForCausalLM" not in ModelRegistry.get_supported_archs():
#     ModelRegistry.register_model("IPTForCausalLM", IPTForCausalLM)
# from vllm.config import ModelConfig


# ModelConfig.is_deepseek_mla = property(adaptor_is_deepseek_mla)
