"""
CosyVoice3 LLM 模块加载器
==========================

通过 hyperpyyaml 加载 CosyVoice3LM (Qwen2-0.5B)，提供 QAT 训练所需的
模型加载和模块元信息。LLM 的 forward 已返回 {"loss": ...}，无需额外 wrapper。

模块加载器模式（适配新模型时参考）:
    1. 每个模块一个 qat_module_{name}.py
    2. 必须导出: load_{name}_model(yaml_path, hf_model_dir) -> nn.Module
    3. 可选导出: MODULE_NAME, DEFAULT_LR, DEFAULT_GRAD_ACCUM, CKPT_FILE
    4. 加载时通过 overrides 禁用其他模块 (flow=None, hift=None)，
       让 hyperpyyaml 仅构建目标模块，节省内存和初始化时间
"""

import os
import sys

# ================================================================
#  模块元信息
# ================================================================

MODULE_NAME = "llm"
DEFAULT_LR = 1e-5
DEFAULT_GRAD_ACCUM = 2
CKPT_FILE = "llm.pt"

# LLM 的 forward(batch, device) 已返回 {"loss": ...}
# 不需要额外 wrapper


# ================================================================
#  模型加载
# ================================================================

def load_llm_model(yaml_path: str, hf_model_dir: str):
    """通过 hyperpyyaml 加载 CosyVoice3LM。

    Parameters
    ----------
    yaml_path : cosyvoice3.yaml 路径
    hf_model_dir : CosyVoice-BlankEN 路径（Qwen2 HF 模型目录）

    Returns
    -------
    nn.Module — CosyVoice3LM 实例
    """
    from hyperpyyaml import load_hyperpyyaml

    overrides = {
        "qwen_pretrain_path": hf_model_dir,
        "flow": None,
        "hift": None,
        "hifigan": None,
    }
    with open(yaml_path) as f:
        configs = load_hyperpyyaml(f, overrides=overrides)
    return configs["llm"]
