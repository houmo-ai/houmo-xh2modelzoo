"""
Common utilities for PaddleOCR-VL examples.

Provides wrapper functions that bridge between xhquant / xh_model_zoo APIs,
replacing the former xhquant_llm.api dependency.
"""

from pathlib import Path
from typing import List, Optional

import torch
from safetensors.torch import load_file as load_safetensors_file


def decode_next_token(tokenizer, logits: torch.Tensor):
    """Decode next token from model logits. Replacement for xhquant_llm.api.decode_next_token."""
    next_token_id = torch.argmax(logits, dim=-1)
    next_token_str = tokenizer.batch_decode(next_token_id, skip_special_tokens=True)
    return next_token_id, next_token_str


def xhquant_llm_init(log_file=None, debug=False, file_mode="w"):
    """Initialize xhquant and xh_model_zoo logger. Replacement for xhquant_llm.api.xhquant_llm_init."""
    from xhquant.api import xhquant_init
    from xh_model_zoo.utils.logger import xh2modelzoo_init_logger

    xhquant_log_file = None
    if log_file is not None:
        log_fname = Path(log_file).stem
        log_suffix = Path(log_file).suffix
        xhquant_log_name = f"{log_fname}_xhquant{log_suffix}"
        xhquant_log_file = str(Path(log_file).with_name(xhquant_log_name))

    xhquant_init(xhquant_log_file, debug=debug)
    xh2modelzoo_init_logger(log_file, "DEBUG" if debug else "INFO", "xhquant_llm", file_mode=file_mode)


def get_root_logger():
    """Get the root logger. Replacement for xhquant_llm.api.get_root_logger."""
    from xh_model_zoo.utils.logger import get_root_logger as _get_root_logger
    return _get_root_logger()


def load_quarot_gptq_state_dict(native_model, state_dict_path: str, strict: bool = False, logger=None) -> None:
    """Load QuaRot/GPTQ state dict into model."""
    from xh_model_zoo.xh_llm.quarot.quantizer_utils import rotation_utils

    rotation_utils.fuse_layer_norms(native_model)
    state_dict = load_safetensors_file(state_dict_path)

    model_state_dict = native_model.state_dict()
    unexpected_keys = [k for k in state_dict if k not in model_state_dict]
    for key in unexpected_keys:
        value = state_dict.pop(key)
        paths = key.split(".")
        if paths[-1] != "quant_weight":
            if logger is not None:
                logger.warning(f"ignore unexpected state dict: {key}")
            continue

        submodule_name = ".".join(paths[:-1])
        submodule = native_model.get_submodule(submodule_name)
        if value.min().item() >= -(2**7) and value.max().item() <= (2**7) - 1:
            value = value.to(torch.int8)
        elif value.min().item() >= -(2**15) and value.max().item() <= (2**15) - 1:
            value = value.to(torch.int16)
        else:
            value = value.to(torch.float32)
        submodule.register_buffer("quant_weight", value, persistent=False)
        if logger is not None:
            logger.debug(f"add quant_weight to {submodule_name}")

    native_model.load_state_dict(state_dict, strict=strict)
