import torch
import torch.nn as nn
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from accelerate.hooks import add_hook_to_module
from accelerate.utils import (
    check_tied_parameters_on_same_device,
    extract_model_from_parallel,
    find_tied_parameters,
    get_balanced_memory,
    get_max_memory,
    load_offloaded_weights,
    offload_weight,
    save_offload_index,
    set_module_tensor_to_device,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def decode_next_token(tokenizer: PreTrainedTokenizerBase, logits: torch.Tensor):
    # logits: (batch_size, 1, vocab_size)
    next_token_id = torch.argmax(logits, dim=-1)
    next_token_str = tokenizer.batch_decode(next_token_id, skip_special_tokens=True)
    return next_token_id, next_token_str


def auto_offload(model: nn.Module, no_split_module_classes=None, device_map="auto"):
    device_map_kwargs = {"no_split_module_classes": []}
    if no_split_module_classes is not None:
        if not isinstance(no_split_module_classes, (tuple, list)):
            no_split_module_classes = [no_split_module_classes]
        device_map_kwargs["no_split_module_classes"].extend(no_split_module_classes)
    # device_map = "balanced_low_0"
    max_memory = None
    target_dtype = torch.float16
    max_memory = get_balanced_memory(
        model,
        dtype=target_dtype,
        # low_zero=(device_map == "balanced_low_0"),
        low_zero=False,
        max_memory=max_memory,
        **device_map_kwargs,
    )
    device_map_kwargs["max_memory"] = max_memory
    device_map = infer_auto_device_map(model, dtype=target_dtype, **device_map_kwargs)
    dispatch_model(model, device_map=device_map)
