# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: utils.py
# Description:
#   Utility helpers for HOUMO AI xh2modelzoo workflows.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from itertools import chain

import torch
import torch.nn as nn

from xhquant.utils.registry import DynamicModule


def get_module_device(module: nn.Module, default: str | torch.device = "cpu") -> torch.device:
    tensor = next(
        chain(module.parameters(recurse=True), module.buffers(recurse=True)),
        None,
    )
    if tensor is None:
        return torch.device(default)
    return tensor.device


def unfold_args(args):
    ## 将输入的List展开
    if not isinstance(args, (list, tuple)):
        return args
    new_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            new_args.extend(arg)
        else:
            new_args.append(arg)
    args = new_args
    return args


def extract_quant_bits(quant_type: str) -> tuple[int | None, int | None]:
    """从 quant_type 字符串中提取 w_bit 和 a_bit。

    Args:
        quant_type: 格式为 "w{w_bit}a{a_bit}..." 的量化类型字符串

    Returns:
        (w_bit, a_bit) 元组，如果解析失败则返回 (None, None)

    Examples:
        >>> extract_quant_bits("w8a8_fp32")
        (8, 8)
        >>> extract_quant_bits("w4a8_int8")
        (4, 8)
        >>> extract_quant_bits("w4a8")
        (4, 8)
    """
    import re

    pattern = r"w(\d+)a(\d+)"
    match = re.search(pattern, quant_type)
    if match:
        w_bit = int(match.group(1))
        a_bit = int(match.group(2))
        return w_bit, a_bit
    return None, None


def format_model_name(cfg):
    model_cfg = cfg["model"]

    # cfg_name = f"{model_name}_{prefill_chunk_length}_{args.context_length // 1024}k-{quant_type}"
    if "model" in cfg:
        chip_arch = cfg["model"].get("chip_arch", None)
    if chip_arch is None:
        chip_arch = cfg.get("chip_arch", None)
    if chip_arch is None:
        raise NotImplementedError(f"chip_arch not in {cfg}")
    chip_arch = chip_arch.lower()

    fileds = []
    if chip_arch.startswith("xh2"):
        fileds.append("xh2")
    elif chip_arch.startswith("yuehui"):
        fileds.append("yh1")
    else:
        raise ValueError(f"Unsupported chip_arch: {chip_arch}")
    model_name = model_cfg["model_name"]
    chip_arch = fileds[-1]
    if model_name.startswith(chip_arch):
        ##yif model_name already starts with chip_arch, do not add prefix
        return
    model_name = model_name.lower()
    fileds.append(model_name)
    quant_prefix = ""
    if "quant_type" in model_cfg["quant_scheme"]:
        quant_type = model_cfg["quant_scheme"]["quant_type"]
        # 从 quant_type 中提取 w_bit 和 a_bit
        w_bit, a_bit = extract_quant_bits(quant_type)
        assert w_bit is not None and a_bit is not None, (
            f"Failed to extract w_bit and a_bit from quant_type: {quant_type}"
        )
        quant_prefix += f"w{w_bit}"
        quant_prefix += f"a{a_bit}"

    if "w_scheme" in model_cfg["quant_scheme"]:
        w_scheme = model_cfg["quant_scheme"]["w_scheme"]
        quant_prefix += f"w{w_scheme['bits']}"
    if "act_scheme" in model_cfg["quant_scheme"]:
        act_scheme = model_cfg["quant_scheme"]["act_scheme"]
        quant_prefix += f"a{act_scheme['bits']}"
    fileds.append(quant_prefix)

    fileds.append(str(model_cfg["prefill_chunk_length"]))
    fileds.append(f"{model_cfg['context_max_length'] // 1024}k")

    model_name = "_".join(fileds)
    cfg["model"]["model_name"] = model_name
    return cfg


def _wrap_no_split_modules(self, device_map: str):
    """
    Get the modules of the model that should not be spit when using device_map. We iterate through the modules to
    get the underlying `_no_split_modules`.

    Args:
        device_map (`str`):
            The device map value. Options are ["auto", "balanced", "balanced_low_0", "sequential"]

    Returns:
        `list[str]`: List of modules that should not be split
    """
    from transformers import PreTrainedModel

    from .register import XHLLM_TRACEABLE_MODULES

    registry = XHLLM_TRACEABLE_MODULES
    _no_split_modules = set()
    modules_to_check = [self]
    while len(modules_to_check) > 0:
        module = modules_to_check.pop(-1)
        if not isinstance(module, DynamicModule):
            continue
        # if the module does not appear in _no_split_modules, we also check the children
        if module.__class__.__name__ not in _no_split_modules:
            if isinstance(module, PreTrainedModel):
                if module._no_split_modules is None:
                    raise ValueError(
                        f"{module.__class__.__name__} does not support `device_map='{device_map}'`. To implement support, the model "
                        "class needs to implement the `_no_split_modules` attribute."
                    )
                else:
                    _wrap_no_split_modules = []
                    for no_split_module in module._no_split_modules:
                        if no_split_module in registry:
                            dm_cls = registry.get(no_split_module)
                            _wrap_no_split_modules.append(dm_cls.__name__)
                    module._no_split_modules = _wrap_no_split_modules
                    _no_split_modules = _no_split_modules | set(module._no_split_modules)
            modules_to_check += list(module.children())
    return list(_no_split_modules)


def hf_auto_offload(hf_model, device_map="auto", max_memory=None):
    from transformers.modeling_utils import _get_device_map

    if hasattr(hf_model, "_xh_auto_offload") and hf_model._xh_auto_offload:
        return
    for _, m in hf_model.named_modules():
        if hasattr(m, "module._hf_hook"):
            return

    max_memory = None
    hf_quantizer = None
    dtype = torch.float16
    keep_in_fp32_regex = None
    _wrap_no_split_modules(hf_model, device_map)
    import inspect

    _get_device_map_signature = inspect.signature(_get_device_map)
    if len(_get_device_map_signature.parameters) <= 4:
        device_map = _get_device_map(hf_model, device_map, max_memory, hf_quantizer)
    else:
        device_map = _get_device_map(hf_model, device_map, max_memory, hf_quantizer, dtype, keep_in_fp32_regex)

    offload_folder = None
    offload_index = None
    offload_buffers = False
    device_map_kwargs = {
        "device_map": device_map,
        "offload_dir": offload_folder,
        "offload_index": offload_index,
        "offload_buffers": offload_buffers,
    }

    device_map_kwargs["skip_keys"] = hf_model._skip_keys_device_placement
    from accelerate import dispatch_model

    dispatch_model(hf_model, **device_map_kwargs)
    hf_model._xh_auto_offload = True


def is_graph_module(model):
    import torch.fx as fx

    return isinstance(model, (fx.GraphModule, fx.Interpreter))


@dataclass
class MemoryInfo:
    VmRSS: float = 0.0
    VmHWM: float = 0.0
    VmSize: float = 0.0
    VmPeak: float = 0.0

    def __repr__(self):
        return f"VmRSS={self.VmRSS:.2f} GB, VmHWM={self.VmHWM:.2f} GB, VmSize={self.VmSize:.2f} GB, VmPeak={self.VmPeak:.2f} GB"


def get_cpu_memory_mb() -> MemoryInfo:
    result = MemoryInfo(VmRSS=0.0, VmHWM=0.0, VmSize=0.0, VmPeak=0.0)

    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "VmPeak:")):
                key, value, unit = line.split()
                key = key.rstrip(":")
                kb = int(value)
                setattr(result, key, kb / 1024 / 1024)

    return result


def get_model_param_buffer_size_gb(model) -> tuple[float, float]:
    param_numel = 0
    param_bytes = 0

    buffer_numel = 0
    buffer_bytes = 0

    for p in model.parameters():
        if p.device.type == "meta":
            continue
        param_numel += p.numel()
        param_bytes += p.numel() * p.element_size()

    for b in model.buffers():
        if b.device.type == "meta":
            continue
        buffer_numel += b.numel()
        buffer_bytes += b.numel() * b.element_size()

    total_numel = param_numel + buffer_numel
    total_bytes = param_bytes + buffer_bytes

    return total_numel / 1000**3, total_bytes / 1024**3
