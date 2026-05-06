# Copyright 2025 HOUMO AI
#
# File: auto_offload.py
# Description:
#   Accelerate-based auto offload helpers for xh2modelzoo develop models.
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

import accelerate
import accelerate.hooks
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


def auto_offload(model: nn.Module, no_split_module_classes=None, device_map=None):
    if device_map is None:
        device_map_kwargs = {"no_split_module_classes": []}
        if no_split_module_classes is not None:
            if not isinstance(no_split_module_classes, (tuple, list)):
                no_split_module_classes = [no_split_module_classes]
            device_map_kwargs["no_split_module_classes"].extend(no_split_module_classes)
        tmp_no_split_module_classes = []
        for name, m in model.named_modules():
            for cls_name in no_split_module_classes:
                if cls_name in [i.__name__ for i in m.__class__.__mro__[:]]:
                    tmp_no_split_module_classes.append(m.__class__.__name__)
        no_split_module_classes = list(set(tmp_no_split_module_classes))
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


def xh_infer_auto_device_map(model: nn.Module, no_split_module_classes=None):
    device_map_kwargs = {"no_split_module_classes": []}
    if no_split_module_classes is not None:
        if not isinstance(no_split_module_classes, (tuple, list)):
            no_split_module_classes = [no_split_module_classes]
        # 保存原始传入的类名
        original_class_names = list(no_split_module_classes)
        device_map_kwargs["no_split_module_classes"].extend(no_split_module_classes)

        # 在模型中查找匹配的类，收集实际类名
        found_class_names = []
        all_module_class_names = set()
        for name, m in model.named_modules():
            module_class_name = m.__class__.__name__
            all_module_class_names.add(module_class_name)
            for cls_name in original_class_names:
                # 检查传入的类名是否在模块的 MRO 中
                mro_names = [x.__name__ for x in m.__class__.__mro__[:]]
                if cls_name in mro_names:
                    found_class_names.append(module_class_name)

        found_class_names = list(set(found_class_names))

        # 如果找到了匹配的类，使用找到的实际类名
        if found_class_names:
            device_map_kwargs["no_split_module_classes"] = found_class_names
        else:
            # 如果没有找到精确匹配，尝试部分匹配（比如 "QMoeBlock" 匹配 "QuantMoeBlock"）
            # 提取关键部分（去掉 "Q" 前缀，保留核心名称如 "MoeBlock"）
            partial_matches = []
            for cls_name in original_class_names:
                # 提取核心名称（去掉开头的 "Q" 或 "Quant" 等前缀）
                core_name = cls_name
                if cls_name.startswith("Q") and len(cls_name) > 1:
                    core_name = cls_name[1:]  # 去掉 "Q" 前缀
                elif cls_name.startswith("Quant"):
                    core_name = cls_name[5:]  # 去掉 "Quant" 前缀

                for module_class_name in all_module_class_names:
                    # 检查模块类名是否包含核心名称（不区分大小写）
                    if core_name.lower() in module_class_name.lower():
                        partial_matches.append(module_class_name)

            if partial_matches:
                device_map_kwargs["no_split_module_classes"] = list(set(partial_matches))
            # 如果还是没有找到，保留原始类名（可能 accelerate 会处理，或者类名本身就是正确的）

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
    return device_map


def remove_offload(module):
    # for name, m in module.named_modules():
    #     if hasattr(m, "_hf_hook"):
    #         hf_hook = m._hf_hook
    #         if isinstance(hf_hook, accelerate.hooks.AlignDevicesHook):
    #             hf_hook.execution_device = "cpu"
    #             hf_hook.pre_forward(m, None, None)

    accelerate.hooks.remove_hook_from_module(module, recurse=True)
    if hasattr(module, "hf_device_map"):
        delattr(module, "hf_device_map")
    # module.forward = nn.Module.forward
