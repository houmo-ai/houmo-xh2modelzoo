import copy
import re
from typing import Any, Dict, List

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from xhquant.api import DeviceType, get_root_logger

def update_cfg_after_quanted(model,yaml_dict):
    # rmsnorm update compute mode
    rmsnorm_update_cfg = yaml_dict['rmsnorm_update_cfg']
    compute_mode = rmsnorm_update_cfg['compute_mode']
    for name,module in model.named_modules():
        if "norm" in name and hasattr(module,'norm'):
            if name in rmsnorm_update_cfg['customize_pre_sub_exp']:
                pre_sub_exp = rmsnorm_update_cfg['customize_pre_sub_exp'][name]
            else:
                pre_sub_exp = 0
            module.norm.update_compute_mode(compute_mode, pre_sub_exp)
    return model

class BaseConverter:
    target_device: DeviceType

    def __init__(self):
        pass

    @staticmethod
    def xh1_hmonnx_compatible(input_names: List[str]):
        input_names = copy.deepcopy(input_names)
        input_names_mapping = {
            "inputs_embeds": "input_1",
            "past_seq_length": "valid_length",
            "current_input_length": "current_length",
        }
        for idx in range(len(input_names)):
            in_name = input_names[idx]
            if in_name in input_names_mapping:
                input_names[idx] = input_names_mapping[in_name]
            else:
                # 匹配past_key_cache_后面跟数字的字符串
                kcache_pattern = r"^past_key_cache_\d+$"  # \d+表示匹配一个或多个数字
                kcache_match = re.match(kcache_pattern, in_name)
                if kcache_match:
                    kcache_idx = kcache_match.group(0).split("_")[-1]
                    input_names[idx] = "model_layers_{}_self_attn_kcache_input".format(kcache_idx)
                else:
                    vcache_pattern = r"^past_value_cache_\d+$"  # \d+表示匹配一个或多个数字
                    vcache_match = re.match(vcache_pattern, in_name)
                    if vcache_match:
                        vcache_idx = vcache_match.group(0).split("_")[-1]
                        input_names[idx] = "model_layers_{}_self_attn_vcache_input".format(vcache_idx)
        return input_names


class HFTransfromersConverter(BaseConverter):
    def __init__(self):
        super().__init__()

    def load_hf_model(self, hf_model_dir: str, **kwargs) -> Any:
        raise NotImplementedError()

    def load_quant_weight(self, quant_weight_path: str, native_hf_model: nn.Module) -> bool:
        logger = get_root_logger()
        archive_file = quant_weight_path
        logger.info(f"Load previously saved checkpoint from: {archive_file}")
        is_safetensors = archive_file.endswith(".safetensors")
        state_dict: Dict[str, Tensor]
        if is_safetensors:
            state_dict = load_safetensors_file(archive_file, device="cpu")
        else:
            state_dict = torch.load(archive_file, weights_only=True, map_location="cpu")

        model_state_dict = native_hf_model.state_dict()
        unexpect_state_dict = []
        for k, v in state_dict.items():
            if k not in model_state_dict:
                unexpect_state_dict.append(k)

        for k in unexpect_state_dict:
            paths = k.split(".")
            if paths[-1] == "quant_weight":
                submodule_name = ".".join(paths[:-1])
                submodule = native_hf_model.get_submodule(submodule_name)
                # submodule = get_submodule(native_model, k)
                v = state_dict[k]
                if v.min().item() >= -pow(2, 7) and v.max().item() <= pow(2, 7) - 1:
                    v = v.to(torch.int8)
                elif v.min().item() >= -pow(2, 15) and v.max().item() <= pow(2, 15) - 1:
                    v = v.to(torch.int16)
                else:
                    v = v.to(torch.float32)
                submodule.register_buffer("quant_weight", v, persistent=False)
                logger.debug(f"add quant_weight to {submodule_name}")
            else:
                logger.warning(f"ignore unexpect state dict: {k}")
            state_dict.pop(k)

        native_hf_model.load_state_dict(state_dict)
        del state_dict
        return True
