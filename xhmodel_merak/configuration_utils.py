import copy
import json
from typing import Optional

from addict import Dict as AttrDict

from xhquant.api import QuantScheme


class BaseAttrDict(AttrDict):
    pass


class BaseConfig:
    """
    以下划线开头的属性,不会出现在json中
    """

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)

        for key, value in list(output.items()):
            if key.startswith("_"):
                output.pop(key)
                continue
            if hasattr(value, "to_dict"):
                output[key] = value.to_dict()
        return output

    @classmethod
    def _convert_dict_to_attrdict(cls, obj):
        """递归地将 dict 转换为 AttrDict"""
        if isinstance(obj, dict) and not isinstance(obj, AttrDict):
            return AttrDict({k: cls._convert_dict_to_attrdict(v) for k, v in obj.items()})
        elif isinstance(obj, (list, tuple)):
            return type(obj)(cls._convert_dict_to_attrdict(item) for item in obj)
        return obj

    @classmethod
    def from_dict(cls, config_dict: dict):
        # 递归地将所有嵌套的 dict 转换为 AttrDict
        converted_dict = cls._convert_dict_to_attrdict(config_dict)
        config = cls(**converted_dict)
        return config

    def to_json_string(self, use_diff: bool = True) -> str:
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"


class BaseModelConfig(BaseConfig):
    def __init__(
        self,
        *,
        model_name: str,
        chip_arch: str = "XH2a",
        model_type: str | None = None,
        quant_scheme: dict | QuantScheme | None = None,
        quant_weight: Optional[str] = None,
        enable: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.chip_arch = chip_arch
        self.model_type = model_type
        self._work_dir: str = None
        self.quant_scheme = quant_scheme
        self.enable_auto_offload = False
        self.quant_weight = quant_weight
        self.enable = enable

    @property
    def work_dir(self):
        return self._work_dir

    @work_dir.setter
    def work_dir(self, value):
        self._work_dir = value


class HFModelConfig(BaseModelConfig):
    def __init__(
        self,
        *,
        model_name: str,
        chip_arch: str = "XH2a",
        model_type: str | None = None,
        quant_scheme: dict | QuantScheme | None = None,
        quant_weight: str | None = None,
        hf_model: str | None = None,
        enable_prefill_chunk: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            chip_arch=chip_arch,
            model_type=model_type,
            quant_scheme=quant_scheme,
            quant_weight=quant_weight,
            **kwargs,
        )
        self.hf_model = hf_model
        self.enable_prefill_chunk = enable_prefill_chunk  # prefill时切分输入为更小的chunk，调试用
        if self.enable_prefill_chunk:
            self.use_cache = True
