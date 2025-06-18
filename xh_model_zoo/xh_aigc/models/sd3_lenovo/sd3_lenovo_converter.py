from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from typing_extensions import override
from xhquant.api import convert_onnx_to_hmonnx, get_root_logger

from ..sd3 import SD3ConvertConfig, SD3Converter
from .sd3_lenovo_diffusion_pipe import SD3LenovoDiffusion3Pipe


@dataclass
class SD3LenovoConvertConfig(SD3ConvertConfig):
    mmdit_quant: bool = True
    t5_quant: bool = True


class SD3LenovoConverter(SD3Converter):
    def __init__(self, pretrained_model_path: str, lenovo_model_path: str, convert_config: SD3LenovoConvertConfig):
        super().__init__(pretrained_model_path, convert_config)
        self.lenovo_model_path = lenovo_model_path

    def load_model(self, pretrained_model_path, convert_config: SD3LenovoConvertConfig):  # type: ignore[override]
        pipe = SD3LenovoDiffusion3Pipe.from_pretrained(
            pretrained_model_path, self.lenovo_model_path, convert_config.mmdit_quant, convert_config.t5_quant
        )
        return pipe

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path: str, convert_config: SD3LenovoConvertConfig, work_dir: str, **kwargs
    ):  # type: ignore[override]
        assert (
            transformers.__version__ == "4.46.0"
        ), "transformers version must be 4.46.0, please pip install transformers==4.46.0"
        lenovo_model = kwargs.get("lenovo_model", "")
        converter = SD3LenovoConverter(pretrained_model_path, lenovo_model, convert_config)
        converter._convert(work_dir)
        return converter
