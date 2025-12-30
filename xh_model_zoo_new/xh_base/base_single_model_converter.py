import torch, torch.nn as nn
from pathlib import Path
from typing import Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field

from xhquant.api import (
    convert_onnx_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    convert_fx_model_to_quanted_model,
    convert_dynamo_model_to_quanted_model,
    QuantGraph,
    HMONNXGoldenInference,
    QuantScheme,
    get_root_logger,
)
from xh_model_zoo_new.core import Converter, ConverterConfig
from xh_model_zoo_new.utils.time_profiler import TimeProfiler
from functools import cached_property


# TODO 实现 convert_onnx_opset?


@dataclass
class BaseSingleModelConverterConfig(ConverterConfig):
    """Config for single model converter with fixed input/output shapes.

    Using kw_only=True allows inputs (required) to be defined after
    parent class fields with default values.
    """

    inputs: List[torch.Tensor]

    model_name: Optional[str] = None
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None
    quant_scheme: QuantScheme = field(default_factory=QuantScheme)  # 重新定义以覆盖父类字段

    def __post_init__(self):
        if isinstance(self.quant_scheme, dict):
            self.quant_scheme = QuantScheme(**self.quant_scheme)
        if isinstance(self.inputs, torch.Tensor):
            self.inputs = [self.inputs]


class BaseSingleModelConverter(Converter):
    config: BaseSingleModelConverterConfig

    def __init__(self, model: Union[str, nn.Module], config: BaseSingleModelConverterConfig):
        super().__init__()
        assert isinstance(model, (str, nn.Module)), "model must be a ONNX file path or a nn.Module"
        self.logger = get_root_logger()
        self.model = model
        self.config = config

    def prepare_inputs(self) -> List[torch.Tensor]:
        if self.config.inputs is not None:
            return self.config.inputs
        raise ValueError("Not support auto generate inputs now")

    @property
    def input_args(self) -> List[torch.Tensor]:
        return self.prepare_inputs()

    @cached_property
    def quanted_model(self) -> QuantGraph:
        input_args = self.prepare_inputs()
        if isinstance(self.model, str):
            return convert_onnx_to_quanted_model(
                self.model, input_args, self.config.quant_scheme.target_device, self.config.get_quant_cfg()
            )
        elif isinstance(self.model, nn.Module):
            return convert_dynamo_model_to_quanted_model(
                self.model, input_args, self.config.quant_scheme.target_device, self.config.get_quant_cfg()
            )

    def export(self, output_dir: str, generate_golden: bool = False) -> Any:
        input_args = self.input_args
        quanted_model = self.quanted_model

        work_dir = Path(output_dir)
        if isinstance(self.model, str):
            model_name = self.config.model_name or Path(self.model).stem
        else:
            model_name = self.config.model_name or self.model.__class__.__name__
        prefix = f"{model_name}-{self.config.quant_scheme.target_device}"
        with TimeProfiler("convert_quanted_model_to_hmonnx", self.logger) as tp:
            out_hmonnx_file = work_dir / "hmonnx" / f"{prefix}.onnx"
            out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
            convert_quanted_model_to_hmonnx(
                quanted_model, input_args, str(out_hmonnx_file), self.config.input_names, self.config.output_names
            )
        if generate_golden:
            with TimeProfiler("generate_golden", self.logger) as tp:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                session = HMONNXGoldenInference(str(out_hmonnx_file))
                session.to(device)
                session.save_golden = True
                session.golden_dir = work_dir / "golden"
                session.step = 0
                session(*input_args)

        return str(out_hmonnx_file)

    @classmethod
    def convert_and_export(
        cls, onnx_file: Union[str, nn.Module], config: BaseSingleModelConverterConfig, output_dir: str, generate_golden: bool = False
    ) -> Any:
        assert isinstance(onnx_file, (str, nn.Module)), "onnx_file must be a ONNX file path or a nn.Module"
        return cls(onnx_file, config).export(output_dir, generate_golden)
