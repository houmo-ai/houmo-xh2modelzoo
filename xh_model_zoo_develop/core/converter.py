import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Type, Any, List, Union
from torch import nn
from xhquant.api import QuantScheme, QuantGraph, FrontendGraph
from abc import abstractmethod, ABC

from .converter_config import ConverterConfig


class Converter(ABC):
    _registry: Dict[str, Type["Converter"]] = {}
    config: ConverterConfig

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        model_keys = getattr(cls, "MODEL_KEYS", None)
        if model_keys is not None:
            if isinstance(model_keys, str):
                model_keys = [model_keys]
            for model_key in model_keys:
                if model_key in cls._registry:
                    raise ValueError(f"Model {model_key} already registered")
                cls._registry[model_key] = cls

    @abstractmethod
    def export(self, output_dir: str, generate_golden: bool, *args, **kwargs) -> Any:
        """
        You need to implement this method in the subclass.
        Export the model to hmonnx format and generate golden data.
        Args:
            output_dir (str): The directory to export the model.
            generate_golden (bool): Whether to generate golden data.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            Any: The result of the export.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def convert_and_export(cls, config: "ConverterConfig", *args, **kwargs) -> Any:
        """
        Convert the model to hmonnx format and export the file.
        You need to implement this method in the subclass.
        """
        raise NotImplementedError

    @classmethod
    def auto_export(cls, architecture: str, *args, **kwargs) -> Any:
        """
        Auto convert the model to hmonnx format and export the file.

        Args:
            architecture (str): The architecture of the model.
        """
        return cls._registry[architecture].convert_and_export(*args, **kwargs)

    @classmethod
    def get_all_support_architectures(cls) -> List[str]:
        return list(cls._registry.keys())
