from typing import Callable, Any
from xh_model_zoo_new.core import InferAdapter
from xhquant.api import HMONNXInference, QuantGraph
import torch.nn as nn


class BaseSingleModelInfer(InferAdapter, nn.Module):
    def __init__(self, model: Callable):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    @classmethod
    def from_hmonnx(cls, onnx_path: str) -> "BaseSingleModelInfer":
        model = HMONNXInference(onnx_path)
        infer = cls(model)
        return infer

    @classmethod
    def from_qmodel(cls, model: QuantGraph) -> "BaseSingleModelInfer":
        infer = cls(model)
        return infer
