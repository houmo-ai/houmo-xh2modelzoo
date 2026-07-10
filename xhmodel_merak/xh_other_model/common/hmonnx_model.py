from typing import Any, List, Optional, Union

import torch
import torch.nn as nn

from xhquant.api import HMONNXGoldenInference as HMONNXInference

from ..builder import MODELS


@MODELS.register_module()
class HMONNXModel(nn.Module):
    def __init__(self, onnx_file: str, save_golden: bool = False, save_golden_dir: Optional[str] = None):
        super().__init__()
        self.save_golden = save_golden
        self.golden_dir = save_golden_dir
        self.session: Optional[HMONNXInference] = None
        self.load_model(onnx_file)

    def _set_device(self, device: torch.device) -> None:
        self.device = device
        if self.session is not None:
            self.session.to(device)

    def to(self, *args, **kwargs) -> nn.Module:
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            def apply_fn(module):
                if not hasattr(module, "_set_device"):
                    return
                module._set_device(device)

            self.apply(apply_fn)
            return super().to(*args, **kwargs)

        if dtype is not None:
            assert False, f"dtype {dtype} is not supported"
        return self

    def load_model(self, model_path: str) -> None:
        self.session = HMONNXInference(model_path)

    def infer(self, *inputs) -> Union[Any, List[Any]]:
        assert self.session is not None
        inputs = [x.to(torch.float16) if x.dtype == torch.float32 else x for x in inputs]
        if self.save_golden:
            assert self.golden_dir is not None
            self.session.save_golden = True
            self.session.save_golden_dir = self.golden_dir
        outs = self.session.forward(*inputs)
        if isinstance(outs, (tuple, list)) and len(outs) == 1:
            outs = outs[0]
        return outs

    def forward(self, *inputs) -> Any:
        return self.infer(*inputs)
