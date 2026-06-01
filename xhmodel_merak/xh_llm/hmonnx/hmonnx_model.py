from pathlib import Path
from typing import Any

import torch

from xhquant.api import HMONNXGraphGoldenInference as HMONNXGoldenInference

from ..device_mixin import DeviceMixin
from ..infer_mixin import GoldenMixin
from ..utils import unfold_args


class HMONNXGolden(HMONNXGoldenInference):
    def __init__(self, hmonnx_path):
        super().__init__(hmonnx_path)
        self.golden_dir = str(Path(hmonnx_path).parent)
        self.legacy_mode = False


class HMONNXModel(GoldenMixin, DeviceMixin):
    def __init__(self, hmonnx):
        self.hmonnx_session = HMONNXGolden(hmonnx)
        self._enable_auto_offload = False
        self._dtype = torch.float16
        self._device = torch.device("cpu")
        self._enable_golden = False

    def _set_device(self, device):
        self._device = device
        self.hmonnx_session.to(device)
        return self

    def _set_dtype(self, dtype):
        self._dtype = dtype
        return self

    def __call__(self, *args: Any) -> Any:
        return self.forward(*args)

    def forward(self, *args):
        args = unfold_args(args)
        return self.hmonnx_session.forward(*args)

    def to_fast(self):
        self.hmonnx_session.to_fast_mode()
        return self

    @property
    def enable_golden(self) -> None:
        return self._enable_golden

    @enable_golden.setter
    def enable_golden(self, enable: bool) -> None:
        self._enable_golden = enable
        self._set_enable_golden(enable)

    def _set_enable_golden(self, enable: bool) -> None:
        self.hmonnx_session.save_golden = enable
        self.hmonnx_session.reset_step()

    def enable_auto_offload(self):
        self._enable_auto_offload = True
        self.hmonnx_session.enable_auto_offload = True

    def update_step(self):
        self.hmonnx_session.update_step()


class HMONNXBaseModel(GoldenMixin, DeviceMixin):
    def __init__(self) -> None:
        self._models: dict[str, HMONNXModel] = {}
        self._enable_golden = False
        self._enable_auto_offload = False
        self._dtype = torch.float16
        self._device = torch.device("cpu")
        self.fast_mode = False

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, HMONNXModel):
            self._models[name] = value
        return super().__setattr__(name, value)

    @property
    def enable_golden(self) -> bool:
        return self._enable_golden

    @enable_golden.setter
    def enable_golden(self, enable: bool) -> None:
        for model in self._models.values():
            model.enable_golden = enable
        self._enable_golden = enable
        self._set_enable_golden(enable)

    def _set_enable_golden(self, enable: bool) -> None:
        pass

    def enable_auto_offload(self):
        for model in self._models.values():
            model.enable_auto_offload = True
        self._enable_auto_offload = True

    def _set_device(self, device):
        self._device = device
        for model in self._models.values():
            model.to(device)
        return self

    def to_fast(self):
        """转换为快速推理模式，返回一个新的模型实例。"""
        # 默认实现直接返回自己，子类可以重写此方法以支持快速推理模式
        if self.fast_mode:
            return self
        self.fast_mode = True
        for model in self._models.values():
            model.to_fast()
        return self
