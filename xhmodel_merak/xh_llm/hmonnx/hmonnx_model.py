import os
import warnings
from pathlib import Path
from typing import Any

import torch

from xhquant.api import HMONNXGoldenInference, get_xhquant_logger
from xhquant.xhonnxruntime.hmonnx_graph_inference import HMONNXCUDAGraphInference
from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

from ..device_mixin import DeviceMixin
from ..infer_mixin import GoldenMixin
from ..utils import unfold_args


class HMONNXGolden(HMONNXGoldenInference):
    def __init__(self, hmonnx_path):
        super().__init__(hmonnx_path)
        self.golden_dir = str(Path(hmonnx_path).parent)
        self.legacy_mode = False


class HMONNXModel(GoldenMixin, DeviceMixin):
    ENV_ENABLE_INFERENCE_V2 = "ENABLE_HMINFERENCE_V2"

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "enable_auto_offload" and isinstance(value, bool) and "hmonnx_session" in self.__dict__:
            self._set_enable_auto_offload(value)
            return
        super().__setattr__(name, value)

    def __init__(
        self,
        hmonnx: str,
        onnx_graph=None,
        enable_golden: bool = False,
        enable_cuda_graph: bool = False,
        enable_auto_offload=False,
        device_map: str | None | torch.device | list[str | torch.device] = None,
        layer_infos=None,
    ):
        self._hmonnx_path = hmonnx
        self._onnx_graph = onnx_graph
        self._session_device_map = device_map
        self._session_layer_infos = layer_infos
        self._enable_cuda_graph = enable_cuda_graph
        self._fast_mode = False

        self._create_hmonnx_session(enable_golden=enable_golden, enable_auto_offload=enable_auto_offload)
        self._enable_auto_offload = bool(enable_auto_offload)
        self._dtype = torch.float16
        self._enable_golden = enable_golden

    def _create_hmonnx_session(self, enable_golden: bool, enable_auto_offload: bool) -> None:
        normalized_device_map = self._normalize_device_map(self._session_device_map)
        use_inference_v2 = self._is_inference_v2_env_enabled()
        if use_inference_v2:
            logger = get_xhquant_logger()
            logger.info(
                "Environment variable %s is enabled; routing %s through HMONNXInferenceV2.",
                self.ENV_ENABLE_INFERENCE_V2,
                self._hmonnx_path,
            )
            session_config = HMONNXInferenceConfig()
            session_config.enable_cuda_graph = self._enable_cuda_graph
            session_config.enable_auto_offload = enable_auto_offload
            session_config.exec_devices = normalized_device_map
            session_config.enable_golden = enable_golden
            session_config.layers = self._session_layer_infos
            if self._onnx_graph is None:
                self.hmonnx_session = HMONNXInferenceV2(self._hmonnx_path, session_config)
            else:
                self.hmonnx_session = HMONNXInferenceV2.from_onnx_graph(
                    self._hmonnx_path, self._onnx_graph, session_config
                )
            self._device = self.hmonnx_session.device
        else:
            logger = get_xhquant_logger()
            warnings.warn(
                (
                    "Legacy HMONNX runtime fallback is deprecated and may be removed in a future release. "
                    f"Set {self.ENV_ENABLE_INFERENCE_V2}=1 to route compatible single-device models through "
                    "HMONNXInferenceV2."
                ),
                FutureWarning,
                stacklevel=1,
            )
            logger.warning(
                "Falling back to legacy HMONNX runtime for %s. Set %s=1 to enable HMONNXInferenceV2.",
                self._hmonnx_path,
                self.ENV_ENABLE_INFERENCE_V2,
            )
            if self.enable_cuda_graph:
                logger = get_xhquant_logger()
                logger.info(f"enable CUDA Graph for {self._hmonnx_path}")
                self.hmonnx_session = HMONNXCUDAGraphInference(self._hmonnx_path)
            else:
                self.hmonnx_session = HMONNXGolden(self._hmonnx_path)
            self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            self.hmonnx_session.to(self._device)

    @staticmethod
    def _normalize_device_map(device_map: str | torch.device | int | list[Any] | tuple[Any, ...] | None) -> list[Any]:
        if device_map is None:
            return []
        if isinstance(device_map, (str, int, torch.device)):
            return [device_map]
        return list(device_map)

    @classmethod
    def _is_inference_v2_env_enabled(cls) -> bool:
        value = os.getenv(cls.ENV_ENABLE_INFERENCE_V2, "")
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @property
    def is_enable_cuda_graph(self) -> bool:
        return self._enable_cuda_graph

    @property
    def enable_cuda_graph(self) -> bool:
        return self._enable_cuda_graph

    def _set_device(self, device: torch.device):
        if device == self._device:
            return self

        if not isinstance(self.hmonnx_session, HMONNXInferenceV2):
            if self._enable_cuda_graph and self.device.type == "cuda":
                logger = get_xhquant_logger()
                logger.warning(
                    f"Model is on {self.device} and CUDA Graph is enabled, ignoring device change to {device}."
                )
                return self
            self.hmonnx_session.to(device)
            self._device = device

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
        self._fast_mode = True
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

    def _set_enable_auto_offload(self, enable: bool) -> None:
        if enable and self.enable_cuda_graph and not isinstance(self.hmonnx_session, HMONNXInferenceV2):
            raise RuntimeError("Cannot enable auto offload when CUDA Graph is enabled.")

        session_auto_offload = getattr(self.hmonnx_session, "enable_auto_offload", None)
        if callable(session_auto_offload):
            try:
                session_auto_offload(enable)
            except TypeError:
                if enable:
                    session_auto_offload()
        else:
            self.hmonnx_session.enable_auto_offload = enable

        self._enable_auto_offload = enable

    def enable_auto_offload(self):
        self._set_enable_auto_offload(True)
        return self

    def update_step(self):
        self.hmonnx_session.update_step()


class HMONNXBaseModel(GoldenMixin, DeviceMixin):
    def __init__(self, **kwargs) -> None:
        self._models: dict[str, HMONNXModel] = {}
        self._enable_golden = False
        self._enable_auto_offload = False
        self._dtype = torch.float16
        self._device = torch.device("cpu")
        self.fast_mode = False
        self._valid_devices: list[torch.device] = []
        device_map = kwargs.get("device_map")
        if device_map is None:
            if torch.cuda.is_available():
                device_map = list(range(torch.cuda.device_count()))
            else:
                device_map = ["cpu"]

        if isinstance(device_map, (str, int, torch.device)):
            device_map = [device_map]

        for device in device_map:
            if isinstance(device, torch.device):
                normalized = device
            elif isinstance(device, int):
                normalized = torch.device(f"cuda:{device}")
            else:
                device_str = str(device).strip().lower()
                if device_str == "cpu":
                    normalized = torch.device("cpu")
                elif device_str.startswith("cuda:"):
                    normalized = torch.device(device_str)
                elif device_str.isdigit():
                    normalized = torch.device(f"cuda:{int(device_str)}")
                else:
                    raise ValueError(f"Unsupported device in device_map: {device!r}")
            self._valid_devices.append(normalized)

        gpu_devices = [device for device in self._valid_devices if device.type == "cuda"]
        cpu_devices = [device for device in self._valid_devices if device.type == "cpu"]

        def _free_cuda_memory(device: torch.device) -> int:
            if not torch.cuda.is_available():
                return 0
            index = device.index if device.index is not None else torch.cuda.current_device()
            try:
                _ = torch.tensor([0], device=index)
                free_memory, _ = torch.cuda.mem_get_info(index)
                return free_memory
            except Exception:
                return 0

        gpu_devices.sort(key=_free_cuda_memory, reverse=True)
        self._valid_devices = gpu_devices + cpu_devices
        self._device = self._valid_devices[0]

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "enable_auto_offload" and isinstance(value, bool) and "_models" in self.__dict__:
            self._set_enable_auto_offload(value)
            return
        if isinstance(value, HMONNXModel):
            self._models[name] = value
        return super().__setattr__(name, value)

    @property
    def enable_golden(self) -> bool:
        return self._enable_golden

    @property
    def enable_cuda_graph(self) -> bool:
        return any(model.enable_cuda_graph for model in self._models.values())

    @enable_golden.setter
    def enable_golden(self, enable: bool) -> None:
        if enable:
            if self.enable_cuda_graph:
                raise RuntimeError("Cannot get golden status when CUDA Graph is enabled.")
        for model in self._models.values():
            model.enable_golden = enable
        self._enable_golden = enable
        self._set_enable_golden(enable)

    def _set_enable_golden(self, enable: bool) -> None:
        pass

    def _set_enable_auto_offload(self, enable: bool) -> None:
        for model in self._models.values():
            model.enable_auto_offload = enable
        self._enable_auto_offload = enable

    def enable_auto_offload(self):
        self._set_enable_auto_offload(True)
        return self

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
