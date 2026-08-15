import ctypes
import gc
import os
import weakref
from importlib.metadata import PackageNotFoundError, version
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.fx as fx
import torch.nn as nn
from packaging.version import Version
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from tqdm import tqdm
from transformers import AutoConfig, GenerationConfig
from transformers.utils.quantization_config import QuantizationMethod

from xhmodel_merak.configuration_utils import BaseAttrDict, BaseModelConfig
from xhmodel_merak.xh_llm._dequant_converter import gptqmodel_torch_qlinear_converter
from xhmodel_merak.xh_llm.infer_mixin import SwitchFXInterpreter
from xhmodel_merak.xh_llm.llm_data_processor import BaseLLMInputProcessor
from xhmodel_merak.xh_llm.register import XHLLM_TRACEABLE_MODULES
from xhquant.api import (
    FXInterpreter,
    PrecisionMode,
    XHExportedGraph,
    XHFrontendGraph,
    XHQuantedGraph,
    eager_qmodel_ptq,
    get_xhquant_logger,
    model_to_eager_qmodel,
    ptq_quantize,
    to_export_graph,
    to_export_hmonnx_v2,
    to_quant_graph,
)
from xhquant.utils import ConfigDict, log_function_call
from xhquant.utils.registry import DynamicModule
from xhquant.xhonnxruntime import AutoOffloadGraphModel

from ._dequant_converter import autoround_torch_qlinear_converter, restore_autoround_qwen3_5_moe_sparse_block
from .device_mixin import DeviceMixin
from .types import LLMModelMeta, LLMModelState, ModelMeta, ModelSwitcher
from .utils import (
    get_cpu_memory_mb,
    get_model_param_buffer_size_gb,
    hf_auto_offload,
    resolve_model_dtype,
    unfold_args,
)
from .wrap_model import wrap_llm_model


if TYPE_CHECKING:
    from .hmonnx import BaseLLMHMONNXModel


class XHBaseModel(DeviceMixin):
    transformers_min_version: Optional[str] = None  # 依赖的transformers库的最低版本要求，默认为None，表示不限制最低版本
    transformers_max_version: Optional[str] = None  # 依赖的transformers库的最高版本要求，默认为None，表示不限制最高版本
    HF_MODEL_CLS: Optional[type] = None  # 该模型对应的 Huggingface 模型类，主要用于版本检查和模型加载
    HF_AUTO_MODEL_CLS: Optional[type] = None  # 该模型对应的 Huggingface 模型自动加载类，主要用于模型加载
    HF_MODEL_DTYPE = torch.float16  # 模型权重的数据类型，默认为 float16

    BUILD_HF_COMPATIBLE_FUNC: Callable  # 默认为None，子类需要提供该函数以支持transformers库的接口兼容
    META_CLS: Optional[type[LLMModelMeta]] = LLMModelMeta  # 该模型对应的Meta类，主要用于模型的元信息管理和版本控制
    HMONNXINFERENCE_CLS: Optional[type["BaseLLMHMONNXModel"]] = (
        None  # 该模型对应的HMONNXInference类，主要用于HMONNX推理
    )
    CONFIG_CLS: Optional[type[BaseModelConfig]] = None  # 该模型对应的配置类，主要用于模型的配置管理

    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.hf_model_dir = config.hf_model
        self.model_type = config.model_type
        self._state = LLMModelState.NONE

        self._wrap_model: DynamicModule | None = None
        self._frontend_model: XHFrontendGraph | None = None
        self._quanted_model: XHQuantedGraph | None = None
        self._exported_model: XHExportedGraph | None = None
        self.hf_compatible_model: nn.Module | None = None
        self.enable_hf_compatible: bool = False
        self.interactive_mode = False
        self._dtype = resolve_model_dtype(config.get("dtype", "float16"))
        self._device = "cpu"
        self._data_processor = None
        wrap_cfg = self.config.to_dict()
        wrap_cfg = BaseAttrDict(wrap_cfg)
        self._inference_model = None
        # 兼容旧代码
        self.wrap_cfg = wrap_cfg

        self._models = {}

    def get_wrap_cfg(self):
        wrap_config = self.wrap_cfg
        if hasattr(self.config, "only_first_block"):
            wrap_config.only_first_block = self.config.only_first_block
        if hasattr(self.config, "max_layers"):
            wrap_config.max_layers = self.config.max_layers
        if os.environ.get("LAYER_TAG_ENABLE", "").strip().lower() in {"1", "true", "yes", "on"}:
            wrap_config.enable_layer_tag = True
        return wrap_config

    def __setattr__(self, name: str, value: Any) -> None:
        old_value = getattr(self, name, None)
        if isinstance(old_value, XHBaseModel) or isinstance(value, XHBaseModel):
            self._models[name] = value
        return super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._models:
            del self._models[name]
        return super().__delattr__(name)

    def get_kvcache_mixin(self):
        raise NotImplementedError(
            "Subclasses must implement get_kvcache_mixin method to return the KVCacheMixin instance for the model."
        )

    def eval(self):
        # 该函数用于将模型设置为评估模式，子类可以根据需要重写该函数以实现特定的评估模式设置
        if self._wrap_model is not None:
            self._wrap_model.eval()
        if self._frontend_model is not None:
            self._frontend_model.eval()
        if self._quanted_model is not None:
            self._quanted_model.eval()
        if self._exported_model is not None:
            self._exported_model.eval()

    @classmethod
    def config_class(cls):
        return cls.CONFIG_CLS

    @property
    def wrap_model(self) -> DynamicModule:
        if self._wrap_model is not None:
            return weakref.proxy(self._wrap_model)
        return self._wrap_model

    @property
    def frontend_model(self) -> XHFrontendGraph:
        if self._frontend_model is not None:
            return weakref.proxy(self._frontend_model)
        return self._frontend_model

    @property
    def quanted_model(self) -> XHQuantedGraph:
        if self._quanted_model is not None:
            return weakref.proxy(self._quanted_model)
        return self._quanted_model

    @property
    def exported_model(self) -> XHExportedGraph:
        if self._exported_model is not None:
            return weakref.proxy(self._exported_model)
        return self._exported_model

    @property
    def work_dir(self) -> str:
        return self.config.work_dir

    @work_dir.setter
    def work_dir(self, work_dir: str):
        self.config.work_dir = work_dir

    ### 模型状态切换
    def set_state(self, state: LLMModelState):
        if state == self._state:
            return
        if state == LLMModelState.WRAP:
            self.to_wrap()
        elif state == LLMModelState.EAGER_FAST:
            self.to_eager_fast()
        elif state == LLMModelState.EAGER_ALIGNED:
            self.to_eager_aligned()
        elif state == LLMModelState.FRONTED:
            self.to_fronted()
        elif state in [LLMModelState.QUANTED_FAST, LLMModelState.QUANTED_ALIGNED]:
            self.to_quanted_fast() if state == LLMModelState.QUANTED_FAST else self.to_quanted_aligned()
        elif state == LLMModelState.QUANTED_DISABLE:
            self.to_quanted_disable()
        elif state == LLMModelState.EXPORTED:
            assert self._state in [
                LLMModelState.NONE,
                LLMModelState.WRAP,
                LLMModelState.FRONTED,
                LLMModelState.QUANTED_FAST,
                LLMModelState.QUANTED_ALIGNED,
                LLMModelState.QUANTED_DISABLE,
            ], f"Invalid state transition: {self._state} -> {state}"
            self.to_exported()

    def _to_wrap(self, hf_model):
        raise NotImplementedError("Subclasses must implement _to_wrap method to support wraping the model.")

    def to_wrap(self, hf_model: Optional[Any] = None):
        if self._state != LLMModelState.NONE:
            return
        hf_model = self.get_native_model() if hf_model is None else hf_model
        logger = get_xhquant_logger()
        for _, sub_model in self._models.items():
            sub_model.to_wrap(hf_model)
        logger.info(f"Converting model {type(self).__name__} to wrap mode...")
        memory_info = get_cpu_memory_mb()
        logger.info(f"Initial CPU memory usage before converting to wrap mode: {str(memory_info)}")
        self._to_wrap(hf_model)
        memory_info = get_cpu_memory_mb()
        logger.info(f"Initial CPU memory usage after converting to wrap mode: {str(memory_info)}")
        self._state = LLMModelState.WRAP

    def _to_eager(self, aligned: bool = True):
        if self._state not in [LLMModelState.NONE, LLMModelState.WRAP]:
            raise RuntimeError(f"Invalid state transition: {self._state} -> {LLMModelState.EAGER}")

        if self._state != LLMModelState.WRAP:
            self.to_wrap()
        for _, module in self._wrap_model.named_modules():
            if hasattr(module, "_eager_forward"):
                module.forward = module._eager_forward
                del module._eager_forward
        quant_config = self.get_quant_cfg()
        eager_qmodel = model_to_eager_qmodel(self._wrap_model, self.config.quant_scheme.target_device, quant_config)
        eager_qmodel_ptq(eager_qmodel, "ALIGNED" if aligned else "FAST", release_unused_parameters=True)

    def to_eager_fast(self):
        if not self.config.enable:
            return
        if self._state == LLMModelState.EAGER_FAST:
            return
        logger = get_xhquant_logger()
        logger.info(f"Converting model {type(self).__name__} to eager fast mode...")
        self._to_eager(aligned=False)
        self._state = LLMModelState.EAGER_FAST

    def to_eager_aligned(self):
        if not self.config.enable:
            return
        if self._state == LLMModelState.EAGER_ALIGNED:
            return
        logger = get_xhquant_logger()
        logger.info(f"Converting model {type(self).__name__} to eager aligned mode...")
        self._to_eager(aligned=True)
        self._state = LLMModelState.EAGER_ALIGNED

    def release_wraped_model(self):
        del self._wrap_model
        self._wrap_model = None

    def _to_fronted(self, wrap_model):
        raise NotImplementedError("Fronted mode is not implemented.")

    def to_fronted(self):
        if not self.config.enable:
            return
        if self._state == LLMModelState.FRONTED:
            return
        if self._state not in [LLMModelState.NONE, LLMModelState.WRAP]:
            raise RuntimeError(f"Invalid state transition: {self._state} -> {LLMModelState.FRONTED}")

        if self._state != LLMModelState.WRAP:
            self.to_wrap()
        for _, sub_model in self._models.items():
            sub_model.to_fronted()
        logger = get_xhquant_logger()
        logger.info(f"Converting model {type(self).__name__} to fronted mode...")
        self._frontend_model = self._to_fronted(self._wrap_model)
        self.release_wraped_model()
        self._state = LLMModelState.FRONTED
        return self._frontend_model

    def get_quant_cfg(self):
        quant_scheme = self.config.quant_scheme if hasattr(self.config, "quant_scheme") else None
        if quant_scheme is not None:
            quant_cfg = quant_scheme.to_dict() if hasattr(quant_scheme, "to_dict") else dict(quant_scheme)
        else:
            quant_cfg = {}
        quant_cfg = ConfigDict(quant_cfg)
        for legacy_key, config_key in (("nodes", "nodes_cfg"), ("ops", "ops_cfg")):
            if legacy_key not in quant_cfg:
                continue
            legacy_overrides = quant_cfg.pop(legacy_key)
            mapped_overrides = quant_cfg.setdefault(config_key, ConfigDict())
            for name, override_cfg in legacy_overrides.items():
                if name in mapped_overrides:
                    continue
                mapped_overrides[name] = (
                    ConfigDict(dict(quant_type=override_cfg))
                    if isinstance(override_cfg, str)
                    else ConfigDict(override_cfg)
                )
        return quant_cfg

    def _to_quanted(self, frontend_model, state, auto_release_unused_parameters: bool = True, infer_shape: bool = True):
        target_device = self.config.chip_arch
        quant_cfg = self.get_quant_cfg()
        quanted_model = to_quant_graph(frontend_model, target_device, quant_cfg)
        if state in [LLMModelState.QUANTED_FAST, LLMModelState.QUANTED_ALIGNED]:
            with self.get_kvcache_mixin().kv_cache_scope(device="meta"):
                data_processor = self.get_data_preprocessor()
                dummy_inputs = self.get_dummy_inputs()
                assert isinstance(dummy_inputs, (dict,)), (
                    f"Dummy inputs should be a dictionary of tensors, but get {type(dummy_inputs)}."
                )
                calib_data = data_processor(dummy_inputs)
                assert isinstance(calib_data, (list, tuple)), (
                    f"Processed dummy inputs should be a list or tuple of tensors, but get {type(calib_data)}."
                )

                calib_data = unfold_args(calib_data)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                ptq_quantize(
                    quanted_model,
                    [calib_data],
                    PrecisionMode.ALIGNED if state == LLMModelState.QUANTED_ALIGNED else PrecisionMode.FAST,
                    [device],
                    auto_release_unused_parameters=auto_release_unused_parameters,
                    infer_shape=infer_shape,
                )
        elif state == LLMModelState.QUANTED_DISABLE:
            pass
        else:
            raise ValueError(f"Invalid quantization state: {state}")
        return quanted_model

    def to_quanted_fast(self):
        if not self.config.enable:
            return
        if self._state == LLMModelState.QUANTED_FAST:
            return
        if self._state not in [LLMModelState.NONE, LLMModelState.WRAP, LLMModelState.FRONTED]:
            raise RuntimeError(f"Invalid state transition: {self._state} -> {LLMModelState.QUANTED_FAST}")
        if self._state != LLMModelState.FRONTED:
            self.to_fronted()
        for _, sub_model in self._models.items():
            sub_model.to_quanted_fast()
        logger = get_xhquant_logger()
        logger.info(f"Converting model {type(self).__name__} to quanted fast mode...")
        self._quanted_model = self._to_quanted(self._frontend_model, LLMModelState.QUANTED_FAST)
        del self._frontend_model
        self._frontend_model = None  # 释放前端模型以节省内存
        self._state = LLMModelState.QUANTED_FAST

    def to_quanted_aligned(self):
        if not self.config.enable:
            return
        if self._state == LLMModelState.QUANTED_ALIGNED:
            return
        if self._state not in [LLMModelState.NONE, LLMModelState.WRAP, LLMModelState.FRONTED]:
            raise RuntimeError(f"Invalid state transition: {self._state} -> {LLMModelState.QUANTED_ALIGNED}")
        if self._state != LLMModelState.FRONTED:
            self.to_fronted()
        for _, sub_model in self._models.items():
            sub_model.to_quanted_aligned()
        logger = get_xhquant_logger()
        logger.info(f"Converting model {type(self).__name__} to quanted aligned mode...")
        self._quanted_model = self._to_quanted(self._frontend_model, LLMModelState.QUANTED_ALIGNED)
        del self._frontend_model
        self._frontend_model = None  # 释放前端模型以节省内存
        self._state = LLMModelState.QUANTED_ALIGNED

    def to_quanted_disable(self):
        if not self.config.enable:
            return
        if self._state == LLMModelState.QUANTED_DISABLE:
            return
        if self._state not in [LLMModelState.NONE, LLMModelState.WRAP, LLMModelState.FRONTED]:
            raise RuntimeError(f"Invalid state transition: {self._state} -> {LLMModelState.QUANTED_FAST}")
        if self._state != LLMModelState.FRONTED:
            self.to_fronted()
        for _, sub_model in self._models.items():
            sub_model.to_quanted_disable()
        logger = get_xhquant_logger()
        logger.info(f"Converting model {type(self).__name__} to quanted disable mode...")
        self._quanted_model = self._to_quanted(self._frontend_model, LLMModelState.QUANTED_DISABLE)
        del self._frontend_model
        self._frontend_model = None  # 释放前端模型以节省内存
        self._state = LLMModelState.QUANTED_DISABLE

    def to_exported(self):
        raise NotImplementedError

    def check_transformer_version(self) -> bool:
        """检查 transformers 库的版本是否在指定范围内。

        Returns:
            如果版本在指定范围内返回 True，否则返回 False

        Raises:
            ImportError: 如果未安装 transformers 库
        """
        try:
            import transformers
        except ImportError as e:
            raise ImportError("未安装 transformers 库，请先安装: pip install transformers") from e

        current_version = Version(transformers.__version__)

        if self.transformers_min_version is not None:
            if current_version < Version(self.transformers_min_version):
                raise RuntimeError(
                    f"当前 transformers 版本 {current_version} 低于模型要求的最低版本 {self.transformers_min_version}。"
                    f"请升级 transformers 库: pip install --upgrade transformers"
                )

        if self.transformers_max_version is not None:
            if current_version > Version(self.transformers_max_version):
                raise RuntimeError(
                    f"当前 transformers 版本 {current_version} 高于模型要求的最高版本 {self.transformers_max_version}。"
                    f"请安装适合的 transformers 版本: pip install transformers=={self.transformers_max_version}"
                )

        return True

    ### 模型加载
    def get_dummy_inputs(self) -> dict[str, str]:
        """
        获取模型的dummy输入，用于生成计算图，返回值必须是一个字典，键是输入名称，值是对应的输入数据。
        """

        raise NotImplementedError()

    def _wraped_post(self, hf_model: Any):
        # 该函数在模型被wrap后调用，子类可以重写该函数以实现特定的wrap后处理逻辑
        pass

    def _wraped_pre(self, hf_model: Any):
        pass

    @staticmethod
    def _trim_cpu_allocator() -> None:
        gc.collect()
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

    def init_wrap_model(self, hf_model: Any) -> Any:
        if hf_model is None:
            hf_model = self.get_native_model()
        # Dtype-sensitive constants (for example KDA decay and absorbed MLA
        # scales) are created while wrapping.  Enforce the xh model dtype
        # before that work rather than casting only the finished wrap model.
        hf_model.to(dtype=self._dtype)
        self._wraped_pre(hf_model)
        self._wrap_model = wrap_llm_model(hf_model, self.get_wrap_cfg())  # id(model) == id(wrap_model)

        def check_wraped(wraped_model: nn.Module):
            """
            检查是否被正确的wrap
            """
            for name, module in wraped_model.named_modules():
                if not isinstance(module, DynamicModule) and type(module) in XHLLM_TRACEABLE_MODULES:
                    raise RuntimeError(
                        f"Module {name}[{type(module)}] is  not wrapped by XHQuant, "
                        f"please use XHLLM_TRACEABLE_MODULES.register_module to wrap it."
                    )

                return self._wrap_model

        check_wraped(self._wrap_model)
        self._wraped_post(hf_model)
        self._trim_cpu_allocator()
        return self._wrap_model

    @classmethod
    def get_hmonnx_inference_cls(cls) -> Optional[type["BaseLLMHMONNXModel"]]:
        cls.HMONNXINFERENCE_CLS.LLM_MODEL_CLS = cls
        return cls.HMONNXINFERENCE_CLS

    @classmethod
    def from_pretrained(cls, config: BaseModelConfig) -> "XHBaseModel":
        return cls(config)

    @classmethod
    def from_hmonnx_meta(cls, meta: LLMModelMeta, **kwargs) -> "XHBaseModel":
        infer_cls = cls.get_hmonnx_inference_cls()
        return infer_cls(meta, **kwargs)

    @classmethod
    def get_hf_auto_model_cls(cls):
        return cls.HF_AUTO_MODEL_CLS

    @classmethod
    def get_hf_model_cls(cls):
        return cls.HF_MODEL_CLS

    @classmethod
    def get_hf_model_dtype(cls):
        return cls.HF_MODEL_DTYPE

    @classmethod
    def untied_weights(cls, module: nn.Module) -> nn.Module:
        param_ids = {}
        duplicate_params = []
        for name, param in module.named_parameters(remove_duplicate=False):
            param_id = id(param)
            if param_id not in param_ids:
                param_ids[param_id] = param
            else:
                duplicate_params.append(name)

        duplicate_params = list(set(duplicate_params))
        for param_name in duplicate_params:
            fields = param_name.split(".")[:-1]
            m_name = ".".join(fields)
            attr_name = param_name.split(".")[-1]
            m = module.get_submodule(m_name)
            param = getattr(m, attr_name)
            setattr(m, attr_name, nn.Parameter(param.clone()))
        return module

    @classmethod
    def _load_hf_model(cls, hf_model_dir: str, **kwargs):
        auto_model_cls = cls.get_hf_auto_model_cls()

        model_dtype = cls.get_hf_model_dtype()
        if "dtype" not in kwargs and "torch_dtype" not in kwargs:
            # Keep the legacy default keyword for model families that still
            # support older Transformers releases. Instance-driven export
            # paths pass the resolved xh dtype explicitly via ``dtype``.
            kwargs["torch_dtype"] = model_dtype
        native_model = auto_model_cls.from_pretrained(hf_model_dir, **kwargs)
        native_model = cls.untied_weights(native_model)
        # if native_model.config.tie_word_embeddings:  # type: ignore
        #     old_torchscript = native_model.config.torchscript  # type: ignore
        #     native_model.config.torchscript = True  # type: ignore
        #     native_model.tie_weights()  # type: ignore
        #     native_model.config.tie_word_embeddings = False  # type: ignore
        #     native_model.config.torchscript = old_torchscript  # type: ignore

        return native_model

    @classmethod
    def _load_quant_weight(cls, quant_weight_path: str, native_hf_model: nn.Module, strict: bool = True) -> bool:
        logger = get_xhquant_logger()
        archive_file = cls._resolve_quant_weight_archive(quant_weight_path)
        logger.info(f"Load previously saved checkpoint from: {archive_file}")
        is_safetensors = archive_file.endswith(".safetensors")
        state_dict: dict[str, Tensor]
        if is_safetensors:
            state_dict = load_safetensors_file(archive_file, device="cpu")
        else:
            state_dict = torch.load(archive_file, weights_only=True, map_location="cpu")

        model_state_dict = native_hf_model.state_dict()
        unexpect_state_dict = []
        for k, _ in state_dict.items():
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

        native_hf_model.load_state_dict(state_dict, strict=strict)
        del state_dict
        return True

    @staticmethod
    def _resolve_quant_weight_archive(quant_weight_path: str) -> str:
        archive_path = Path(quant_weight_path)
        if not archive_path.is_dir():
            return str(archive_path)

        preferred_names = (
            "quant_weight.pt",
            "quantized_weight.pt",
            "model.pt",
            "pytorch_model.bin",
        )
        for name in preferred_names:
            candidate = archive_path / name
            if candidate.is_file():
                return str(candidate)

        candidates = sorted(
            path for pattern in ("*.pt", "*.pth", "*.bin") for path in archive_path.glob(pattern) if path.is_file()
        )
        if len(candidates) == 1:
            return str(candidates[0])
        if not candidates:
            raise FileNotFoundError(
                f"quant_weight directory {archive_path} does not contain a torch checkpoint "
                "(.pt, .pth, or .bin). safetensors/GPTQ HF directories are not supported here."
            )

        candidate_list = ", ".join(str(path) for path in candidates)
        raise ValueError(
            f"quant_weight directory {archive_path} contains multiple checkpoint candidates; "
            f"please pass one file explicitly. Candidates: {candidate_list}"
        )

    @classmethod
    def _load_gptqmodel(cls, hf_model_dir: str, device_map="cpu", **kwargs):
        try:
            from gptqmodel import GPTQModel
        except ImportError:
            candidate_roots = [
                Path(hf_model_dir).resolve().parents[2] if len(Path(hf_model_dir).resolve().parents) >= 3 else None,
                Path.home() / "gptqmodel",
            ]
            for candidate_root in candidate_roots:
                if candidate_root is None:
                    continue
                package_init = candidate_root / "gptqmodel" / "__init__.py"
                if package_init.exists():
                    import sys

                    if str(candidate_root) not in sys.path:
                        sys.path.insert(0, str(candidate_root))
                    sys.modules.pop("gptqmodel", None)
                    from gptqmodel import GPTQModel

                    break
            else:
                raise

        trust_remote_code = bool(kwargs.pop("trust_remote_code", True))
        backend = kwargs.pop("backend", "torch")
        # ``offload_to_disk`` is a quantization-loop option. Older and
        # third-party quantized checkpoints often omit it, and GPTQModel's
        # legacy default is True. Never let an inference/export load silently
        # inherit that default; callers that intentionally need it can still
        # opt in explicitly.
        kwargs.setdefault("offload_to_disk", False)
        kwargs.setdefault("offload_to_disk_path", None)
        if "dtype" not in kwargs:
            kwargs["dtype"] = cls.get_hf_model_dtype()
        valid_string_device_maps = {
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        }
        load_kwargs = {
            "backend": backend,
            "trust_remote_code": trust_remote_code,
            **kwargs,
        }

        if isinstance(device_map, dict):
            load_kwargs["device_map"] = device_map
        elif isinstance(device_map, str):
            if device_map in valid_string_device_maps:
                load_kwargs["device_map"] = device_map
            elif device_map != "meta":
                load_kwargs["device"] = device_map
        elif device_map is not None:
            load_kwargs["device"] = device_map

        if "device_map" in load_kwargs and "device" not in load_kwargs:
            load_kwargs["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

        logger = get_xhquant_logger()
        logger.info(f"GPTQModel load: {load_kwargs}")
        try:
            q_model = GPTQModel.load(hf_model_dir, **load_kwargs)
        except TypeError:
            load_kwargs.pop("backend", None)
            q_model = GPTQModel.load(hf_model_dir, **load_kwargs)

        # hf_model = cast(Union[Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM], q_model.model.eval())
        hf_model = q_model.model
        qcfg = getattr(hf_model.config, "quantization_config", None)
        if isinstance(qcfg, dict):
            try:
                from transformers.utils.quantization_config import GPTQConfig

                hf_model.config.quantization_config = GPTQConfig.from_dict(qcfg)
            except Exception:
                pass
        return hf_model

    @classmethod
    def _dequantize_gptqmodel_hf_model(cls, native_hf_model):
        from transformers.utils import is_gptqmodel_available

        assert is_gptqmodel_available(), "We need gptqmodel to dequantize auto-gptq model"
        converter = gptqmodel_torch_qlinear_converter
        from gptqmodel.nn_modules.qlinear import PackableQuantLinear

        logger = get_xhquant_logger()
        logger.info(f"Start Dequantizing GPTQModel")

        dequant_modules = []
        for name, module in native_hf_model.named_modules():  # type: ignore
            if isinstance(module, PackableQuantLinear):
                dequant_modules.append((name, module))

        for name, module in tqdm(dequant_modules, desc="Dequantizing GPTQModel"):
            devices = list(
                {tensor.device for tensor in chain(module.parameters(recurse=True), module.buffers(recurse=True))}
            )
            if len(devices) > 1:
                raise NotImplementedError(f"expected module {name} on a single device but got {devices}")

            original_device = devices[0] if devices else torch.device("cpu")
            if original_device.type != "cpu":
                module.to("cpu")
            converter(module)
            if original_device.type != "cpu":
                module.to(original_device)

        logger.info("Dequantizing GPTQModel Finished")
        return native_hf_model

    @classmethod
    def _dequantize_awq_hf_model(cls, native_hf_model: nn.Module):
        assert native_hf_model.config.quantization_config.quant_method == QuantizationMethod.AWQ
        hf_model = native_hf_model
        try:
            from awq.modules.linear.gemm import WQLinear_GEMM
            from awq.utils.packing_utils import reverse_awq_order, unpack_awq
        except ImportError:
            WQLinear_GEMM = None
            reverse_awq_order = None
            unpack_awq = None

        for name, module in hf_model.named_modules():
            if isinstance(module, WQLinear_GEMM):
                if hasattr(module, "weight"):
                    continue
                bits = module.w_bit
                group_size = module.group_size
                iweight = module.qweight
                izeros = module.qzeros
                scales = module.scales

                iweight, izeros = unpack_awq(iweight, izeros, bits)
                # Reverse the order of the iweight and izeros tensors
                iweight, izeros = reverse_awq_order(iweight, izeros, bits)

                # overflow checks
                iweight = torch.bitwise_and(iweight, (2**bits) - 1)
                izeros = torch.bitwise_and(izeros, (2**bits) - 1)

                # fp16 weights
                scales = scales.repeat_interleave(group_size, dim=0)
                izeros = izeros.repeat_interleave(group_size, dim=0)

                # quant weight and weight
                quant_weight = iweight - izeros
                weight = quant_weight * scales
                quant_weight = quant_weight.t().contiguous()
                weight = weight.t().contiguous()

                iweight = None
                izeros = None
                scales = None
                if hasattr(module, "qweight"):
                    delattr(module, "qweight")
                if hasattr(module, "qzeros"):
                    delattr(module, "qzeros")
                if hasattr(module, "scales"):
                    delattr(module, "scales")

                module.register_parameter("weight", nn.Parameter(weight))
                module.register_buffer("quant_weight", quant_weight)
                quant_weight = None
                weight = None
                # module.forward = types.MethodType(linear_forward, module)
                module.__class__ = nn.Linear

        if hf_model.config.tie_word_embeddings:
            hf_model.config.torchscript = True
            hf_model.tie_weights()
            hf_model.config.tie_word_embeddings = False

        hf_model.quantization_method = None  # type: ignore
        hf_model._is_hf_initialized = False  # type: ignore
        return hf_model

    @classmethod
    def _get_gptq_quant_linear_converters(cls):
        """Return installed GPTQ QuantLinear implementations and converters."""
        from ._dequant_converter import (
            general_qlinear_converter,
            gptqmodel_torch_qlinear_converter,
            qlinear_cuda_old_converter,
        )

        converter_by_type = {}
        unsupported_base_types = []

        try:
            from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear as GeneralQuantLinear
            from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear as CudaOldQuantLinear
            from auto_gptq.nn_modules.qlinear.qlinear_exllama import QuantLinear as ExllamaQuantLinear
            from auto_gptq.nn_modules.qlinear.qlinear_exllamav2 import QuantLinear as Exllamav2QuantLinear
            from auto_gptq.nn_modules.qlinear.qlinear_marlin import QuantLinear as AutoGPTQMarlinQuantLinear

            converter_by_type[GeneralQuantLinear] = general_qlinear_converter
            converter_by_type[CudaOldQuantLinear] = qlinear_cuda_old_converter
            unsupported_base_types.extend([ExllamaQuantLinear, Exllamav2QuantLinear, AutoGPTQMarlinQuantLinear])
        except ImportError:
            pass

        try:
            from gptqmodel.nn_modules.qlinear import BaseQuantLinear
            from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

            converter_by_type[TorchQuantLinear] = gptqmodel_torch_qlinear_converter
            for module_path, class_name in (
                ("gptqmodel.nn_modules.qlinear.torch_fused", "TorchFusedQuantLinear"),
                ("gptqmodel.nn_modules.qlinear.gemm_hf_kernel", "HFKernelLinear"),
            ):
                try:
                    module = __import__(module_path, fromlist=[class_name])
                    converter_by_type[getattr(module, class_name)] = gptqmodel_torch_qlinear_converter
                except (ImportError, AttributeError):
                    pass
            unsupported_base_types.append(BaseQuantLinear)
        except ImportError:
            pass

        return converter_by_type, tuple(unsupported_base_types)

    @classmethod
    def _dequantize_gptq_hf_model(cls, native_hf_model: nn.Module):
        """
        Abandon this method in future. Should unify 'gptqmodel' quant model
        load and dequantization logic. Currently, Gemma4 is not registered
        in our custom gptqmodel yet, so we need to load it via hf transformers
        and dequantize it via this method.
        """
        hf_model = native_hf_model
        converter_by_type, unsupported_base_types = cls._get_gptq_quant_linear_converters()
        dequant_linears = []
        unsupported_linears = []
        for name, module in hf_model.named_modules():
            converter = next(
                (
                    candidate_converter
                    for quant_linear_type, candidate_converter in converter_by_type.items()
                    if isinstance(module, quant_linear_type)
                ),
                None,
            )
            if converter is not None:
                dequant_linears.append((name, module, converter))
            elif unsupported_base_types and isinstance(module, unsupported_base_types):
                unsupported_linears.append((name, type(module)))

        if unsupported_linears:
            details = ", ".join(f"{name}: {module_type.__name__}" for name, module_type in unsupported_linears[:20])
            raise NotImplementedError(f"GPTQ dequantization is not implemented for: {details}")

        pbar = tqdm(dequant_linears, desc="Dequantizing GPTQ model")
        for name, module, converter in pbar:
            pbar.set_description(f"Dequantizing GPTQ: {name}")
            converter(module)

        hf_model.quantization_method = None  # type: ignore
        hf_model._is_hf_initialized = False  # type: ignore
        return hf_model

    @classmethod
    def _dequantize_gptqmodel(cls, native_hf_model: nn.Module) -> nn.Module:
        raise ValueError(
            "The _dequantize_gptqmodel method is deprecated, please use _dequantize_gptqmodel_hf_model instead."
        )
        """
        Dequantize a GPTQ model loaded via GPTQModel library.
        This handles models loaded through `_load_gptqmodel` which contain QuantLinear layers.
        """
        from gptqmodel.nn_modules.qlinear import PackableQuantLinear

        from xhmodel_merak.xh_llm._dequant_converter import gptqmodel_torch_qlinear_converter

        hf_model = native_hf_model
        torch_linear_cls = [PackableQuantLinear]
        try:
            from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedQuantLinear

            torch_linear_cls.append(TorchFusedQuantLinear)
        except Exception:
            pass

        # Find and convert all QuantLinear modules
        dequant_linears = []
        for name, module in hf_model.named_modules():
            if isinstance(module, tuple(torch_linear_cls)):
                dequant_linears.append((name, module))

        if dequant_linears:
            logger = get_xhquant_logger()
            pbar = tqdm(dequant_linears, desc="Dequantizing GPTQ model")
            for name, module in pbar:
                pbar.set_description(f"Dequantizing GPTQ: {name}")
                gptqmodel_torch_qlinear_converter(module)

        # Clear quantization metadata
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "quantization_config"):
            hf_model.config.quantization_config = None

        if hasattr(hf_model, "quantization_method"):
            hf_model.quantization_method = None
        if hasattr(hf_model, "_is_hf_initialized"):
            hf_model._is_hf_initialized = False

        return hf_model

    @staticmethod
    def _get_quantization_method(quantization_config: Any) -> str | None:
        # AutoConfig 已经读取了 config.json；这里直接使用其中的 quantization_config.quant_method。
        if quantization_config is None:
            return None
        if isinstance(quantization_config, dict):
            quant_method = quantization_config.get("quant_method")
        else:
            quant_method = getattr(quantization_config, "quant_method", None)
        quant_method = getattr(quant_method, "value", quant_method)
        return str(quant_method).lower() if quant_method is not None else None

    @staticmethod
    def _compressed_tensors_value(value: Any) -> Any:
        return getattr(value, "value", value)

    @classmethod
    def _convert_compressed_tensors_linear(
        cls,
        module_name: str,
        module: nn.Linear,
    ) -> nn.Linear:
        def _expand_compressed_tensors_group_param(
            module_name: str,
            value: Tensor,
            weight_shape: torch.Size,
            group_size: int,
            param_name: str,
        ) -> Tensor:
            if value.shape[-1] * group_size < weight_shape[-1]:
                raise RuntimeError(
                    f"compressed-tensors {param_name} shape mismatch for {module_name}: "
                    f"{tuple(value.shape)} cannot cover weight shape {tuple(weight_shape)} with group_size={group_size}."
                )
            return value.repeat_interleave(group_size, dim=-1)[..., : weight_shape[-1]]

        min_compressed_tensors_version = Version("0.15.0")
        try:
            compressed_tensors_version = Version(version("compressed-tensors"))
        except PackageNotFoundError as e:
            raise ImportError("未安装 compressed-tensors 库，请先安装: pip install compressed-tensors>=0.15.0") from e

        """
        compressed-tensors 0.15.0后对量化线性层实现做了调整。这里的实现不保证兼容0.15.0以下版本。
        """
        if compressed_tensors_version < min_compressed_tensors_version:
            raise RuntimeError(
                f"当前 compressed-tensors 版本 {compressed_tensors_version} 低于要求的最低版本 "
                f"{min_compressed_tensors_version}。请升级 compressed-tensors 库: "
                "pip install --upgrade 'compressed-tensors>=0.15.0'"
            )

        from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
        from compressed_tensors.quantization.lifecycle.forward import dequantize

        scheme = getattr(module, "quantization_scheme", None)
        weights = getattr(scheme, "weights", None)
        if scheme is None or weights is None:
            raise NotImplementedError(f"Cannot find compressed-tensors quantization_scheme.weights for {module_name}.")

        format_str = str(cls._compressed_tensors_value(getattr(scheme, "format", None))).lower()
        if format_str != "pack-quantized":
            raise NotImplementedError(
                f"compressed-tensors format {format_str!r} is not supported for {module_name}; "
                "only 'pack-quantized' is supported."
            )

        num_bits = int(getattr(weights, "num_bits"))
        if num_bits not in (4, 8):
            raise NotImplementedError(
                f"compressed-tensors num_bits={num_bits} is not supported for {module_name}; "
                "only int4/int8 pack-quantized weights are supported."
            )

        strategy = str(cls._compressed_tensors_value(getattr(weights, "strategy", None))).lower()
        if strategy not in {"group", "tensor_group"}:
            raise NotImplementedError(
                f"compressed-tensors strategy {strategy!r} is not supported for {module_name}; "
                "only group/tensor_group pack-quantized weights are supported."
            )

        g_idx = getattr(module, "weight_g_idx", None)
        if g_idx is not None and g_idx.device.type != "meta" and not torch.any(g_idx == -1):
            raise NotImplementedError(
                f"compressed-tensors actorder with weight_g_idx is not supported for {module_name}."
            )

        packed = module.weight_packed.detach()
        scale = module.weight_scale.detach()
        weight_shape = torch.Size(int(dim) for dim in module.weight_shape.detach().cpu().tolist())
        group_size = int(getattr(weights, "group_size"))

        # transformers 加载 compressed-tensors 后，量化 Linear 仍是 nn.Linear，
        # 但 dense weight 尚未 materialize，压缩权重保存在 weight_packed/weight_scale 成员中。
        unpacked = unpack_from_int32(packed, num_bits, weight_shape).contiguous()
        zero_point = getattr(module, "weight_zero_point", None)
        zero_point = zero_point.detach() if zero_point is not None else None
        if zero_point is not None and zero_point.dtype == torch.int32:
            original_zp_shape = torch.Size((*weight_shape[:-1], scale.shape[-1]))
            zero_point = unpack_from_int32(zero_point, num_bits, original_zp_shape, packed_dim=0).contiguous()

        expanded_scale = _expand_compressed_tensors_group_param(
            module_name, scale, weight_shape, group_size, "weight_scale"
        )
        expanded_zero_point = None
        if zero_point is not None:
            expanded_zero_point = _expand_compressed_tensors_group_param(
                module_name, zero_point, weight_shape, group_size, "weight_zero_point"
            )

        # zero point 先作用到整数域：dense_weight == (unpacked - zero_point) * scale。
        quant_weight_int = unpacked.to(torch.int16)
        if expanded_zero_point is not None:
            quant_weight_int = quant_weight_int - expanded_zero_point.to(torch.int16)
        if quant_weight_int.min() < -128 or quant_weight_int.max() > 127:
            raise RuntimeError(
                f"compressed-tensors quant_weight for {module_name} exceeds int8 range after zero point subtraction."
            )
        quant_weight = quant_weight_int.to(torch.int8).contiguous()
        dense_weight = (quant_weight.to(expanded_scale.dtype) * expanded_scale).contiguous()

        reference_weight = dequantize(
            x_q=unpacked,
            scale=scale,
            zero_point=zero_point,
            args=weights,
            g_idx=g_idx,
            dtype=dense_weight.dtype,
        ).contiguous()
        if not torch.allclose(dense_weight.float(), reference_weight.float(), rtol=1e-2, atol=1e-2):
            max_diff = torch.max(torch.abs(dense_weight.float() - reference_weight.float())).item()
            raise RuntimeError(
                f"compressed-tensors dequantization check failed for {module_name}: max_abs_diff={max_diff}"
            )

        if tuple(dense_weight.shape) != (module.out_features, module.in_features):
            raise RuntimeError(
                f"compressed-tensors dense weight shape mismatch for {module_name}: "
                f"weight={tuple(dense_weight.shape)}, expected={(module.out_features, module.in_features)}"
            )

        # 用父节点 setattr 替换模块，得到后续 wrap/export 预期的普通 nn.Linear，
        # 同时保留未缩放整数权重 quant_weight 作为 buffer。
        new_linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=dense_weight.device,
            dtype=dense_weight.dtype,
        )
        new_linear.weight = nn.Parameter(dense_weight, requires_grad=False)
        if module.bias is not None:
            new_linear.bias = nn.Parameter(
                module.bias.detach().to(device=dense_weight.device, dtype=dense_weight.dtype).contiguous(),
                requires_grad=False,
            )
        new_linear.register_buffer("quant_weight", quant_weight)
        return new_linear

    @classmethod
    def _dequantize_compressed_tensors_hf_model(cls, native_hf_model: nn.Module) -> nn.Module:
        hf_model = native_hf_model
        compressed_linears = [
            (name, module)
            for name, module in hf_model.named_modules()
            if isinstance(module, nn.Linear) and hasattr(module, "weight_packed") and hasattr(module, "weight_scale")
        ]
        if not compressed_linears:
            raise NotImplementedError(
                "No compressed-tensors pack-quantized nn.Linear modules with weight_packed/weight_scale were found."
            )

        for module_name, module in tqdm(compressed_linears, desc="Dequantizing compressed-tensors Linear modules"):
            new_linear = cls._convert_compressed_tensors_linear(module_name, module)
            if "." in module_name:
                parent_name, child_name = module_name.rsplit(".", 1)
                parent = hf_model.get_submodule(parent_name)
            else:
                parent = hf_model
                child_name = module_name
            setattr(parent, child_name, new_linear)

        logger = get_xhquant_logger()
        logger.info(f"Converted {len(compressed_linears)} compressed-tensors Linear modules to nn.Linear.")
        return hf_model

    @classmethod
    def _dequantize_autoround_hf_model(cls, native_hf_model: nn.Module):
        hf_model = native_hf_model
        logger = get_xhquant_logger()

        try:
            from auto_round.utils.weight_handler import (
                check_and_mark_quantized_module,
                convert_module_to_hp_if_necessary,
            )

            detected_types = check_and_mark_quantized_module(hf_model)
            if len(detected_types) > 0:
                hf_model = convert_module_to_hp_if_necessary(
                    hf_model, dtype=cls.HF_MODEL_DTYPE, device="cpu", to_cpu=False
                )
        except ImportError:
            logger.warning("auto_round weight_handler is unavailable; falling back to manual QuantLinear conversion.")

        autoround_torch_quantlinear_classes = []
        for module_path in (
            "auto_round_extension.torch.qlinear_torch",
            "auto_round_extension.torch.qlinear_torch_zp",
        ):
            try:
                module = __import__(module_path, fromlist=["QuantLinear"])
                autoround_torch_quantlinear_classes.append(module.QuantLinear)
            except Exception:
                continue

        dequant_linears = []
        if autoround_torch_quantlinear_classes:
            quantlinear_types = tuple(autoround_torch_quantlinear_classes)
            for _, module in hf_model.named_modules():
                if isinstance(module, quantlinear_types):
                    dequant_linears.append(module)
        memory_info = get_cpu_memory_mb()
        logger.info(f"Found {len(dequant_linears)} AutoRound torch QuantLinear modules to dequantize.")

        logger.info(f"Initial CPU memory usage before autoround dequantization: {str(memory_info)}")

        for _, module in enumerate(tqdm(dequant_linears, desc="Dequantizing AutoRound QuantLinear modules")):
            autoround_torch_qlinear_converter(module)
        dequant_linears.clear()

        autoround_moe_block_names = [
            module_name
            for module_name, module in hf_model.named_modules()
            if type(module).__name__ == "LinearQwen3_5MoeSparseMoeBlock"
        ]
        logger.info(
            "Found %d AutoRound LinearQwen3_5MoeSparseMoeBlock modules to restore.",
            len(autoround_moe_block_names),
        )
        for module_name in autoround_moe_block_names:
            module = hf_model.get_submodule(module_name) if module_name else hf_model
            restored_module = restore_autoround_qwen3_5_moe_sparse_block(module)
            if module_name:
                parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
                parent = hf_model.get_submodule(parent_name) if parent_name else hf_model
                setattr(parent, child_name, restored_module)
            else:
                hf_model = restored_module

        autoround_moe_block_names.clear()
        gc.collect()

        memory_info = get_cpu_memory_mb()
        logger.info(f"Initial CPU memory usage After autoround dequantization: {str(memory_info)}")

        hf_model.quantization_method = None  # type: ignore
        hf_model._is_hf_initialized = False  # type: ignore
        return hf_model

    @classmethod
    def _dequantize_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs):
        # abandoned function
        raise ValueError(
            """
            The _dequantize_hf_model method is deprecated.If model structure is not modified after quantization, 
            get_hf_model() will automatically dequantize the model. If model structure adjustment is needed after 
            loading gptqmodel, please implement your own _postprocess_gptqmodel_structure method. 
            ref: xhmodel_merak/xh_llm/models/glm_4_moe_lite/glm_4_moe_lite_model.py
            """
        )
        return None

    @classmethod
    def _postprocess_gptqmodel_structure(cls, native_hf_model: nn.Module, **kwargs) -> nn.Module:
        return native_hf_model

    @classmethod
    def dequantize_hf_model(cls, hf_model, quantization_config):
        quant_method = cls._get_quantization_method(quantization_config)
        if quant_method == "awq":
            hf_model = cls._dequantize_awq_hf_model(hf_model)
        elif quant_method == "gptq":
            hf_model = cls._dequantize_gptq_hf_model(hf_model)
        elif quant_method == "compressed-tensors":
            hf_model = cls._dequantize_compressed_tensors_hf_model(hf_model)
        elif quant_method in {"auto-round", "auto_round", "autoround"}:
            hf_model = cls._dequantize_autoround_hf_model(hf_model)
        else:
            raise NotImplementedError(f"Dequantize not implemented for quantization method: {quant_method}")
        return hf_model

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
        quantization_config = (
            config.get("quantization_config", None)
            if isinstance(config, dict)
            else getattr(config, "quantization_config", None)
        )
        quant_method = cls._get_quantization_method(quantization_config)

        if quantization_config is None:
            native_hf_model = cls._load_hf_model(hf_model_dir, **kwargs)
            if quant_weight is not None and len(quant_weight) > 0:
                cls._load_quant_weight(quant_weight, native_hf_model)
            return native_hf_model

        if quant_weight is not None and len(quant_weight) > 0:
            raise RuntimeError(
                "Model is already quantized, quant_weight should be None or empty when loading quantized model."
            )

        if quant_method == "awq":
            hf_model = cls._load_hf_model(hf_model_dir, **kwargs)
            hf_model = cls._dequantize_awq_hf_model(hf_model)
        elif quant_method == "gptq":
            """
            GPTQModel在量化某些模型时，会调整模型结构，这导致gptqmodel量化模型和原始hf模型结构不一致，
            无法进行后续Wrap和Fronted等转换，_postprocess_gptqmodel_structure函数的作用是对伪量化后
            的gptqmodel模型结构进行调整，使其与原始hf模型结构一致。如果适配的新模型也存在类似问题，需要
            实现自己的_postprocess_gptqmodel_structure函数，覆盖基类行为
            """
            hf_model = cls._load_gptqmodel(hf_model_dir, **kwargs)
            hf_model = cls._dequantize_gptqmodel_hf_model(hf_model)
            hf_model = cls._postprocess_gptqmodel_structure(hf_model, **kwargs)
        elif quant_method == "compressed-tensors":
            """
            适配comrpessed-tensors量化模型
            """
            # TODO: 量化时改变结构的MoE compressed-tensors量化模型加载权重missing/unexpected问题
            hf_model = cls._load_hf_model(hf_model_dir, **kwargs)
            hf_model = cls._dequantize_compressed_tensors_hf_model(hf_model)
        elif quant_method in {"auto-round", "auto_round", "autoround"}:
            hf_model = cls._load_hf_model(hf_model_dir, **kwargs)
            hf_model = cls._dequantize_autoround_hf_model(hf_model)
        else:
            raise NotImplementedError(f"Dequantize not implemented for quantization method: {quant_method}")
        return hf_model

    def get_native_model(self):
        resume_from = self.config.quant_weight
        kwargs = {}
        if self.config.enable_auto_offload:
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = "cpu"
        native_hf_model = self.get_hf_model(
            self.hf_model_dir,
            quant_weight=resume_from,
            dtype=self.dtype,
            **kwargs,
        )
        hf_model_cls = self.get_hf_model_cls()
        assert isinstance(native_hf_model, hf_model_cls), (
            f"The model is not {hf_model_cls.__name__}, but {type(native_hf_model)}"
        )

        return native_hf_model

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        """
        仅仅加载模型结构,不初始化权重,不占用显存
        """
        from accelerate import init_empty_weights

        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            # Fallback for older transformers versions
            no_init_weights = init_empty_weights

        config = AutoConfig.from_pretrained(hf_model_dir)
        with no_init_weights(), init_empty_weights():
            auto_model_cls = cls.HF_AUTO_MODEL_CLS
            model_dtype = cls.HF_MODEL_DTYPE
            if "dtype" not in kwargs:
                kwargs["dtype"] = model_dtype
            hf_model: nn.Module = auto_model_cls.from_config(
                config,
                **kwargs,
            )
            if hf_model.can_generate():
                try:
                    hf_model.generation_config = GenerationConfig.from_pretrained(hf_model_dir)
                except OSError:
                    logger = get_xhquant_logger()
                    logger.info(
                        "Generation config file not found, using a generation config created from the model config."
                    )
                pass
        return hf_model

    def get_empty_native_model(self):
        native_hf_model = self.get_empty_hf_model(self.hf_model_dir, dtype=self.dtype)
        hf_model_cls = self.get_hf_model_cls()
        assert isinstance(native_hf_model, hf_model_cls), (
            f"The model is not {hf_model_cls.__name__}, but {type(native_hf_model)}"
        )
        return native_hf_model

    @classmethod
    def get_compatible_model(cls, hf_model_dir, **kwargs):
        return cls.get_empty_hf_model(hf_model_dir, **kwargs)

    def get_compatible_native_model(self):
        return self.get_compatible_model(self.hf_model_dir, dtype=self.dtype)

    ### 模型推理
    def _set_inference_model(self):
        inference_model = self.get_inference_model()
        current_inference_model = self._inference_model if hasattr(self, "_inference_model") else None
        source_model = None
        if current_inference_model is not None:
            if isinstance(current_inference_model, FXInterpreter):
                source_model = current_inference_model.module
            elif isinstance(current_inference_model, AutoOffloadGraphModel):
                source_model = current_inference_model

        if source_model is not None and source_model == inference_model:
            return
        enable_auto_offload = self.config.enable_auto_offload
        if self._state in [LLMModelState.EAGER_ALIGNED, LLMModelState.EAGER_FAST, LLMModelState.WRAP]:
            if self.config.enable_auto_offload:
                hf_auto_offload(inference_model, dtype=self.dtype)
        elif self._state in [
            LLMModelState.FRONTED,
            LLMModelState.QUANTED_DISABLE,
            LLMModelState.QUANTED_FAST,
            LLMModelState.QUANTED_ALIGNED,
        ]:
            if enable_auto_offload:
                auto_offload_max_memory = self.get_wrap_cfg().get("auto_offload_max_memory", None)
                inference_model = AutoOffloadGraphModel.from_graph_model(
                    inference_model, max_memory=auto_offload_max_memory
                )
            else:
                inference_model.to(device=self.device, dtype=self.dtype)
                if self.interactive_mode:
                    assert isinstance(inference_model, fx.GraphModule)
                    inference_model = FXInterpreter(inference_model)

        if self._inference_model is not None:
            if AutoOffloadGraphModel.is_auto_offload_model(self._inference_model):
                AutoOffloadGraphModel.remove_auto_offload_model(self._inference_model)
                self._inference_model = None
        self._inference_model = inference_model

    def prepare_for_inference(self, *args, **kwargs):
        for _, sub_model in self._models.items():
            sub_model.prepare_for_inference()
        inference_model = self.get_inference_model()
        if self._inference_model is not None and self._inference_model == inference_model:
            return
        if self._state in [LLMModelState.EAGER_ALIGNED, LLMModelState.EAGER_FAST, LLMModelState.WRAP]:
            if self.config.enable_auto_offload:
                hf_auto_offload(inference_model, dtype=self.dtype)
        elif self._state in [
            LLMModelState.FRONTED,
            LLMModelState.QUANTED_DISABLE,
            LLMModelState.QUANTED_FAST,
            LLMModelState.QUANTED_ALIGNED,
        ]:
            if self.config.enable_auto_offload:
                auto_offload_max_memory = self.get_wrap_cfg().get("auto_offload_max_memory", None)
                if isinstance(inference_model, ModelSwitcher):
                    raise RuntimeError(
                        "Auto offload is not supported for ModelSwitcher yet, please disable auto offload or use a single model."
                    )

                inference_model = AutoOffloadGraphModel.from_graph_model(
                    inference_model, max_memory=auto_offload_max_memory
                )
            else:
                if self.interactive_mode:
                    if isinstance(inference_model, ModelSwitcher):
                        inference_model = SwitchFXInterpreter(inference_model)
                    else:
                        assert isinstance(inference_model, fx.GraphModule)
                        inference_model = FXInterpreter(inference_model)

        self._inference_model = inference_model

    def release_inference_model(self):
        for _, sub_model in self._models.items():
            sub_model.release_inference_model()
        if self._inference_model is not None:
            inference_model = self._inference_model
            if AutoOffloadGraphModel.is_auto_offload_model(inference_model):
                AutoOffloadGraphModel.remove_auto_offload_model(inference_model)
            elif isinstance(inference_model, FXInterpreter):
                pass
            self._inference_model = None

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        raise NotImplementedError("Subclasses of BaseLLMModel must implement _get_data_preprocessor method.")

    def get_data_preprocessor(self) -> BaseLLMInputProcessor:
        """对输入做处理，返回计算图需要的输入"""
        if self._data_processor is None:
            self._data_processor = self._get_data_preprocessor()
        self._data_processor.to(self.device, self.dtype)
        return self._data_processor

    def __call__(self, *args, **kwargs) -> Any:
        infer_model = None
        if self._state == LLMModelState.NONE:
            raise RuntimeError("Model is not ready for generation, please set state to fronted or quanted.")
        self._set_inference_model()
        assert self._inference_model is not None, (
            "Inference model is not prepared, please call prepare_for_inference first."
        )
        if self.enable_hf_compatible:
            if self.hf_compatible_model is None:
                hf_model = self.get_empty_hf_model(
                    self.hf_model_dir,
                    dtype=self.dtype,
                )
                # 从类中直接获取函数，避免自动绑定 self
                hf_compatible_model = type(self).build_hf_compatible_model(hf_model, self)
                assert isinstance(hf_compatible_model, self.get_hf_model_cls())
                hf_compatible_model.to(device=self.device, dtype=self.dtype)
                self.hf_compatible_model = hf_compatible_model
            infer_model = self.hf_compatible_model
        else:
            infer_model = self
        out = infer_model.forward(*args, **kwargs)
        return out

    def forward(self, *args, **kwargs):
        raise RuntimeError("Direct call is not allowed for BaseModel, please use generate method for inference.")

    def generate(self, *args, **kwargs):
        infer_model = None
        if self._state == LLMModelState.NONE:
            raise RuntimeError("Model is not ready for generation, please set state to fronted or quanted.")
        elif self._state in [LLMModelState.EAGER_FAST, LLMModelState.EAGER_ALIGNED]:
            infer_model = self._wrap_model
        else:
            if self.hf_compatible_model is None:
                hf_model = self.get_compatible_native_model()
                # 从类中直接获取函数，避免自动绑定 self
                hf_compatible_model = type(self).build_hf_compatible_model(hf_model, self)
                assert isinstance(hf_compatible_model, self.get_hf_model_cls()), (
                    f"Expected hf compatible model of type {self.get_hf_model_cls().__name__}, but got {type(hf_compatible_model).__name__}"
                )
                if not self.config.enable_auto_offload:
                    hf_compatible_model.to(device=self.device, dtype=self.dtype)
                self.hf_compatible_model = hf_compatible_model
            infer_model = self.hf_compatible_model
        assert infer_model is not None
        out = infer_model.generate(*args, **kwargs)
        return out

    @classmethod
    def build_hf_compatible_model(cls, hf_model: nn.Module, *args, **kwargs) -> nn.Module:
        return cls.BUILD_HF_COMPATIBLE_FUNC(hf_model, *args, **kwargs)

    def get_inference_model(self):
        inference_model = None
        if self._state == LLMModelState.WRAP:
            inference_model = self._wrap_model
        elif self._state == LLMModelState.FRONTED:
            inference_model = self._frontend_model
        elif self._state in [LLMModelState.QUANTED_FAST, LLMModelState.QUANTED_ALIGNED, LLMModelState.QUANTED_DISABLE]:
            inference_model = self._quanted_model
        elif self._state == LLMModelState.EXPORTED:
            inference_model = self._exported_model
        elif self._state in [LLMModelState.EAGER_FAST, LLMModelState.EAGER_ALIGNED]:
            inference_model = self._wrap_model
        elif self._state == LLMModelState.NONE:
            pass
        else:
            raise RuntimeError(f"Invalid model state: {self._state}")
        if isinstance(inference_model, ModelSwitcher):
            inference_model = inference_model.activate_model
        return inference_model

    def _set_device(self, device: torch.device | str | None):
        self._device = device
        if self.config.enable_auto_offload:
            return
        inference_model = self.get_inference_model()
        if inference_model is not None:
            if device is not None:
                self._device = device
                if self._state in [
                    LLMModelState.WRAP,
                    LLMModelState.EAGER_FAST,
                    LLMModelState.EAGER_ALIGNED,
                    LLMModelState.FRONTED,
                    LLMModelState.QUANTED_DISABLE,
                    LLMModelState.QUANTED_FAST,
                    LLMModelState.QUANTED_ALIGNED,
                ]:
                    inference_model.to(device)

    def _set_dtype(self, dtype: torch.dtype | str | None):
        if dtype is None:
            return
        self._dtype = resolve_model_dtype(dtype)
        inference_model = self.get_inference_model()
        if inference_model is None:
            return
        if self._state in [
            LLMModelState.WRAP,
            LLMModelState.EAGER_FAST,
            LLMModelState.EAGER_ALIGNED,
        ]:
            if not self.config.enable_auto_offload:
                inference_model = inference_model.to(self._dtype)
        else:
            inference_model = inference_model.to(self._dtype)

    # 模型导出
    @classmethod
    def get_export_metadata_cls(cls):
        return cls.META_CLS

    def get_export_cfg(self) -> dict[str, list[str]]:
        raise NotImplementedError(
            "Subclasses of BaseLLMModel must implement get_export_cfg method to provide export configuration."
        )

    @log_function_call()
    def _export_hmonnx(self, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.config.work_dir = str(output_dir)
        model_name = self.config.model_name
        if model_name is None or len(model_name) == 0:
            raise ValueError("Model name is not specified in config, please set model_name in config before exporting.")
        output_hmonnx_file = str(Path(output_dir) / f"{self.config.model_name}.onnx")
        logger = get_xhquant_logger()
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.fixed()

        data_processor = self.get_data_preprocessor()
        dummy_inputs = self.get_dummy_inputs()
        assert isinstance(dummy_inputs, (dict,)), "Dummy inputs should be a dictionary, but get {type(dummy_inputs)}."
        inputs = data_processor(dummy_inputs)
        assert isinstance(
            inputs,
            (list, tuple),
        ), "Processed dummy inputs should be a list, tuple, but get {type(inputs)}."
        inputs = unfold_args(inputs)
        if not self._quanted_model.is_fixed():
            raise ValueError("Quantized model is not fixed, Please call `fixed` first.")
        self._quanted_model.to("cpu")
        prefill_exported_model = to_export_graph(self._quanted_model, inputs)

        logger.info(f"Exporting to HMONNX format for {model_name} .........")

        # 导出的文件格式必须是 hmquant_{model_name}_{prefill/decode}_with_act.onnx
        export_cfg = self.get_export_cfg()
        # normalize_onnx_name=True, 会修改导出的hmonnx文件路径
        exported_hmonnx_file = to_export_hmonnx_v2(
            prefill_exported_model, inputs, str(output_hmonnx_file), export_cfg, normalize_onnx_name=True
        )
        return exported_hmonnx_file

    def export_hmonnx(self) -> ModelMeta:
        raise NotImplementedError(
            "HMONNX export is not implemented for BaseLLMModel, please implement _export_hmonnx method in subclass."
        )


class XHSubModel(XHBaseModel):
    def get_compatible_native_model(self):
        # 单独调试辅助子模型时，不能加载空模型
        hf_model = super().get_native_model()
        if self.config.enable_auto_offload:
            hf_auto_offload(hf_model, dtype=self.dtype)
        return hf_model
