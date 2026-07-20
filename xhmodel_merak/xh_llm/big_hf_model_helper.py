"""超大 Hugging Face 模型的低峰值内存 HMONNX 分层导出工具。

本模块负责按 placeholder 模块边界拆分主图和子图、从 safetensors 按需物化权重、
分别量化并导出子图，以及为后续将子图回填到 HMONNX 主图准备节点契约信息。
"""

import copy
import glob
import hashlib
import inspect
import json
import multiprocessing as mp
import operator
import os
import re
import shutil
import threading
from collections import defaultdict
from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Type

import onnx
import torch
import torch.nn as nn
from safetensors import safe_open
from torch import Tensor
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.quantizers.auto import AutoHfQuantizer

from xhquant.api import (
    FrontendGraph,
    PrecisionMode,
    get_xhquant_logger,
    ptq_quantize,
    to_export_hmonnx_from_quanted_graph,
    to_frontend_graph,
    to_quant_graph,
)
from xhquant.core.graph.graph_module_pipe import annotate_pipe_split
from xhquant.utils.registry import DynamicModule, _DMRegistryCls

from ._dequant_converter import ensure_gptqmodel_unpack_buffers
from .base_model import XHBaseModel
from .wrap_model import wrap_llm_model


_FX_PLACEHOLDER_REGISTRY_LOCK = threading.RLock()
_PROCESS_PLACEHOLDER_EXPORTER = None
_PROCESS_EMPTY_HF_MODEL = None
_KEEP_EXPORT_TMP_ENV = "XH2MODELZOO_KEEP_EXPORT_TMP"
_PLACEHOLDER_EXPORT_OOM_HINT = "Reduce XH2MODELZOO_EXPORT_WORKERS or make more GPUs visible to the export process."


def _placeholder_export_worker_count(value: Any = None) -> int:
    """Resolve placeholder subgraph export parallelism."""
    if value is None:
        value = os.environ.get("XH2MODELZOO_EXPORT_WORKERS", 1)
    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid placeholder export worker count: {value!r}") from exc
    return max(1, workers)


def _cleanup_big_model_export_temporary_files(value: Any = None) -> bool:
    """Resolve whether successful big-model exports remove temporary files."""
    from_environment = value is None
    if value is None:
        value = os.environ.get(_KEEP_EXPORT_TMP_ENV, "0")
    if isinstance(value, bool):
        return not value if from_environment else value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        enabled = True
    elif normalized in {"0", "false", "no", "off"}:
        enabled = False
    else:
        raise ValueError(
            f"Invalid {_KEEP_EXPORT_TMP_ENV} value: {value!r}; expected one of 1/true/yes/on or 0/false/no/off."
        )
    return not enabled if from_environment else enabled


def _placeholder_export_execution_devices() -> list[str]:
    """Return visible devices used to spread placeholder export worker tasks."""
    if torch.cuda.is_available():
        return [f"cuda:{device_index}" for device_index in range(torch.cuda.device_count())]
    return ["cpu"]


def _set_placeholder_export_execution_device(device: str | torch.device | None) -> str:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        if resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(resolved_device)
    return str(resolved_device)


def _is_cuda_oom_exception(exc: BaseException) -> bool:
    visited = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        message = str(current).lower()
        if "cuda" in message and "out of memory" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _serializable_meta_template(value: Any) -> Any:
    """Convert FX/FakeTensor arguments into spawn-safe plain meta tensors."""
    value = _node_arg_meta_value(value)
    if value is None:
        return None
    if isinstance(value, Tensor) or (hasattr(value, "shape") and hasattr(value, "dtype")):
        return torch.empty(tuple(int(dim) for dim in value.shape), dtype=value.dtype, device="meta")
    if isinstance(value, tuple):
        return tuple(_serializable_meta_template(item) for item in value)
    if isinstance(value, list):
        return [_serializable_meta_template(item) for item in value]
    return value


class WeightMapping:
    """记录模型权重文件位置以及 meta tensor 与参数名之间的映射。"""

    weight_dir: str
    weight_map: dict[str, str]
    tensor_id_to_param_names: dict[int, str]


class GPTQModelQuantizedModelPreprocessor:
    """Apply GPTQModel's quantized module layout to an existing meta HF model.

    The input model has already been constructed by ``get_empty_hf_model``.  This
    helper only applies GPTQModel's model-definition/Defuser rules and replaces
    checkpoint-backed ``Linear`` modules with the per-layer QuantLinear selected
    by GPTQModel.  Packed weights remain on meta and are materialized later by
    ``BigHFModelExportHelper`` one placeholder at a time.
    """

    CONFIG_MARKER = "_xh_gptqmodel_custom_preprocessed"

    def __init__(self, model_dir: str | Path, *, dtype: torch.dtype = torch.float16) -> None:
        self.model_dir = str(Path(model_dir))
        self.dtype = dtype
        try:
            import defuser
            from gptqmodel import QuantizeConfig
            from gptqmodel.models._const import DEVICE
            from gptqmodel.models.auto import check_and_get_model_definition
            from gptqmodel.quantization.config import dynamic_get
            from gptqmodel.utils.backend import BACKEND
            from gptqmodel.utils.importer import select_quant_linear
        except ImportError as exc:
            raise ImportError(
                "GPTQ quantized big-model loading requires GPTQModel and Defuser. "
                "AWQ and AutoRound do not use these dependencies."
            ) from exc

        self._defuser = defuser
        self._device = DEVICE
        self._backend = BACKEND
        self._dynamic_get = dynamic_get
        self._select_quant_linear = select_quant_linear
        self.quant_config = QuantizeConfig.from_pretrained(self.model_dir)
        self.model_definition = check_and_get_model_definition(
            self.model_dir,
            trust_remote_code=True,
        )
        self._candidate_cache: dict[tuple[Any, ...], list[type[nn.Module]]] = {}
        self._class_cache: dict[tuple[Any, ...], type[nn.Module]] = {}

    def preprocess(
        self,
        hf_model: PreTrainedModel,
        skip_module_prefixes: Optional[list[str]] = None,
    ) -> tuple[str, ...]:
        """Mutate an already-empty HF model into GPTQModel's meta module tree."""
        with torch.device("meta"):
            self.model_definition.before_model_load(
                self.model_definition,
                load_quantized_model=True,
            )
            self._defuser.convert_model(hf_model, cleanup_original=True)
            replaced = self._replace_quant_linears(hf_model, skip_module_prefixes=skip_module_prefixes)

        hf_model.config.quantization_config = self.quant_config
        setattr(hf_model.config, self.CONFIG_MARKER, True)
        hf_model._meta_quantized_module_count = len(replaced)
        hf_model._meta_quantized_module_names = tuple(replaced)
        return tuple(replaced)

    def _checkpoint_quantized_modules(self) -> set[str]:
        index_path = Path(self.model_dir) / "model.safetensors.index.json"
        if index_path.is_file():
            with index_path.open(encoding="utf-8") as fin:
                tensor_names = json.load(fin).get("weight_map", {}).keys()
                module_names = {name.removesuffix(".qweight") for name in tensor_names if name.endswith(".qweight")}
        else:
            module_names = set()
            for checkpoint_path in Path(self.model_dir).glob("*.safetensors"):
                with safe_open(checkpoint_path, framework="pt", device="cpu") as reader:
                    module_names.update(
                        name.removesuffix(".qweight") for name in reader.keys() if name.endswith(".qweight")
                    )

        normalize = getattr(self.model_definition, "normalize_quantized_module_names", None)
        if normalize is not None:
            module_names = set(normalize(module_names))
        if not module_names:
            raise RuntimeError(f"No GPTQ qweight tensors were found under {self.model_dir!r}.")
        return module_names

    def _model_config(self, hf_model: PreTrainedModel):
        config = hf_model.config
        sub_configs = getattr(config, "sub_configs", {}) or {}
        if getattr(self.model_definition, "config_class", None) == sub_configs.get("text_config"):
            return config.get_text_config()
        return config

    def _allowed_modules(self, hf_model: PreTrainedModel) -> set[str]:
        simple_layer_modules = self.model_definition.simple_layer_modules(
            self._model_config(hf_model),
            self.quant_config,
        )
        allowed_suffixes = {suffix for module_group in simple_layer_modules for suffix in module_group}
        layer_prefixes = tuple(self.model_definition.extract_layers_node())
        ignored_prefixes = tuple(
            [self.model_definition.lm_head] + list(self.model_definition.get_base_modules(hf_model))
        )

        allowed = set()
        for module_name, module in hf_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.quant_config.lm_head and module_name == self.model_definition.lm_head:
                allowed.add(module_name)
                continue
            if not module_name.startswith(layer_prefixes):
                continue
            if module_name.startswith(ignored_prefixes):
                continue
            if any(module_name.endswith(suffix) for suffix in allowed_suffixes):
                allowed.add(module_name)
        return allowed

    def _effective_quant_config(self, module_name: str) -> dict[str, Any] | None:
        overrides = self._dynamic_get(self.quant_config.dynamic, module_name=module_name)
        if overrides is False:
            return None
        overrides = overrides or {}
        return {
            "bits": overrides.get("bits", self.quant_config.bits),
            "group_size": overrides.get("group_size", self.quant_config.group_size),
            "desc_act": overrides.get("desc_act", self.quant_config.desc_act),
            "sym": overrides.get("sym", self.quant_config.sym),
            "pack_dtype": overrides.get("pack_dtype", self.quant_config.pack_dtype),
        }

    def _quant_linear_class(
        self,
        module_name: str,
        linear: nn.Linear,
        effective: dict[str, Any],
    ) -> type[nn.Module]:
        config_key = (
            effective["bits"],
            effective["group_size"],
            effective["desc_act"],
            effective["sym"],
            effective["pack_dtype"],
            self.quant_config.format,
            self.quant_config.quant_method,
        )
        class_key = (*config_key, linear.in_features, linear.out_features)
        if class_key in self._class_cache:
            return self._class_cache[class_key]

        candidates = self._candidate_cache.get(config_key)
        if candidates is None:
            candidates = self._select_quant_linear(
                **effective,
                backend=self._backend.AUTO,
                format=self.quant_config.format,
                quant_method=self.quant_config.quant_method,
                pack=False,
                dynamic=None,
                device=self._device.CPU,
                dtype=self.dtype,
                multi_select=True,
                adapter=self.quant_config.adapter,
            )
            self._candidate_cache[config_key] = candidates

        for linear_cls in candidates:
            valid, _ = linear_cls.validate(
                **effective,
                in_features=linear.in_features,
                out_features=linear.out_features,
                device=self._device.CPU,
                dtype=self.dtype,
                adapter=self.quant_config.adapter,
            )
            if valid:
                self._class_cache[class_key] = linear_cls
                return linear_cls
        raise ValueError(f"GPTQModel found no compatible QuantLinear for {module_name!r}: {effective}.")

    def _new_quant_linear(self, module_name: str, linear: nn.Linear) -> nn.Module:
        effective = self._effective_quant_config(module_name)
        if effective is None:
            raise ValueError(f"Module is disabled by GPTQ dynamic config: {module_name!r}.")
        linear_cls = self._quant_linear_class(module_name, linear, effective)
        quant_linear = linear_cls(
            **effective,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            name=module_name,
            lm_head_name=self.model_definition.lm_head,
            backend=self._backend.AUTO,
            register_buffers=True,
            adapter=self.quant_config.adapter,
        )
        quant_linear.device = torch.device("meta")
        return quant_linear

    @staticmethod
    def _is_within_module_prefix(module_name: str, module_prefixes: tuple[str, ...]) -> bool:
        for prefix in module_prefixes:
            if prefix == "":
                return True
            if module_name == prefix or module_name.startswith(prefix + "."):
                return True
        return False

    def _replace_quant_linears(
        self,
        hf_model: PreTrainedModel,
        skip_module_prefixes: Optional[list[str]] = None,
    ) -> list[str]:
        checkpoint_modules = self._checkpoint_quantized_modules()
        allowed_modules = self._allowed_modules(hf_model)
        skip_prefixes = tuple(skip_module_prefixes or [])
        invalid: list[str] = []
        replaced: list[str] = []

        for module_name in sorted(checkpoint_modules):
            if skip_prefixes and self._is_within_module_prefix(module_name, skip_prefixes):
                continue
            if module_name not in allowed_modules:
                invalid.append(f"{module_name} (excluded by module_tree)")
                continue
            try:
                linear = hf_model.get_submodule(module_name)
            except AttributeError:
                invalid.append(f"{module_name} (missing)")
                continue
            if not isinstance(linear, nn.Linear):
                invalid.append(f"{module_name} ({type(linear).__name__})")
                continue
            if self._effective_quant_config(module_name) is None:
                invalid.append(f"{module_name} (disabled by dynamic config)")
                continue
            hf_model.set_submodule(module_name, self._new_quant_linear(module_name, linear))
            replaced.append(module_name)

        if invalid:
            raise RuntimeError(
                f"{len(invalid)} checkpoint QuantLinear modules do not match GPTQModel's empty-model rules: "
                + ", ".join(invalid[:20])
            )
        return replaced

    @classmethod
    @torch.inference_mode()
    def prepare_loaded_module(
        cls,
        module: nn.Module,
        quant_config: Any,
        skip_module_prefixes: Optional[list[str]] = None,
    ) -> int:
        """Convert loaded GPTQ v1 qzeros to the runtime v2 representation."""
        try:
            from gptqmodel.nn_modules.qlinear import BaseQuantLinear
            from gptqmodel.quantization.config import FORMAT
            from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module
        except ImportError as exc:
            raise ImportError("Preparing GPTQModel weights requires GPTQModel.") from exc

        checkpoint_format = getattr(quant_config, "format", None)
        checkpoint_format = getattr(checkpoint_format, "value", checkpoint_format)
        if str(checkpoint_format).lower() != str(FORMAT.GPTQ.value).lower():
            return 0

        skip_prefixes = tuple(skip_module_prefixes or [])
        converted = 0
        for module_name, submodule in module.named_modules():
            if skip_prefixes and cls._is_within_module_prefix(module_name, skip_prefixes):
                continue
            if not isinstance(submodule, BaseQuantLinear):
                continue
            qzeros = getattr(submodule, "qzeros", None)
            # A meta QuantLinear is still an unloaded skeleton.  Marking its
            # format as v2 now would make a later streamed v1 checkpoint load
            # skip the required conversion.
            if qzeros is None or getattr(qzeros, "is_meta", False):
                continue
            if not submodule.REQUIRES_FORMAT_V2 or submodule.qzero_format() == 2:
                continue
            # Keep this identical to GPTQModel.load(): its checkpoint-level
            # v1 -> v2 conversion uses QuantizeConfig.bits for every dynamic
            # QuantLinear, rather than the per-layer override.  Using the
            # module's effective bits here changes every dynamic qzeros buffer
            # and makes the streamed model differ from GPTQModel's runtime
            # representation.
            convert_gptq_v1_to_v2_format_module(
                module=submodule,
                bits=int(quant_config.bits),
                pack_dtype=quant_config.pack_dtype,
            )
            converted += 1
        return converted


def load_weight_from_safetensor(weight_map: dict[str, str], param_names: list[str]) -> list[Tensor]:
    """按请求顺序从一个或多个 safetensors 分片中加载权重到 CPU。

    先按分片文件聚合参数名，避免为同一个分片反复打开文件；返回值顺序严格与
    ``param_names`` 一致，便于调用方直接与参数名列表配对。
    """
    parameters_files = defaultdict(list)
    logger = get_xhquant_logger()
    for param_name in param_names:
        file = weight_map[param_name]
        parameters_files[file].append(param_name)

    parameters = {}
    for file_path, file_p_names in parameters_files.items():
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for full_param_name in file_p_names:
                tensor = f.get_tensor(full_param_name)
                logger.info(f"Loaded {full_param_name}: {tensor.shape}, dtype: {tensor.dtype}")
                parameters[full_param_name] = tensor

    return [parameters[p_name] for p_name in param_names]


def _flatten_tensor_meta_values(value: Any) -> list[Any]:
    """递归提取输出 meta 中所有 tensor-like 叶子，用于生成输出契约。"""
    if value is None:
        return []
    if isinstance(value, Tensor):
        return [value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return [value]
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_flatten_tensor_meta_values(item))
        return values
    return []


def _node_arg_meta_value(value: Any) -> Any:
    """将 FX Node 参数解引用为其 ``meta['val']``，普通值保持不变。"""
    meta = getattr(value, "meta", None)
    if isinstance(meta, dict) and "val" in meta:
        return meta["val"]
    return value


def _flatten_input_fake_tensor_values(value: Any) -> list[Any]:
    """提取节点输入中的 fake tensor，并保留 list 类型缓存作为单个逻辑输入。"""
    value = _node_arg_meta_value(value)
    if value is None:
        return []
    if isinstance(value, Tensor):
        return [value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return [value]
    if isinstance(value, list):
        # A cache list is one logical call argument, but its FX Nodes still
        # need to be resolved to their fake-tensor metadata before being used
        # as example inputs for the standalone subgraph.
        return [[_node_arg_meta_value(item) for item in value]]
    if isinstance(value, tuple):
        values = []
        for item in value:
            values.extend(_flatten_input_fake_tensor_values(item))
        return values
    return []


def _placeholder_input_fake_tensors(args: Any) -> list[Any]:
    """按照原调用参数层级构造子图 tracing 所需的 fake tensor 输入模板。

    list 通常表示一个需要整体传入 wrapper 的缓存参数，因此作为一层结构保留；
    tuple 则对应位置参数集合，需要递归重建其中的每个参数。
    """
    return _flatten_input_fake_tensor_values(args)


def _rebuild_value_from_flat_inputs(template: Any, flat_inputs: tuple[Any, ...], cursor: list[int]) -> Any:
    value = _node_arg_meta_value(template)
    if value is None:
        return None
    if isinstance(value, Tensor) or (hasattr(value, "shape") and hasattr(value, "dtype")):
        if cursor[0] >= len(flat_inputs):
            raise RuntimeError("Not enough flattened inputs to rebuild placeholder call arguments.")
        rebuilt = flat_inputs[cursor[0]]
        cursor[0] += 1
        return rebuilt
    if isinstance(value, tuple):
        return tuple(_rebuild_value_from_flat_inputs(item, flat_inputs, cursor) for item in value)
    if isinstance(value, list):
        if cursor[0] >= len(flat_inputs):
            raise RuntimeError("Not enough flattened inputs to rebuild placeholder call arguments.")
        rebuilt = flat_inputs[cursor[0]]
        cursor[0] += 1
        return rebuilt
    return value


class _FlattenedInputAdapter(nn.Module):
    """Expose flat graph inputs while rebuilding the placeholder's nested call."""

    def __init__(self, module: nn.Module, args_template: Any, expected_outputs: int | None = None):
        super().__init__()
        self.module = module
        self.__dict__["args_template"] = args_template
        self.__dict__["num_flat_inputs"] = len(_placeholder_input_fake_tensors(args_template))
        self.__dict__["expected_outputs"] = expected_outputs

    @staticmethod
    def _adapt_outputs(outputs: Any, expected_outputs: int | None) -> Any:
        if expected_outputs is None or expected_outputs < 0:
            return outputs
        if expected_outputs == 1:
            if isinstance(outputs, (tuple, list)):
                if not outputs:
                    raise RuntimeError("Placeholder module returned no outputs; expected 1.")
                return outputs[0]
            return outputs
        if not isinstance(outputs, (tuple, list)):
            raise RuntimeError(f"Placeholder module returned a single output; expected {expected_outputs}.")
        if len(outputs) < expected_outputs:
            raise RuntimeError(f"Placeholder module returned {len(outputs)} outputs; expected {expected_outputs}.")
        return tuple(outputs[:expected_outputs])

    def _forward_flat(self, flat_inputs):
        cursor = [0]
        rebuilt_args = tuple(_rebuild_value_from_flat_inputs(item, flat_inputs, cursor) for item in self.args_template)
        if cursor[0] != self.num_flat_inputs:
            raise RuntimeError(f"Unused flattened placeholder inputs: consumed {cursor[0]} of {self.num_flat_inputs}.")
        outputs = self.module(*rebuilt_args)
        return self._adapt_outputs(outputs, self.expected_outputs)

    def forward(self, *flat_inputs):
        return self._forward_flat(flat_inputs)


def _make_flattened_input_adapter(
    module: nn.Module,
    args_template: Any,
    expected_outputs: int | None = None,
) -> _FlattenedInputAdapter:
    """Create an adapter whose concrete forward signature matches its flat inputs."""
    adapter = _FlattenedInputAdapter(module, args_template, expected_outputs)
    num_inputs = adapter.num_flat_inputs
    if num_inputs > 16:
        raise RuntimeError(f"Placeholder has {num_inputs} flat inputs; at most 16 are supported.")
    arg_names = [f"arg{idx}" for idx in range(num_inputs)]
    signature = ", ".join(["self", *arg_names])
    tuple_expr = ", ".join(arg_names)
    if num_inputs == 1:
        tuple_expr += ","
    namespace: dict[str, Any] = {}
    exec(f"def forward({signature}):\n    return self._forward_flat(({tuple_expr}))", {}, namespace)
    adapter.__class__ = type(
        f"FlattenedInputAdapter{num_inputs}",
        (_FlattenedInputAdapter,),
        {"forward": namespace["forward"]},
    )
    return adapter


def _fake_tensors_to_zeros(value: Any) -> Any:
    """将 fake/meta tensor 输入模板转换为同 shape、dtype 的 CPU 零值校准数据。"""
    if value is None:
        return None
    if isinstance(value, Tensor) or (hasattr(value, "shape") and hasattr(value, "dtype")):
        return torch.zeros(tuple(value.shape), dtype=value.dtype)
    if isinstance(value, tuple):
        return tuple(_fake_tensors_to_zeros(item) for item in value)
    if isinstance(value, list):
        return [_fake_tensors_to_zeros(item) for item in value]
    return value


def _fake_tensor_shape_summary(value: Any) -> Any:
    """Return a lightweight nested shape summary for export diagnostics."""
    if isinstance(value, Tensor) or (hasattr(value, "shape") and hasattr(value, "dtype")):
        return (tuple(value.shape), str(value.dtype))
    if isinstance(value, (tuple, list)):
        return type(value)(_fake_tensor_shape_summary(item) for item in value)
    return value


def _flatten_calib_data_nested_lists(value: Any) -> list[Any]:
    """Flatten nested lists in calib data into a single flat list.

    Only ``list`` containers are expanded; tensors and other values are kept
    as leaves so a nested ``past_conv_cache`` list becomes individual inputs.
    """
    flat: list[Any] = []
    if isinstance(value, list):
        for item in value:
            flat.extend(_flatten_calib_data_nested_lists(item))
    else:
        flat.append(value)
    return flat


def _module_name_to_filename(name: str) -> str:
    """将完整 module target 转换为稳定且可用作文件名的字符串。"""
    filename = re.sub(r"[^0-9A-Za-z_.-]+", "_", name)
    filename = filename.replace(".", "_").strip("_") or "decoder_layer"
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    return f"{filename}_{digest}"


def _validate_exported_placeholder_hmonnx(
    hmonnx_file: str | Path,
    *,
    expected_inputs: int,
    expected_outputs: int | None,
) -> dict[str, Any]:
    """Check a worker result before it is accepted by the parent process."""
    hmonnx_path = Path(hmonnx_file)
    if not hmonnx_path.is_file():
        raise RuntimeError(f"Placeholder HMONNX was not created: {hmonnx_path}")
    model = onnx.load(str(hmonnx_path), load_external_data=False)
    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        external_data = {entry.key: entry.value for entry in initializer.external_data}
        location = external_data.get("location")
        if not location:
            raise RuntimeError(f"External initializer {initializer.name!r} has no location in {hmonnx_path}.")
        external_path = hmonnx_path.parent / location
        if not external_path.is_file():
            raise RuntimeError(f"External data file is missing for {initializer.name!r}: {external_path}")
        offset = int(external_data.get("offset", 0))
        length = int(external_data.get("length", 0))
        if offset < 0 or length < 0 or offset + length > external_path.stat().st_size:
            raise RuntimeError(
                f"External data range is invalid for {initializer.name!r} in {hmonnx_path}: "
                f"offset={offset}, length={length}, file_size={external_path.stat().st_size}."
            )
    actual_inputs = len(model.graph.input)
    actual_outputs = len(model.graph.output)
    if actual_inputs != expected_inputs:
        raise RuntimeError(
            f"Placeholder HMONNX input count mismatch for {hmonnx_path}: "
            f"expected {expected_inputs}, got {actual_inputs}."
        )
    if expected_outputs is not None and actual_outputs != expected_outputs:
        raise RuntimeError(
            f"Placeholder HMONNX output count mismatch for {hmonnx_path}: "
            f"expected {expected_outputs}, got {actual_outputs}."
        )
    onnx.checker.check_model(str(hmonnx_path))
    return {
        "path": str(hmonnx_path),
        "inputs": actual_inputs,
        "outputs": actual_outputs,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
    }


def _init_placeholder_export_process(
    big_hf_model_cls,
    hf_model_dir: str,
    empty_hf_model_factory,
) -> None:
    """Initialize one spawn worker with an isolated meta model and exporter."""
    global _PROCESS_EMPTY_HF_MODEL, _PROCESS_PLACEHOLDER_EXPORTER

    big_hf_model_cls.initialize_process_worker()
    empty_hf_model = empty_hf_model_factory(hf_model_dir)
    model_root = getattr(empty_hf_model, "model", None)
    if model_root is not None and hasattr(model_root, "visual"):
        model_root.visual = None
    big_hf_model_cls._preprocess_quantized_hf_model(empty_hf_model, hf_model_dir)

    exporter = big_hf_model_cls.__new__(big_hf_model_cls)
    exporter._hf_model_dir = hf_model_dir
    exporter._weight_mapping = WeightMapping()
    exporter._weight_mapping.weight_dir = hf_model_dir
    exporter._weight_mapping.weight_map = exporter._read_weight_map()
    exporter._weight_mapping.tensor_id_to_param_names = {}

    _PROCESS_EMPTY_HF_MODEL = empty_hf_model
    _PROCESS_PLACEHOLDER_EXPORTER = exporter


def _export_placeholder_process_task(task: dict[str, Any]) -> dict[str, Any]:
    """Materialize and export one placeholder target inside an isolated process."""
    exporter = _PROCESS_PLACEHOLDER_EXPORTER
    empty_hf_model = _PROCESS_EMPTY_HF_MODEL
    if exporter is None or empty_hf_model is None:
        raise RuntimeError("Placeholder export worker was not initialized.")

    module_target = task["module_target"]
    execution_device = _set_placeholder_export_execution_device(task.get("execution_device"))
    loaded_placeholder_m = exporter._load_place_holder_module_once(empty_hf_model, module_target)
    outputs = []
    try:
        for mode in task["modes"]:
            exporter._export_loaded_place_holder_layer_from_template(
                module_target,
                loaded_placeholder_m,
                mode["args_template"],
                task["target_device"],
                mode["wrap_cfg"],
                task["quant_cfg"],
                mode["output_dir"],
                execution_device=execution_device,
                expected_outputs=mode["expected_outputs"],
            )
            output_hmonnx_file = Path(mode["output_dir"]) / f"{_module_name_to_filename(module_target)}.onnx"
            outputs.append(
                _validate_exported_placeholder_hmonnx(
                    output_hmonnx_file,
                    expected_inputs=mode["expected_inputs"],
                    expected_outputs=mode["expected_outputs"],
                )
            )
        return {"module_target": module_target, "outputs": outputs, "pid": os.getpid()}
    finally:
        exporter._release_place_holder_module(loaded_placeholder_m)
        del loaded_placeholder_m


def _raise_placeholder_process_failure(module_target: str, exc: BaseException) -> None:
    if isinstance(exc, CancelledError):
        raise RuntimeError(f"Placeholder export worker for {module_target!r} was cancelled.") from exc
    message = f"Placeholder export worker for {module_target!r} failed: {exc}"
    if _is_cuda_oom_exception(exc):
        message = f"{message}. {_PLACEHOLDER_EXPORT_OOM_HINT}"
    raise RuntimeError(message) from exc


def _placeholder_output_meta(value: Any) -> tuple[list[list[int]], list[str], int]:
    """从 FX 输出 meta 生成 ``PlaceHolderModule`` 的 shape/dtype/数量契约。"""
    if isinstance(value, (tuple, list)) and any(isinstance(item, (tuple, list)) for item in value):
        raise RuntimeError(
            "Nested placeholder outputs are not supported because flattening would lose the output structure."
        )
    tensor_values = _flatten_tensor_meta_values(value)
    if not tensor_values:
        return [], [], -1
    output_shapes = [list(tensor_value.shape) for tensor_value in tensor_values]
    output_dtypes = [str(tensor_value.dtype).removeprefix("torch.") for tensor_value in tensor_values]
    return output_shapes, output_dtypes, len(tensor_values)


def _append_call_arg(args: list[Any], value: Any) -> None:
    args.append(value)


def _merge_call_args_kwargs_as_args(module, args, kwargs) -> tuple[Any, ...]:
    """按 ``forward`` 签名将 kwargs 规范化为有序位置参数。

    HMONNX placeholder 和独立子图通过位置顺序连接，因此必须为缺失但位于最后一个
    已提供 kwarg 之前的可选参数补 ``None``，保证主图与子图输入编号一致。
    """
    merged_args = []
    for arg in args:
        _append_call_arg(merged_args, arg)
    if not kwargs:
        return tuple(merged_args)

    signature = inspect.signature(module.forward)
    positional_names = [
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    provided_positional_indices = [
        index for index, name in enumerate(positional_names[len(args) :], start=len(args)) if name in kwargs
    ]
    if provided_positional_indices:
        last_provided_index = max(provided_positional_indices)
        for name in positional_names[len(args) : last_provided_index + 1]:
            _append_call_arg(merged_args, kwargs.get(name, None))
    for name, value in kwargs.items():
        if name not in positional_names:
            _append_call_arg(merged_args, value)
    return tuple(merged_args)


def _rebuild_value_from_flat_inputs(template: Any, flat_inputs: tuple[Any, ...], cursor: list[int]) -> Any:
    """依据 meta 模板将扁平输入恢复为模块原始调用结构。

    ``cursor`` 使用单元素列表保存可变游标，使递归调用共享同一消费位置。
    list 类型缓存被视为一个整体输入，tuple 则按元素递归恢复。
    """
    value = _node_arg_meta_value(template)
    if value is None:
        return None
    if isinstance(value, Tensor) or (hasattr(value, "shape") and hasattr(value, "dtype")):
        if cursor[0] >= len(flat_inputs):
            raise RuntimeError("Not enough flattened inputs to rebuild decoder layer call arguments.")
        rebuilt_value = flat_inputs[cursor[0]]
        cursor[0] += 1
        return rebuilt_value
    if isinstance(value, tuple):
        return tuple(_rebuild_value_from_flat_inputs(item, flat_inputs, cursor) for item in value)
    if isinstance(value, list):
        if cursor[0] >= len(flat_inputs):
            raise RuntimeError("Not enough flattened inputs to rebuild decoder layer call arguments.")
        rebuilt_value = flat_inputs[cursor[0]]
        cursor[0] += 1
        return rebuilt_value
    return value


def _get_module_tensor_parent(module, tensor_name: str):
    """定位点分隔参数名对应的直接父模块和局部属性名。"""
    parent_name, _, child_name = tensor_name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    return parent, child_name


def _replace_module_parameter(module, name: str, value: Tensor) -> None:
    """用 CPU 权重物化 meta parameter，同时保留目标 dtype 和梯度属性。"""
    parent, child_name = _get_module_tensor_parent(module, name)
    param = parent._parameters[child_name]
    if tuple(param.shape) != tuple(value.shape):
        raise RuntimeError(
            f"Shape mismatch for parameter {name}: expected {tuple(param.shape)}, got {tuple(value.shape)}"
        )
    parent._parameters[child_name] = nn.Parameter(
        value.to(device="cpu", dtype=param.dtype),
        requires_grad=param.requires_grad,
    )


def _replace_module_buffer(module, name: str, value: Tensor) -> None:
    """用 CPU 权重物化 meta buffer，同时校验 shape 并保留目标 dtype。"""
    parent, child_name = _get_module_tensor_parent(module, name)
    buffer = parent._buffers[child_name]
    if tuple(buffer.shape) != tuple(value.shape):
        raise RuntimeError(
            f"Shape mismatch for buffer {name}: expected {tuple(buffer.shape)}, got {tuple(value.shape)}"
        )
    parent._buffers[child_name] = value.to(device="cpu", dtype=buffer.dtype)


def _build_cpu_module_for_buffer_initialization(module: nn.Module) -> nn.Module:
    """Recreate a module on CPU using constructor values available on the instance.

    Required constructor arguments are resolved from the module first and then
    from its config. Unsupported constructors fail explicitly instead of leaving
    non-persistent buffers backed by uninitialized ``to_empty`` storage.
    """
    signature = inspect.signature(type(module).__init__)
    args = []
    kwargs = {}
    config = getattr(module, "config", None)
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name == "device":
            value = torch.device("cpu")
        elif hasattr(module, name):
            value = getattr(module, name)
        elif config is not None and hasattr(config, name):
            value = getattr(config, name)
        elif name == "config" and config is not None:
            value = config
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            raise RuntimeError(
                f"Cannot reconstruct {type(module).__name__} to initialize non-persistent buffers: "
                f"missing constructor argument {name!r}."
            )

        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[name] = value
    return type(module)(*args, **kwargs)


def _copy_module_shared_params(module: nn.Module) -> nn.Module:
    """深拷贝模块结构，但与原模块共享 tensor/parameter/buffer 底层数据。

    prefill 和 decode 导出需要独立执行 wrap 等结构修改；共享权重数据可以避免为同一
    placeholder 模块额外复制一份大权重，同时隔离结构层面的原地修改。
    """

    def _share_tensors(obj: Any, memo: dict[int, Any], seen: set[int]) -> None:
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, nn.Parameter):
            memo.setdefault(obj_id, nn.Parameter(obj.data, requires_grad=obj.requires_grad))
            return
        if torch.is_tensor(obj):
            memo.setdefault(obj_id, obj)
            return
        if isinstance(obj, nn.Module):
            for value in obj.__dict__.values():
                _share_tensors(value, memo, seen)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                _share_tensors(key, memo, seen)
                _share_tensors(value, memo, seen)
            return
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                _share_tensors(item, memo, seen)

    memo: dict[int, Any] = {}
    _share_tensors(module, memo, set())
    return copy.deepcopy(module, memo)


@contextmanager
def _without_fx_placeholder_module_type(module_type: type):
    """临时取消模块的 FX 叶子注册，使独立子图 tracing 能展开模块内部计算。"""
    from xhquant.nn import FX_LEAF_MODULES

    with _FX_PLACEHOLDER_REGISTRY_LOCK:
        module_type_name = module_type.__name__
        removed_module = FX_LEAF_MODULES._module_dict.pop(module_type_name, None)
        try:
            yield
        finally:
            if removed_module is not None:
                FX_LEAF_MODULES._module_dict[module_type_name] = removed_module


def _normalize_type_names(type_names: Optional[list[str] | str], default: list[str]) -> tuple[str, ...]:
    """将可选的单个或多个类型名规范化为不可变 tuple。"""
    if type_names is None:
        return tuple(default)
    if isinstance(type_names, str):
        return (type_names,)
    return tuple(type_names)


def _node_attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return default


def _external_data_entries(initializer: onnx.TensorProto) -> dict[str, str]:
    return {item.key: item.value for item in initializer.external_data}


def _set_external_data_entries(initializer: onnx.TensorProto, entries: dict[str, str]) -> None:
    del initializer.external_data[:]
    for key, value in entries.items():
        item = initializer.external_data.add()
        item.key = key
        item.value = str(value)
    initializer.data_location = onnx.TensorProto.EXTERNAL


def _external_data_int(entries: dict[str, str], key: str, default: int) -> int:
    value = entries.get(key)
    return default if value in (None, "") else int(value)


def _resolve_main_external_data_file(model: onnx.ModelProto, model_path: Path) -> tuple[str, Path]:
    for initializer in model.graph.initializer:
        location = _external_data_entries(initializer).get("location", "")
        if location:
            return location, model_path.parent / location
    location = f"{model_path.stem}_external_data"
    return location, model_path.parent / location


def _append_file_slice(source_file: Path, target_file: Path, offset: int, length: int) -> int:
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_offset = target_file.stat().st_size if target_file.exists() else 0
    remaining = length
    with open(source_file, "rb") as source, open(target_file, "ab") as target:
        source.seek(offset)
        while remaining > 0:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"Unexpected end of external data file: {source_file}")
            target.write(chunk)
            remaining -= len(chunk)
    return target_offset


def _merge_external_data(
    subgraph_model: onnx.ModelProto,
    subgraph_path: Path,
    main_location: str,
    main_file: Path,
) -> None:
    for initializer in subgraph_model.graph.initializer:
        entries = _external_data_entries(initializer)
        location = entries.get("location", "")
        if not location:
            continue
        subgraph_dir = subgraph_path.parent.resolve()
        source_file = (subgraph_dir / location).resolve()
        try:
            source_file.relative_to(subgraph_dir)
        except ValueError as exc:
            raise RuntimeError(f"Subgraph external data location escapes model directory: {location!r}") from exc
        if not source_file.exists():
            raise FileNotFoundError(f"Subgraph external data file does not exist: {source_file}")
        source_offset = _external_data_int(entries, "offset", 0)
        source_length = _external_data_int(entries, "length", source_file.stat().st_size - source_offset)
        target_offset = _append_file_slice(source_file, main_file, source_offset, source_length)
        entries.update(location=main_location, offset=str(target_offset), length=str(source_length))
        _set_external_data_entries(initializer, entries)


def _prefix_subgraph(subgraph_model: onnx.ModelProto, prefix: str) -> None:
    graph = subgraph_model.graph
    boundary_names = {value.name for value in graph.input} | {value.name for value in graph.output}

    def prefixed(name: str) -> str:
        if not name or name in boundary_names or name.startswith(f"{prefix}/"):
            return name
        return f"{prefix}/{name}"

    for node in graph.node:
        if node.name:
            node.name = prefixed(node.name)
        for index, name in enumerate(node.input):
            node.input[index] = prefixed(name)
        for index, name in enumerate(node.output):
            node.output[index] = prefixed(name)
    for initializer in graph.initializer:
        initializer.name = prefixed(initializer.name)
    for value_info in graph.value_info:
        value_info.name = prefixed(value_info.name)


def _merge_opset_imports(main_model: onnx.ModelProto, subgraph_model: onnx.ModelProto) -> None:
    main_versions = {opset.domain: opset.version for opset in main_model.opset_import}
    for subgraph_opset in subgraph_model.opset_import:
        existing_version = main_versions.get(subgraph_opset.domain)
        if existing_version is None:
            main_model.opset_import.add(domain=subgraph_opset.domain, version=subgraph_opset.version)
            main_versions[subgraph_opset.domain] = subgraph_opset.version
            continue
        if existing_version != subgraph_opset.version:
            raise RuntimeError(
                f"Incompatible ONNX opset versions for domain {subgraph_opset.domain!r}: "
                f"main={existing_version}, subgraph={subgraph_opset.version}"
            )


def _check_name_conflicts(
    main_graph: onnx.GraphProto,
    placeholder: onnx.NodeProto,
    tuple_getitems: list[tuple[int, onnx.NodeProto]],
    subgraph: onnx.GraphProto,
    replacements: dict[str, str],
) -> None:
    removed_node_ids = {id(placeholder), *(id(node) for _, node in tuple_getitems)}
    removed_tensor_names = set(placeholder.output)
    for _, node in tuple_getitems:
        removed_tensor_names.update(node.output)

    main_node_names = {node.name for node in main_graph.node if id(node) not in removed_node_ids and node.name}
    subgraph_node_names = [node.name for node in subgraph.node if node.name]
    duplicate_node_names = {name for name in subgraph_node_names if subgraph_node_names.count(name) > 1}
    if duplicate_node_names:
        raise RuntimeError(f"Duplicate node names in subgraph: {sorted(duplicate_node_names)[:10]}")
    conflicts = sorted(set(subgraph_node_names) & main_node_names)
    if conflicts:
        raise RuntimeError(f"Subgraph node name conflicts: {conflicts[:10]}")

    main_tensor_names = {
        name
        for node in main_graph.node
        if id(node) not in removed_node_ids
        for name in (*node.input, *node.output)
        if name and name not in removed_tensor_names
    }
    main_tensor_names.update(value.name for value in main_graph.input if value.name)
    main_tensor_names.update(value.name for value in main_graph.output if value.name)
    main_tensor_names.update(initializer.name for initializer in main_graph.initializer if initializer.name)
    main_tensor_names.update(value.name for value in main_graph.value_info if value.name)

    subgraph_tensor_names = {
        replacements.get(name, name) for node in subgraph.node for name in (*node.input, *node.output) if name
    }
    subgraph_tensor_names.update(
        replacements.get(initializer.name, initializer.name) for initializer in subgraph.initializer if initializer.name
    )
    subgraph_tensor_names.update(
        replacements.get(value.name, value.name) for value in subgraph.value_info if value.name
    )
    allowed = set(placeholder.input) | {node.output[0] for _, node in tuple_getitems if node.output}
    conflicts = sorted((subgraph_tensor_names & main_tensor_names) - allowed)
    if conflicts:
        raise RuntimeError(f"Subgraph tensor name conflicts: {conflicts[:10]}")


def replace_placeholder_node(
    placeholder: onnx.NodeProto,
    main_model: onnx.ModelProto,
    subgraph_file: str | Path,
    main_external_location: str,
    main_external_file: Path,
) -> None:
    """Replace one tuple-valued HMONNX placeholder with an exported subgraph."""
    subgraph_path = Path(subgraph_file)
    subgraph_model = onnx.load(str(subgraph_path), load_external_data=False)
    _merge_external_data(subgraph_model, subgraph_path, main_external_location, main_external_file)
    _prefix_subgraph(subgraph_model, subgraph_path.stem)
    _merge_opset_imports(main_model, subgraph_model)

    main_graph = main_model.graph
    subgraph = subgraph_model.graph
    if len(placeholder.input) != len(subgraph.input):
        raise RuntimeError(
            f"Placeholder input count mismatch for {placeholder.name}: "
            f"{len(placeholder.input)} != {len(subgraph.input)}"
        )
    if len(placeholder.output) != 1:
        raise RuntimeError(f"Expected PlaceHolder to have one tuple output, got {len(placeholder.output)}")

    tuple_getitems = []
    for node in main_graph.node:
        if node.op_type != "TupleGetItem" or node.domain != "ai.houmo.xh2a":
            continue
        if placeholder.output[0] not in node.input:
            continue
        output_index = int(_node_attr(node, "index", -1))
        if output_index < 0:
            raise RuntimeError(f"TupleGetItem node {node.name} has no valid index attribute.")
        tuple_getitems.append((output_index, node))
    output_indices = [output_index for output_index, _ in tuple_getitems]
    duplicate_indices = {index for index in output_indices if output_indices.count(index) > 1}
    unexpected_indices = {index for index in output_indices if index >= len(subgraph.output)}
    if duplicate_indices or unexpected_indices:
        raise RuntimeError(
            f"TupleGetItem indices for {placeholder.name} must be unique valid subgraph output indices; "
            f"duplicates={sorted(duplicate_indices)}, unexpected={sorted(unexpected_indices)}"
        )

    replacements = {value.name: placeholder.input[index] for index, value in enumerate(subgraph.input)}
    for output_index, tuple_getitem in tuple_getitems:
        replacements[subgraph.output[output_index].name] = tuple_getitem.output[0]
    consumed_output_indices = set(output_indices)
    for output_index, output in enumerate(subgraph.output):
        if output_index in consumed_output_indices or output.name in replacements:
            continue
        replacements[output.name] = f"{subgraph_path.stem}/{output.name}"

    _check_name_conflicts(main_graph, placeholder, tuple_getitems, subgraph, replacements)
    for node in subgraph.node:
        for index, name in enumerate(node.input):
            node.input[index] = replacements.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = replacements.get(name, name)

    removed_node_ids = {id(placeholder), *(id(node) for _, node in tuple_getitems)}
    merged_nodes = []
    inserted = False
    for node in main_graph.node:
        if id(node) in removed_node_ids:
            if node is placeholder:
                merged_nodes.extend(subgraph.node)
                inserted = True
            continue
        merged_nodes.append(node)
    if not inserted:
        raise RuntimeError(f"Failed to find placeholder node {placeholder.name} in main graph.")

    del main_graph.node[:]
    main_graph.node.extend(merged_nodes)
    main_graph.initializer.extend(subgraph.initializer)
    main_graph.value_info.extend(subgraph.value_info)


def replace_placeholders_with_subgraphs(
    hmonnx_file: str | Path,
    subgraphs_by_content: dict[str, str | Path],
    logger,
    *,
    cleanup_temporary_files: bool = False,
) -> None:
    """Replace all placeholders transactionally and validate the merged HMONNX.

    When ``cleanup_temporary_files`` is enabled, remove the main graph backups
    only after the merged model passes ONNX validation. Placeholder subgraph
    directories are owned and cleaned by ``BigHFModelExportHelper``.
    """
    model_path = Path(hmonnx_file)
    model = onnx.load(str(model_path), load_external_data=False)
    main_location, main_external_file = _resolve_main_external_data_file(model, model_path)

    placeholders = []
    for node in model.graph.node:
        if node.op_type != "PlaceHolder" or node.domain != "ai.houmo.xh2a":
            continue
        content = _node_attr(node, "content", "")
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        subgraph_file = subgraphs_by_content.get(content)
        if subgraph_file is None:
            raise FileNotFoundError(f"No HMONNX subgraph registered for placeholder {content!r}.")
        if not Path(subgraph_file).exists():
            raise FileNotFoundError(f"Placeholder HMONNX file does not exist for {content}: {subgraph_file}")
        placeholders.append((node.name, content, Path(subgraph_file)))

    logger.info(f"Found {len(placeholders)} PlaceHolder nodes in {model_path}")
    if not placeholders:
        return

    backup_file = model_path.with_suffix(model_path.suffix + ".bak")
    external_backup_file = main_external_file.with_name(main_external_file.name + ".bak")
    shutil.copy2(model_path, backup_file)
    had_external_data = main_external_file.exists()
    if had_external_data:
        shutil.copy2(main_external_file, external_backup_file)
    try:
        progress = tqdm(placeholders, desc="Replace HMONNX placeholders")
        for placeholder_name, content, subgraph_file in progress:
            placeholder = next(
                (node for node in model.graph.node if node.name == placeholder_name and node.op_type == "PlaceHolder"),
                None,
            )
            if placeholder is None:
                raise RuntimeError(f"Failed to find placeholder node {placeholder_name} in current graph.")
            replace_placeholder_node(
                placeholder,
                model,
                subgraph_file,
                main_location,
                main_external_file,
            )
            progress.set_description(content)

        remaining = [node.name for node in model.graph.node if node.op_type == "PlaceHolder"]
        if remaining:
            raise RuntimeError(f"Graph still has PlaceHolder nodes: {remaining[:10]}")
        onnx.save(model, str(model_path))
        onnx.checker.check_model(str(model_path))
    except Exception as exc:
        logger.exception(f"Failed to replace HMONNX placeholders in {model_path}, rolling back: {exc}")
        logger.error(f"Restoring HMONNX backup: {backup_file} -> {model_path}")
        shutil.copy2(backup_file, model_path)
        if had_external_data:
            logger.error(f"Restoring external data backup: {external_backup_file} -> {main_external_file}")
            shutil.copy2(external_backup_file, main_external_file)
        elif main_external_file.exists():
            logger.error(f"Removing newly created external data after rollback: {main_external_file}")
            main_external_file.unlink()
        raise

    if cleanup_temporary_files:
        backup_file.unlink(missing_ok=True)
        external_backup_file.unlink(missing_ok=True)


class BigHFModelExportHelper:
    """分层导出超大 Hugging Face 模型的 HMONNX 图。

    当模型无法一次性完成权重加载、量化和导出时，本类会将
    ``placeholder_type_names`` 指定的模块（通常是 MLP/MOE) 作为独立导出单元:

    1. 在完整模型的前端图中，将匹配模块替换为 ``PlaceHolderModule``，从而导出
       只保留模块输入、输出契约的主图。
    2. 按模块从 safetensors 文件中加载真实权重，必要时执行反量化，并逐个完成
       前端转换、PTQ 量化和 HMONNX 子图导出；已导出的模块会及时释放以控制峰值内存。
    3. 通过 placeholder 的 ``content`` 属性关联模块路径，为后续将主图中的
       ``PlaceHolder`` 节点替换为对应子图提供入口。

    初始化时仅物化并加载 placeholder 子树之外的参数（例如 embedding、最终 norm
    和 lm_head）。placeholder 模块的参数保持在 meta device 上，直到其子图被导出时
    才按需加载。预量化模型会先通过 Transformers quantizer 完成结构预处理，非
    placeholder 模块立即反量化，placeholder 模块则在逐层加载时反量化。

    Args:
        hf_model_dir: Hugging Face 模型目录，其中应包含 safetensors 权重及可选的
            ``model.safetensors.index.json``。
        hf_model: 已创建并完成量化结构预处理的 ``PreTrainedModel``。大模型场景通常
            在 meta device 上初始化，以避免构造阶段加载全部权重。
        placeholder_type_names: 作为独立子图导出的模块类型名。可传单个字符串或
           字符串列表；未指定时默认匹配 ``DecoderLayer``。对于 ``DynamicModule``，
            同时支持按其有效基类名称匹配。
        cleanup_temporary_files: 是否在主图与子图成功合并并通过 ONNX 校验后，删除
            placeholder 子图目录及主图备份文件。未显式指定时由环境变量
            ``XH2MODELZOO_KEEP_EXPORT_TMP`` 控制；默认清理，设为真时保留。
    """

    RUNTIME_PLACEHOLDER_REPLACEMENTS: dict[str, type[DynamicModule]] = {}
    PLACEHOLDER_TYPES = []

    @classmethod
    def initialize_process_worker(cls) -> None:
        """Register model-specific wrappers in a fresh spawn interpreter."""

    def __init__(
        self,
        hf_model_dir,
        hf_model: PreTrainedModel,
        placeholder_type_names: Optional[list[str] | str] = None,
        cleanup_temporary_files: bool | None = None,
    ) -> None:
        self._placeholder_type_names = _normalize_type_names(placeholder_type_names, ["DecoderLayer"])
        self._placeholder_types = set()
        self._cleanup_temporary_files = _cleanup_big_model_export_temporary_files(cleanup_temporary_files)

        for _, module in hf_model.named_modules():
            if self._is_placeholder_module(module):
                self._placeholder_types.add(type(module))
        self._log_placeholder_types("init")
        if not self._placeholder_types:
            raise RuntimeError(
                "BigHFModelExportHelper found no placeholder modules for configured types "
                f"{list(self._placeholder_type_names)}. Refusing to continue because this "
                "would materialize the full model."
            )

        self._hf_model = hf_model
        self._hf_model_dir = hf_model_dir
        self._weight_mapping = WeightMapping()
        self._load_weight_mapping(hf_model)

        # 如果Linear是量化版本的，需要反量化
        config = self._hf_model.config
        if hasattr(config, "quantization_config"):
            quantization_config = config.quantization_config
            # 调用XHBaseModel.dequantize_hf_model 反量化不在placeholder_types中的模块。
            # placeholder（如 DecoderLayer）子树的权重此时仍为 meta，其反量化延后到
            # export_decoder_layers 中逐层进行；这里先把 placeholder 子树临时从模块树
            # 摘除，避免 dequantize_hf_model 触碰仍为 meta 的 placeholder QuantLinear。
            with self._detach_placeholder_modules(hf_model):
                XHBaseModel.dequantize_hf_model(hf_model, quantization_config)

    @classmethod
    def register_placeholder(cls, registry: _DMRegistryCls, hf_model: Optional[torch.nn.Module] = None):
        pass

    # 初始化与权重物化

    @staticmethod
    def _preprocess_quantized_hf_model(
        hf_model: PreTrainedModel,
        hf_model_dir: str | Path | None = None,
        skip_module_prefixes: Optional[list[str]] = None,
    ) -> None:
        """Preprocess a pre-quantized empty model with the matching loader.

        AWQ and AutoRound retain Transformers' standard ``AutoHfQuantizer``
        flow. GPTQ checkpoints use GPTQModel's model definition, checkpoint
        allow-list, dynamic per-layer precision, and QuantLinear selection.
        """
        config = hf_model.config
        if not hasattr(config, "quantization_config"):
            return
        quant_config = config.quantization_config
        quant_method = XHBaseModel._get_quantization_method(quant_config)
        if quant_method == "gptq":
            model_dir = hf_model_dir or getattr(config, "_name_or_path", None)
            if not model_dir:
                raise ValueError("hf_model_dir is required to preprocess a GPTQModel empty model.")
            preprocessor = GPTQModelQuantizedModelPreprocessor(model_dir)
            replaced = preprocessor.preprocess(hf_model, skip_module_prefixes=skip_module_prefixes)
            get_xhquant_logger().info(
                "Replaced %d checkpoint-backed Linear modules using GPTQModel rules.",
                len(replaced),
            )
            XHBaseModel._trim_cpu_allocator()
            return

        hf_quantizer = AutoHfQuantizer.from_config(
            quant_config,
            pre_quantized=True,
        )
        hf_quantizer.device_map = None
        config.quantization_config = hf_quantizer.quantization_config
        # ``init_empty_weights`` only covers the original HF model construction.
        # Quantizer preprocessing happens afterwards and may replace every Linear
        # with a packed QuantLinear whose constructor calls factory functions such
        # as ``torch.zeros`` without an explicit device.  Without this context,
        # Qwen3.5-MoE AutoRound temporarily materializes all qweight/qzeros/scales
        # buffers on CPU (roughly 16 GiB) even though the source skeleton is meta.
        # Keep the structural conversion itself on meta; real tensors are loaded
        # later for non-placeholder modules or one placeholder at a time.
        with torch.device("meta"):
            hf_quantizer.preprocess_model(hf_model)
        # Some third-party quantized modules allocate their packed tensors on
        # an explicit device and therefore ignore the default meta context. Keep
        # this conversion as a defensive cleanup for those implementations. The
        # big-model loader materializes the required non-placeholder tensors, or
        # one placeholder at a time, after this structural conversion.
        hf_model.to_empty(device="meta")
        XHBaseModel._trim_cpu_allocator()

    @staticmethod
    def _prepare_loaded_gptqmodel_modules(
        module: nn.Module,
        quantization_config: Any,
        model_config: Any = None,
        skip_module_prefixes: Optional[list[str]] = None,
    ) -> int:
        config = model_config or getattr(module, "config", None)
        if not bool(getattr(config, GPTQModelQuantizedModelPreprocessor.CONFIG_MARKER, False)):
            return 0
        converted = GPTQModelQuantizedModelPreprocessor.prepare_loaded_module(
            module,
            quantization_config,
            skip_module_prefixes=skip_module_prefixes,
        )
        skip_prefixes = tuple(skip_module_prefixes or [])
        initialized_unpack_buffers = 0
        try:
            from gptqmodel.nn_modules.qlinear import BaseQuantLinear
        except ImportError as exc:
            raise ImportError("Preparing GPTQModel unpack buffers requires GPTQModel.") from exc

        for module_name, submodule in module.named_modules():
            if skip_prefixes and GPTQModelQuantizedModelPreprocessor._is_within_module_prefix(
                module_name,
                skip_prefixes,
            ):
                continue
            if not isinstance(submodule, BaseQuantLinear):
                continue
            qzeros = getattr(submodule, "qzeros", None)
            if qzeros is None or getattr(qzeros, "is_meta", False):
                continue
            before = all(
                getattr(submodule, buffer_name, None) is not None
                for buffer_name in ("wf_unsqueeze_zero", "wf_unsqueeze_neg_one")
            )
            ensure_gptqmodel_unpack_buffers(submodule)
            after = all(
                getattr(submodule, buffer_name, None) is not None
                for buffer_name in ("wf_unsqueeze_zero", "wf_unsqueeze_neg_one")
            )
            if not before and after:
                initialized_unpack_buffers += 2
        if initialized_unpack_buffers:
            get_xhquant_logger().info(
                "Initialized %d GPTQModel lazy unpack buffers.",
                initialized_unpack_buffers,
            )
        return converted

    def _is_placeholder_module(self, module: nn.Module) -> bool:
        """判断模块的实际类型或 DynamicModule 基类是否命中 placeholder 配置。"""
        if type(module) in self._placeholder_types:
            return True
        return bool(self._placeholder_type_name_candidates(module) & set(self._placeholder_type_names))

    @staticmethod
    def _placeholder_type_name_candidates(module: nn.Module) -> set[str]:
        """返回模块可用于 placeholder 匹配的实际类型名和有效基类名。"""
        type_names = {type(module).__name__}
        placeholder_type_name = getattr(module, "PLACEHOLDER_TYPE_NAME", None)
        if placeholder_type_name:
            type_names.add(str(placeholder_type_name))
        if isinstance(module, DynamicModule):
            ignored_types = {DynamicModule, nn.Module, object}
            for base_type in type(module).__mro__[1:]:
                if base_type in ignored_types:
                    continue
                type_names.add(base_type.__name__)
        return type_names

    @classmethod
    def resolve_placeholder_prefixes(
        cls,
        hf_model: PreTrainedModel,
        placeholder_type_names: Optional[list[str] | str] = None,
    ) -> list[str]:
        """在构造 ``BigHFModelExportHelper`` 前解析需要从主图量化预处理中跳过的模块。"""
        configured_type_names = set(_normalize_type_names(placeholder_type_names, ["DecoderLayer"]))
        return [
            name
            for name, module in hf_model.named_modules()
            if cls._placeholder_type_name_candidates(module) & configured_type_names
        ]

    def replace_runtime_placeholder_modules(self, hf_model: nn.Module) -> int:
        """Replace third-party fused placeholder leaves with pure lightweight modules."""
        replacements = self.RUNTIME_PLACEHOLDER_REPLACEMENTS
        if not replacements:
            return 0

        replaced = 0
        for parent in list(hf_model.modules()):
            for child_name, child_module in list(parent._modules.items()):
                if child_module is None:
                    continue
                placeholder_cls = replacements.get(type(child_module).__name__)
                if placeholder_cls is None:
                    continue
                placeholder = object.__new__(placeholder_cls)
                nn.Module.__init__(placeholder)
                parent._modules[child_name] = placeholder
                replaced += 1
        get_xhquant_logger().info(f"Replaced {replaced} fused runtime modules with lightweight placeholders.")
        return replaced

    def prepare_loaded_placeholder_module(self, module: nn.Module, wrap_cfg) -> nn.Module:
        """Adapt a materialized third-party placeholder before tracing its subgraph."""
        return module

    def repair_linear_cache_input_meta(self, fronted_graph_module: FrontendGraph) -> int:
        """Restore split q/k/v cache metadata lost by nested-list FX inputs."""
        config = getattr(self._hf_model.config, "text_config", self._hf_model.config)
        required = (
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_value_head_dim",
        )
        if not all(hasattr(config, name) for name in required):
            return 0

        key_dim = int(config.linear_num_key_heads) * int(config.linear_key_head_dim)
        value_dim = int(config.linear_num_value_heads) * int(config.linear_value_head_dim)
        kernel_size = int(config.linear_conv_kernel_dim)
        recurrent_tail = (
            int(config.linear_num_value_heads),
            int(config.linear_key_head_dim),
            int(config.linear_value_head_dim),
        )
        conv_nodes = [
            node
            for node in fronted_graph_module.graph.nodes
            if node.op == "placeholder" and str(node.name).startswith("past_conv_cache_")
        ]
        recurrent_nodes = [
            node
            for node in fronted_graph_module.graph.nodes
            if node.op == "placeholder" and str(node.name).startswith("past_recurrent_state_")
        ]

        repaired = 0
        for index, node in enumerate(conv_nodes):
            old_value = node.meta.get("val")
            dtype = getattr(old_value, "dtype", torch.float16)
            batch_size = int(getattr(old_value, "shape", (1,))[0])
            width = value_dim if index % 3 == 2 else key_dim
            value = torch.empty((batch_size, width, kernel_size), dtype=dtype, device="meta")
            node.meta["val"] = value
            node.meta["tensor_meta"] = _extract_tensor_metadata(value)
            repaired += 1
        for node in recurrent_nodes:
            old_value = node.meta.get("val")
            dtype = getattr(old_value, "dtype", torch.float16)
            batch_size = int(getattr(old_value, "shape", (1,))[0])
            value = torch.empty((batch_size, *recurrent_tail), dtype=dtype, device="meta")
            node.meta["val"] = value
            node.meta["tensor_meta"] = _extract_tensor_metadata(value)
            repaired += 1
        if repaired:
            get_xhquant_logger().info("Repaired metadata for %d flattened linear-cache graph inputs.", repaired)
        return repaired

    def _log_placeholder_types(self, stage: str) -> None:
        """记录配置的类型名与当前已解析出的实际模块类型。"""
        logger = get_xhquant_logger()
        logger.info(
            f"BigHFModelExportHelper placeholder types ({stage}): configured={list(self._placeholder_type_names)}, "
            f"resolved={sorted(placeholder_type.__name__ for placeholder_type in self._placeholder_types)}"
        )

    @contextmanager
    def _detach_placeholder_modules(self, hf_model: PreTrainedModel):
        """临时将 placeholder 子模块从模块树摘除，退出时恢复。

        用于把 ``dequantize_hf_model`` 的作用范围限制在非 placeholder 模块上，避免它
        遍历到权重仍为 meta 的 placeholder 子树（如 DecoderLayer）。
        """
        # 只摘除最顶层的 placeholder 模块（不进入 placeholder 子树内部继续查找）。
        detached: list[tuple[nn.Module, str, nn.Module]] = []
        for parent in hf_model.modules():
            for child_name, child_module in list(parent._modules.items()):
                if child_module is not None and self._is_placeholder_module(child_module):
                    detached.append((parent, child_name, child_module))
        seen: set[tuple[int, str]] = set()
        unique_detached = []
        for parent, child_name, child_module in detached:
            key = (id(parent), child_name)
            if key in seen:
                continue
            seen.add(key)
            unique_detached.append((parent, child_name, child_module))

        try:
            for parent, child_name, _ in unique_detached:
                setattr(parent, child_name, nn.Module())
            yield
        finally:
            for parent, child_name, child_module in unique_detached:
                setattr(parent, child_name, child_module)

    def _read_weight_map(self) -> dict[str, str]:
        """建立完整 tensor 名到 safetensors 分片文件的映射。

        优先读取 Hugging Face 分片索引；单文件或无索引模型则扫描所有
        safetensors 文件中的 key。
        """
        weight_map = {}
        index_path = Path(self._hf_model_dir) / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                data = json.load(f)
                for k, v in data["weight_map"].items():
                    weight_map[k] = str(Path(self._hf_model_dir) / v)
        else:
            safetensor_files = glob.glob(str(Path(self._hf_model_dir) / "*.safetensors"))
            for f in safetensor_files:
                with safe_open(f, framework="pt") as ptr:
                    for k in ptr.keys():
                        weight_map[k] = f
        return weight_map

    def _placeholder_prefixes(self, hf_model: PreTrainedModel) -> list[str]:
        """收集所有 placeholder 模块在完整模型中的路径前缀。"""
        return [name for name, module in hf_model.named_modules() if self._is_placeholder_module(module)]

    @staticmethod
    def _within_placeholder_module(module_name: str, placeholder_prefixes: list[str]) -> bool:
        """判断模块路径是否等于或位于任一 placeholder 子树内。"""
        for prefix in placeholder_prefixes:
            if prefix == "":
                return True
            if module_name == prefix or module_name.startswith(prefix + "."):
                return True
        return False

    @classmethod
    def _within_placeholder_tensor(cls, tensor_name: str, placeholder_prefixes: list[str]) -> bool:
        """根据 tensor 的父模块路径判断其是否属于 placeholder 子树。"""
        module_name = tensor_name.rpartition(".")[0]
        return cls._within_placeholder_module(module_name, placeholder_prefixes)

    def _collect_non_placeholder_tensors(
        self, hf_model: PreTrainedModel, placeholder_prefixes: list[str]
    ) -> tuple[dict[str, nn.Parameter], dict[str, Tensor]]:
        """收集初始化阶段需要立即物化的非 placeholder 参数和 buffer。"""
        uninitialized_params = {
            full_name: param
            for full_name, param in hf_model.named_parameters()
            if not self._within_placeholder_tensor(full_name, placeholder_prefixes)
        }
        uninitialized_buffers = {
            full_name: buffer
            for full_name, buffer in hf_model.named_buffers()
            if not self._within_placeholder_tensor(full_name, placeholder_prefixes)
        }
        return uninitialized_params, uninitialized_buffers

    def _materialize_non_placeholder_modules(self, hf_model: PreTrainedModel, placeholder_prefixes: list[str]) -> int:
        """在 CPU 上为空壳模块分配非 placeholder tensor 存储。

        ``to_empty`` 只分配存储而不初始化真实权重；持久化参数随后从 safetensors
        覆盖。未写入权重文件的 non-persistent buffer 则通过同类型 CPU 模块恢复其
        构造函数初始化值。
        """
        materialized_module_names = []
        for module_name, module in hf_model.named_modules():
            if self._within_placeholder_module(module_name, placeholder_prefixes):
                continue
            direct_params = list(module.named_parameters(recurse=False))
            direct_buffers = list(module.named_buffers(recurse=False))
            direct_tensors = [tensor for _, tensor in [*direct_params, *direct_buffers]]
            if not any(tensor is not None and tensor.is_meta for tensor in direct_tensors):
                continue
            module.to_empty(device="cpu", recurse=False)

            # rotary/cache 等 non-persistent buffer 通常不在权重文件中，需要使用模块
            # 构造函数生成有效初值，不能保留 to_empty 得到的未初始化存储。
            non_persistent_buffer_names = {
                buffer_name
                for buffer_name, buffer in direct_buffers
                if buffer_name in module._non_persistent_buffers_set and buffer is not None
            }
            if non_persistent_buffer_names:
                cpu_module_instance = _build_cpu_module_for_buffer_initialization(module)
                for buffer_name in non_persistent_buffer_names:
                    initialized_buffer = cpu_module_instance.get_buffer(buffer_name)
                    expected_buffer = dict(direct_buffers)[buffer_name]
                    if tuple(initialized_buffer.shape) != tuple(expected_buffer.shape):
                        raise RuntimeError(
                            f"Shape mismatch while initializing non-persistent buffer {module_name}.{buffer_name}: "
                            f"expected {tuple(expected_buffer.shape)}, got {tuple(initialized_buffer.shape)}"
                        )
                    module._buffers[buffer_name] = initialized_buffer.to(device="cpu", dtype=expected_buffer.dtype)
            materialized_module_names.append(module_name)

        return len(materialized_module_names)

    @staticmethod
    def _remove_output_embedding_shared_tensors(
        hf_model: PreTrainedModel,
        uninitialized_params: dict[str, nn.Parameter],
        uninitialized_buffers: dict[str, Tensor],
    ) -> None:
        """从待加载集合移除与输入 embedding 共享的输出层 tensor。

        tied embedding 只应加载一次；否则重复替换 output weight 会破坏参数共享关系，
        并额外占用一份词表权重内存。
        """
        if not bool(getattr(hf_model.config, "tie_word_embeddings", False)):
            return

        hf_model.tie_weights()
        input_embedding = hf_model.get_input_embeddings()
        output_embedding = hf_model.get_output_embeddings()
        if input_embedding is None or output_embedding is None:
            return

        input_weight = getattr(input_embedding, "weight", None)
        output_weight = getattr(output_embedding, "weight", None)
        if output_weight is not None and output_weight.is_meta and input_weight is not None:
            output_embedding.weight = input_weight
        if output_embedding is input_embedding:
            return

        shared_tensor_ids = {
            id(tensor)
            for _, tensor in list(input_embedding.named_parameters(recurse=True))
            + list(input_embedding.named_buffers(recurse=True))
        }
        output_tensor_names = set()
        output_embedding_names = [name for name, module in hf_model.named_modules() if module is output_embedding]
        for module_name in output_embedding_names:
            prefix = f"{module_name}." if module_name else ""
            output_tensor_names.update(
                f"{prefix}{local_name}"
                for local_name, tensor in output_embedding.named_parameters(recurse=True)
                if id(tensor) in shared_tensor_ids
            )
            output_tensor_names.update(
                f"{prefix}{local_name}"
                for local_name, tensor in output_embedding.named_buffers(recurse=True)
                if id(tensor) in shared_tensor_ids
            )

        for tensor_name in output_tensor_names:
            uninitialized_params.pop(tensor_name, None)
            uninitialized_buffers.pop(tensor_name, None)

    @staticmethod
    def _remove_buffers_missing_from_weight_map(
        hf_model: PreTrainedModel,
        uninitialized_buffers: dict[str, Tensor],
        weight_map: dict[str, str],
    ) -> None:
        """Skip non-persistent buffers and reject missing persistent buffers."""
        for buffer_name in list(uninitialized_buffers):
            if buffer_name in weight_map:
                continue
            parent, child_name = _get_module_tensor_parent(hf_model, buffer_name)
            if child_name in parent._non_persistent_buffers_set:
                uninitialized_buffers.pop(buffer_name, None)
                continue
            raise KeyError(f"Persistent buffer {buffer_name!r} is missing from the safetensors weight map.")

    def _load_weight_mapping(self, hf_model: PreTrainedModel):
        """初始化权重索引并物化主图所需的非 placeholder 权重。"""
        logger = get_xhquant_logger()
        weight_map = self._read_weight_map()
        self._weight_mapping.weight_map = weight_map
        self._weight_mapping.weight_dir = self._hf_model_dir

        placeholder_prefixes = self._placeholder_prefixes(hf_model)
        uninitialized_params, uninitialized_buffers = self._collect_non_placeholder_tensors(
            hf_model, placeholder_prefixes
        )

        materialized_count = self._materialize_non_placeholder_modules(hf_model, placeholder_prefixes)
        logger.info(f"Materialized {materialized_count} non-placeholder modules from meta tensors.")

        self._remove_output_embedding_shared_tensors(hf_model, uninitialized_params, uninitialized_buffers)
        self._remove_buffers_missing_from_weight_map(hf_model, uninitialized_buffers, weight_map)

        self._load_non_placeholder_modules_from_safetensor(
            hf_model, weight_map, uninitialized_params, uninitialized_buffers
        )
        config = hf_model.config
        if hasattr(config, "quantization_config"):
            converted = self._prepare_loaded_gptqmodel_modules(
                hf_model,
                config.quantization_config,
                skip_module_prefixes=placeholder_prefixes,
            )
            if converted:
                logger.info("Converted %d loaded GPTQModel QuantLinear modules to runtime format.", converted)
        if bool(getattr(hf_model.config, "tie_word_embeddings", False)):
            # Parameter materialization replaces objects, so restore aliases after
            # loading to keep a separate lm_head bound to the loaded embedding.
            hf_model.tie_weights()

        # 保留尚未物化的 meta parameter 身份映射，供后续按模块加载或外部流程定位。
        param_ids = {}
        for name, param in hf_model.named_parameters():
            if param.is_meta:
                param_ids[id(param)] = name
        self._weight_mapping.tensor_id_to_param_names = param_ids

    def _load_non_placeholder_modules_from_safetensor(
        self,
        hf_model: PreTrainedModel,
        weight_map: dict[str, str],
        uninitialized_params: dict[str, nn.Parameter],
        uninitialized_buffers: dict[str, Tensor],
    ) -> None:
        """加载所有不在 placeholder 子树内的模块的真实权重。

        placeholder 类型（如 attention、MLP/MoE）的权重会在子图导出阶段按需加载，
        因此这里跳过 placeholder 模块及其子模块，只物化其余仍为 meta 的
        参数/缓冲区（如 embedding、最终 norm、lm_head 等）。
        """
        if not uninitialized_params and not uninitialized_buffers:
            return
        param_full_names = list(uninitialized_params.keys())
        buffer_full_names = list(uninitialized_buffers.keys())
        tensors = load_weight_from_safetensor(weight_map, param_full_names + buffer_full_names)
        param_tensors = tensors[: len(param_full_names)]
        buffer_tensors = tensors[len(param_full_names) :]
        for full_name, tensor in zip(param_full_names, param_tensors, strict=True):
            _replace_module_parameter(hf_model, full_name, tensor)
        for full_name, tensor in zip(buffer_full_names, buffer_tensors, strict=True):
            _replace_module_buffer(hf_model, full_name, tensor)

    # Placeholder 类型注册与前端图改写

    @classmethod
    def create_split_tag(
        cls, hf_wrap_model: PreTrainedModel, type_name_or_types: Optional[str | Type | list[str | type]] = None
    ) -> PreTrainedModel:
        """按指定模块类型在 wrap 模型中标注 pipeline split 边界。"""
        if type_name_or_types is None:
            type_name_or_types = ["DecoderLayer"]
        if not isinstance(type_name_or_types, (list, tuple)):
            type_name_or_types = [type_name_or_types]
        split_module_types = []
        for _, m in hf_wrap_model.named_modules():
            for type_name_or_type in type_name_or_types:
                if isinstance(type_name_or_type, type) and type(m) is type_name_or_type:
                    split_module_types.append(type(m))
                elif isinstance(type_name_or_type, str) and type(m).__name__ == type_name_or_type:
                    split_module_types.append(type(m))

        split_module_types = list(set(split_module_types))
        logger = get_xhquant_logger()
        logger.info(f"split by {[module_type.__name__ for module_type in split_module_types]}")
        annotate_pipe_split(hf_wrap_model, tuple(split_module_types))
        return hf_wrap_model

    def register_layer_as_placeholder(self, hf_model: nn.Module):
        """解析并注册 FX 叶子模块类型，阻止主图 tracing 展开其内部计算。

        wrap 前后模块的实际类型可能发生变化，因此调用方可重复调用；已注册类型会
        自动去重。
        """
        from xhquant.nn import FX_LEAF_MODULES

        for _, module in hf_model.named_modules():
            if self._is_placeholder_module(module):
                self._placeholder_types.add(type(module))
        self._log_placeholder_types("register_layer_as_placeholder")

        with _FX_PLACEHOLDER_REGISTRY_LOCK:
            for placeholder_type in self._placeholder_types:
                if placeholder_type.__name__ not in FX_LEAF_MODULES:
                    FX_LEAF_MODULES.register_module(module=placeholder_type)

    def strip_unwrapped_placeholder_members(self, hf_model: nn.Module) -> int:
        """Remove meta tensors from placeholder leaves that cannot become DynamicModules.

        Some third-party fused modules (notably AutoRound fused MoE blocks)
        reject dynamic subclass creation.  They can still remain FX leaves in
        the main graph, but their registered meta parameters would make the
        frontend graph's final ``.to()`` fail.  The separately retained
        placeholder template model owns the tensors used for subgraph export,
        so clearing the main-graph leaf is safe.
        """
        stripped = 0
        for module in list(hf_model.modules()):
            if isinstance(module, DynamicModule) or not self._is_placeholder_module(module):
                continue
            module._modules.clear()
            module._parameters.clear()
            module._buffers.clear()
            stripped += 1
        get_xhquant_logger().info(f"Stripped registered members from {stripped} unwrapped placeholder leaves.")
        return stripped

    def register_layer_as_place_holder(self, fronted_graph_module: FrontendGraph):
        """将前端图中的目标模块替换为带完整 I/O 契约的 ``PlaceHolderModule``。"""
        import xhquant.nn as xhnn

        self.repair_linear_cache_input_meta(fronted_graph_module)

        for node in fronted_graph_module.graph.nodes:
            if node.op == "call_module":
                m = fronted_graph_module.get_submodule(str(node.target))
                if type(m) in self._placeholder_types:
                    # 主图和独立子图必须使用相同的位置输入顺序，避免 kwargs 在导出后
                    # 丢失名称语义而造成端口错连。
                    node.args = _merge_call_args_kwargs_as_args(m, node.args, node.kwargs)
                    node.kwargs = {}
                    val = node.meta["val"]
                    output_shapes, output_dtypes, num_outputs = _placeholder_output_meta(val)
                    place_holder_m = xhnn.PlaceHolderModule(
                        output_shapes=output_shapes,
                        output_dtypes=output_dtypes,
                        num_outputs=num_outputs,
                        content=node.target,
                    )
                    place_holder_m.__dict__["owner"] = m
                    place_holder_m.__dict__["input_fake_tensors"] = _flatten_input_fake_tensor_values(node.args)
                    fronted_graph_module.set_submodule(node.target, place_holder_m)
                    if num_outputs == 1:
                        # PlaceHolderModule 统一返回 list；单输出原模块的下游仍期望 tensor，
                        # 因此插入 getitem(0) 保持原前端图的数据类型和使用方式不变。
                        node.meta["val"] = [val]
                        with fronted_graph_module.graph.inserting_after(node):
                            getitem_node = fronted_graph_module.graph.call_function(operator.getitem, args=(node, 0))
                        getitem_node.meta = {"val": val}
                        node.replace_all_uses_with(getitem_node)
                        getitem_node.args = (node, 0)

        fronted_graph_module.graph.lint()
        fronted_graph_module.recompile()

    # Placeholder 子图加载、量化与导出

    def _load_module_from_safetensor(self, module, module_prefix: str) -> dict[str, int]:
        """按完整模块前缀加载并物化一个 placeholder 模块的全部持久化 tensor。"""
        weight_map = self._weight_mapping.weight_map
        full_param_names = []
        local_param_names = []
        missing_param_names = []
        for local_name, _ in module.named_parameters(recurse=True):
            full_name = f"{module_prefix}.{local_name}" if module_prefix else local_name
            if full_name in weight_map:
                full_param_names.append(full_name)
                local_param_names.append(local_name)
            else:
                # Checking only ``is_meta`` is insufficient: third-party
                # QuantLinear constructors may allocate a real CPU zero tensor.
                # Such a tensor would look materialized while never receiving
                # checkpoint data. Every placeholder parameter must therefore
                # be backed by an explicit safetensors key.
                missing_param_names.append(full_name)

        full_buffer_names = []
        local_buffer_names = []
        missing_persistent_buffer_names = []
        missing_non_persistent_buffers: dict[nn.Module, list[tuple[str, Tensor]]] = defaultdict(list)
        for local_name, buffer in module.named_buffers(recurse=True):
            full_name = f"{module_prefix}.{local_name}" if module_prefix else local_name
            if full_name in weight_map:
                full_buffer_names.append(full_name)
                local_buffer_names.append(local_name)
                continue
            parent, child_name = _get_module_tensor_parent(module, local_name)
            if child_name in parent._non_persistent_buffers_set:
                if getattr(buffer, "is_meta", False):
                    missing_non_persistent_buffers[parent].append((child_name, buffer))
                continue
            # Persistent buffers are checkpoint state just like parameters.
            # Reject missing keys even when a constructor happened to allocate
            # a non-meta placeholder value on CPU.
            missing_persistent_buffer_names.append(full_name)

        if missing_param_names or missing_persistent_buffer_names:
            details = []
            if missing_param_names:
                details.append("parameters=" + ", ".join(missing_param_names[:20]))
            if missing_persistent_buffer_names:
                details.append("persistent_buffers=" + ", ".join(missing_persistent_buffer_names[:20]))
            raise RuntimeError(
                f"Placeholder {module_prefix!r} has tensors absent from the safetensors weight map: "
                + "; ".join(details)
            )

        for parent, buffer_specs in missing_non_persistent_buffers.items():
            cpu_parent = _build_cpu_module_for_buffer_initialization(parent)
            for child_name, expected_buffer in buffer_specs:
                initialized_buffer = cpu_parent.get_buffer(child_name)
                if tuple(initialized_buffer.shape) != tuple(expected_buffer.shape):
                    raise RuntimeError(
                        f"Shape mismatch while initializing non-persistent buffer {module_prefix}.{child_name}: "
                        f"expected {tuple(expected_buffer.shape)}, got {tuple(initialized_buffer.shape)}"
                    )
                parent._buffers[child_name] = initialized_buffer.to(device="cpu", dtype=expected_buffer.dtype)

        tensors = load_weight_from_safetensor(weight_map, full_param_names + full_buffer_names)
        param_tensors = tensors[: len(full_param_names)]
        buffer_tensors = tensors[len(full_param_names) :]

        for local_name, tensor in zip(local_param_names, param_tensors, strict=True):
            _replace_module_parameter(module, local_name, tensor)
        for local_name, tensor in zip(local_buffer_names, buffer_tensors, strict=True):
            _replace_module_buffer(module, local_name, tensor)

        missing_meta_params = [
            f"{module_prefix}.{local_name}" if module_prefix else local_name
            for local_name, param in module.named_parameters(recurse=True)
            if getattr(param, "is_meta", False)
        ]
        if missing_meta_params:
            raise RuntimeError(
                "Failed to materialize DecoderLayer parameters from safetensors; missing keys: "
                + ", ".join(missing_meta_params[:20])
            )

        missing_meta_buffers = [
            f"{module_prefix}.{local_name}" if module_prefix else local_name
            for local_name, buffer in module.named_buffers(recurse=True)
            if getattr(buffer, "is_meta", False)
        ]
        if missing_meta_buffers:
            raise RuntimeError(
                "Failed to materialize DecoderLayer buffers from safetensors; missing keys: "
                + ", ".join(missing_meta_buffers[:20])
            )

        return {
            "parameter_keys": len(full_param_names),
            "buffer_keys": len(full_buffer_names),
            "initialized_non_persistent_buffers": sum(len(items) for items in missing_non_persistent_buffers.values()),
            "missing_parameter_keys": 0,
            "missing_persistent_buffer_keys": 0,
        }

    @staticmethod
    def _collect_place_holder_modules(fronted_graph_module: FrontendGraph):
        """按图中出现顺序收集 ``(FX node, PlaceHolderModule)``。"""
        import xhquant.nn as xhnn

        place_modules = []
        for node in fronted_graph_module.graph.nodes:
            if node.op != "call_module":
                continue
            m = fronted_graph_module.get_submodule(str(node.target))
            if isinstance(m, xhnn.PlaceHolderModule):
                place_modules.append((node, m))
        return place_modules

    def _load_place_holder_module_once(self, empty_hf_model_for_placeholder, module_target: str):
        """复制、物化并按需反量化一个待导出的 placeholder 模块。"""
        hf_model = empty_hf_model_for_placeholder
        # 深拷贝隔离后续 wrap/to_empty 等原地修改，模板模型始终保持可重复加载状态。
        placeholder_m = empty_hf_model_for_placeholder.get_submodule(module_target)
        placeholder_m = copy.deepcopy(placeholder_m)
        load_audit = self._load_module_from_safetensor(placeholder_m, module_target)
        config = hf_model.config
        if hasattr(config, "quantization_config"):
            quantization_config = config.quantization_config
            self._prepare_loaded_gptqmodel_modules(
                placeholder_m,
                quantization_config,
                model_config=config,
            )
            placeholder_m = XHBaseModel.dequantize_hf_model(placeholder_m, quantization_config)
        placeholder_m._placeholder_load_audit = load_audit
        return placeholder_m

    def _export_loaded_place_holder_layer(
        self,
        node,
        m,
        loaded_placeholder_m,
        target_device,
        wrap_cfg,
        quant_cfg,
        output_dir: str | Path,
    ) -> None:
        """使用指定模式的输入契约量化并导出一个已加载 placeholder 模块。"""
        # 节点 meta 记录了主图 tracing 时的真实 shape/dtype，是构造子图 dummy 输入的
        # 唯一依据；这里同时统一 kwargs，确保导出端口顺序稳定。
        node.args = _merge_call_args_kwargs_as_args(m, node.args, node.kwargs)
        node.kwargs = {}
        self._export_loaded_place_holder_layer_from_template(
            str(node.target),
            loaded_placeholder_m,
            node.args,
            target_device,
            wrap_cfg,
            quant_cfg,
            output_dir,
            expected_outputs=getattr(m, "num_outputs", None),
        )

    def _export_loaded_place_holder_layer_from_template(
        self,
        module_target: str,
        loaded_placeholder_m,
        args_template,
        target_device,
        wrap_cfg,
        quant_cfg,
        output_dir: str | Path,
        execution_device: str | None = None,
        expected_outputs: int | None = None,
    ) -> None:
        """Export one mode from a spawn-safe static input template."""
        execution_device = _set_placeholder_export_execution_device(execution_device)
        # prefill/decode 会分别修改模块结构；共享权重的结构副本可避免第二份权重开销。
        placeholder_m = _copy_module_shared_params(loaded_placeholder_m)
        placeholder_m = self.prepare_loaded_placeholder_module(placeholder_m, wrap_cfg)

        input_fake_tensors = _placeholder_input_fake_tensors(args_template)
        get_xhquant_logger().info(
            "Placeholder %s standalone input contract: %s",
            module_target,
            _fake_tensor_shape_summary(input_fake_tensors),
        )
        with _without_fx_placeholder_module_type(type(placeholder_m)):
            # 主图需要把该类型视为叶子，而子图恰好相反：必须临时展开模块内部计算。
            wrap_llm_model(placeholder_m, wrap_cfg)
            adapted_placeholder_m = _make_flattened_input_adapter(placeholder_m, args_template, expected_outputs)
            placeholder_fronted_graph_module = to_frontend_graph(adapted_placeholder_m, "TorchFX", input_fake_tensors)
        placeholder_quanted_model = to_quant_graph(placeholder_fronted_graph_module, target_device, quant_cfg)

        # 使用 fake tensor 的 shape/dtype 创建零值校准数据，避免执行完整主图收集样本。
        calib_data = _fake_tensors_to_zeros(input_fake_tensors)
        calib_data = _flatten_calib_data_nested_lists(calib_data)
        ptq_quantize(
            placeholder_quanted_model,
            [calib_data],
            PrecisionMode.ALIGNED,
            [execution_device],
            auto_release_unused_parameters=True,
            infer_shape=False,
        )
        fname = _module_name_to_filename(module_target)
        output_hmonnx_file = str(Path(output_dir) / f"{fname}.onnx")
        to_export_hmonnx_from_quanted_graph(
            placeholder_quanted_model,
            calib_data,
            str(output_hmonnx_file),
            save_as_external_data=True,
        )
        # 尽早解除图对象和已物化权重的引用，防止逐层导出时峰值内存持续累积。
        del placeholder_quanted_model
        del placeholder_fronted_graph_module
        try:
            placeholder_m.to_empty(device="meta")
        except Exception as exc:
            get_xhquant_logger().warning(
                f"Failed to release exported placeholder module {module_target} to meta device: {exc}"
            )
        del placeholder_m

    @staticmethod
    def _release_place_holder_module(module) -> int:
        """将模块权重退回 meta device，并主动回收 CPU/GPU 缓存。"""
        try:
            module.to_empty(device="meta")
        except Exception as exc:
            get_xhquant_logger().warning(f"Failed to release placeholder module to meta device: {exc}")
        import gc

        collected = gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return collected

    def export_place_holder_layers(
        self,
        fronted_graph_module: FrontendGraph,
        empty_hf_model_for_placeholder: PreTrainedModel,
        target_device,
        wrap_cfg,
        quant_cfg,
        output_dir: str | Path,
    ):
        """逐个导出单张前端图中的所有 placeholder 子图。"""
        place_modules = self._collect_place_holder_modules(fronted_graph_module)
        pbar = tqdm(place_modules, desc="Export PlaceHolder layers")
        logger = get_xhquant_logger()
        for node, m in pbar:
            pbar.set_description(f"export {node.target} as PlaceHolder")
            logger.info(f"*************** export {node.target} as PlaceHolder ***************")
            loaded_placeholder_m = self._load_place_holder_module_once(
                empty_hf_model_for_placeholder,
                str(node.target),
            )
            self._export_loaded_place_holder_layer(
                node,
                m,
                loaded_placeholder_m,
                target_device,
                wrap_cfg,
                quant_cfg,
                output_dir,
            )
            self._release_place_holder_module(loaded_placeholder_m)
            del loaded_placeholder_m

    def export_prefill_decode_placeholder_layers(
        self,
        prefill_fronted_graph_module: FrontendGraph,
        decode_fronted_graph_module: FrontendGraph,
        empty_hf_model_for_placeholder: PreTrainedModel,
        target_device,
        prefill_wrap_cfg,
        decode_wrap_cfg,
        quant_cfg,
        prefill_output_dir: str | Path,
        decode_output_dir: str | Path,
        placeholder_export_workers: int | None = None,
        empty_hf_model_factory=None,
    ):
        """按 module target 复用权重，分别导出 prefill 和 decode placeholder 子图。

        同名模块只从 safetensors 加载和反量化一次，再针对两种调用契约各生成一份
        HMONNX，从而降低磁盘读取、反量化开销和峰值内存。
        """
        prefill_place_modules = self._collect_place_holder_modules(prefill_fronted_graph_module)
        decode_place_modules = self._collect_place_holder_modules(decode_fronted_graph_module)
        prefill_modules_by_target = {str(node.target): (node, m) for node, m in prefill_place_modules}
        decode_modules_by_target = {str(node.target): (node, m) for node, m in decode_place_modules}

        # 使用有序并集：优先保持 prefill 图顺序，再追加 decode 独有模块。
        module_targets = list(prefill_modules_by_target)
        module_targets.extend(target for target in decode_modules_by_target if target not in prefill_modules_by_target)

        logger = get_xhquant_logger()
        placeholder_export_workers = _placeholder_export_worker_count(placeholder_export_workers)
        logger.info("Export prefill/decode PlaceHolder layers with %d worker(s)", placeholder_export_workers)
        reuse_existing = os.environ.get("XH2MODELZOO_REUSE_BIG_MODEL_PLACEHOLDERS", "").lower() in {
            "1",
            "true",
            "yes",
        }

        def can_reuse(path: Path, placeholder_module) -> bool:
            if not reuse_existing or not path.is_file():
                return False
            expected_outputs = getattr(placeholder_module, "num_outputs", None)
            if expected_outputs is None:
                return True
            try:
                existing_model = onnx.load(str(path), load_external_data=False)
            except Exception as exc:
                logger.warning("Cannot reuse placeholder HMONNX %s: %s", path, exc)
                return False
            actual_outputs = len(existing_model.graph.output)
            if actual_outputs != int(expected_outputs):
                logger.info(
                    "Regenerating placeholder HMONNX %s because output count changed: %d != %d",
                    path,
                    actual_outputs,
                    int(expected_outputs),
                )
                return False
            return True

        def export_module_target(module_target: str) -> str:
            logger.info(f"*************** export {module_target} as PlaceHolder ***************")
            placeholder_filename = f"{_module_name_to_filename(module_target)}.onnx"
            prefill_hmonnx_file = Path(prefill_output_dir) / placeholder_filename
            decode_hmonnx_file = Path(decode_output_dir) / placeholder_filename
            prefill_module = prefill_modules_by_target.get(module_target)
            decode_module = decode_modules_by_target.get(module_target)
            export_prefill = prefill_module is not None and not can_reuse(prefill_hmonnx_file, prefill_module[1])
            export_decode = decode_module is not None and not can_reuse(decode_hmonnx_file, decode_module[1])
            if not export_prefill and not export_decode:
                logger.info("Reusing existing prefill/decode placeholder HMONNX for %s", module_target)
                return module_target
            loaded_placeholder_m = self._load_place_holder_module_once(
                empty_hf_model_for_placeholder,
                module_target,
            )
            try:
                if export_prefill:
                    node, m = prefill_modules_by_target[module_target]
                    self._export_loaded_place_holder_layer(
                        node,
                        m,
                        loaded_placeholder_m,
                        target_device,
                        prefill_wrap_cfg,
                        quant_cfg,
                        prefill_output_dir,
                    )
                if export_decode:
                    node, m = decode_modules_by_target[module_target]
                    self._export_loaded_place_holder_layer(
                        node,
                        m,
                        loaded_placeholder_m,
                        target_device,
                        decode_wrap_cfg,
                        quant_cfg,
                        decode_output_dir,
                    )
                return module_target
            finally:
                self._release_place_holder_module(loaded_placeholder_m)
                del loaded_placeholder_m

        def process_task(module_target: str, execution_device: str | None = None) -> dict[str, Any] | None:
            placeholder_filename = f"{_module_name_to_filename(module_target)}.onnx"
            prefill_hmonnx_file = Path(prefill_output_dir) / placeholder_filename
            decode_hmonnx_file = Path(decode_output_dir) / placeholder_filename
            prefill_module = prefill_modules_by_target.get(module_target)
            decode_module = decode_modules_by_target.get(module_target)
            export_prefill = prefill_module is not None and not can_reuse(prefill_hmonnx_file, prefill_module[1])
            export_decode = decode_module is not None and not can_reuse(decode_hmonnx_file, decode_module[1])
            if not export_prefill and not export_decode:
                logger.info("Reusing existing prefill/decode placeholder HMONNX for %s", module_target)
                return None

            modes = []
            for name, module_entry, should_export, wrap_cfg, output_dir in (
                ("prefill", prefill_module, export_prefill, prefill_wrap_cfg, prefill_output_dir),
                ("decode", decode_module, export_decode, decode_wrap_cfg, decode_output_dir),
            ):
                if not should_export or module_entry is None:
                    continue
                node, placeholder_module = module_entry
                merged_args = _merge_call_args_kwargs_as_args(placeholder_module, node.args, node.kwargs)
                args_template = _serializable_meta_template(merged_args)
                expected_outputs = getattr(placeholder_module, "num_outputs", None)
                if expected_outputs is not None and int(expected_outputs) < 0:
                    expected_outputs = None
                modes.append(
                    {
                        "name": name,
                        "args_template": args_template,
                        "expected_inputs": len(_placeholder_input_fake_tensors(args_template)),
                        "expected_outputs": expected_outputs,
                        "wrap_cfg": copy.deepcopy(wrap_cfg),
                        "output_dir": str(output_dir),
                    }
                )
            return {
                "module_target": module_target,
                "modes": modes,
                "target_device": target_device,
                "execution_device": execution_device,
                "quant_cfg": copy.deepcopy(quant_cfg),
            }

        if placeholder_export_workers == 1 or len(module_targets) <= 1:
            pbar = tqdm(module_targets, desc="Export prefill/decode PlaceHolder layers")
            for module_target in pbar:
                pbar.set_description(f"export {module_target} as PlaceHolder")
                export_module_target(module_target)
        else:
            if empty_hf_model_factory is None:
                raise ValueError("empty_hf_model_factory is required for process-level placeholder export.")
            execution_devices = _placeholder_export_execution_devices()
            tasks = [
                task
                for index, module_target in enumerate(module_targets)
                if (task := process_task(module_target, execution_devices[index % len(execution_devices)])) is not None
            ]
            device_counts = {device: 0 for device in execution_devices}
            for task in tasks:
                device_counts[task["execution_device"]] = device_counts.get(task["execution_device"], 0) + 1
            logger.info(
                "Starting %d isolated placeholder export process(es) for %d target(s) across execution devices: %s",
                placeholder_export_workers,
                len(tasks),
                device_counts,
            )
            with ProcessPoolExecutor(
                max_workers=placeholder_export_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_placeholder_export_process,
                initargs=(type(self), self._hf_model_dir, empty_hf_model_factory),
            ) as executor:
                futures = {
                    executor.submit(_export_placeholder_process_task, task): task["module_target"] for task in tasks
                }
                pbar = tqdm(as_completed(futures), total=len(futures), desc="Export prefill/decode PlaceHolder layers")
                for future in pbar:
                    module_target = futures[future]
                    pbar.set_description(f"exported {module_target} as PlaceHolder")
                    try:
                        result = future.result()
                    except Exception as exc:
                        for pending_future in futures:
                            if pending_future is not future:
                                pending_future.cancel()
                        _raise_placeholder_process_failure(module_target, exc)
                    if result["module_target"] != module_target:
                        raise RuntimeError(
                            "Placeholder worker returned target "
                            f"{result['module_target']!r}, expected {module_target!r}."
                        )

    # HMONNX 主图与子图组装

    def _replace_placeholder(self, placeholder_node, subgraph_hmonnx_file: str):
        """单节点替换需要主图和外部数据上下文，请使用批量替换接口。"""
        raise RuntimeError("Use replace_hmonnx_placeholders_with_subgraphs() to preserve HMONNX external data.")

    def replace_hmonnx_placeholders_with_subgraphs(
        self,
        hmonnx_file: str,
        cleanup_temporary_files: bool | None = None,
    ) -> None:
        """扫描 HMONNX 主图并使用 ``content`` 指向的真实子图替换 placeholder。

        ``cleanup_temporary_files`` 可覆盖构造时的全局设置。清理仅发生在子图合并
        成功并通过 ONNX 校验之后；失败时会保留 placeholder 和备份文件用于回滚与排查。
        """
        logger = get_xhquant_logger()
        logger.info(f"************ Replace PlaceHolder for {hmonnx_file} ************")

        if cleanup_temporary_files is None:
            cleanup_temporary_files = getattr(self, "_cleanup_temporary_files", None)
        if cleanup_temporary_files is None:
            cleanup_temporary_files = _cleanup_big_model_export_temporary_files()

        hmonnx_path = Path(hmonnx_file)
        placeholders_path = hmonnx_path.parent / "placeholders"
        subgraphs_by_content = {}
        for subgraph_file in placeholders_path.glob("*.onnx"):
            subgraphs_by_content[subgraph_file.stem] = subgraph_file

        # ``content`` stores the module target, while files use its normalized form.
        class _NormalizedSubgraphMap(dict):
            def get(self, key, default=None):
                return super().get(_module_name_to_filename(key), default)

        replace_placeholders_with_subgraphs(
            hmonnx_file,
            _NormalizedSubgraphMap(subgraphs_by_content),
            logger,
            cleanup_temporary_files=cleanup_temporary_files,
        )
        if cleanup_temporary_files and placeholders_path.exists():
            shutil.rmtree(placeholders_path, ignore_errors=False)
            logger.info("Removed temporary placeholder HMONNX directory: %s", placeholders_path)
