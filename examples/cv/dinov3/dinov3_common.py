# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for DINOv3 LT-DETR export, quantization, search, and evaluation.

This module targets the HMSW-4200 model named in the Jira discussion:
``dinov3/vitt16-ltdetr-coco``. It is intentionally separate from the existing
backbone-only DINOv3 scripts because this model is an object detector with
``images`` input and ``labels``, ``boxes``, ``scores`` outputs.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# 全局默认值 & 常量
# ═══════════════════════════════════════════════════════════════════════════════
import argparse
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# 路径常量：所有产物默认放在 examples/cv/dinov3/ 下
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "dinov3/vitt16-ltdetr-coco"
DEFAULT_IMAGE_SIZE = 640
DEFAULT_QUANT_TYPE = "w8a8h1_sefp"
DEFAULT_ONNX = SCRIPT_DIR / "onnx" / "dinov3-vitt16-ltdetr-coco_is640.onnx"
DEFAULT_WORK_DIR = SCRIPT_DIR / "work_dirs" / "dinov3-vitt16-ltdetr-coco"
DEFAULT_MODEL_CACHE_DIR = SCRIPT_DIR / "models"
DEFAULT_DATA_CACHE_DIR = SCRIPT_DIR / "data_cache"
DEFAULT_DEBUG_OUTPUT_DIR = SCRIPT_DIR / "debug"
DEFAULT_DEBUG_INPUT_DIR = DEFAULT_DEBUG_OUTPUT_DIR / "debug_inputs" / "dinov3-vitt16-ltdetr-coco"
DEFAULT_COMPARE_DIR = SCRIPT_DIR / "compare"
DEFAULT_COCO_EVAL_DIR = SCRIPT_DIR / "coco_eval"
DEFAULT_SEARCH_DIR = SCRIPT_DIR / "mixed_precision_search"
W16_WEIGHT_SCHEMA = {"bits": 16, "fp_mode": "sefp", "hidden_bit": True}
W16_ACT_SCHEMA = {"bits": 16}
NO_PROXY_ENV_NAMES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 配置文件加载与校验
# 支持 JSON/YAML 格式的混合精度配置，校验 nodes/ops 映射合法性
# ═══════════════════════════════════════════════════════════════════════════════


class OptionalDependencyError(RuntimeError):
    """当可选依赖（如 PyYAML）未安装时抛出。"""


def read_mapping_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON or YAML mapping file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise OptionalDependencyError("PyYAML is required for YAML mixed precision configs.") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected {config_path} to contain a mapping, got {type(data).__name__}")
    return data


def _validate_override_mapping(name: str, value: Any) -> dict[str, str | dict[str, Any]]:
    """校验混精配置中的 nodes/ops 映射格式，返回归一化后的 dict。"""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"mixed precision '{name}' must be a mapping")

    normalized: dict[str, str | dict[str, Any]] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"mixed precision '{name}' keys must be non-empty strings")
        if isinstance(item, str):
            if not item:
                raise ValueError(f"mixed precision '{name}.{key}' quant type must be non-empty")
            normalized[key] = item
        elif isinstance(item, dict):
            normalized[key] = dict(item)
        else:
            raise ValueError(f"mixed precision '{name}.{key}' must be a quant type string or mapping")
    return normalized


def load_mixed_precision_config(path: str | Path | None) -> dict[str, Any]:
    """Load and normalize the mixed precision config used by QuantScheme.

    Supported JSON/YAML shape:

    {
      "nodes": {"node_name": "w16a16h1_sefp"},
      "ops": {"MatMul": "w16a16h1_sefp"},
      "input_enable_fp32": false,
      "output_enable_fp32": true
    }
    """
    if path is None:
        return {"nodes": {}, "ops": {}}

    raw = read_mapping_file(path)
    allowed = {"nodes", "ops", "input_enable_fp32", "output_enable_fp32"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unsupported mixed precision keys: {unknown}")

    normalized: dict[str, Any] = {
        "nodes": _validate_override_mapping("nodes", raw.get("nodes", {})),
        "ops": _validate_override_mapping("ops", raw.get("ops", {})),
    }
    for flag in ("input_enable_fp32", "output_enable_fp32"):
        if flag in raw:
            if not isinstance(raw[flag], bool):
                raise ValueError(f"mixed precision '{flag}' must be a boolean")
            normalized[flag] = raw[flag]
    return normalized


# ═══════════════════════════════════════════════════════════════════════════════
# QuantScheme 参数构建 & ONNX 节点名转换
# 将 CLI 参数和混精配置组装成 xhquant API 所需的 QuantScheme kwargs
# ═══════════════════════════════════════════════════════════════════════════════

def make_quant_scheme_kwargs(
    *,
    quant_type: str,
    mixed_precision: Mapping[str, Any] | None = None,
    input_enable_fp32: bool = False,
    output_enable_fp32: bool = True,
) -> dict[str, Any]:
    """Build kwargs for xhquant.api.QuantScheme from CLI/config inputs."""
    mp = dict(mixed_precision or {})
    return {
        "quant_type": quant_type,
        "nodes": dict(mp.get("nodes", {})),
        "ops": dict(mp.get("ops", {})),
        "input_enable_fp32": bool(mp.get("input_enable_fp32", input_enable_fp32)),
        "output_enable_fp32": bool(mp.get("output_enable_fp32", output_enable_fp32)),
    }


# ONNX 节点名 → xhquant 内部名：去除斜杠、驼峰转下划线、小写

def onnx_name_to_xhquant_name(name: str) -> str:
    """Approximate xhquant's imported node naming for ONNX slash-style names."""
    normalized = name.strip("/")
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", normalized)
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", normalized).strip("_").lower()
    return f"_{normalized}" if normalized else name


# 根据精度模式（weight/activation/both）构建单节点的精度 schema
# MatMul 的权重在 i_cfg_2（第二个输入），其余算子在 w_cfg

def split_precision_node_config(mode: str, op_type: str | None = None) -> dict[str, Any]:
    """
    根据精度模式（weight/te_activation/both）和算子类型，构建单节点的精度覆盖 schema。
    MatMul 的权重在 i_cfg_2（第二个输入），其余算子在 w_cfg。
    """
    if mode not in {"weight", "te_activation", "both"}:
        raise ValueError(f"Unsupported precision mode: {mode}")
    config: dict[str, Any] = {}
    if mode in {"weight", "both"}:
        if op_type == "MatMul":
            config["act_schema_2"] = dict(W16_ACT_SCHEMA)
        else:
            config["w_schema"] = dict(W16_WEIGHT_SCHEMA)
    if mode in {"te_activation", "both"}:
        config["act_schema"] = dict(W16_ACT_SCHEMA)
    return config


# 从 ONNX 模型中收集所有节点名 → op_type 的映射
def collect_mixed_precision_node_ops(onnx_path: str | Path) -> dict[str, str]:
    """从 ONNX 模型中收集所有节点名 → op_type 的映射，用于确定每个节点的算子类型。"""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    return {onnx_name_to_xhquant_name(node.name): node.op_type for node in model.graph.node if node.name}


# 按模块分组收集 ONNX TE/weighted 节点，用于批量生成分组搜索配置。
# 分组逻辑：backbone_blocks / backbone_sta / decoder_layers / decoder_heads 等。
# activation-only 节点不生成配置，也不进入后续搜索。

def collect_mixed_precision_groups(onnx_path: str | Path) -> dict[str, list[str]]:
    """按模块分组收集 ONNX TE/weighted 节点名，用于批量生成分组搜索配置。"""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    groups: dict[str, set[str]] = {}

    def add(group: str, node_name: str) -> None:
        """辅助函数：将节点名添加到指定分组的 set 中。"""
        groups.setdefault(group, set()).add(node_name)

    te_weighted_ops = {"Conv", "Gemm", "MatMul"}
    for node in model.graph.node:
        raw_name = node.name
        xh_name = onnx_name_to_xhquant_name(raw_name)
        if node.op_type not in te_weighted_ops:
            continue

        if raw_name.startswith("/backbone/blocks."):
            add("backbone_blocks_weighted", xh_name)
            match = re.search(r"/backbone/blocks\.(\d+)/", raw_name)
            if match:
                block_id = int(match.group(1))
                add(f"backbone_blocks_{block_id // 3 * 3}_{block_id // 3 * 3 + 2}_weighted", xh_name)
        if raw_name.startswith("/backbone/sta/"):
            add("backbone_sta_weighted", xh_name)
        if raw_name.startswith("/backbone/"):
            add("backbone_all_weighted", xh_name)
        if raw_name.startswith("/decoder/decoder/layers."):
            add("decoder_layers_weighted", xh_name)
        if raw_name.startswith("/decoder/decoder/dec_bbox_head") or raw_name.startswith(
            "/decoder/decoder/dec_score_head"
        ):
            add("decoder_heads_weighted", xh_name)
        if raw_name.startswith("/decoder/"):
            add("decoder_all_weighted", xh_name)

    return {group: sorted(nodes) for group, nodes in sorted(groups.items()) if nodes}


# ═══════════════════════════════════════════════════════════════════════════════
# 分组搜索配置生成器
# 自动按模块分组（backbone/decoder/heads），每组生成 weight/te_activation/both 三种配置。
# 只覆盖 TE/weighted 节点，不再生成 activation-only 配置。
# ═══════════════════════════════════════════════════════════════════════════════

def write_mixed_precision_search_configs(args: argparse.Namespace) -> Path:
    """write-search-configs 子命令入口。按模块分组生成 TE/weighted 混精配置 JSON，输出 manifest 索引文件。"""
    out_dir = Path(args.out_dir).expanduser().resolve()
    config_dir = out_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    groups = collect_mixed_precision_groups(args.onnx)
    node_ops = collect_mixed_precision_node_ops(args.onnx)
    weighted_modes = ("weight", "te_activation", "both")
    manifest: list[dict[str, Any]] = []

    for group_name, nodes in groups.items():
        for mode in weighted_modes:
            case_name = f"{group_name}_{mode}"
            config = {
                "nodes": {
                    node_name: split_precision_node_config(mode, op_type=node_ops.get(node_name))
                    for node_name in nodes
                },
                "ops": {},
                "input_enable_fp32": False,
                "output_enable_fp32": True,
            }
            config_path = config_dir / f"{case_name}.json"
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
            manifest.append(
                {
                    "case": case_name,
                    "group": group_name,
                    "mode": mode,
                    "node_count": len(nodes),
                    "config": str(config_path),
                }
            )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Mixed precision search configs: {config_dir}")
    print(f"Manifest: {manifest_path}")
    return manifest_path


# ═══════════════════════════════════════════════════════════════════════════════
# LightlyTrain 模型加载 & ONNX 导出
# 通过 lightly-train SDK 加载预训练 LT-DETR 并导出 ONNX
# 注意：lightly-train 是 AGPL 许可，不要 vendor 其源码
# ═══════════════════════════════════════════════════════════════════════════════

def set_lightly_cache_env(model_cache_dir: str | Path, data_cache_dir: str | Path) -> None:
    """Keep LightlyTrain downloads under examples/cv/dinov3 unless explicitly overridden."""
    model_cache = Path(model_cache_dir).expanduser().resolve()
    data_cache = Path(data_cache_dir).expanduser().resolve()
    model_cache.mkdir(parents=True, exist_ok=True)
    data_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIGHTLY_TRAIN_MODEL_CACHE_DIR", str(model_cache))
    os.environ.setdefault("LIGHTLY_TRAIN_DATA_CACHE_DIR", str(data_cache))


def require_lightly_train():
    """检查 lightly_train 是否可导入，不可用时抛出友好的安装提示。"""
    try:
        import lightly_train
    except ImportError as exc:
        raise OptionalDependencyError(
            "lightly_train is required for this command. Install it in the active environment, "
            "for example: python -m pip install lightly-train==0.15.1"
        ) from exc
    return lightly_train


def _resolve_device_arg(device: str) -> str | None:
    """将 CLI 的 device 参数转为 lightly_train 可接受的值（auto → None）。"""
    if device == "auto":
        return None
    return device


def resolve_runtime_device(device: str) -> str:
    """解析运行时设备：auto 时自动检测 CUDA 可用性，否则直接返回指定设备。"""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_lightly_model(args: argparse.Namespace):
    """设置缓存环境变量后，通过 lightly_train SDK 加载预训练模型。"""
    set_lightly_cache_env(args.model_cache_dir, args.data_cache_dir)
    lightly_train = require_lightly_train()
    return lightly_train.load_model(args.model_name, device=_resolve_device_arg(args.device))


def export_onnx(args: argparse.Namespace) -> Path:
    """export-onnx 子命令入口。加载 LightlyTrain 模型并导出 ONNX，支持 simplify/verify/opset 等选项。"""
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    model = load_lightly_model(args)
    simplify = not args.no_simplify
    if simplify and importlib.util.find_spec("onnxslim") is None:
        print("onnxslim is not installed; exporting without simplification.")
        simplify = False
    model.export_onnx(
        out,
        precision=args.precision,
        batch_size=args.batch_size,
        dynamic_batch_size=args.dynamic_batch,
        opset_version=args.opset,
        simplify=simplify,
        verify=not args.no_verify,
        num_channels=args.num_channels,
    )
    print(f"ONNX exported: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX I/O 自动检测 & HMONNX 量化导出
# detect_onnx_io: 自动解析 ONNX 输入输出名和 shape
# quantize: 基础量化子命令，支持混精配置和 golden 推理
# ═══════════════════════════════════════════════════════════════════════════════

def _onnx_dim_to_int(dim: Any, fallback: int) -> int:
    """将 ONNX dimension 对象转为 int，dim_value 为 0 时返回 fallback。"""
    if getattr(dim, "dim_value", 0):
        return int(dim.dim_value)
    return fallback


def detect_onnx_io(
    onnx_path: str | Path,
    *,
    batch_size: int = 1,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[str, tuple[int, ...], list[str]]:
    """自动检测 ONNX 模型的输入名/shape 和输出名列表，用于构建推理 session。"""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    initializer_names = {init.name for init in model.graph.initializer}
    graph_inputs = [inp for inp in model.graph.input if inp.name not in initializer_names]
    if len(graph_inputs) != 1:
        names = [inp.name for inp in graph_inputs]
        raise ValueError(f"Expected one non-initializer ONNX input, got {names}")

    net_input = graph_inputs[0]
    dims = list(net_input.type.tensor_type.shape.dim)
    if len(dims) != 4:
        raise ValueError(f"Expected NCHW input for {net_input.name!r}, got rank {len(dims)}")

    shape = (
        _onnx_dim_to_int(dims[0], batch_size),
        _onnx_dim_to_int(dims[1], 3),
        _onnx_dim_to_int(dims[2], image_size),
        _onnx_dim_to_int(dims[3], image_size),
    )
    output_names = [output.name for output in model.graph.output]
    return net_input.name, shape, output_names


def build_quant_config(args: argparse.Namespace, mixed_precision: Mapping[str, Any] | None = None):
    """根据 CLI 参数和混精配置，构建 xhquant 的 QuantConfig 对象。"""
    from xhquant.api import DeviceType, QuantScheme, create_quant_config

    quant_scheme_kwargs = make_quant_scheme_kwargs(
        quant_type=args.quant_type,
        mixed_precision=mixed_precision,
        input_enable_fp32=args.input_enable_fp32,
        output_enable_fp32=args.output_enable_fp32,
    )
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, **quant_scheme_kwargs)
    return create_quant_config(quant_scheme)


# quantize 子命令入口：ONNX → HMONNX，可选生成 golden 推理数据

def quantize(args: argparse.Namespace) -> Path:
    """quantize 子命令入口。将 ONNX 转为 HMONNX，可选生成 golden 推理数据。"""
    import torch
    from xhquant.api import DeviceType, HMONNXGoldenInference, convert_onnx_to_hmonnx

    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    input_name, input_shape, output_names = detect_onnx_io(
        onnx_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    if args.output_names:
        output_names = [name for name in args.output_names.split(",") if name]

    mixed_precision = load_mixed_precision_config(args.mixed_precision_config)
    quant_config = build_quant_config(args, mixed_precision)

    out = Path(args.out).expanduser().resolve() if args.out else default_hmonnx_path(onnx_path, args)
    out.parent.mkdir(parents=True, exist_ok=True)

    sample_input = torch.randn(*input_shape, dtype=torch.float32)
    print(f"Quantizing ONNX: {onnx_path}")
    print(f"Input: {input_name} shape={tuple(sample_input.shape)}")
    print(f"Outputs: {output_names}")
    print(f"Quant type: {args.quant_type}; nodes={len(mixed_precision['nodes'])}; ops={len(mixed_precision['ops'])}")
    convert_onnx_to_hmonnx(
        str(onnx_path),
        [sample_input],
        DeviceType.XH2a,
        str(out),
        quant_config=quant_config,
        input_names=[input_name],
        output_names=output_names,
    )
    print(f"HMONNX exported: {out}")

    if not args.skip_golden:
        session = HMONNXGoldenInference(str(out))
        session.to(args.device)
        session.save_golden = True
        session.golden_dir = str(out.parent / f"golden_{args.quant_type}")
        session.step = 0
        session(sample_input.to(device=args.device, dtype=torch.float16))
        print(f"Golden generated: {session.golden_dir}")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 推理 Runner 封装
# make_onnx_runner / make_hmonnx_runner：创建可复用的推理闭包
# 避免每次推理都重新加载模型，提升评测效率
# ═══════════════════════════════════════════════════════════════════════════════

def default_hmonnx_path(onnx_path: Path, args: argparse.Namespace) -> Path:
    """根据 ONNX 路径和量化类型，生成默认的 HMONNX 输出路径。"""
    work_dir = Path(args.work_dir).expanduser().resolve() / onnx_path.stem / "hmonnx"
    return work_dir / f"{onnx_path.stem}_{args.quant_type}_XH2a.onnx"


def _as_numpy(value: Any) -> np.ndarray:
    """将 tensor/ndarray 统一转为 numpy，兼容 torch.Tensor 和 np.ndarray。"""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def preprocess_image_batch(model: Any, image: str | Path) -> tuple[Any, dict[str, Any]]:
    """用 LightlyTrain 模型的预处理管线处理单张图片，返回 (batch, metadata)。"""
    import torch

    x, metadata = model.preprocess_image(image)
    with torch.no_grad():
        batch = model.preprocess_batch(x.unsqueeze(0))
    return batch, metadata


def make_input(args: argparse.Namespace) -> Path:
    """make-input 子命令入口。将真实图片预处理为 images.npy 用于 debug/compare。"""
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_lightly_model(args)
    model.eval()
    model.deploy()
    batch, metadata = preprocess_image_batch(model, args.image)

    input_path = out_dir / "images.npy"
    metadata_path = out_dir / "metadata.json"
    np.save(input_path, _as_numpy(batch).astype(np.float32))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved debug input: {input_path}")
    print(f"Saved metadata: {metadata_path}")
    return out_dir


def make_onnx_runner(onnx_path: str | Path):
    """创建 ONNX Runtime 推理闭包，返回 run(batch) 函数，避免重复创建 session。"""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def run(batch: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """执行 ONNX Runtime 推理，返回 (labels, boxes, scores) 的 numpy 元组。"""
        outputs = session.run(None, {input_name: _as_numpy(batch).astype(np.float32)})
        return tuple(np.asarray(output) for output in outputs)  # type: ignore[return-value]

    return run


def run_onnx(onnx_path: str | Path, batch: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """便捷函数：创建 ONNX runner 并立即执行一次推理。"""
    return make_onnx_runner(onnx_path)(batch)


def make_hmonnx_runner(hmonnx_path: str | Path, device: str):
    """创建 HMONNX 推理闭包，返回 run(batch) 函数，session 只加载一次。"""
    import torch
    from xhquant.api import HMONNXGoldenInference

    device = resolve_runtime_device(device)
    session = HMONNXGoldenInference(str(hmonnx_path))
    session.to(device)

    def run(batch: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """执行 HMONNX 推理，返回 (labels, boxes, scores) 的 numpy 元组。"""
        output = session(batch.to(device=device, dtype=torch.float16))
        if not isinstance(output, (tuple, list)):
            output = (output,)
        return tuple(_as_numpy(item) for item in output)  # type: ignore[return-value]

    return run


def run_hmonnx(hmonnx_path: str | Path, batch: Any, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """便捷函数：创建 HMONNX runner 并立即执行一次推理。"""
    return make_hmonnx_runner(hmonnx_path, device)(batch)


def metric_summary(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float | int | list[int]]:
    """计算两个 numpy 数组的相似度指标：cosine、MSE、max_abs_diff 等。用于对比量化前后输出。"""
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.shape != rhs.shape:
        return {"shape_match": 0, "lhs_shape": list(lhs.shape), "rhs_shape": list(rhs.shape)}

    if np.issubdtype(lhs.dtype, np.integer) or np.issubdtype(rhs.dtype, np.integer):
        return {
            "shape_match": 1,
            "exact_match_rate": float(np.mean(lhs == rhs)),
            "num_elements": int(lhs.size),
        }

    lhs_f = lhs.astype(np.float32).reshape(-1)
    rhs_f = rhs.astype(np.float32).reshape(-1)
    diff = lhs_f - rhs_f
    denom = float(np.linalg.norm(lhs_f) * np.linalg.norm(rhs_f))
    cos = float(np.dot(lhs_f, rhs_f) / denom) if denom > 0 else 1.0
    return {
        "shape_match": 1,
        "cosine_similarity": cos,
        "mse": float(np.mean(diff * diff)),
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 检测结果后处理 & 单图对比
# detections_from_outputs: 从模型输出提取检测框
# compare: 子命令，对比 Torch/ONNX/HMONNX 在同一张图上的输出差异
# ═══════════════════════════════════════════════════════════════════════════════

def detections_from_outputs(
    outputs: Sequence[np.ndarray],
    *,
    threshold: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    """从模型输出 (labels, boxes, scores) 中提取检测结果，按 score 过滤和排序。"""
    labels, boxes, scores = (np.asarray(item) for item in outputs)
    labels = labels[0]
    boxes = boxes[0]
    scores = scores[0]
    keep = np.where(scores > threshold)[0]
    if keep.size == 0:
        keep = np.argsort(scores)[::-1][:max_detections]
    else:
        keep = keep[np.argsort(scores[keep])[::-1]][:max_detections]

    rows = []
    for idx in keep:
        rows.append(
            {
                "label": int(labels[idx]),
                "score": float(scores[idx]),
                "box": [float(x) for x in boxes[idx].tolist()],
            }
        )
    return rows


def draw_detections(image: str | Path, detections: Sequence[Mapping[str, Any]], out: str | Path) -> None:
    """在图片上绘制检测框和标签，保存为 JPEG。用于可视化对比。"""
    from PIL import Image, ImageDraw

    img = Image.open(image).convert("RGB").resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
    draw = ImageDraw.Draw(img)
    for det in detections:
        x0, y0, x1, y1 = det["box"]
        draw.rectangle((x0, y0, x1, y1), outline="red", width=2)
        draw.text((x0, max(0, y0 - 12)), f"{det['label']} {det['score']:.3f}", fill="red")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def compare(args: argparse.Namespace) -> Path:
    """compare 子命令入口。对比 Torch/ONNX/HMONNX 在同一张图上的输出差异，生成对比报告和可视化。"""
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_lightly_model(args)
    model.eval()
    model.deploy()
    batch, _metadata = preprocess_image_batch(model, args.image)

    import torch

    with torch.no_grad():
        torch_outputs = tuple(_as_numpy(item) for item in model(batch))

    report: dict[str, Any] = {
        "model": args.model_name,
        "image": str(args.image),
        "threshold": args.threshold,
        "torch_detections": detections_from_outputs(
            torch_outputs,
            threshold=args.threshold,
            max_detections=args.max_detections,
        ),
        "metrics": {},
    }

    if args.onnx:
        onnx_outputs = run_onnx(args.onnx, batch)
        report["onnx_detections"] = detections_from_outputs(
            onnx_outputs,
            threshold=args.threshold,
            max_detections=args.max_detections,
        )
        report["metrics"]["onnx_vs_torch"] = {
            "labels": metric_summary(torch_outputs[0], onnx_outputs[0]),
            "boxes": metric_summary(torch_outputs[1], onnx_outputs[1]),
            "scores": metric_summary(torch_outputs[2], onnx_outputs[2]),
        }

    if args.hmonnx:
        hmonnx_outputs = run_hmonnx(args.hmonnx, batch, args.device)
        report["hmonnx_detections"] = detections_from_outputs(
            hmonnx_outputs,
            threshold=args.threshold,
            max_detections=args.max_detections,
        )
        report["metrics"]["hmonnx_vs_torch"] = {
            "labels": metric_summary(torch_outputs[0], hmonnx_outputs[0]),
            "boxes": metric_summary(torch_outputs[1], hmonnx_outputs[1]),
            "scores": metric_summary(torch_outputs[2], hmonnx_outputs[2]),
        }
        draw_detections(args.image, report["hmonnx_detections"], out_dir / "hmonnx_detections.jpg")

    draw_detections(args.image, report["torch_detections"], out_dir / "torch_detections.jpg")
    report_path = out_dir / "compare.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Compare report: {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# COCO mAP 评测
# run_torch_with_original_size: 用原始分辨率跑 Torch 推理（更准确）
# scale_resized_boxes_to_original: 将 640x640 输出的 bbox 缩放回原图尺寸
# map_contiguous_label_to_category_id: 模型连续标签 → COCO category_id
# outputs_to_coco_rows: 模型输出 → COCO 预测格式（image_id, category_id, bbox, score）
# ═══════════════════════════════════════════════════════════════════════════════

def run_torch_with_original_size(model: Any, batch: Any, metadata: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用原始分辨率跑 Torch 推理（传入 orig_target_size），比 resize 到 640 更准确。"""
    import torch

    orig_target_size = torch.tensor(
        [[int(metadata["orig_h"]), int(metadata["orig_w"])]],
        dtype=torch.int64,
        device=batch.device,
    )
    with torch.no_grad():
        outputs = model(batch, orig_target_size=orig_target_size)
    return tuple(_as_numpy(item) for item in outputs)  # type: ignore[return-value]


def scale_resized_boxes_to_original(
    outputs: Sequence[np.ndarray],
    metadata: Mapping[str, Any],
    input_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将 640×640 输出的 bbox 坐标缩放回原始图片尺寸。"""
    labels, boxes, scores = (np.asarray(item).copy() for item in outputs)
    in_h, in_w = input_hw
    orig_h = float(metadata["orig_h"])
    orig_w = float(metadata["orig_w"])
    boxes[..., [0, 2]] *= orig_w / float(in_w)
    boxes[..., [1, 3]] *= orig_h / float(in_h)
    return labels, boxes, scores


def map_contiguous_label_to_category_id(label: int, category_ids: Sequence[int] | None) -> int:
    """将模型的连续标签（0,1,2...）映射为 COCO 的 category_id。"""
    if category_ids is None:
        return label
    if 0 <= label < len(category_ids):
        return int(category_ids[label])
    return label


def outputs_to_coco_rows(
    outputs: Sequence[np.ndarray],
    *,
    image_id: int,
    score_threshold: float,
    max_detections: int,
    category_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """将模型输出转为 COCO 预测格式（image_id, category_id, bbox, score）。"""
    labels, boxes, scores = (np.asarray(item) for item in outputs)
    labels = labels[0]
    boxes = boxes[0]
    scores = scores[0]
    keep = np.where(scores >= score_threshold)[0]
    keep = keep[np.argsort(scores[keep])[::-1]][:max_detections]

    rows: list[dict[str, Any]] = []
    for idx in keep:
        x0, y0, x1, y1 = [float(v) for v in boxes[idx].tolist()]
        label = int(labels[idx])
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": map_contiguous_label_to_category_id(label, category_ids),
                "bbox": [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)],
                "score": float(scores[idx]),
            }
        )
    return rows


def select_coco_image_ids(
    image_ids: Sequence[int],
    *,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int = 0,
) -> list[int]:
    """从 COCO image ID 列表中按 sample_size 随机采样，支持 limit 截断。"""
    selected = list(image_ids)
    if sample_size is not None:
        if sample_size < 1:
            raise ValueError("--sample-size must be positive")
        if sample_size < len(selected):
            rng = random.Random(sample_seed)
            selected = sorted(rng.sample(selected, sample_size))
    if limit is not None:
        selected = selected[:limit]
    return selected


# coco_eval 子命令入口：对 Torch/ONNX/HMONNX 做 COCO bbox mAP 评测
# 支持 --sample-size 做随机子集评测（如 sample200），用于快速验证混精效果

def coco_eval(args: argparse.Namespace) -> Path:
    """coco-eval 子命令入口。对 Torch/ONNX/HMONNX 做 COCO bbox mAP 评测，支持子集采样。"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    coco_gt = COCO(str(args.annotations))
    image_ids = coco_gt.getImgIds()
    if args.skip_missing_images:
        image_root = Path(args.images_dir)
        filtered_image_ids = []
        for image_id in image_ids:
            info = coco_gt.loadImgs([image_id])[0]
            if (image_root / info["file_name"]).exists():
                filtered_image_ids.append(image_id)
        missing = len(image_ids) - len(filtered_image_ids)
        if missing:
            print(f"Skipping {missing} COCO images missing from {image_root}")
        image_ids = filtered_image_ids
    image_ids = select_coco_image_ids(
        image_ids,
        limit=args.limit,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    if not image_ids:
        raise RuntimeError("No COCO images selected for evaluation.")
    category_ids = sorted(coco_gt.getCatIds())

    model = load_lightly_model(args)
    model.eval()
    model.deploy()
    if args.backend == "onnx" and not args.onnx:
        raise ValueError("--onnx is required for --backend onnx")
    if args.backend == "hmonnx" and not args.hmonnx:
        raise ValueError("--hmonnx is required for --backend hmonnx")
    onnx_runner = make_onnx_runner(args.onnx) if args.backend == "onnx" else None
    hmonnx_runner = make_hmonnx_runner(args.hmonnx, args.device) if args.backend == "hmonnx" else None

    predictions: list[dict[str, Any]] = []
    for index, image_id in enumerate(image_ids, start=1):
        info = coco_gt.loadImgs([image_id])[0]
        image_path = Path(args.images_dir) / info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"COCO image not found: {image_path}")

        batch, metadata = preprocess_image_batch(model, image_path)
        input_hw = (int(batch.shape[-2]), int(batch.shape[-1]))

        if args.backend == "torch":
            outputs = run_torch_with_original_size(model, batch, metadata)
        elif args.backend == "onnx":
            assert onnx_runner is not None
            outputs = scale_resized_boxes_to_original(onnx_runner(batch), metadata, input_hw)
        elif args.backend == "hmonnx":
            assert hmonnx_runner is not None
            outputs = scale_resized_boxes_to_original(hmonnx_runner(batch), metadata, input_hw)
        else:
            raise ValueError(f"Unsupported backend: {args.backend}")

        predictions.extend(
            outputs_to_coco_rows(
                outputs,
                image_id=image_id,
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
                category_ids=category_ids,
            )
        )
        if index % args.log_every == 0:
            print(f"Processed {index}/{len(image_ids)} COCO images")

    predictions_path = out_dir / f"{args.backend}_predictions.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")

    if not predictions:
        raise RuntimeError("No predictions were generated; lower --score-threshold or inspect model outputs.")

    coco_dt = coco_gt.loadRes(str(predictions_path))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metric_names = [
        "AP_50_95",
        "AP_50",
        "AP_75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    metrics = {name: float(value) for name, value in zip(metric_names, evaluator.stats.tolist(), strict=True)}
    metrics.update(
        {
            "backend": args.backend,
            "num_images": len(image_ids),
            "predictions": str(predictions_path),
            "sample_seed": args.sample_seed if args.sample_size is not None else None,
            "sample_size": args.sample_size,
            "reference_lightly_train_coco_val_AP_50_95": 0.498,
        }
    )
    metrics_path = out_dir / f"{args.backend}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"COCO metrics: {metrics_path}")
    return metrics_path




# ═══════════════════════════════════════════════════════════════════════════════
# 调试 & 配置工具
# write-quant-config: 混精 JSON → xhquant Python config
# debug-nodes: 逐节点精度调试，对比 fp32/quantized 输出差异
# ═══════════════════════════════════════════════════════════════════════════════


def write_xhquant_config(args: argparse.Namespace) -> Path:
    """write-quant-config 子命令入口。将混精 JSON/YAML 配置转为 xhquant Python config 文件。"""
    from xhquant.api import Config

    mixed_precision = load_mixed_precision_config(args.mixed_precision_config)
    quant_config = build_quant_config(args, mixed_precision)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    Config(quant_config).dump(str(out))
    print(f"XHQuant config written: {out}")
    return out


def build_debug_command_args(args: argparse.Namespace, quant_config_path: str | Path | None = None) -> list[str]:
    """构建 xhquant OnnxQuantDebugCommand 的命令行参数列表。"""
    cmd_args = [
        "--onnx",
        str(args.onnx),
        "--input",
        str(args.input_dir),
        "--target-device",
        "XH2a",
        "--quant-type",
        str(quant_config_path or args.quant_type),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.compare_all_nodes:
        cmd_args.append("--compare-all-nodes")
    if args.only_compare_output:
        cmd_args.append("--only-compare-output")
    if args.disable_aligned:
        cmd_args.append("--disable-aligned")
    if args.resume:
        cmd_args.append("--resume")
    return cmd_args


def debug_nodes(args: argparse.Namespace) -> None:
    """debug-nodes 子命令入口。逐节点对比 fp32/quantized 输出差异，定位精度瓶颈。"""
    if args.image:
        input_args = argparse.Namespace(**vars(args))
        input_args.out_dir = args.input_dir
        make_input(input_args)

    quant_config_path = None
    if args.mixed_precision_config:
        quant_config_path = Path(args.output_dir).expanduser().resolve() / "quant_config_from_mixed_precision.py"
        config_args = argparse.Namespace(**vars(args))
        config_args.out = quant_config_path
        write_xhquant_config(config_args)

    cmd_args = build_debug_command_args(args, quant_config_path=quant_config_path)
    print("OnnxQuantDebugCommand args:")
    print(" ".join(cmd_args))
    if args.dry_run:
        return

    from xhquant.cli.onnx_quant_debug_v2 import OnnxQuantDebugCommand

    OnnxQuantDebugCommand(cmd_args).run()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 参数定义
# add_common_model_args: 模型加载参数（model-name/device/cache）
# add_quant_args: 量化参数（quant-type/mixed-precision-config）
# ═══════════════════════════════════════════════════════════════════════════════

def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    """添加通用模型参数：model-name、device、cache 目录。"""
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="LightlyTrain model name or checkpoint path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Model/runtime device.")
    parser.add_argument(
        "--model-cache-dir",
        default=str(DEFAULT_MODEL_CACHE_DIR),
        help="LightlyTrain model cache directory.",
    )
    parser.add_argument(
        "--data-cache-dir",
        default=str(DEFAULT_DATA_CACHE_DIR),
        help="LightlyTrain data cache directory.",
    )


def add_quant_args(parser: argparse.ArgumentParser) -> None:
    """添加量化参数：quant-type、mixed-precision-config、fp32 I/O 开关。"""
    parser.add_argument("--quant-type", default=DEFAULT_QUANT_TYPE, help="Base quantization type.")
    parser.add_argument(
        "--mixed-precision-config",
        default=None,
        help="JSON/YAML file with nodes/ops overrides for mixed precision.",
    )
    parser.add_argument("--input-enable-fp32", action="store_true", help="Keep input stubs in fp32.")
    parser.add_argument(
        "--output-enable-fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep output stubs in fp32. Enabled by default for detection boxes/scores.",
    )



# ═══════════════════════════════════════════════════════════════════════════════
# 子命令注册
# 可用子命令：
#   export-onnx       导出 ONNX
#   quantize          ONNX → HMONNX 基础量化
#   make-input        生成调试用输入数据
#   compare           单图对比 Torch/ONNX/HMONNX
#   coco-eval         COCO mAP 评测
#   write-quant-config   混精 JSON → xhquant config
#   write-search-configs 生成分组搜索配置
#   debug-nodes       逐节点精度调试
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """构建完整的 CLI 参数解析器，注册所有子命令。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-onnx", help="Export LightlyTrain LT-DETR to ONNX.")
    add_common_model_args(export_parser)
    export_parser.add_argument("--out", default=str(DEFAULT_ONNX), help="Output ONNX path.")
    export_parser.add_argument("--precision", default="fp32", choices=["auto", "fp32", "fp16"])
    export_parser.add_argument("--batch-size", type=int, default=1)
    export_parser.add_argument("--dynamic-batch", action="store_true", help="Export a dynamic batch dimension.")
    export_parser.add_argument("--opset", type=int, default=17)
    export_parser.add_argument("--num-channels", type=int, default=None)
    export_parser.add_argument("--no-simplify", action="store_true")
    export_parser.add_argument("--no-verify", action="store_true")
    export_parser.set_defaults(func=export_onnx)

    quant_parser = subparsers.add_parser("quantize", help="Convert ONNX to HMONNX.")
    add_quant_args(quant_parser)
    quant_parser.add_argument("--onnx", default=str(DEFAULT_ONNX), help="Input ONNX path.")
    quant_parser.add_argument("--out", default=None, help="Output HMONNX path.")
    quant_parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), help="Directory for generated HMONNX.")
    quant_parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    quant_parser.add_argument("--batch-size", type=int, default=1)
    quant_parser.add_argument("--output-names", default=None, help="Comma-separated ONNX output names override.")
    quant_parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Golden inference device.")
    quant_parser.add_argument("--skip-golden", action="store_true")
    quant_parser.set_defaults(func=quantize)

    input_parser = subparsers.add_parser("make-input", help="Create a real images.npy input for debug.")
    add_common_model_args(input_parser)
    input_parser.add_argument("--image", required=True, help="Image used for debug/compare.")
    input_parser.add_argument("--out-dir", default=str(DEFAULT_DEBUG_INPUT_DIR))
    input_parser.set_defaults(func=make_input)

    compare_parser = subparsers.add_parser("compare", help="Compare PyTorch, ONNX, and HMONNX outputs.")
    add_common_model_args(compare_parser)
    compare_parser.add_argument("--image", required=True)
    compare_parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    compare_parser.add_argument("--hmonnx", default=None)
    compare_parser.add_argument("--out-dir", default=str(DEFAULT_COMPARE_DIR))
    compare_parser.add_argument("--threshold", type=float, default=0.6)
    compare_parser.add_argument("--max-detections", type=int, default=20)
    compare_parser.set_defaults(func=compare)

    coco_parser = subparsers.add_parser("coco-eval", help="Evaluate torch/ONNX/HMONNX on COCO bbox mAP.")
    add_common_model_args(coco_parser)
    coco_parser.add_argument("--backend", required=True, choices=["torch", "onnx", "hmonnx"])
    coco_parser.add_argument("--images-dir", required=True, help="COCO val images directory.")
    coco_parser.add_argument("--annotations", required=True, help="COCO instances_val*.json path.")
    coco_parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    coco_parser.add_argument("--hmonnx", default=None)
    coco_parser.add_argument("--out-dir", default=str(DEFAULT_COCO_EVAL_DIR))
    coco_parser.add_argument("--score-threshold", type=float, default=0.001)
    coco_parser.add_argument("--max-detections", type=int, default=300)
    coco_parser.add_argument("--limit", type=int, default=None, help="Optional image limit for smoke tests.")
    coco_parser.add_argument("--sample-size", type=int, default=None, help="Deterministic random COCO image sample size.")
    coco_parser.add_argument("--sample-seed", type=int, default=0, help="Seed for --sample-size.")
    coco_parser.add_argument("--log-every", type=int, default=50)
    coco_parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Filter annotations to images that exist under --images-dir before applying --limit.",
    )
    coco_parser.set_defaults(func=coco_eval)


    cfg_parser = subparsers.add_parser("write-quant-config", help="Translate mixed precision JSON/YAML to xhquant config.")
    add_quant_args(cfg_parser)
    cfg_parser.add_argument("--out", required=True, help="Output Python config file.")
    cfg_parser.set_defaults(func=write_xhquant_config)

    search_cfg_parser = subparsers.add_parser(
        "write-search-configs",
        help="Write grouped mixed precision search configs for LT-DETR.",
    )
    search_cfg_parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    search_cfg_parser.add_argument("--out-dir", default=str(DEFAULT_SEARCH_DIR))
    search_cfg_parser.set_defaults(func=write_mixed_precision_search_configs)

    debug_parser = subparsers.add_parser("debug-nodes", help="Run xhquant ONNX node debug on a real input.")
    add_common_model_args(debug_parser)
    add_quant_args(debug_parser)
    debug_parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    debug_parser.add_argument("--image", default=None, help="Optional image. If set, images.npy is generated first.")
    debug_parser.add_argument("--input-dir", default=str(DEFAULT_DEBUG_INPUT_DIR))
    debug_parser.add_argument("--output-dir", default=str(DEFAULT_DEBUG_OUTPUT_DIR))
    debug_parser.add_argument("--compare-all-nodes", action="store_true")
    debug_parser.add_argument("--only-compare-output", action="store_true")
    debug_parser.add_argument("--disable-aligned", action="store_true")
    debug_parser.add_argument("--resume", action="store_true")
    debug_parser.add_argument("--dry-run", action="store_true", help="Print debug command args without running.")
    debug_parser.set_defaults(func=debug_nodes)

    return parser


def main(argv: Sequence[str] | None = None) -> Any:
    """CLI 入口；无参数运行时默认执行 export-onnx。"""
    if (argv is None and len(sys.argv) == 1) or (argv is not None and len(argv) == 0):
        argv = ["export-onnx"]
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
