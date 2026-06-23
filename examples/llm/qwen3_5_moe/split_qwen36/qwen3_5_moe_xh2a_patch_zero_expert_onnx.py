from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from safetensors.torch import safe_open

from xh_model_zoo.xh_llm.models.qwen3_5_moe.qwen3_5_moe_converter import Qwen3_5MoeConverterXH2a


DEFAULT_MODEL = "/data01/datasets/qwen36moe-no-rotate-attn8-shared8-n256-iter400"
DEFAULT_WORK_DIR = (
    "work_dirs/qwen36moe-no-rotate-attn8-shared8-n256-iter400-split-moe-premoe-w8a8h0_sefp-"
    "experts-w4a8h0_sefp"
)
LINEAR_NAMES = ("gate_proj", "up_proj", "down_proj")
GPTQ_PREFIXES = ("model.language_model.layers.", "model.layers.")
EXPERT_RE = re.compile(r"layer_(?P<layer>\d{3})/expert_(?P<expert>\d{3})\.onnx$")


def _parse_index_spec(spec: str, upper: int) -> set[int]:
    spec = str(spec).strip().lower()
    if spec in {"all", "*"}:
        return set(range(upper))
    values: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            values.update(range(int(start_s), int(end_s) + 1))
        else:
            values.add(int(part))
    invalid = sorted(value for value in values if value < 0 or value >= upper)
    if invalid:
        raise ValueError(f"Index spec {spec!r} has out-of-range values: {invalid}")
    return values


def _tensor_dtype(np_dtype: np.dtype) -> int:
    if np_dtype == np.dtype("float16"):
        return onnx.TensorProto.FLOAT16
    if np_dtype == np.dtype("float32"):
        return onnx.TensorProto.FLOAT
    if np_dtype == np.dtype("int8"):
        return onnx.TensorProto.INT8
    raise TypeError(f"Unsupported dtype: {np_dtype}")


def _replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    for idx, init in enumerate(model.graph.initializer):
        if init.name != name:
            continue
        replacement = numpy_helper.from_array(np.ascontiguousarray(array), name=name)
        replacement.data_type = _tensor_dtype(array.dtype)
        del model.graph.initializer[idx]
        model.graph.initializer.insert(idx, replacement)
        return
    raise KeyError(f"Initializer {name!r} not found")


def _qweight_is_all_zero(model: onnx.ModelProto) -> bool:
    qweight_inits = [init for init in model.graph.initializer if init.name.endswith(".qweight")]
    if not qweight_inits:
        return False
    return any(np.count_nonzero(numpy_helper.to_array(init)) == 0 for init in qweight_inits)


def _has_non_ssfp_linear(model: onnx.ModelProto) -> bool:
    for node in model.graph.node:
        if node.op_type != "Linear":
            continue
        mode_attr = next((attr for attr in node.attribute if attr.name == "mode"), None)
        if mode_attr is None or mode_attr.s.decode(errors="replace").lower() != "ssfp":
            return True
    return False


def _set_linear_mode_ssfp(model: onnx.ModelProto) -> None:
    for node in model.graph.node:
        if node.op_type != "Linear":
            continue
        mode_attr = next((attr for attr in node.attribute if attr.name == "mode"), None)
        if mode_attr is None:
            mode_attr = node.attribute.add()
            mode_attr.name = "mode"
        mode_attr.s = b"ssfp"


def _expected_shape_from_onnx_qweight(model: onnx.ModelProto, linear_name: str) -> tuple[int, int]:
    init = next(item for item in model.graph.initializer if item.name == f"{linear_name}.qweight")
    dims = tuple(int(dim) for dim in init.dims)
    if len(dims) != 3 or dims[1] != 64:
        raise ValueError(f"Unsupported {linear_name}.qweight shape: {dims}")
    return dims[2], dims[0] * dims[1]


def _find_gptq_tensor_file(
    tensor_index: dict[str, Path],
    layer_idx: int,
    expert_idx: int,
    linear_name: str,
) -> str:
    suffix = f"{layer_idx}.mlp.experts.{expert_idx}.{linear_name}"
    for prefix in GPTQ_PREFIXES:
        key_prefix = f"{prefix}{suffix}"
        safetensors_file = tensor_index.get(f"{key_prefix}.qweight")
        if safetensors_file is not None:
            return key_prefix
    raise KeyError(f"Missing GPTQ tensors for layer={layer_idx} expert={expert_idx} {linear_name}")


def _build_tensor_index(safetensors_files: list[Path]) -> dict[str, Path]:
    tensor_index: dict[str, Path] = {}
    for safetensors_file in safetensors_files:
        with safe_open(str(safetensors_file), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if ".mlp.experts." not in key:
                    continue
                if key.endswith((".qweight", ".qzeros", ".scales", ".g_idx")):
                    tensor_index[key] = safetensors_file
    return tensor_index


def _read_indexed_tensor(tensor_index: dict[str, Path], key: str) -> torch.Tensor:
    safetensors_file = tensor_index.get(key)
    if safetensors_file is None:
        raise KeyError(f"Missing GPTQ tensor {key}")
    with safe_open(str(safetensors_file), framework="pt", device="cpu") as reader:
        return reader.get_tensor(key)


def _make_hmonnx_initializers(
    tensor_index: dict[str, Path],
    key_prefix: str,
    expected_shape: tuple[int, int],
    qweight_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    qweight = _read_indexed_tensor(tensor_index, f"{key_prefix}.qweight")
    qzeros = _read_indexed_tensor(tensor_index, f"{key_prefix}.qzeros")
    scales = _read_indexed_tensor(tensor_index, f"{key_prefix}.scales")
    g_idx = _read_indexed_tensor(tensor_index, f"{key_prefix}.g_idx")

    unpacked = Qwen3_5MoeConverterXH2a._unpack_gptqmodel_qweight(
        qweight,
        qzeros,
        g_idx,
        expected_shape,
    )
    q_np = unpacked.numpy().astype(np.int16, copy=False)
    scales_np = scales.numpy()
    g_idx_np = g_idx.numpy().astype(np.int64, copy=False)[: expected_shape[1]]

    sign = np.sign(scales_np).astype(np.int16, copy=False)
    sign[sign == 0] = 1
    signed = q_np * sign[g_idx_np, : expected_shape[0]].T
    signed = np.clip(signed, -8, 7).astype(np.int8, copy=False)
    hmonnx_qweight = signed.reshape(expected_shape[0], expected_shape[1] // 64, 64).transpose(1, 2, 0)
    if tuple(hmonnx_qweight.shape) != tuple(qweight_shape):
        raise ValueError(f"Packed qweight shape {hmonnx_qweight.shape} != expected {qweight_shape}")

    scale_or_exp = np.abs(scales_np.astype(np.float32, copy=False))
    scale_or_exp[scale_or_exp <= 1.1e-5] = 1.0
    scale_or_exp = scale_or_exp.astype(np.float16).reshape(qweight_shape[0], 1, qweight_shape[2])
    return hmonnx_qweight, scale_or_exp


def _patch_one_expert(
    onnx_path: Path,
    tensor_index: dict[str, Path],
    layer_idx: int,
    expert_idx: int,
    *,
    only_zero: bool,
    backup: bool,
    dry_run: bool,
) -> bool:
    model = onnx.load(str(onnx_path), load_external_data=True)
    needs_weight_patch = _qweight_is_all_zero(model)
    needs_mode_patch = _has_non_ssfp_linear(model)
    if only_zero and not needs_weight_patch and not needs_mode_patch:
        return False

    updated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if needs_weight_patch or not only_zero:
        for linear_name in LINEAR_NAMES:
            q_init = next(item for item in model.graph.initializer if item.name == f"{linear_name}.qweight")
            qweight_shape = tuple(int(dim) for dim in q_init.dims)
            expected_shape = _expected_shape_from_onnx_qweight(model, linear_name)
            key_prefix = _find_gptq_tensor_file(
                tensor_index,
                layer_idx,
                expert_idx,
                linear_name,
            )
            updated[linear_name] = _make_hmonnx_initializers(
                tensor_index,
                key_prefix,
                expected_shape,
                qweight_shape,
            )

    if dry_run:
        return True
    if backup:
        backup_path = onnx_path.with_suffix(onnx_path.suffix + ".pre_direct_patch")
        if not backup_path.exists():
            shutil.copy2(onnx_path, backup_path)
    for linear_name, (qweight, scale_or_exp) in updated.items():
        _replace_initializer(model, f"{linear_name}.qweight", qweight)
        _replace_initializer(model, f"{linear_name}.scale_or_exp", scale_or_exp)
    _set_linear_mode_ssfp(model)
    onnx.save(model, str(onnx_path))
    return True


def _iter_expert_paths(work_dir: Path, layers: set[int], experts: set[int]) -> Iterable[tuple[Path, int, int]]:
    experts_dir = work_dir / "hmonnx" / "experts"
    for path in sorted(experts_dir.glob("layer_*/expert_*.onnx")):
        match = EXPERT_RE.search(str(path))
        if match is None:
            continue
        layer_idx = int(match.group("layer"))
        expert_idx = int(match.group("expert"))
        if layer_idx in layers and expert_idx in experts:
            yield path, layer_idx, expert_idx


def _count_nonzero_qweights(onnx_path: Path) -> list[int]:
    model = onnx.load(str(onnx_path), load_external_data=True)
    return [
        int(np.count_nonzero(numpy_helper.to_array(init)))
        for init in model.graph.initializer
        if init.name.endswith(".qweight")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly patch zero routed-expert HMONNX ONNX initializers from GPTQModel safetensors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GPTQModel directory containing model*.safetensors")
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR, help="Split-MoE work directory")
    parser.add_argument("--layers", default="all", help="Layer ids: all, 0, 0,1,2 or 0-3")
    parser.add_argument("--experts", default="all", help="Expert ids: all, 0, 0,1,2 or 0-7")
    parser.add_argument("--num-layers", type=int, default=40)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate initializers even when qweights are already nonzero; mode is fixed to ssfp by default",
    )
    parser.add_argument("--backup", action="store_true", help="Write .pre_direct_patch backups before modifying files")
    parser.add_argument("--dry-run", action="store_true", help="Report how many files would be patched without writing")
    parser.add_argument("--verify-counts", action="store_true", help="Print qweight nonzero counts after each patch")
    return parser.parse_args()


def main() -> None:
    start = time.time()
    args = parse_args()
    model_dir = Path(args.model)
    work_dir = Path(args.work_dir)
    safetensors_files = sorted(model_dir.glob("model*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No model*.safetensors files found in {model_dir}")
    tensor_index = _build_tensor_index(safetensors_files)
    print(f"Indexed {len(tensor_index)} GPTQ expert tensors from {len(safetensors_files)} safetensors files")

    layers = _parse_index_spec(args.layers, args.num_layers)
    experts = _parse_index_spec(args.experts, args.num_experts)
    expert_paths = list(_iter_expert_paths(work_dir, layers, experts))
    print(f"Scanning {len(expert_paths)} expert ONNX files under {work_dir}")

    patched = 0
    skipped = 0
    failures: list[tuple[str, str]] = []
    for path, layer_idx, expert_idx in expert_paths:
        try:
            changed = _patch_one_expert(
                path,
                tensor_index,
                layer_idx,
                expert_idx,
                only_zero=not args.force,
                backup=args.backup,
                dry_run=args.dry_run,
            )
            if changed:
                patched += 1
                if args.verify_counts and not args.dry_run:
                    print(f"patched layer={layer_idx} expert={expert_idx} counts={_count_nonzero_qweights(path)}")
                else:
                    print(f"patched layer={layer_idx} expert={expert_idx} path={path}")
            else:
                skipped += 1
        except Exception as exc:
            failures.append((str(path), str(exc)))
            print(f"FAILED layer={layer_idx} expert={expert_idx} path={path}: {exc}")

    print("\nPatch summary:")
    print(f"  patched: {patched}")
    print(f"  skipped: {skipped}")
    print(f"  failures: {len(failures)}")
    print(f"  elapsed_s: {time.time() - start:.2f}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
