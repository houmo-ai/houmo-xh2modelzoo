import argparse
import json
import logging
import os.path as osp
import random
import shutil
import sys
import os
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
from onnxsim import simplify


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEROBOT_SRC = REPO_ROOT.parent / "lerobot" / "src"


def ensure_lerobot_importable(lerobot_src: str | Path | None = None) -> Path:
    lerobot_src_path = Path(lerobot_src) if lerobot_src is not None else DEFAULT_LEROBOT_SRC
    lerobot_src_path = lerobot_src_path.resolve()
    if not lerobot_src_path.exists():
        raise FileNotFoundError(
            f"LeRobot src path not found: {lerobot_src_path}. "
            "Please pass --lerobot_src to point to lerobot/src."
        )
    if str(lerobot_src_path) not in sys.path:
        sys.path.insert(0, str(lerobot_src_path))
    return lerobot_src_path


from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config


ORIGIN_WORKDIR = Path("work_dirs/smolvla_action")
ONNX_DIR = ORIGIN_WORKDIR / "smolvla_action_onnx"
HMONNX_DIR = ORIGIN_WORKDIR / "smolvla_action_hmonnx"
DEFAULT_PREFILL_META = Path("work_dirs/smolvla_llm_kvcache") / "meta_info.json"
ONNX_DIR.mkdir(parents=True, exist_ok=True)
HMONNX_DIR.mkdir(parents=True, exist_ok=True)


class suppress_logging_internal_errors:
    """Temporarily suppress Python logging internal formatter exceptions."""

    def __enter__(self):
        self._old_raise_exceptions = logging.raiseExceptions
        logging.raiseExceptions = False
        return self

    def __exit__(self, exc_type, exc, tb):
        logging.raiseExceptions = self._old_raise_exceptions
        return False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_lerobot_modules(lerobot_src: str | Path | None = None):
    ensure_lerobot_importable(lerobot_src)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    return PreTrainedConfig, SmolVLAPolicy


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_hf_cached_model_path(model_id_or_path: str) -> str:
    model_path = Path(model_id_or_path)
    if model_path.exists():
        return str(model_path.resolve())

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_cache_dir = cache_root / f"models--{model_id_or_path.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    refs_main = repo_cache_dir / "refs" / "main"

    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = snapshots_dir / revision
        if snapshot_dir.exists():
            return str(snapshot_dir.resolve())

    if snapshots_dir.exists():
        snapshot_candidates = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()])
        if snapshot_candidates:
            return str(snapshot_candidates[-1].resolve())

    return model_id_or_path


def configure_hf_offline_env():
    os.environ.setdefault("HUGGINGFACE_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_policy_config_offline(model_path: str, PreTrainedConfig):
    config = PreTrainedConfig.from_pretrained(model_path)
    if hasattr(config, "vlm_model_name") and isinstance(config.vlm_model_name, str):
        config.vlm_model_name = resolve_hf_cached_model_path(config.vlm_model_name)
    return config


@torch.no_grad()
def load_smolvla_policy(model_path: str, device: torch.device, lerobot_src: str | Path | None = None):
    configure_hf_offline_env()
    PreTrainedConfig, SmolVLAPolicy = get_lerobot_modules(lerobot_src)
    config = load_policy_config_offline(model_path, PreTrainedConfig)
    if hasattr(config, "device"):
        config.device = str(device)
    policy = SmolVLAPolicy.from_pretrained(model_path, config=config, strict=False)
    policy.eval()
    policy.to(device)
    return policy


def save_hf_artifacts(model_path: str, work_dir: Path) -> str | None:
    hf_config_dir = work_dir / "hf_config"
    hf_config_dir.mkdir(parents=True, exist_ok=True)

    copied_any = False
    local_model_dir = Path(model_path)
    hf_config_files = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "merges.txt",
        "vocab.json",
    ]
    if local_model_dir.exists():
        for cfg_file in hf_config_files:
            src_file = local_model_dir / cfg_file
            if src_file.exists():
                shutil.copyfile(src_file, hf_config_dir / cfg_file)
                copied_any = True

    if copied_any:
        return str(hf_config_dir)
    return None


def load_prefill_meta(prefill_meta_path: str | Path | None) -> dict | None:
    if prefill_meta_path is None:
        return None

    meta_path = Path(prefill_meta_path)
    if not meta_path.is_absolute():
        meta_path = Path.cwd() / meta_path
    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_prefix_length(policy: nn.Module, args) -> int:
    flow_model = policy.model
    config_prefix_length = int(getattr(flow_model.config, "prefix_length", 0) or 0)
    if args.prefix_length is not None and args.prefix_length > 0:
        return args.prefix_length
    if config_prefix_length > 0:
        return config_prefix_length

    prefill_meta = load_prefill_meta(args.prefill_meta)
    if prefill_meta is not None:
        meta_prefix_length = int(prefill_meta.get("prefix_length", 0) or 0)
        if meta_prefix_length > 0:
            return meta_prefix_length

        cache_shapes = prefill_meta.get("cache_shapes", {})
        if cache_shapes:
            first_shape = next(iter(cache_shapes.values()))
            if isinstance(first_shape, list) and len(first_shape) >= 2 and int(first_shape[1]) > 0:
                return int(first_shape[1])

    raise ValueError(
        "Unable to infer prefix_length for action export. "
        "Please pass --prefix_length explicitly, or ensure --prefill_meta points to a valid prefill meta_info.json."
    )


def build_cache_input_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for layer_idx in range(num_layers):
        names.append(f"past_key_{layer_idx}")
        names.append(f"past_value_{layer_idx}")
    return names


class SmolVLAActionPart(nn.Module):
    def __init__(self, flow_model: nn.Module):
        super().__init__()
        self.flow_model = flow_model
        self.num_layers = flow_model.vlm_with_expert.num_vlm_layers

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        *flat_cache: torch.Tensor,
    ) -> torch.Tensor:
        past_key_values: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx in range(self.num_layers):
            past_key_values[layer_idx] = {
                "key_states": flat_cache[layer_idx * 2],
                "value_states": flat_cache[layer_idx * 2 + 1],
            }

        x_t = x_t.to(dtype=self.flow_model.action_in_proj.weight.dtype)
        timestep = timestep.to(device=x_t.device, dtype=torch.float32)
        prefix_pad_masks = prefix_pad_masks.to(device=x_t.device, dtype=torch.bool)
        return self.flow_model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )


@torch.no_grad()
def build_dummy_inputs(policy: nn.Module, device: torch.device, export_dtype: torch.dtype, prefix_length: int):
    flow_model = policy.model
    config = flow_model.config
    text_config = flow_model.vlm_with_expert.config.text_config

    batch_size = 1

    x_t = torch.randn(
        (batch_size, config.chunk_size, config.max_action_dim),
        device=device,
        dtype=export_dtype,
    )
    timestep = torch.ones((batch_size,), device=device, dtype=torch.float32)
    prefix_pad_masks = torch.ones((batch_size, prefix_length), device=device, dtype=torch.bool)

    num_layers = flow_model.vlm_with_expert.num_vlm_layers
    cache_shape = (batch_size, prefix_length, text_config.num_key_value_heads, text_config.head_dim)
    flat_cache: list[torch.Tensor] = []
    for _ in range(num_layers):
        flat_cache.append(torch.randn(cache_shape, device=device, dtype=export_dtype))
        flat_cache.append(torch.randn(cache_shape, device=device, dtype=export_dtype))

    return x_t, timestep, prefix_pad_masks, tuple(flat_cache)


def evaluate_constant_node(node: gs.Node, memo: dict[str, np.ndarray | None]):
    if node.op == "Constant":
        const_value = node.attrs.get("value")
        if isinstance(const_value, gs.Constant):
            return np.asarray(const_value.values)
        if hasattr(const_value, "values"):
            return np.asarray(const_value.values)
        return np.asarray(const_value)

    if node.op == "Shape":
        input_tensor_shape = getattr(node.inputs[0], "shape", None)
        if input_tensor_shape is not None and all(isinstance(dim, int) for dim in input_tensor_shape):
            return np.asarray(input_tensor_shape, dtype=np.int64)

    input_values = [extract_constant_values(input_tensor, memo) for input_tensor in node.inputs]
    if any(value is None for value in input_values):
        return None

    if node.op == "Identity":
        return np.asarray(input_values[0])
    if node.op == "Cast":
        dtype = onnx.helper.tensor_dtype_to_np_dtype(node.attrs["to"])
        return np.asarray(input_values[0], dtype=dtype)
    if node.op == "Mul":
        return np.asarray(input_values[0]) * np.asarray(input_values[1])
    if node.op == "Add":
        return np.asarray(input_values[0]) + np.asarray(input_values[1])
    if node.op == "Sub":
        return np.asarray(input_values[0]) - np.asarray(input_values[1])
    if node.op == "Div":
        return np.asarray(input_values[0]) / np.asarray(input_values[1])
    if node.op == "Pow":
        return np.power(np.asarray(input_values[0]), np.asarray(input_values[1]))
    if node.op == "Reciprocal":
        return np.reciprocal(np.asarray(input_values[0], dtype=np.float64))
    if node.op == "Neg":
        return -np.asarray(input_values[0])
    if node.op == "Equal":
        return np.equal(np.asarray(input_values[0]), np.asarray(input_values[1]))
    if node.op == "Where":
        return np.where(np.asarray(input_values[0]), np.asarray(input_values[1]), np.asarray(input_values[2]))
    if node.op == "Unsqueeze":
        values = np.asarray(input_values[0])
        axes = node.attrs.get("axes")
        if axes is None and len(input_values) > 1:
            axes = np.asarray(input_values[1]).reshape(-1).tolist()
        if axes is None:
            return None
        for axis in sorted(int(axis) for axis in axes):
            values = np.expand_dims(values, axis=axis)
        return values
    if node.op == "Expand":
        values = np.asarray(input_values[0])
        target_shape = np.asarray(input_values[1], dtype=np.int64).reshape(-1).tolist()
        return np.broadcast_to(values, target_shape)
    if node.op == "Concat":
        axis = int(node.attrs.get("axis", 0))
        return np.concatenate([np.asarray(value) for value in input_values], axis=axis)
    if node.op == "Slice":
        data = np.asarray(input_values[0])
        starts = np.asarray(input_values[1], dtype=np.int64).reshape(-1)
        ends = np.asarray(input_values[2], dtype=np.int64).reshape(-1)
        if len(input_values) > 3:
            axes = np.asarray(input_values[3], dtype=np.int64).reshape(-1)
        else:
            axes = np.arange(starts.size, dtype=np.int64)
        if len(input_values) > 4:
            steps = np.asarray(input_values[4], dtype=np.int64).reshape(-1)
        else:
            steps = np.ones(starts.size, dtype=np.int64)

        slices = [slice(None)] * data.ndim
        for start, end, axis, step in zip(starts, ends, axes, steps, strict=False):
            axis = int(axis)
            if axis < 0:
                axis += data.ndim
            slices[axis] = slice(int(start), int(end), int(step))
        return data[tuple(slices)]
    if node.op == "Shape":
        return np.asarray(np.asarray(input_values[0]).shape, dtype=np.int64)
    if node.op == "ConstantOfShape":
        shape = np.asarray(input_values[0], dtype=np.int64).reshape(-1).tolist()
        fill_value = node.attrs.get("value")
        if isinstance(fill_value, gs.Constant):
            fill_value = fill_value.values
        if hasattr(fill_value, "values"):
            fill_value = fill_value.values
        fill_array = np.asarray(0.0 if fill_value is None else fill_value)
        scalar_value = fill_array.reshape(-1)[0] if fill_array.size > 0 else 0.0
        return np.full(shape, scalar_value, dtype=fill_array.dtype if fill_array.size > 0 else np.float32)

    return None


def extract_constant_values(tensor, memo: dict[str, np.ndarray | None] | None = None):
    if memo is None:
        memo = {}

    if isinstance(tensor, gs.Constant):
        return tensor.values
    tensor_name = getattr(tensor, "name", None)
    if tensor_name in memo:
        return memo[tensor_name]
    if hasattr(tensor, "inputs") and len(tensor.inputs) == 1:
        producer = tensor.inputs[0]
        memo[tensor_name] = evaluate_constant_node(producer, memo)
        return memo[tensor_name]
    return None


def rewrite_unsqueeze_axes_as_attrs(graph: gs.Graph):
    for node in graph.nodes:
        if node.op != "Unsqueeze":
            continue
        if "axes" in node.attrs or len(node.inputs) < 2:
            continue

        axes_values = extract_constant_values(node.inputs[1])
        if axes_values is None:
            continue

        if hasattr(axes_values, "tolist"):
            axes_values = axes_values.tolist()
        if not isinstance(axes_values, list):
            axes_values = [int(axes_values)]
        else:
            axes_values = [int(axis) for axis in axes_values]

        node.attrs["axes"] = axes_values
        node.inputs = [node.inputs[0]]


def rewrite_cumsum_axis_as_constant(graph: gs.Graph):
    for node in graph.nodes:
        if node.op != "CumSum" or len(node.inputs) < 2:
            continue

        axis_values = extract_constant_values(node.inputs[1])
        if axis_values is None:
            continue

        axis_array = np.array(axis_values)
        if axis_array.size != 1:
            continue
        axis_constant = gs.Constant(
            name=f"{node.name}_axis_const",
            values=np.array(int(axis_array.reshape(-1)[0]), dtype=np.int64),
        )
        node.inputs[1] = axis_constant


def rewrite_input_tensor_as_constant(graph: gs.Graph, op_names: tuple[str, ...], input_index: int = 1):
    constant_cache: dict[str, np.ndarray | None] = {}
    for node in graph.nodes:
        if node.op not in op_names or len(node.inputs) <= input_index:
            continue

        input_values = extract_constant_values(node.inputs[input_index], constant_cache)
        if input_values is None:
            continue

        constant_values = np.asarray(input_values)
        node.inputs[input_index] = gs.Constant(
            name=f"{node.name}_const_input_{input_index}",
            values=constant_values,
        )


def fold_constant_only_nodes(graph: gs.Graph, op_names: tuple[str, ...]):
    constant_cache: dict[str, np.ndarray | None] = {}
    for node in graph.nodes:
        if node.op not in op_names or len(node.outputs) != 1:
            continue

        folded_value = evaluate_constant_node(node, constant_cache)
        if folded_value is None:
            continue

        constant_output = gs.Constant(
            name=f"{node.name}_folded_const",
            values=np.asarray(folded_value),
        )
        original_output = node.outputs[0]
        for consumer in list(original_output.outputs):
            consumer.inputs = [constant_output if input_tensor is original_output else input_tensor for input_tensor in consumer.inputs]
        node.outputs = []


def canonicalize_onnx_graph(input_onnx_file: Path, output_onnx_file: Path):
    model = onnx.load(str(input_onnx_file))
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: onnx shape inference failed, keep original model: {exc}")
    graph = gs.import_onnx(model)
    rewrite_unsqueeze_axes_as_attrs(graph)
    rewrite_cumsum_axis_as_constant(graph)
    rewrite_input_tensor_as_constant(
        graph,
        ("ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceL1", "ReduceL2"),
        input_index=1,
    )
    rewrite_input_tensor_as_constant(graph, ("Reshape",), input_index=1)
    rewrite_input_tensor_as_constant(graph, ("Pow",), input_index=1)
    rewrite_input_tensor_as_constant(graph, ("Expand",), input_index=1)
    rewrite_input_tensor_as_constant(graph, ("Split",), input_index=1)
    rewrite_input_tensor_as_constant(graph, ("Slice",), input_index=1)
    rewrite_input_tensor_as_constant(graph, ("Slice",), input_index=2)
    rewrite_input_tensor_as_constant(graph, ("Slice",), input_index=3)
    rewrite_input_tensor_as_constant(graph, ("Slice",), input_index=4)
    fold_constant_only_nodes(graph, ("Pow", "Expand", "Unsqueeze", "Concat"))
    graph.cleanup().toposort()
    canonical_model = gs.export_onnx(graph)
    onnx.save(canonical_model, str(output_onnx_file))


def simplify_onnx_with_fallback(
    input_onnx_file: Path,
    simplified_onnx_file: Path,
    test_input_shapes: dict[str, list[int]],
):
    canonicalize_onnx_graph(input_onnx_file, simplified_onnx_file)
    canonical_model = onnx.load(str(simplified_onnx_file))
    try:
        model_simplified, check = simplify(canonical_model, test_input_shapes=test_input_shapes)
        if not check:
            print("Warning: onnxsim check failed, keeping canonicalized ONNX.")
            return
        onnx.save(model_simplified, str(simplified_onnx_file))
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: onnxsim failed, fallback to canonicalized ONNX: {exc}")


def export_action(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    export_dtype = torch.float16 if device.type == "cuda" else torch.float32

    policy = load_smolvla_policy(args.model_path, device, args.lerobot_src)
    flow_model = policy.model
    if flow_model._rtc_enabled():
        raise ValueError(
            "当前 action 导出脚本只覆盖 sample_actions() 的非 RTC 分支，即直接调用 denoise_step()。"
        )

    prefix_length = resolve_prefix_length(policy, args)

    action_model = SmolVLAActionPart(flow_model)
    action_model.eval()
    action_model.to(device=device, dtype=export_dtype)

    x_t, timestep, prefix_pad_masks, flat_cache = build_dummy_inputs(
        policy,
        device,
        export_dtype,
        prefix_length,
    )

    work_dir = ORIGIN_WORKDIR
    work_dir.mkdir(parents=True, exist_ok=True)
    hf_config_dir = save_hf_artifacts(args.model_path, work_dir)

    input_names = ["x_t", "timestep", "prefix_pad_masks", *build_cache_input_names(action_model.num_layers)]
    output_names = ["v_t"]

    temp_onnx_file = ONNX_DIR / f"{args.output_name}.onnx"
    simplified_onnx_file = ONNX_DIR / f"{args.output_name}_simplified.onnx"
    out_hmonnx_file = HMONNX_DIR / f"{args.output_name}_xh2.onnx"

    model_inputs = (x_t, timestep, prefix_pad_masks, *flat_cache)

    print("Exporting SmolVLA action denoise branch to ONNX...")
    torch.onnx.export(
        action_model,
        model_inputs,
        str(temp_onnx_file),
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
        verbose=False,
        do_constant_folding=True,
    )

    print("Simplifying ONNX...")
    test_input_shapes = {
        "x_t": list(x_t.shape),
        "timestep": list(timestep.shape),
        "prefix_pad_masks": list(prefix_pad_masks.shape),
    }
    for name, tensor in zip(input_names[3:], flat_cache, strict=False):
        test_input_shapes[name] = list(tensor.shape)
    simplify_onnx_with_fallback(temp_onnx_file, simplified_onnx_file, test_input_shapes)

    print("Converting ONNX to HMONNX for XH2A...")
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)

    calib_inputs = tuple(tensor.detach().cpu() for tensor in model_inputs)
    with suppress_logging_internal_errors():
        convert_onnx_to_hmonnx(
            str(simplified_onnx_file),
            calib_inputs,
            out_hmonnx_file=osp.join(str(out_hmonnx_file)),
            device_type="XH2A",
            quant_config=quant_config,
        )

    cache_shape = list(flat_cache[0].shape)
    meta_info = {
        "model_path": args.model_path,
        "input_names": input_names,
        "output_names": output_names,
        "quant_type": args.quant_type,
        "onnx_file": str(simplified_onnx_file.relative_to(work_dir)),
        "hmonnx_file": str(out_hmonnx_file.relative_to(work_dir)),
        "notes": [
            "该脚本导出 sample_actions() 中非 RTC 分支的 denoise_step()。",
            "输入 past_key_* / past_value_* 来自 smolvla_export_llm_kvcache_xh2a.py 的 prefill 输出。",
            "该脚本内部包含 embed_suffix、llm decode 和 action_out_proj，输出单步 v_t。",
        ],
        "chunk_size": flow_model.config.chunk_size,
        "max_action_dim": flow_model.config.max_action_dim,
        "prefix_length": prefix_length,
        "num_layers": action_model.num_layers,
        "cache_shape_per_tensor": cache_shape,
    }
    if hf_config_dir is not None:
        meta_info["hf_config_dir"] = str(Path(hf_config_dir).relative_to(work_dir))

    meta_info_file = work_dir / "meta_info.json"
    with open(meta_info_file, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4, ensure_ascii=False)

    print(f"ONNX saved to: {simplified_onnx_file}")
    print(f"HMONNX saved to: {out_hmonnx_file}")
    print(f"Meta info saved to: {meta_info_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SmolVLA action denoise branch to ONNX/HMONNX")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="SmolVLA policy path / repo id. Action export requires a full SmolVLA policy.",
    )
    parser.add_argument(
        "--lerobot_src",
        type=str,
        default=str(DEFAULT_LEROBOT_SRC),
        help="Path to lerobot/src",
    )
    parser.add_argument("--device", type=str, default="auto", help="Export device: auto/cpu/cuda")
    parser.add_argument(
        "--prefix_length",
        type=int,
        default=None,
        help="Override prefix length. If omitted, the script tries config.prefix_length, then --prefill_meta.",
    )
    parser.add_argument(
        "--prefill_meta",
        type=str,
        default=str(DEFAULT_PREFILL_META),
        help="Path to smolvla llm kvcache meta_info.json used to infer prefix_length when config.prefix_length <= 0.",
    )
    parser.add_argument("--output_name", type=str, default="smolvla_action", help="Output file stem")
    parser.add_argument("--quant_type", type=str, default="w8a8h1_sefp", help="xhquant quant type")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--seed", type=int, default=42)
    export_action(parser.parse_args())