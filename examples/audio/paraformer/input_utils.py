import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx


def resolve_path(path_str: str) -> Path:
    return Path(os.path.expandvars(path_str)).expanduser()


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_npy(path: Path) -> np.ndarray:
    ensure_file(path, "npy")
    return np.load(path)


def save_npy(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, array)


def get_onnx_input_specs(onnx_path: Path) -> List[Tuple[str, List[Optional[int]]]]:
    model = onnx.load(str(onnx_path))
    specs: List[Tuple[str, List[Optional[int]]]] = []
    for value in model.graph.input:
        dims: List[Optional[int]] = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            else:
                dims.append(None)
        specs.append((value.name, dims))
    return specs


def get_onnx_input_elem_types(onnx_path: Path) -> Dict[str, int]:
    model = onnx.load(str(onnx_path))
    elem_types: Dict[str, int] = {}
    for value in model.graph.input:
        elem_types[value.name] = int(value.type.tensor_type.elem_type)
    return elem_types


def _cast_to_elem_type(array: np.ndarray, elem_type: Optional[int]) -> np.ndarray:
    if elem_type is None:
        return array
    dtype_map = {
        1: np.float32,
        10: np.float16,
        11: np.float64,
        6: np.int32,
        7: np.int64,
        3: np.int8,
        9: np.bool_,
    }
    target = dtype_map.get(elem_type)
    if target is None or array.dtype == target:
        return array
    return array.astype(target)


def lengths_to_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    lengths = lengths.astype(np.int32, copy=False).reshape(-1)
    positions = np.arange(max_len, dtype=np.int32)[None, :]
    return (positions < lengths[:, None]).astype(np.float32)


def pad_or_trim(array: np.ndarray, target_len: int, axis: int = 1, pad_value: float = 0.0) -> np.ndarray:
    current = array.shape[axis]
    if current == target_len:
        return array
    if current > target_len:
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(0, target_len)
        return array[tuple(slices)]

    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, target_len - current)
    return np.pad(array, pad_width, mode="constant", constant_values=pad_value)


def build_encoder_inputs(onnx_path: Path, inputs_dir: Path, float_dtype: str = "fp16") -> Dict[str, np.ndarray]:
    speech = load_npy(inputs_dir / "speech.npy")
    lengths_path = inputs_dir / "speech_lengths.npy"
    if lengths_path.exists():
        speech_lengths = load_npy(lengths_path)
    else:
        speech_lengths = np.array([speech.shape[1]], dtype=np.int32)

    if float_dtype in {"fp16", "float16"}:
        speech = speech.astype(np.float16)
    elif float_dtype in {"fp32", "float32"}:
        speech = speech.astype(np.float32)
    else:
        raise ValueError(f"unsupported float dtype: {float_dtype}")

    return build_encoder_inputs_from_arrays(onnx_path, speech, speech_lengths)


def build_encoder_inputs_from_arrays(
    onnx_path: Path,
    speech: np.ndarray,
    speech_lengths: np.ndarray,
) -> Dict[str, np.ndarray]:
    input_specs = get_onnx_input_specs(onnx_path)
    elem_types = get_onnx_input_elem_types(onnx_path)
    inputs: Dict[str, np.ndarray] = {}
    for name, _ in input_specs:
        lower = name.lower()
        if "speech" in lower and "length" not in lower:
            if "mask" in lower:
                mask = lengths_to_mask(speech_lengths, max_len=int(speech.shape[1]))
                inputs[name] = _cast_to_elem_type(mask, elem_types.get(name))
            else:
                inputs[name] = _cast_to_elem_type(speech, elem_types.get(name))
        elif "length" in lower or lower.endswith("len"):
            inputs[name] = _cast_to_elem_type(speech_lengths, elem_types.get(name))
        else:
            raise ValueError(f"Unsupported encoder input: {name}")
    return inputs


def _shape_from_dims(dims: List[Optional[int]], batch_size: int) -> Tuple[int, ...]:
    shape: List[int] = []
    for axis, dim in enumerate(dims):
        if dim is not None:
            shape.append(int(dim))
        elif axis == 0:
            shape.append(batch_size)
        else:
            raise ValueError(f"cannot infer non-batch dynamic dim from {dims}")
    return tuple(shape)


def init_decoder_caches(onnx_path: Path, batch_size: int = 1) -> List[np.ndarray]:
    specs = get_onnx_input_specs(onnx_path)
    elem_types = get_onnx_input_elem_types(onnx_path)
    caches: List[np.ndarray] = []
    for name, dims in specs:
        if not name.startswith("in_cache_"):
            continue
        shape = _shape_from_dims(dims, batch_size)
        cache = np.zeros(shape, dtype=np.float32)
        caches.append(_cast_to_elem_type(cache, elem_types.get(name)))
    return caches


def build_decoder_inputs_from_arrays(
    onnx_path: Path,
    enc: np.ndarray,
    enc_len: np.ndarray,
    acoustic_embeds: np.ndarray,
    acoustic_embeds_len: np.ndarray,
    caches: Optional[List[np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    input_specs = get_onnx_input_specs(onnx_path)
    elem_types = get_onnx_input_elem_types(onnx_path)
    cache_values = caches if caches is not None else init_decoder_caches(onnx_path, batch_size=int(enc.shape[0]))

    enc_target_len = int(enc.shape[1])
    token_target_len = int(acoustic_embeds.shape[1])
    for name, dims in input_specs:
        lower = name.lower()
        if lower == "enc" and len(dims) > 1 and dims[1] is not None:
            enc_target_len = int(dims[1])
        elif lower == "acoustic_embeds" and len(dims) > 1 and dims[1] is not None:
            token_target_len = int(dims[1])

    enc = pad_or_trim(enc, enc_target_len, axis=1)
    enc_len = np.minimum(enc_len.astype(np.int32, copy=False), enc_target_len)
    acoustic_embeds = pad_or_trim(acoustic_embeds, token_target_len, axis=1)
    acoustic_embeds_len = np.minimum(acoustic_embeds_len.astype(np.int32, copy=False), token_target_len)

    inputs: Dict[str, np.ndarray] = {}
    for name, dims in input_specs:
        lower = name.lower()
        if lower == "enc":
            inputs[name] = _cast_to_elem_type(enc, elem_types.get(name))
        elif lower == "enc_mask":
            mask = lengths_to_mask(enc_len, max_len=int(enc.shape[1]))
            inputs[name] = _cast_to_elem_type(mask, elem_types.get(name))
        elif lower == "enc_len":
            inputs[name] = _cast_to_elem_type(enc_len, elem_types.get(name))
        elif lower == "acoustic_embeds":
            inputs[name] = _cast_to_elem_type(acoustic_embeds, elem_types.get(name))
        elif lower == "pre_token_mask":
            mask = lengths_to_mask(acoustic_embeds_len, max_len=int(acoustic_embeds.shape[1]))
            inputs[name] = _cast_to_elem_type(mask, elem_types.get(name))
        elif lower == "acoustic_embeds_len":
            inputs[name] = _cast_to_elem_type(acoustic_embeds_len, elem_types.get(name))
        elif name.startswith("in_cache_"):
            cache_index = int(name.split("_")[-1])
            cache = cache_values[cache_index]
            if dims and any(dim is not None for dim in dims):
                target_shape = _shape_from_dims(dims, batch_size=int(enc.shape[0]))
                if cache.shape != target_shape:
                    cache = np.zeros(target_shape, dtype=cache.dtype)
            inputs[name] = _cast_to_elem_type(cache, elem_types.get(name))
        else:
            raise ValueError(f"Unsupported decoder input: {name}")
    return inputs