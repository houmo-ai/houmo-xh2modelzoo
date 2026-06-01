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
    if target is None:
        return array
    if array.dtype != target:
        return array.astype(target)
    return array


def pad_or_trim(array: np.ndarray, target_len: int, axis: int = 1, pad_value: float = 0.0) -> np.ndarray:
    current = array.shape[axis]
    if current == target_len:
        return array
    if current > target_len:
        slc = [slice(None)] * array.ndim
        slc[axis] = slice(0, target_len)
        return array[tuple(slc)]

    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, target_len - current)
    return np.pad(array, pad_width, mode="constant", constant_values=pad_value)


def reshape_to_rank(array: np.ndarray, rank: int) -> np.ndarray:
    while array.ndim < rank:
        array = np.expand_dims(array, axis=1)
    return array


def make_pre_token_mask(token_len: int, max_len: int) -> np.ndarray:
    mask = np.zeros((1, max_len), dtype=np.int32)
    mask[:, :token_len] = 1
    return mask


def build_encoder_inputs(
    onnx_path: Path,
    inputs_dir: Path,
    float_dtype: str = "fp16",
) -> Dict[str, np.ndarray]:
    speech = load_npy(inputs_dir / "speech.npy")
    if speech.dtype != np.float32 and speech.dtype != np.float16:
        speech = speech.astype(np.float32)

    if float_dtype in {"fp16", "float16"}:
        speech = speech.astype(np.float16)
    elif float_dtype in {"fp32", "float32"}:
        speech = speech.astype(np.float32)

    speech_len = np.array([speech.shape[1]], dtype=np.int32)
    mask_dtype = np.float16 if speech.dtype == np.float16 else np.float32
    speech_mask = np.ones((speech.shape[0], speech.shape[1]), dtype=mask_dtype)

    input_specs = get_onnx_input_specs(onnx_path)
    inputs: Dict[str, np.ndarray] = {}
    for name, dims in input_specs:
        lower = name.lower()
        if "length" in lower or lower.endswith("len"):
            inputs[name] = speech_len
        elif "mask" in lower:
            mask = reshape_to_rank(speech_mask, len(dims) if dims else speech_mask.ndim)
            inputs[name] = mask
        elif "speech" in lower:
            inputs[name] = speech
        else:
            raise ValueError(f"Unsupported encoder input: {name}")

    return inputs


def _infer_target_len(dims: List[Optional[int]], fallback: int) -> int:
    if len(dims) >= 2 and dims[1] is not None:
        return int(dims[1])
    return fallback


def build_decoder_inputs_from_arrays(
    onnx_path: Path,
    enc: np.ndarray,
    enc_mask: Optional[np.ndarray],
    pre_acoustic_embeds: np.ndarray,
    pre_token_mask: np.ndarray,
    max_token_len: int = 100,
) -> Dict[str, np.ndarray]:
    input_specs = get_onnx_input_specs(onnx_path)
    elem_types = get_onnx_input_elem_types(onnx_path)

    enc_len = np.array([enc.shape[1]], dtype=np.int32)

    if pre_token_mask.ndim == 1:
        pre_token_mask = pre_token_mask.reshape(1, -1)

    token_len = int(pre_token_mask.sum())
    token_len = max(token_len, 1)

    inputs: Dict[str, np.ndarray] = {}
    for name, dims in input_specs:
        lower = name.lower()
        if "enc" in lower and "mask" not in lower and "len" not in lower:
            inputs[name] = _cast_to_elem_type(enc, elem_types.get(name))
        elif "enc_mask" in lower or ("mask" in lower and "enc" in lower):
            if enc_mask is None:
                base = np.ones((enc.shape[0], 1, enc.shape[1]), dtype=np.float32)
            else:
                base = enc_mask
            mask = reshape_to_rank(base, len(dims) if dims else base.ndim)
            inputs[name] = _cast_to_elem_type(mask, elem_types.get(name))
        elif "enc_len" in lower or ("length" in lower and "pre" not in lower):
            inputs[name] = _cast_to_elem_type(enc_len, elem_types.get(name))
        elif "pre_acoustic" in lower:
            target_len = _infer_target_len(dims, max_token_len)
            embeds = pad_or_trim(pre_acoustic_embeds, target_len, axis=1, pad_value=0.0)
            inputs[name] = _cast_to_elem_type(embeds, elem_types.get(name))
        elif "pre_token_mask" in lower:
            target_len = _infer_target_len(dims, max_token_len)
            mask = pad_or_trim(pre_token_mask, target_len, axis=1, pad_value=0)
            inputs[name] = _cast_to_elem_type(mask, elem_types.get(name))
        elif "pre_token_length" in lower or ("pre" in lower and "length" in lower):
            length = np.array([token_len], dtype=np.int32)
            inputs[name] = _cast_to_elem_type(length, elem_types.get(name))
        else:
            raise ValueError(f"Unsupported decoder input: {name}")

    return inputs


def build_decoder_inputs(
    onnx_path: Path,
    inputs_dir: Path,
    max_token_len: int = 100,
) -> Dict[str, np.ndarray]:
    enc = load_npy(inputs_dir / "enc.npy")
    enc_mask_path = inputs_dir / "enc_mask.npy"
    enc_mask = load_npy(enc_mask_path) if enc_mask_path.exists() else None
    pre_acoustic_embeds = load_npy(inputs_dir / "pre_acoustic_embeds.npy")
    pre_token_mask = load_npy(inputs_dir / "pre_token_mask.npy")

    return build_decoder_inputs_from_arrays(
        onnx_path,
        enc,
        enc_mask,
        pre_acoustic_embeds,
        pre_token_mask,
        max_token_len=max_token_len,
    )


def build_encoder_mask(encoder_inputs: Dict[str, np.ndarray], enc: np.ndarray) -> np.ndarray:
    base = None
    for name, arr in encoder_inputs.items():
        if "mask" in name.lower():
            base = arr
            break

    if base is None:
        length = None
        for name, arr in encoder_inputs.items():
            lower = name.lower()
            if "length" in lower or lower.endswith("len"):
                length = int(arr[0])
                break
        if length is None:
            length = enc.shape[1]
        base = np.ones((enc.shape[0], length), dtype=np.float32)

    if base.ndim == 2:
        base = base[:, None, :]
    elif base.ndim > 3:
        base = np.squeeze(base)
        if base.ndim == 2:
            base = base[:, None, :]

    if base.shape[-1] != enc.shape[1]:
        base = pad_or_trim(base, enc.shape[1], axis=-1, pad_value=0.0)

    return base.astype(np.float32)
