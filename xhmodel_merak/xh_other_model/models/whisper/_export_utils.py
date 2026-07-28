from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn


GB = int(2**30)
LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 1.8)
MASK_FILL = -65504.0


class Decoder(nn.Module):
    def __init__(self, model, proj_out, config=None):
        super().__init__()
        self.config = config
        self.model = model
        self.proj_out = proj_out

    def forward(
        self,
        decoder_input_ids,
        cache_position,
        past_len,
        current_len,
        mask_atten=None,
        encoder_attention_mask=None,
        k_cache_list=None,
        v_cache_list=None,
        k_list=None,
        v_list=None,
    ):
        hidden_state, k_cache_list, v_cache_list = self.model.decoder(
            input_ids=decoder_input_ids,
            k_list=k_list,
            v_list=v_list,
            position_ids=cache_position,
            k_cache=k_cache_list,
            v_cache=v_cache_list,
            past_len=past_len,
            current_len=current_len,
            mask_atten=mask_atten,
            encoder_attention_mask=encoder_attention_mask,
        )
        output = self.proj_out(hidden_state)
        return output, k_cache_list, v_cache_list


def select_torch_device(device: str) -> str:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return str(device)
    return "cpu"


def to_xh_device_type(target_device: str, device_type_cls: Any) -> Any:
    if target_device != "XH2a":
        raise ValueError(
            f"Whisper workflow currently supports target_device='XH2a', got {target_device!r}"
        )
    return device_type_cls.XH2a


def flatten_inputs(inputs: Any) -> list[Any]:
    flattened: list[Any] = []
    for arg in inputs:
        if isinstance(arg, (list, tuple)):
            flattened.extend(arg)
        else:
            flattened.append(arg)
    return flattened


def run_hmonnx_golden(hmonnx_file: Path, golden_dir: Path, device: str, inputs: Sequence[Any]) -> None:
    from xhquant.api import HMONNXGoldenInference

    session = HMONNXGoldenInference(str(hmonnx_file))
    session.to(device)
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    session.step = 0
    session(*inputs)


def jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def build_encoder_input_features(
    model_dir: str, torch_device: str, audio_path: str,
) -> torch.Tensor:
    """Build encoder input features from a real audio file via WhisperProcessor.

    Reads the audio at its native sample rate, resamples to 16 kHz, and
    generates mel features through the Whisper processor. Raises FileNotFoundError
    if audio_path does not exist.
    """
    import librosa
    import soundfile as sf
    from transformers import WhisperProcessor

    if not Path(audio_path).is_file():
        raise FileNotFoundError(
            f"Encoder audio file not found: {audio_path}. "
            "Please set export.components.encoder.audio_path in the YAML config "
            "to an absolute path, or run from the repository root directory."
        )

    processor = WhisperProcessor.from_pretrained(model_dir)
    audio_array, sr = sf.read(audio_path)
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    features = processor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features
    return features.to(dtype=torch.float32, device=torch_device)


def build_decoder_export_payload(
    *,
    prompt_token_ids: list[int],
    cache_position: list[int],
    past_len: int,
    num_heads: int,
    head_dim: int,
    embed_dim: int,
    max_source_positions: int,
    num_decode_layers: int,
) -> dict[str, Any]:
    decoder_input_ids = torch.tensor([prompt_token_ids])
    cache_position_tensor = torch.tensor([cache_position])
    past_len_tensor = torch.tensor([past_len])

    k_cache = [
        torch.ones([1, num_heads, embed_dim, head_dim], dtype=torch.float16) * MASK_FILL
        for _ in range(num_decode_layers)
    ]
    v_cache = [
        torch.ones([1, num_heads, embed_dim, head_dim], dtype=torch.float16) * MASK_FILL
        for _ in range(num_decode_layers)
    ]
    k_list = [
        torch.ones([1, num_heads, max_source_positions, head_dim], dtype=torch.float16) * MASK_FILL
        for _ in range(num_decode_layers)
    ]
    v_list = [
        torch.ones([1, num_heads, max_source_positions, head_dim], dtype=torch.float16) * MASK_FILL
        for _ in range(num_decode_layers)
    ]

    encoder_attention_mask = torch.zeros(
        (1, 1, 1, max_source_positions), dtype=torch.float16
    )

    cache_len = decoder_input_ids.shape[1]
    mask_atten = torch.ones((1, num_heads, cache_len, embed_dim)).half()
    mask_atten[:, :, :, past_len + cache_len:] *= MASK_FILL
    current_len = torch.tensor([cache_len])

    warp_inp = (
        decoder_input_ids,
        cache_position_tensor,
        past_len_tensor,
        current_len,
        mask_atten,
        encoder_attention_mask,
        k_cache,
        v_cache,
        k_list,
        v_list,
    )

    inputs_names = [
        "decoder_input_ids",
        "cache_position",
        "past_len",
        "current_len",
        "mask_atten",
        "encoder_attention_mask",
    ]
    for i in range(num_decode_layers):
        inputs_names.append(f"k_cache_{i}")
    for i in range(num_decode_layers):
        inputs_names.append(f"v_cache_{i}")
    for i in range(num_decode_layers):
        inputs_names.append(f"key_state_{i}")
    for i in range(num_decode_layers):
        inputs_names.append(f"value_state_{i}")

    output_names = ["logits"]
    for i in range(num_decode_layers):
        output_names.append(f"newk_cache_{i}")
    for i in range(num_decode_layers):
        output_names.append(f"newv_cache_{i}")

    return {
        "warp_inp": warp_inp,
        "inputs_names": inputs_names,
        "output_names": output_names,
    }


def build_decoder_golden_inputs(
    *,
    prompt_token_ids: list[int],
    cache_position: list[int],
    past_len: int,
    num_heads: int,
    head_dim: int,
    embed_dim: int,
    max_source_positions: int,
    num_decode_layers: int,
    torch_device: str,
    cache_tensor_cls: type,
) -> list[Any]:
    device = torch_device
    decoder_input_ids = torch.tensor([prompt_token_ids], dtype=torch.int32, device=device)
    cache_position_tensor = torch.tensor([cache_position], dtype=torch.int32, device=device)
    past_len_tensor = torch.tensor([past_len], dtype=torch.int32, device=device)

    k_cache = [
        cache_tensor_cls(
            torch.ones([1, num_heads, embed_dim, head_dim], dtype=torch.float16, device=device)
            * MASK_FILL
        )
        for _ in range(num_decode_layers)
    ]
    v_cache = [
        cache_tensor_cls(
            torch.ones([1, num_heads, embed_dim, head_dim], dtype=torch.float16, device=device)
            * MASK_FILL
        )
        for _ in range(num_decode_layers)
    ]
    k_list = [
        torch.ones([1, num_heads, max_source_positions, head_dim], dtype=torch.float16, device=device)
        * MASK_FILL
        for _ in range(num_decode_layers)
    ]
    v_list = [
        torch.ones([1, num_heads, max_source_positions, head_dim], dtype=torch.float16, device=device)
        * MASK_FILL
        for _ in range(num_decode_layers)
    ]

    encoder_attention_mask = torch.zeros(
        (1, 1, 1, max_source_positions), dtype=torch.float16, device=device
    )

    cache_len = len(prompt_token_ids)
    mask_atten = torch.ones((1, num_heads, cache_len, embed_dim), dtype=torch.float16, device=device)
    mask_atten[:, :, :, past_len + cache_len:] *= MASK_FILL
    current_len = torch.tensor([cache_len], dtype=torch.int32, device=device)

    hm_inputs: list[Any] = [
        decoder_input_ids,
        cache_position_tensor,
        past_len_tensor,
        current_len,
        mask_atten,
        encoder_attention_mask,
    ]
    for i in range(num_decode_layers):
        hm_inputs.append(k_cache[i])
    for i in range(num_decode_layers):
        hm_inputs.append(v_cache[i])
    for i in range(num_decode_layers):
        hm_inputs.append(k_list[i])
    for i in range(num_decode_layers):
        hm_inputs.append(v_list[i])
    return hm_inputs


def simplify_onnx(onnx_model: Any) -> tuple[Any, bool]:
    import onnxsim

    skipped_optimizers = [
        "fuse_pad_into_conv",
        "fuse_consecutive_slices",
        "eliminate_common_subexpression",
        "fuse_qkv",
    ]
    if onnx_model.ByteSize() <= LARGE_MODEL_SIZE_THRESHOLD:
        onnx_model_sim, checked = onnxsim.simplify(onnx_model, skipped_optimizers=skipped_optimizers)
    else:
        from xhquant.utils.onnxsim_large_model import simplify_large_onnx

        onnx_model_sim, checked = simplify_large_onnx(
            onnx_model, skipped_optimizers=skipped_optimizers
        )
    if checked:
        onnx_model = onnx_model_sim
    return onnx_model, checked
