import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from xhquant.api import HMONNXGoldenInference

from export_utils import (
    default_hmonnx_path,
    resolve_decoder_onnx,
    resolve_encoder_onnx,
    resolve_manifest,
    resolve_predictor_onnx,
)
from input_utils import (
    build_decoder_inputs_from_arrays,
    build_encoder_inputs,
    build_encoder_mask,
    ensure_file,
    get_onnx_input_specs,
    make_pre_token_mask,
    pad_or_trim,
    resolve_path,
)
from post_decode import post_decode
from utils import OrtInferSession

logger = logging.getLogger(__name__)


def _load_ort_model(model_path: Path, device_id: int, threads: int) -> OrtInferSession:
    ensure_file(model_path, "onnx model")
    return OrtInferSession(str(model_path), device_id, threads)


def _load_hmonnx(model_path: Path, device: str) -> HMONNXGoldenInference:
    ensure_file(model_path, "hmonnx model")
    session = HMONNXGoldenInference(str(model_path))
    session.to(device)
    return session


def _select_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _to_torch_inputs(arrays: List[np.ndarray], device: str, float_dtype: str) -> List[torch.Tensor]:
    if float_dtype in {"fp16", "float16"}:
        float_type = torch.float16
    elif float_dtype in {"fp32", "float32"}:
        float_type = torch.float32
    else:
        raise ValueError(f"unsupported float dtype: {float_dtype}")

    tensors: List[torch.Tensor] = []
    for arr in arrays:
        tensor = torch.from_numpy(arr)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=float_type)
        tensors.append(tensor.to(device))
    return tensors


def _decode_text(decoder_out: np.ndarray, token_len: int) -> str:
    if decoder_out.shape[1] > token_len:
        decoder_out = decoder_out[:, :token_len]
    preds = post_decode(decoder_out, np.array([token_len]))
    return "".join(preds[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="directory containing encoder/predictor/decoder onnx",
    )
    parser.add_argument(
        "--export-manifest",
        type=str,
        default="",
        help="export manifest json path (default: examples/audio/funasr/exports/latest.json if exists)",
    )
    parser.add_argument(
        "--inputs-dir",
        type=str,
        default="examples/audio/funasr/inputs",
        help="directory containing speech.npy",
    )
    parser.add_argument("--encoder", type=str, default="", help="encoder onnx path")
    parser.add_argument("--predictor", type=str, default="", help="predictor onnx path")
    parser.add_argument("--decoder", type=str, default="", help="decoder onnx path")
    parser.add_argument(
        "--encoder-hmonnx",
        type=str,
        default="",
        help="quantized encoder hmonnx path",
    )
    parser.add_argument(
        "--decoder-hmonnx",
        type=str,
        default="",
        help="quantized decoder hmonnx path",
    )
    parser.add_argument(
        "--encoder-quant-type",
        type=str,
        default="w8a8_sefp",
        help="encoder quant type for default hmonnx path",
    )
    parser.add_argument(
        "--decoder-quant-type",
        type=str,
        default="w8a8",
        help="decoder quant type for default hmonnx path",
    )
    parser.add_argument(
        "--max-token-len",
        type=int,
        default=100,
        help="pad length for pre_acoustic_embeds and pre_token_mask",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="device for hmonnx inference",
    )
    parser.add_argument(
        "--hmonnx-float-dtype",
        type=str,
        default="fp16",
        help="float dtype for hmonnx inputs: fp16/fp32",
    )
    parser.add_argument(
        "--onnx-float-dtype",
        type=str,
        default="fp32",
        help="float dtype for onnxruntime inputs: fp16/fp32",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare hmonnx outputs with original onnx outputs",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=-1,
        help="onnxruntime device id (use -1 for CPU)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="onnxruntime intra-op threads",
    )
    args = parser.parse_args()

    model_dir = resolve_path(args.model_dir) if args.model_dir else None
    inputs_dir = resolve_path(args.inputs_dir)
    speech_path = inputs_dir / "speech.npy"
    ensure_file(speech_path, "speech.npy")

    manifest_path = resolve_manifest(args.export_manifest)
    model_dir_str = str(model_dir) if model_dir else ""
    encoder_path = resolve_encoder_onnx(args.encoder, model_dir_str, manifest_path)
    predictor_path = resolve_predictor_onnx(args.predictor, model_dir_str, manifest_path)
    decoder_path = resolve_decoder_onnx(args.decoder, model_dir_str, manifest_path)

    encoder_hmonnx = resolve_path(args.encoder_hmonnx) if args.encoder_hmonnx else default_hmonnx_path(encoder_path, args.encoder_quant_type)
    decoder_hmonnx = resolve_path(args.decoder_hmonnx) if args.decoder_hmonnx else default_hmonnx_path(decoder_path, args.decoder_quant_type)

    ort_encoder = _load_ort_model(encoder_path, args.device_id, args.threads)
    ort_predictor = _load_ort_model(predictor_path, args.device_id, args.threads)
    ort_decoder = _load_ort_model(decoder_path, args.device_id, args.threads)

    device = _select_device(args.device)
    hmonnx_encoder = _load_hmonnx(encoder_hmonnx, device)
    hmonnx_decoder = _load_hmonnx(decoder_hmonnx, device)

    encoder_inputs_dict = build_encoder_inputs(encoder_path, inputs_dir, float_dtype=args.hmonnx_float_dtype)
    input_specs = get_onnx_input_specs(encoder_path)
    encoder_inputs_list = [encoder_inputs_dict[name] for name, _ in input_specs]
    encoder_inputs_torch = _to_torch_inputs(encoder_inputs_list, device, args.hmonnx_float_dtype)

    encoder_outputs_q = hmonnx_encoder(*encoder_inputs_torch)
    if isinstance(encoder_outputs_q, (list, tuple)):
        enc_q_t = encoder_outputs_q[0]
        enc_mask_q_t = encoder_outputs_q[1] if len(encoder_outputs_q) > 1 else None
    else:
        enc_q_t = encoder_outputs_q
        enc_mask_q_t = None

    enc_q = enc_q_t.float().detach().cpu().numpy()
    if enc_mask_q_t is None:
        enc_mask_q = build_encoder_mask(encoder_inputs_dict, enc_q)
    else:
        enc_mask_q = enc_mask_q_t.float().detach().cpu().numpy()

    pre_acoustic_q, pre_token_length_q = ort_predictor([enc_q, enc_mask_q])
    token_len_q = int(pre_token_length_q[0])
    token_len_q = max(token_len_q, 1)

    pre_acoustic_q = pad_or_trim(pre_acoustic_q, args.max_token_len, axis=1, pad_value=0.0)
    pre_token_mask_q = make_pre_token_mask(token_len_q, args.max_token_len)

    decoder_inputs_q = build_decoder_inputs_from_arrays(
        decoder_path,
        enc_q,
        enc_mask_q,
        pre_acoustic_q,
        pre_token_mask_q,
        max_token_len=args.max_token_len,
    )
    decoder_input_names = [name for name, _ in get_onnx_input_specs(decoder_path)]
    decoder_inputs_q_list = [decoder_inputs_q[name] for name in decoder_input_names]
    decoder_inputs_q_torch = _to_torch_inputs(decoder_inputs_q_list, device, args.hmonnx_float_dtype)

    decoder_out_q = hmonnx_decoder(*decoder_inputs_q_torch)
    if isinstance(decoder_out_q, (list, tuple)):
        decoder_out_q = decoder_out_q[0]
    decoder_out_q = decoder_out_q.float().detach().cpu().numpy()

    print("[hmonnx]", _decode_text(decoder_out_q, token_len_q))

    if not args.compare:
        return

    encoder_inputs_ref = build_encoder_inputs(encoder_path, inputs_dir, float_dtype=args.onnx_float_dtype)
    encoder_input_list_ref = [encoder_inputs_ref[name] for name in ort_encoder.get_input_names()]
    encoder_outputs_ref = ort_encoder(encoder_input_list_ref)
    if len(encoder_outputs_ref) == 1:
        enc_ref = encoder_outputs_ref[0]
        enc_mask_ref = build_encoder_mask(encoder_inputs_ref, enc_ref)
    else:
        enc_ref, enc_mask_ref = encoder_outputs_ref[:2]
    pre_acoustic_ref, pre_token_length_ref = ort_predictor([enc_ref, enc_mask_ref])
    token_len_ref = int(pre_token_length_ref[0])
    token_len_ref = max(token_len_ref, 1)

    pre_acoustic_ref = pad_or_trim(pre_acoustic_ref, args.max_token_len, axis=1, pad_value=0.0)
    pre_token_mask_ref = make_pre_token_mask(token_len_ref, args.max_token_len)

    decoder_inputs_ref = build_decoder_inputs_from_arrays(
        decoder_path,
        enc_ref,
        enc_mask_ref,
        pre_acoustic_ref,
        pre_token_mask_ref,
        max_token_len=args.max_token_len,
    )
    decoder_inputs_ref_list = [decoder_inputs_ref[name] for name in decoder_input_names]
    decoder_out_ref = ort_decoder(decoder_inputs_ref_list)[0]

    compare_len = min(decoder_out_ref.shape[1], decoder_out_q.shape[1], token_len_ref, token_len_q)
    diff = np.max(np.abs(decoder_out_ref[:, :compare_len] - decoder_out_q[:, :compare_len]))
    print("[onnx] ", _decode_text(decoder_out_ref, token_len_ref))
    print(f"max_abs_diff: {diff:.6f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()
