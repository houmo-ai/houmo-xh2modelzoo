import argparse
import logging
from typing import List

import numpy as np
import torch
from xhquant.api import HMONNXGoldenInference

from export_utils import default_hmonnx_path, resolve_decoder_onnx, resolve_encoder_onnx, resolve_manifest
from input_utils import build_encoder_inputs_from_arrays, get_onnx_input_specs
from runtime_utils import OrtInferSession, build_decoder_inputs_from_encoder_outputs, decode_ids, extract_full_features, load_model, tokens_to_text

LOGGER = logging.getLogger(__name__)


def _select_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _to_torch_inputs(arrays: List[np.ndarray], device: str, float_dtype: str) -> List[torch.Tensor]:
    if float_dtype in {"fp16", "float16"}:
        target = torch.float16
    elif float_dtype in {"fp32", "float32"}:
        target = torch.float32
    else:
        raise ValueError(f"unsupported float dtype: {float_dtype}")

    tensors: List[torch.Tensor] = []
    for array in arrays:
        tensor = torch.from_numpy(array)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=target)
        tensors.append(tensor.to(device))
    return tensors


def _run_hmonnx(session: HMONNXGoldenInference, arrays: List[np.ndarray], device: str, float_dtype: str) -> List[np.ndarray]:
    outputs = session(*_to_torch_inputs(arrays, device, float_dtype))
    if not isinstance(outputs, (tuple, list)):
        outputs = [outputs]
    return [value.detach().cpu().numpy() for value in outputs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data02/datasets/funasr/Paraformer")
    parser.add_argument("--model-revision", default="v2.0.4")
    parser.add_argument("--audio", default="/data02/datasets/funasr/Paraformer/example/asr_example.wav")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--export-manifest", default="")
    parser.add_argument("--encoder", default="examples/audio/paraformer/exports/mask_safe_static/model.onnx")
    parser.add_argument("--decoder", default="examples/audio/paraformer/exports/mask_safe_static/decoder.onnx")
    parser.add_argument("--encoder-hmonnx", default="work_dirs/paraformer_static_encoder/hmonnx/model_w8a8_sefp_XH2a.onnx")
    parser.add_argument("--decoder-hmonnx", default="work_dirs/paraformer_static_decoder/hmonnx/decoder_w8a8_XH2a.onnx")
    parser.add_argument("--encoder-quant-type", default="w8a8_sefp")
    parser.add_argument("--decoder-quant-type", default="w8a8")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hmonnx-float-dtype", default="fp16")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--device-id", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    manifest_path = resolve_manifest(args.export_manifest)
    encoder_path = resolve_encoder_onnx(args.encoder, args.model_dir, manifest_path)
    decoder_path = resolve_decoder_onnx(args.decoder, args.model_dir, manifest_path)
    encoder_hmonnx = args.encoder_hmonnx or str(default_hmonnx_path(encoder_path, args.encoder_quant_type))
    decoder_hmonnx = args.decoder_hmonnx or str(default_hmonnx_path(decoder_path, args.decoder_quant_type))

    model = load_model(args.model, args.model_revision, device="cpu")
    decoder_input_names = [name for name, _ in get_onnx_input_specs(decoder_path)]

    device = _select_device(args.device)
    encoder_session = HMONNXGoldenInference(encoder_hmonnx)
    encoder_session.to(device)
    decoder_session = HMONNXGoldenInference(decoder_hmonnx)
    decoder_session.to(device)

    ref_encoder = OrtInferSession(str(encoder_path), device_id=args.device_id, intra_op_num_threads=args.threads) if args.compare else None
    ref_decoder = OrtInferSession(str(decoder_path), device_id=args.device_id, intra_op_num_threads=args.threads) if args.compare else None

    speech, speech_lengths = extract_full_features(model, args.audio)
    max_abs_diff = 0.0
    encoder_input_dict = build_encoder_inputs_from_arrays(encoder_path, speech, speech_lengths)
    encoder_inputs = [encoder_input_dict[name] for name, _ in get_onnx_input_specs(encoder_path)]
    encoder_outputs = _run_hmonnx(encoder_session, encoder_inputs, device, args.hmonnx_float_dtype)
    enc_q, enc_len_q, alphas_q = encoder_outputs[:3]
    prepared_q = build_decoder_inputs_from_encoder_outputs(model, decoder_path, enc_q, enc_len_q, alphas_q)
    
    prepared_q["decoder_inputs"]["pre_token_mask"] = prepared_q["decoder_inputs"]["pre_token_mask"][:,:20]
    
    decoder_outputs_q = _run_hmonnx(
        decoder_session,
        [prepared_q["decoder_inputs"][name] for name in decoder_input_names],
        device,
        args.hmonnx_float_dtype,
    )
    logits_q, sample_ids_q = decoder_outputs_q[:2]
    text_q = tokens_to_text(decode_ids(model, sample_ids_q[0], int(prepared_q["acoustic_embeds_len"][0])))

    print("[hmonnx]", text_q)
    if args.compare:
        ref_encoder_outputs = ref_encoder(encoder_inputs)
        enc_ref, enc_len_ref, alphas_ref = ref_encoder_outputs[:3]
        prepared_ref = build_decoder_inputs_from_encoder_outputs(model, decoder_path, enc_ref, enc_len_ref, alphas_ref)
        decoder_outputs_ref = ref_decoder([prepared_ref["decoder_inputs"][name] for name in decoder_input_names])
        logits_ref, sample_ids_ref = decoder_outputs_ref[:2]
        text_ref = tokens_to_text(decode_ids(model, sample_ids_ref[0], int(prepared_ref["acoustic_embeds_len"][0])))
        compare_len = min(logits_ref.shape[1], logits_q.shape[1], int(prepared_ref["acoustic_embeds_len"][0]), int(prepared_q["acoustic_embeds_len"][0]))
        if compare_len > 0:
            max_abs_diff = float(np.max(np.abs(logits_ref[:, :compare_len] - logits_q[:, :compare_len])))
        print("[onnx]", text_ref)
        print(f"max_abs_diff: {max_abs_diff:.6f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()