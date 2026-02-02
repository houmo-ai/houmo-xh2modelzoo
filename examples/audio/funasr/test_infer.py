import argparse
import logging
from pathlib import Path

import numpy as np

from export_utils import resolve_decoder_onnx, resolve_encoder_onnx, resolve_manifest, resolve_predictor_onnx
from input_utils import (
    build_decoder_inputs_from_arrays,
    build_encoder_inputs,
    build_encoder_mask,
    ensure_file,
    make_pre_token_mask,
    pad_or_trim,
    resolve_path,
    save_npy,
)
from post_decode import post_decode
from utils import OrtInferSession

logger = logging.getLogger(__name__)


def _load_ort_model(model_path: Path, device_id: int, threads: int) -> OrtInferSession:
    ensure_file(model_path, "onnx model")
    return OrtInferSession(str(model_path), device_id, threads)


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
        help="directory containing speech.npy (and where outputs can be saved)",
    )
    parser.add_argument("--encoder", type=str, default="", help="encoder onnx path")
    parser.add_argument("--predictor", type=str, default="", help="predictor onnx path")
    parser.add_argument("--decoder", type=str, default="", help="decoder onnx path")
    parser.add_argument(
        "--max-token-len",
        type=int,
        default=100,
        help="pad length for pre_acoustic_embeds and pre_token_mask",
    )
    parser.add_argument(
        "--float-dtype",
        type=str,
        default="fp32",
        help="float dtype for encoder inputs: fp16/fp32",
    )
    parser.add_argument(
        "--save-inputs",
        action="store_true",
        help="save enc/enc_mask/pre_* inputs into inputs-dir",
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

    encoder = _load_ort_model(encoder_path, args.device_id, args.threads)
    predictor = _load_ort_model(predictor_path, args.device_id, args.threads)
    decoder = _load_ort_model(decoder_path, args.device_id, args.threads)

    encoder_inputs = build_encoder_inputs(encoder_path, inputs_dir, float_dtype=args.float_dtype)
    input_list = [encoder_inputs[name] for name in encoder.get_input_names()]
    encoder_outputs = encoder(input_list)
    if len(encoder_outputs) == 1:
        enc = encoder_outputs[0]
        enc_mask = build_encoder_mask(encoder_inputs, enc)
    else:
        enc, enc_mask = encoder_outputs[:2]
    pre_acoustic_embeds, pre_token_length = predictor([enc, enc_mask])

    token_len = int(pre_token_length[0])
    if token_len <= 1:
        logger.info("No words detected.")
        return

    pre_acoustic_embeds = pad_or_trim(pre_acoustic_embeds, args.max_token_len, axis=1, pad_value=0.0)
    pre_token_mask = make_pre_token_mask(token_len, args.max_token_len)

    if args.save_inputs:
        save_npy(inputs_dir / "enc.npy", enc)
        save_npy(inputs_dir / "enc_mask.npy", enc_mask)
        save_npy(inputs_dir / "pre_acoustic_embeds.npy", pre_acoustic_embeds)
        save_npy(inputs_dir / "pre_token_mask.npy", pre_token_mask)

    decoder_inputs = build_decoder_inputs_from_arrays(
        decoder_path,
        enc,
        enc_mask,
        pre_acoustic_embeds,
        pre_token_mask,
        max_token_len=args.max_token_len,
    )
    input_list = [decoder_inputs[name] for name in decoder.get_input_names()]
    outputs = decoder(input_list)
    decoder_out = outputs[0]

    if decoder_out.shape[1] > token_len:
        decoder_out = decoder_out[:, :token_len]

    preds = post_decode(decoder_out, np.array([token_len]))
    print("".join(preds[0]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()
