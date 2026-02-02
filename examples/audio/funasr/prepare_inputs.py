import argparse
import shutil

from export_utils import resolve_encoder_onnx, resolve_manifest, resolve_predictor_onnx
from input_utils import (
    build_encoder_inputs,
    build_encoder_mask,
    ensure_file,
    ensure_dir,
    make_pre_token_mask,
    pad_or_trim,
    resolve_path,
    save_npy,
)
from utils import OrtInferSession


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="directory containing encoder.onnx and predictor.onnx",
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
        help="directory to save prepared inputs",
    )
    parser.add_argument(
        "--speech",
        type=str,
        default="",
        help="speech.npy path (defaults to <inputs-dir>/speech.npy)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="",
        help="encoder ONNX path (defaults to <model-dir>/encoder.onnx)",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        default="",
        help="predictor ONNX path (defaults to <model-dir>/predictor.onnx)",
    )
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
    ensure_dir(inputs_dir)

    default_speech_path = inputs_dir / "speech.npy"
    speech_path = resolve_path(args.speech) if args.speech else default_speech_path
    manifest_path = resolve_manifest(args.export_manifest)
    model_dir_str = str(model_dir) if model_dir else ""
    encoder_path = resolve_encoder_onnx(args.encoder, model_dir_str, manifest_path)
    predictor_path = resolve_predictor_onnx(args.predictor, model_dir_str, manifest_path)

    ensure_file(speech_path, "speech.npy")
    if speech_path != default_speech_path:
        shutil.copyfile(speech_path, default_speech_path)
    ensure_file(encoder_path, "encoder onnx")
    ensure_file(predictor_path, "predictor onnx")

    encoder = OrtInferSession(str(encoder_path), args.device_id, args.threads)
    predictor = OrtInferSession(str(predictor_path), args.device_id, args.threads)

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
    token_len = max(token_len, 1)

    pre_acoustic_embeds = pad_or_trim(pre_acoustic_embeds, args.max_token_len, axis=1, pad_value=0.0)
    pre_token_mask = make_pre_token_mask(token_len, args.max_token_len)

    save_npy(inputs_dir / "enc.npy", enc)
    save_npy(inputs_dir / "enc_mask.npy", enc_mask)
    save_npy(inputs_dir / "pre_acoustic_embeds.npy", pre_acoustic_embeds)
    save_npy(inputs_dir / "pre_token_mask.npy", pre_token_mask)

    print(f"saved: {inputs_dir / 'enc.npy'}")
    print(f"saved: {inputs_dir / 'enc_mask.npy'}")
    print(f"saved: {inputs_dir / 'pre_acoustic_embeds.npy'}")
    print(f"saved: {inputs_dir / 'pre_token_mask.npy'}")


if __name__ == "__main__":
    main()
