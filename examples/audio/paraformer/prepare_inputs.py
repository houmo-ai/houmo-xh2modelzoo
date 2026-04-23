import argparse

from export_utils import resolve_decoder_onnx, resolve_encoder_onnx, resolve_manifest
from input_utils import ensure_dir, resolve_path, save_npy
from runtime_utils import build_full_utterance_inputs, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data02/datasets/funasr/Paraformer")
    parser.add_argument("--model-revision", default="v2.0.4")
    parser.add_argument("--audio", default="/data02/datasets/funasr/Paraformer/example/asr_example.wav")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--export-manifest", default="")
    parser.add_argument("--encoder", default="")
    parser.add_argument("--decoder", default="")
    parser.add_argument("--inputs-dir", default="examples/audio/paraformer/inputs")
    parser.add_argument("--device-id", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    inputs_dir = resolve_path(args.inputs_dir)
    ensure_dir(inputs_dir)
    manifest_path = resolve_manifest(args.export_manifest)
    encoder_onnx = resolve_encoder_onnx(args.encoder, args.model_dir, manifest_path)
    decoder_onnx = resolve_decoder_onnx(args.decoder, args.model_dir, manifest_path)

    model = load_model(args.model, args.model_revision, device="cpu")
    prepared = build_full_utterance_inputs(
        model,
        args.audio,
        encoder_onnx,
        decoder_onnx,
        device_id=args.device_id,
        threads=args.threads,
    )

    save_npy(inputs_dir / "speech.npy", prepared["speech"])
    save_npy(inputs_dir / "speech_lengths.npy", prepared["speech_lengths"])
    save_npy(inputs_dir / "enc.npy", prepared["enc"])
    save_npy(inputs_dir / "enc_len.npy", prepared["enc_len"])
    save_npy(inputs_dir / "alphas.npy", prepared["alphas"])
    save_npy(inputs_dir / "acoustic_embeds.npy", prepared["acoustic_embeds"])
    save_npy(inputs_dir / "acoustic_embeds_len.npy", prepared["acoustic_embeds_len"])
    for name, value in prepared["decoder_inputs"].items():
        if name.startswith("in_cache_"):
            save_npy(inputs_dir / f"{name}.npy", value)

    print(f"saved inputs to {inputs_dir}")


if __name__ == "__main__":
    main()