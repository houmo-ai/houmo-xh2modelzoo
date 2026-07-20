import argparse
from pathlib import Path

from xhmodel_merak.xh_other_model.models.sensevoice_small import runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with a migrated SenseVoiceSmall HMONNX.")
    parser.add_argument("audio_files", nargs="+", help="Audio files to transcribe.")
    parser.add_argument("--export-dir", required=True, help="Directory containing export_meta_info.json.")
    parser.add_argument("--assets-dir", default="", help="Optional frontend/token asset override.")
    parser.add_argument("--tokens", default="", help="Optional tokens.json or SentencePiece model.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", choices=runtime.LANGUAGE_MAP, default="auto")
    parser.add_argument("--textnorm", choices=runtime.TEXTNORM_MAP, default="woitn")
    parser.add_argument("--fast", action="store_true", help="Enable xhquant fast mode (experimental for this graph).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = runtime.resolve_export_artifacts(args.export_dir)
    if "hmonnx_file" not in artifacts:
        raise FileNotFoundError("The export metadata does not contain a HMONNX artifact")
    hmonnx_file = Path(artifacts["hmonnx_file"])
    assets_dir = Path(args.assets_dir).expanduser().resolve() if args.assets_dir else Path(artifacts["assets_dir"])
    frontend = runtime.build_frontend(assets_dir)
    token_file = runtime.find_token_file(assets_dir, args.tokens or None)
    token_list = runtime.load_tokens(token_file)

    for audio_value in args.audio_files:
        audio_file = Path(audio_value.replace("\\", "/")).expanduser().resolve()
        if not audio_file.is_file():
            print(f"Audio file not found: {audio_file}")
            continue
        sample = runtime.Sample(
            audio=str(audio_file),
            text="",
            language=args.language,
            textnorm=args.textnorm,
        )
        waveform = runtime.load_audio_any(sample, target_sr=int(frontend.cfg.fs))
        feat, feat_len = runtime.extract_features(frontend, waveform)
        inputs = runtime.make_inputs_for_sample(feat, feat_len, args.language, args.textnorm)
        logits, output_lengths = runtime.run_hmonnx(
            hmonnx_file,
            inputs,
            device=args.device,
            fast=args.fast,
        )
        output_length = int(output_lengths[0]) if hasattr(output_lengths, "__len__") else int(output_lengths)
        token_ids = runtime.ctc_greedy_decode(logits[0], output_length)
        raw_text = runtime.decode_token_ids(token_ids, token_list)
        print(f"{audio_file}: {runtime.strip_rich_tags(raw_text)}")


if __name__ == "__main__":
    main()
