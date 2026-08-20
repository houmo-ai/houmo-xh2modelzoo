from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, DynamicCache

from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import (
    MiniCPMO45HMONNXRuntime,
    ensure_tts_sampling_config,
    normalize_minicpmo_video,
    patch_audio_attention_return_compat,
    patch_dynamic_cache_seen_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniCPM-o-4.5 HMONNX video demo")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--question", default="请简短描述视频内容。")
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/minicpm_o_4_5_hmonnx_demo"))
    parser.add_argument("--exec-device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--generate-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--output-audio-path", type=Path)
    parser.add_argument("--output-speech-tokens-path", type=Path)
    return parser.parse_args()


def validate_audio(path: Path) -> dict[str, float | int | bool]:
    waveform, sample_rate = sf.read(path)
    samples = np.asarray(waveform, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    duration = float(samples.size / sample_rate) if sample_rate else 0.0
    if not np.isfinite(samples).all() or duration <= 0 or rms <= 1e-5:
        raise RuntimeError(f"TTS produced an invalid or silent WAV: {path}")
    return {"sample_rate": sample_rate, "duration": duration, "rms": rms, "non_silent": True}


def prepare_output_paths(output_dir: Path, audio_path: Path, speech_tokens_path: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    speech_tokens_path.parent.mkdir(parents=True, exist_ok=True)
    return audio_path, speech_tokens_path


def token2wav_asset_path(model_dir: Path) -> Path:
    return model_dir / "assets" / "token2wav"


def initialize_hmonnx_host(
    host,
    *,
    model_dir: Path,
    work_dir: Path,
    runtime_type: Callable = MiniCPMO45HMONNXRuntime,
):
    if hasattr(host, "init_tts"):
        host.init_tts(model_dir=str(token2wav_asset_path(model_dir)))
    return runtime_type(work_dir, host)


def token2wav_metadata(runtime) -> dict[str, dict[str, str] | dict[str, int]]:
    return {
        "backends": runtime.token2wav_backends,
        "execution_counts": runtime.token2wav_execution_counts,
    }


def main() -> None:
    args = parse_args()
    if args.generate_audio and args.ref_audio is None:
        raise ValueError("--ref-audio is required with --generate-audio")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    host = AutoModel.from_pretrained(
        args.model_dir,
        config=ensure_tts_sampling_config(config),
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    ).eval()
    patch_dynamic_cache_seen_tokens(DynamicCache)
    patch_audio_attention_return_compat(host)
    runtime = initialize_hmonnx_host(
        host,
        model_dir=Path(args.model_dir),
        work_dir=args.work_dir,
    )
    runtime.set_exec_device(args.exec_device)
    contents = normalize_minicpmo_video(args.video, include_audio=True, stack_frames=1)
    messages = [{"role": "user", "content": [*contents, args.question]}]
    if args.generate_audio:
        ref_audio, _ = librosa.load(args.ref_audio, sr=16000, mono=True)
        messages.insert(0, host.get_sys_prompt(ref_audio=ref_audio, mode="omni", language=args.language))
    runtime.reset_state()
    output_audio_path = args.output_audio_path or args.output_dir / "output.wav"
    output_speech_tokens_path = args.output_speech_tokens_path or args.output_dir / "speech_tokens.json"
    output_audio_path, output_speech_tokens_path = prepare_output_paths(
        args.output_dir,
        output_audio_path,
        output_speech_tokens_path,
    )
    answer = runtime.chat(
        msgs=messages,
        tokenizer=tokenizer,
        omni_mode=True,
        sampling=False,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        generate_audio=args.generate_audio,
        use_tts_template=args.generate_audio,
        output_audio_path=str(output_audio_path) if args.generate_audio else None,
        max_slice_nums=1,
    )
    text = answer if isinstance(answer, str) else str(answer)
    token_ids = np.asarray(tokenizer.encode(text, add_special_tokens=False)).reshape(-1).tolist()
    (args.output_dir / "output.txt").write_text(text + "\n", encoding="utf-8")
    metadata = {"text": text, "text_token_ids": token_ids, "tts_requested": args.generate_audio}
    if args.generate_audio:
        speech_tokens = runtime.last_speech_token_ids
        if speech_tokens is None or speech_tokens.numel() == 0:
            raise RuntimeError("TTS completed without speech tokens")
        output_speech_tokens_path.parent.mkdir(parents=True, exist_ok=True)
        output_speech_tokens_path.write_text(
            json.dumps({"token_ids": speech_tokens.reshape(-1).tolist()}, indent=2), encoding="utf-8"
        )
        metadata["speech_token_count"] = int(speech_tokens.numel())
        metadata["audio"] = validate_audio(output_audio_path)
        metadata["token2wav"] = token2wav_metadata(runtime)
        metadata["tts_status"] = "success"
    else:
        metadata["tts_status"] = "not_requested"
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    runtime.release()
    print(text)


if __name__ == "__main__":
    main()
