from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import librosa
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, DynamicCache

from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import (
    ensure_tts_sampling_config,
    normalize_minicpmo_video,
    patch_audio_attention_return_compat,
    patch_dynamic_cache_seen_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniCPM-o-4.5 Hugging Face video demo")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--question", default="请简短描述视频内容。")
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/minicpm_o_4_5_hf_demo"))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--generate-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--output-audio-path", type=Path)
    return parser.parse_args()


def token2wav_asset_path(model_dir: Path) -> Path:
    return model_dir / "assets" / "token2wav"


ModelT = TypeVar("ModelT")


def prepare_hf_model(
    model: ModelT,
    *,
    cache_type: type = DynamicCache,
    patch_dynamic_cache: Callable[[type], None] = patch_dynamic_cache_seen_tokens,
    patch_audio_attention: Callable[[ModelT], None] = patch_audio_attention_return_compat,
) -> ModelT:
    patch_dynamic_cache(cache_type)
    patch_audio_attention(model)
    return model


def main() -> None:
    args = parse_args()
    if args.generate_audio and args.ref_audio is None:
        raise ValueError("--ref-audio is required with --generate-audio")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model = prepare_hf_model(
        AutoModel.from_pretrained(
            args.model_dir,
            config=ensure_tts_sampling_config(config),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
        ).eval()
    )
    contents = normalize_minicpmo_video(args.video, include_audio=True, stack_frames=1)
    messages = [{"role": "user", "content": [*contents, args.question]}]
    if args.generate_audio:
        model.init_tts(model_dir=str(token2wav_asset_path(Path(args.model_dir))))
        ref_audio, _ = librosa.load(args.ref_audio, sr=16000, mono=True)
        messages.insert(0, model.get_sys_prompt(ref_audio=ref_audio, mode="omni", language=args.language))
    output_audio_path = args.output_audio_path or args.output_dir / "output.wav"
    answer = model.chat(
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "output.txt").write_text(text + "\n", encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(
        json.dumps({"video": str(args.video), "question": args.question, "text": text}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(text)


if __name__ == "__main__":
    main()
