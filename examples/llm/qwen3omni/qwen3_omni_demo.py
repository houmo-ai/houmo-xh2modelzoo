# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni native HF demo — text / vision / audio / multimodal conversation."""

import argparse
from pathlib import Path

import soundfile as sf
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

try:
    from qwen_omni_utils import process_mm_info
except ImportError:

    def process_mm_info(conversation, use_audio_in_video=False):
        audios, images, videos = [], [], []
        for turn in conversation:
            for item in turn.get("content", []):
                tp = item.get("type")
                if tp == "audio":
                    audios.append(item.get("audio"))
                elif tp == "image":
                    images.append(item.get("image"))
                elif tp == "video":
                    videos.append(item.get("video"))
        return audios, images, videos


SCRIPT_DIR = Path(__file__).resolve().parent


def build_conversation(case: str):
    """Return (conversation, use_audio_in_video) for the given case."""
    image_path = str(SCRIPT_DIR / "data" / "cars.jpg")
    audio_path = str(SCRIPT_DIR / "data" / "cough.wav")

    if case == "text":
        return [
            {"role": "user", "content": [{"type": "text", "text": "请用一句话介绍你自己。"}]},
        ], False
    elif case == "vision":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "请描述这张图。"},
                ],
            },
        ], False
    elif case == "audio":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "请描述你听到了什么。"},
                ],
            },
        ], False
    else:  # multimodal
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "What can you see and hear? Answer in one short sentence."},
                ],
            },
        ], True


def main(args):
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model,
        dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model)

    conversation, use_audio_in_video = build_conversation(args.case)

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    torch.cuda.empty_cache()

    text_ids, audio = model.generate(
        **inputs,
        speaker="Ethan",
        thinker_return_dict_in_generate=True,
        use_audio_in_video=use_audio_in_video,
        max_new_tokens=args.max_new_tokens,
    )

    output_text = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(f"[demo] case={args.case}")
    print(f"[demo] output text: {output_text}")

    if audio is not None:
        out_wav = Path(args.work_dir) / "demo_output.wav"
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), audio.reshape(-1).detach().cpu().tolist(), samplerate=24000)
        print(f"[demo] audio saved to {out_wav}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-Omni native HF demo")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni/demo")
    parser.add_argument("--case", type=str, default="text", choices=["text", "vision", "audio", "multimodal"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
