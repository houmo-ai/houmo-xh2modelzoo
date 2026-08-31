from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import librosa
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

from xhmodel_merak.xh_other_model.models.funaudiochat.constant import (
    AUDIO_TEMPLATE,
    DEFAULT_S2M_GEN_KWARGS,
    DEFAULT_SP_GEN_KWARGS,
    SPOKEN_S2M_PROMPT,
)
from xhmodel_merak.xh_other_model.models.funaudiochat.register import register_funaudiochat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the float FunAudioChat model")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_funaudiochat()
    config = AutoConfig.from_pretrained(args.model_dir)
    config.audio_config.crq_transformer_config["torch_dtype"] = torch.float16
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir,
        config=config,
        torch_dtype=torch.float16,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()

    sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
    sp_gen_kwargs["text_greedy"] = True
    model.sp_gen_kwargs.update(sp_gen_kwargs)
    gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
    gen_kwargs["max_new_tokens"] = args.max_new_tokens

    waveform = librosa.load(args.audio, sr=16000)[0]
    conversation = [
        {"role": "system", "content": SPOKEN_S2M_PROMPT},
        {"role": "user", "content": AUDIO_TEMPLATE},
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text,
        audio=[waveform],
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(model.device)
    generate_ids, audio_ids = model.generate(**inputs, **gen_kwargs)
    text_ids = generate_ids[:, inputs.input_ids.shape[1] :]
    print("generate_text:", processor.decode(text_ids[0], skip_special_tokens=True))
    if audio_ids is not None:
        print("generate_audio_tokens:", int(audio_ids.shape[-1]))


if __name__ == "__main__":
    main()
