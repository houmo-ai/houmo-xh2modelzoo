import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForTextToWaveform, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor


def _read_prompt(prompt: str) -> str:
    prompt_path = Path(prompt)
    if prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8")
    return prompt


def _load_image(image_path: str | None):
    if not image_path:
        return None
    return Image.open(image_path).convert("RGB")


def _load_audio(audio_path: str | None):
    if not audio_path:
        return None

    audio, _ = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.ascontiguousarray(audio)


def _build_conversation(prompt: str, image_path: str | None, audio_path: str | None):
    content = []
    if image_path:
        content.append({"type": "image", "image": image_path})
    if audio_path:
        content.append({"type": "audio", "audio": audio_path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _prepare_inputs(processor, conversation, image_path, audio_path, model, use_audio_in_video):
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text,
        audio=[_load_audio(audio_path)] if audio_path else None,
        images=[_load_image(image_path)] if image_path else None,
        videos=None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    return inputs.to(model.device).to(model.dtype)


def _decode_text(processor, text_result, input_length: int) -> str:
    output_text = processor.batch_decode(
        text_result.sequences[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0] if output_text else ""


def _save_audio(audio, output_audio_path: str):
    if audio is None:
        return

    output_path = Path(output_audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        output_path,
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )
    logger.info(f"Saved generated audio to {output_path}")


def main(args):
    model_dir = str(Path(args.model_dir).resolve())
    prompt = _read_prompt(args.prompt)
    conversation = _build_conversation(prompt, args.image_path, args.audio_path)
    model = AutoModelForTextToWaveform.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
    # model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    #     model_dir,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )
    assert isinstance(model, Qwen3OmniMoeForConditionalGeneration), (
        f"Expected model type Qwen3OmniMoeForConditionalGeneration, but got {type(model)}"
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_dir)

    inputs = _prepare_inputs(
        processor=processor,
        conversation=conversation,
        image_path=args.image_path,
        audio_path=args.audio_path,
        model=model,
        use_audio_in_video=True,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.no_grad():
        text_result, audio = model.generate(
            **inputs,
            speaker="Ethan",
            use_audio_in_video=True,
            thinker_max_new_tokens=args.max_new_tokens,
            thinker_return_dict_in_generate=True,
        )

    content = _decode_text(processor, text_result, inputs["input_ids"].shape[1])
    logger.info(content)
    _save_audio(audio, args.output_audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Debug Qwen3 Omni model with native HF implementation")
    parser.add_argument("--model-dir", type=str, default="data/models/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--image-path", type=str, default="examples_merak/llm/qwen3_omni_moe/data/cars.jpg")
    parser.add_argument("--audio-path", type=str, default="examples_merak/llm/qwen3_omni_moe/data/cough.wav")
    parser.add_argument("--prompt", type=str, default="What can you see and hear? Answer in one short sentence.")
    parser.add_argument("--output-audio", type=str, default="output.wav")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    main(args)
