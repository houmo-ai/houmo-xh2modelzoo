import argparse
from pathlib import Path
import wave

import numpy as np
import torch
from PIL import Image
from transformers import TextStreamer

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhquant.api import get_xhquant_logger, xhquant_init


def _load_wav_without_soundfile(audio_path: str) -> np.ndarray:
    with wave.open(audio_path, "rb") as wav_file:
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        num_frames = wav_file.getnframes()
        pcm_bytes = wav_file.readframes(num_frames)

    if sample_width == 1:
        audio = np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / float(1 << 15)
    elif sample_width == 4:
        audio = np.frombuffer(pcm_bytes, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio)


def _load_audio(audio_path: str | None):
    if not audio_path:
        return None

    try:
        import soundfile as sf
    except ModuleNotFoundError:
        if Path(audio_path).suffix.lower() == ".wav":
            return _load_wav_without_soundfile(audio_path)
        raise ModuleNotFoundError("soundfile is required for non-wav audio inputs.")

    audio, _ = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.ascontiguousarray(audio)


def main(args):
    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()
    is_golden = getattr(args, "golden", False)
    is_fast = getattr(args, "fast", False)
    hmonnx_model = AutoLLMHONNXModel.from_pretrained(args.config)
    tokenizer = hmonnx_model.get_tokenizer()
    processor = hmonnx_model.get_tf_processor()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    content = []
    if args.image_path:
        content.append({"type": "image", "image": Image.open(args.image_path).convert("RGB")})
    if args.audio_path:
        content.append({"type": "audio", "audio": _load_audio(args.audio_path), "sampling_rate": args.audio_sampling_rate})
    content.append({"type": "text", "text": args.prompt})
    messages = [{"role": "user", "content": content}]

    model_inputs = processor.apply_chat_template(messages)
    model_inputs = model_inputs.to(device)
    streamer = TextStreamer(tokenizer)
    hmonnx_model.to(device)
    if is_golden:
        hmonnx_model.enable_golden = True
        logger.warning("Golden outputs should be generated in aligned precision for stability.")
    elif is_fast:
        hmonnx_model.to_fast()
    with LLMInferenceContextManager(hmonnx_model):
        generated_ids = hmonnx_model.generate(
            **model_inputs,
            max_new_tokens=2 if is_golden else args.max_new_tokens,
            streamer=streamer,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    print(tokenizer.decode(output_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="work_dirs/gemma4_e2b_it_latest/golden_meta_info.json")
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--audio-path", type=str, default="")
    parser.add_argument("--audio-sampling-rate", type=int, default=16000)
    parser.add_argument("--prompt", type=str, default="Describe this input.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    parser.add_argument("--golden", action="store_true", help="save golden outputs")
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())
