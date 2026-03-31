import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
import torch
from transformers import TextStreamer

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, LLMInferenceContextManager, LLMModelState
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen3_omni_moe import XHQwen3OmniMoeVisionEncoderModel, XHQwen3OmniVisualConfig


def _build_conversation(prompt: str, image_path: str | None, audio_path: str | None):
    content = []
    if image_path:
        content.append({"type": "image", "image": image_path})
    if audio_path:
        content.append({"type": "audio", "audio": audio_path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _read_prompt(prompt: str) -> str:
    prompt_path = Path(prompt)
    if prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8")
    return prompt


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
    logger = get_xhquant_logger()
    logger.info(f"Saved generated audio to {output_path}")


def main(args):
    if not args.config:
        raise ValueError("Either --config or --model must be specified.")

    cfg = Config.fromfile(args.config)
    cfg_name = Path(args.config).stem if args.config else f"{Path(args.model).name}_{args.model_type.lower()}"
    if args.debug:
        cfg_name += "_debug"

    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(work_dir / f"generate_{args.eval_type}.log")

    xhquant_init(log_file, args.debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()
    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    cfg.dump(work_dir / f"{cfg_name}.py")

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")

    prompt = _read_prompt(args.prompt)
    conversation = _build_conversation(prompt, args.image_path, args.audio_path)

    model_cfg: XHQwen3OmniVisualConfig = AutoLLMConfig.from_pretrained(cfg.model)
    assert type(model_cfg).__name__ == "XHQwen3OmniVisualConfig", (
        f"Expected model config type XHQwen3OmniVisualConfig, but got {type(model_cfg).__name__}"
    )
    model_cfg.work_dir = str(work_dir)  # visual部分需要一个work_dir，存放中间的onnx
    model_cfg.enable_auto_offload = args.auto_offload  # 是否启用自动显存卸载
    xh_model: XHQwen3OmniMoeVisionEncoderModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHQwen3OmniMoeVisionEncoderModel", (
        f"Expected model config type XHQwen3OmniMoeVisionEncoderModel, but got {type(xh_model).__name__}"
    )
    xh_model.set_state(LLMModelState.from_string(args.eval_type))

    processor = xh_model.get_tf_processor()
    tokenizer = processor.tokenizer
    model_inputs = processor.apply_chat_template(conversation, enable_thinking=False).to(device).to(xh_model.dtype)
    streamer = TextStreamer(tokenizer=tokenizer)

    xh_model.to(device=device, dtype=dtype)
    xh_model.eval()
    contexts = [
        TimeProfiler("generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(xh_model),
        torch.no_grad(),
    ]
    with ContextManagers(contexts):
        text_result, audio = xh_model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    content = _decode_text(processor, text_result, model_inputs["input_ids"].shape[1])
    logger.info(content)
    if args.output_audio is None or len(args.output_audio) == 0:
        args.output_audio = str(Path(work_dir) / "output.wav")
    _save_audio(audio, args.output_audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/qwen3_omni/30B-A3B-Instruct/xh2a_qwen3-omni-30b-a3b_instruct_visual_w8a8h1_sefp_256_2k_560x560.py",
    )

    parser.add_argument("--eval-type", type=str, default="wrap", choices=LLMModelState.get_all_values())
    parser.add_argument("--image-path", type=str, default="examples_merak/llm/qwen3_omni_moe/data/cars.jpg")
    parser.add_argument("--audio-path", type=str, default="examples_merak/llm/qwen3_omni_moe/data/cough.wav")
    parser.add_argument("--prompt", type=str, default="What can you see and hear? Answer in one short sentence.")
    parser.add_argument("--output-audio", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--auto-offload", action="store_true", help="Whether to enable auto offload, only for debug and development"
    )
    args = parser.parse_args()
    main(args)
