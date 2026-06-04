import argparse
from pathlib import Path

import torch
from transformers import TextStreamer

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler

try:
    from validate_hm_release_layout import ensure_step_artifact_links
except ImportError:
    ensure_step_artifact_links = None


def _resolve_export_dir(config: str) -> Path:
    path = Path(config).resolve()
    if path.is_dir():
        return path
    return path.parent


def _build_text_inputs(hmonnx_model, prompt: str, think: bool, device: str):
    tokenizer = hmonnx_model.get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    return tokenizer, tokenizer([text], return_tensors="pt", truncation=True).to(device)


def _build_multimodal_inputs(hmonnx_model, prompt: str, image_path: str, think: bool, device: str):
    processor = hmonnx_model.get_tf_processor()
    tokenizer = processor.tokenizer
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    return tokenizer, processor.apply_chat_template(messages, enable_thinking=think).to(device)


def main(args):
    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()
    hmonnx_model = AutoLLMHONNXModel.from_pretrained(args.config)
    logger.info(f"Resolved HMONNX model type: {type(hmonnx_model).__name__}")

    if args.auto_offload and hasattr(hmonnx_model, "enable_auto_offload"):
        hmonnx_model.enable_auto_offload = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = args.prompt
    if Path(prompt).is_file():
        prompt = Path(prompt).read_text()

    use_multimodal = bool(args.image_path) and Path(args.image_path).exists() and hasattr(hmonnx_model, "get_tf_processor")
    if use_multimodal:
        tokenizer, model_inputs = _build_multimodal_inputs(hmonnx_model, prompt, args.image_path, args.think, device)
    else:
        tokenizer, model_inputs = _build_text_inputs(hmonnx_model, prompt, args.think, device)

    streamer = TextStreamer(tokenizer)
    hmonnx_model.to(device)
    if args.fast and hasattr(hmonnx_model, "to_fast"):
        hmonnx_model.to_fast()
    if args.golden and hasattr(hmonnx_model, "enable_golden"):
        hmonnx_model.enable_golden = True

    contexts = [
        TimeProfiler("hmonnx_generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(hmonnx_model, devices=[device]),
    ]
    with ContextManagers(contexts):
        generated_ids = hmonnx_model.generate(
            **model_inputs,
            max_new_tokens=2 if args.golden else args.max_new_tokens,
            streamer=streamer,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    logger.info(f"{'-' * 20} content {'-' * 20}")
    logger.info(content)

    if args.golden and ensure_step_artifact_links is not None:
        export_dir = _resolve_export_dir(args.config)
        if (export_dir / "golden_meta_info.json").is_file():
            ensure_step_artifact_links(export_dir)
            logger.info(f"Ensured HM golden step artifact links under {export_dir}")
        else:
            logger.warning(f"Skip HM step artifact link repair; golden_meta_info.json not found under {export_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to exported golden_meta_info.json or meta file")
    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    parser.add_argument("--debug", action="store_true", help="run in debug mode")
    parser.add_argument("--image-path", type=str, default="./data/images/demo_qwen3_vl.jpeg")
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--think", action="store_true", help="enable think mode")
    parser.add_argument("--golden", action="store_true", help="save golden outputs")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--auto-offload", action="store_true", help="Whether to enable auto offload, only for debug and development"
    )
    main(parser.parse_args())
