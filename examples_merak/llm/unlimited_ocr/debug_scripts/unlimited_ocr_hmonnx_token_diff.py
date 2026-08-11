"""Compare HF native and HMONNX greedy tokens for Unlimited-OCR.

This is a lightweight diagnosis harness for HMONNX quality regressions. It runs
the same base/no-crop prompt through the localized HF model and the exported
HMONNX runtime, then prints the generated token ids/text for the first few
greedy decode steps. If token 0 already differs, investigate prefill/quant or
visual scatter. If token 0 matches and later tokens diverge, investigate decode
graph, KV cache, position, or sliding-window behavior.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import GenerationConfig, GenerationMixin

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_unlimitedocr import UnlimitedOCRForCausalLM
from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_unlimitedocr_patch import unlimited_ocr_patch
from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_processor import XHUnlimitedOCRProcessor
from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_visual_model import UnlimitedOCRBaseVisualModel
from xhquant.api import xhquant_init


PROMPTS = {
    "markdown": "<image>\\n<|grounding|>Convert the document to markdown. ",
    "free-ocr": "<image>\\nFree OCR. ",
}
DEFAULT_HMONNX_SEARCH_ROOT = Path("work_dirs/_unlimited_ocr_xh2a_calib_debug")


def resolve_hmonnx_config(path: str) -> str:
    if path:
        return path
    candidates = sorted(
        DEFAULT_HMONNX_SEARCH_ROOT.glob("*/golden_meta_info.json"),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No golden_meta_info.json found under {DEFAULT_HMONNX_SEARCH_ROOT}. "
            "Pass --hmonnx-config explicitly."
        )
    return str(candidates[-1])


def ensure_generation_support(model, hf_model_dir: str) -> None:
    model_cls = type(model)
    if not issubclass(model_cls, GenerationMixin):
        patched_cls = type(
            f"{model_cls.__name__}WithGeneration",
            (model_cls, GenerationMixin),
            {"__module__": model_cls.__module__},
        )
        model.__class__ = patched_cls
    if getattr(model, "generation_config", None) is None:
        try:
            model.generation_config = GenerationConfig.from_pretrained(hf_model_dir)
        except OSError:
            model.generation_config = GenerationConfig.from_model_config(model.config)


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def print_sequence(name: str, tokenizer, input_len: int, output_ids: torch.Tensor, scores=None) -> list[int]:
    generated = output_ids[0, input_len:].detach().cpu().tolist()
    print(f"\n[{name}] generated {len(generated)} token(s)")
    for index, token_id in enumerate(generated):
        piece = decode_token(tokenizer, int(token_id))
        if scores is not None and index < len(scores):
            top_values, top_indices = torch.topk(scores[index][0].detach().float().cpu(), k=min(5, scores[index].shape[-1]))
            top = ", ".join(
                f"{int(tok)}:{decode_token(tokenizer, int(tok))!r}:{float(val):.3f}"
                for tok, val in zip(top_indices, top_values, strict=False)
            )
            print(f"  {index:02d}: {int(token_id):>6} {piece!r}  top5=[{top}]")
        else:
            print(f"  {index:02d}: {int(token_id):>6} {piece!r}")
    text = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    print(f"[{name}] text: {text!r}")
    return [int(token_id) for token_id in generated]


def run_hf(args, tokenizer, processor, prompt: str, device: str):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    inputs = processor.process(prompt, args.image_path, device=device)
    input_ids = inputs["input_ids"]
    images_ori = inputs["images_ori"].to(dtype)
    images_seq_mask = inputs["images_seq_mask"]
    images_spatial_crop = inputs["images_spatial_crop"]
    images_crop = torch.zeros((1, 3, args.image_size, args.image_size), device=device, dtype=dtype)

    model = UnlimitedOCRForCausalLM.from_pretrained(
        args.hf_model,
        dtype=dtype,
        trust_remote_code=False,
    ).to(device).eval()
    ensure_generation_support(model, args.hf_model)
    original_sliding_window = getattr(model.config, "sliding_window", None)
    original_sliding_window_size = getattr(model.config, "sliding_window_size", None)
    model.config._ring_window = original_sliding_window_size or original_sliding_window
    model.config.sliding_window = None
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        output = model.generate(
            input_ids=input_ids,
            images=[(images_crop, images_ori)],
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    model.config.sliding_window = original_sliding_window
    return input_ids, output.sequences, output.scores


def run_hmonnx(args, prompt: str, device: str):
    hmonnx_model = AutoLLMHONNXModel.from_pretrained(args.hmonnx_config)
    if args.hf_embedding_probe:
        hf_model = UnlimitedOCRForCausalLM.from_pretrained(
            args.hf_model,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=False,
        )
        hmonnx_model.embed_tokens = hf_model.get_input_embeddings().to(device=device, dtype=hmonnx_model.dtype).eval()
        del hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.hf_visual_probe:
        hf_visual_model = UnlimitedOCRForCausalLM.from_pretrained(
            args.hf_model,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=False,
        )
        hf_visual = UnlimitedOCRBaseVisualModel(unlimited_ocr_patch(hf_visual_model)).to(
            device=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).eval()
        hf_visual.config = SimpleNamespace(crop_mode=False, image_size=args.image_size)
        hf_visual.device = torch.device(device)
        hf_visual.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        hmonnx_model.visual = hf_visual
    hmonnx_model = hmonnx_model.to(device)
    if args.fast:
        hmonnx_model.to_fast()
    if args.auto_offload:
        hmonnx_model.enable_auto_offload = True
    processor = hmonnx_model.get_tf_processor()
    tokenizer = processor.tokenizer
    inputs = processor.process(prompt, args.image_path, device=device)
    with torch.no_grad(), LLMInferenceContextManager(hmonnx_model):
        output = hmonnx_model.generate(
            input_ids=inputs["input_ids"],
            images_ori=inputs["images_ori"],
            images_crop=inputs.get("images_crop"),
            images_seq_mask=inputs["images_seq_mask"],
            images_spatial_crop=inputs["images_spatial_crop"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    return tokenizer, inputs["input_ids"], output.sequences, output.scores


def main(args):
    xhquant_init(None, args.debug)
    args.hmonnx_config = resolve_hmonnx_config(args.hmonnx_config)
    prompt = args.prompt or PROMPTS[args.prompt_mode]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    print(f"hmonnx_config={args.hmonnx_config}")
    print(f"image={args.image_path}")
    print(f"prompt={prompt!r}")

    h_tokenizer, h_input_ids, h_sequences, h_scores = run_hmonnx(args, prompt, device)
    hf_processor = XHUnlimitedOCRProcessor(
        h_tokenizer,
        image_token_id=args.image_token_id,
        image_size=args.image_size,
        base_size=args.image_size,
        patch_size=args.patch_size,
        downsample_ratio=args.downsample_ratio,
        crop_mode=False,
    )
    hf_input_ids, hf_sequences, hf_scores = run_hf(args, h_tokenizer, hf_processor, prompt, device)

    print("\n[input]")
    print(f"hf input shape={tuple(hf_input_ids.shape)}")
    print(f"hmonnx input shape={tuple(h_input_ids.shape)}")
    print(f"input ids equal={bool(torch.equal(hf_input_ids.cpu(), h_input_ids.cpu()))}")

    hf_tokens = print_sequence("HF", h_tokenizer, hf_input_ids.shape[1], hf_sequences, hf_scores)
    h_tokens = print_sequence("HMONNX", h_tokenizer, h_input_ids.shape[1], h_sequences, h_scores)

    first_diff = None
    for index, (hf_token, h_token) in enumerate(zip(hf_tokens, h_tokens, strict=False)):
        if hf_token != h_token:
            first_diff = index
            break
    print("\n[comparison]")
    if first_diff is None:
        print(f"first {min(len(hf_tokens), len(h_tokens))} token(s) match")
    else:
        print(f"first diff at generated token {first_diff}: HF={hf_tokens[first_diff]} HMONNX={h_tokens[first_diff]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", type=str, default="data/models/Unlimited-OCR")
    parser.add_argument(
        "--hmonnx-config",
        type=str,
        default="",
    )
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--prompt-mode", choices=sorted(PROMPTS), default="free-ocr")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--image-token-id", type=int, default=128815)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--downsample-ratio", type=int, default=4)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--auto-offload", action="store_true")
    parser.add_argument("--hf-embedding-probe", action="store_true")
    parser.add_argument("--hf-visual-probe", action="store_true")
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())