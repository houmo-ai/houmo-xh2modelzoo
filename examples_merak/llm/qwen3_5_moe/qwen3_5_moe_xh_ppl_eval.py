import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


def _build_hf_compatible_model(hmonnx_model, device: str):
    llm_model_cls = hmonnx_model.LLM_MODEL_CLS
    hf_model = llm_model_cls._get_hf_model_for_compatible(hmonnx_model.hf_model_dir)
    compatible_model = llm_model_cls.build_hf_compatible_model(hf_model, hmonnx_model)
    compatible_model.to(device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    compatible_model.eval()
    hmonnx_model.hf_compatible_model = compatible_model
    return compatible_model


def _load_eval_text(logger) -> str:
    logger.info("Loading wikitext-2 test split...")
    try:
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(dataset["text"])
    except Exception as exc:  # pragma: no cover
        logger.warning(f"datasets unavailable ({exc}); falling back to synthetic smoke text")
        return "The quick brown fox jumps over the lazy dog. " * 500


def evaluate_ppl(hmonnx_model, hf_model, input_ids: torch.Tensor, context_length: int, stride: int, device: str) -> float:
    total_nll = 0.0
    total_tokens = 0
    seq_len = input_ids.shape[1]
    start_pos = max(1, context_length - 1)

    for target_pos in tqdm(range(start_pos, seq_len, stride), desc="PPL eval"):
        ctx_start = max(0, target_pos - (context_length - 1))
        context = input_ids[:, ctx_start:target_pos].to(device)
        target = input_ids[:, target_pos].to(device)

        with torch.no_grad():
            if hasattr(hmonnx_model, "prefill_only"):
                logits = hmonnx_model.prefill_only(context)
                if logits is None:
                    raise RuntimeError("prefill_only returned None")
                logits = logits[:, -1, :]
            else:
                outputs = hf_model(input_ids=context, use_cache=False)
                logits = outputs.logits[:, -1, :]

        log_probs = F.log_softmax(logits.float(), dim=-1)
        nll = F.nll_loss(log_probs, target, reduction="sum")
        total_nll += nll.item()
        total_tokens += 1

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def main(args):
    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()

    hmonnx_model = AutoLLMHONNXModel.from_pretrained(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hmonnx_model.to(device)
    if args.fast and hasattr(hmonnx_model, "to_fast"):
        hmonnx_model.to_fast()

    compatible_model = _build_hf_compatible_model(hmonnx_model, device)
    tokenizer = hmonnx_model.get_tokenizer()

    text = _load_eval_text(logger)
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    if args.max_samples is not None:
        input_ids = input_ids[:, : args.max_samples + 1]

    prefill_chunk_length = getattr(hmonnx_model.meta_info.model_config, "prefill_chunk_length", None)
    meta_context_length = getattr(hmonnx_model.meta_info.model_config, "context_max_length", None)
    context_length = args.context_length or prefill_chunk_length or meta_context_length or 2048
    stride = args.stride if args.stride > 0 else max(1, context_length // 2)

    contexts = [
        TimeProfiler("ppl_eval", logger),
        LLMInferenceContextManager(hmonnx_model, devices=[device]),
    ]
    if device == "cuda":
        contexts.insert(1, MemoryTracker(device=device, name="ppl_eval", logger=logger))

    with ContextManagers(contexts):
        ppl = evaluate_ppl(hmonnx_model, compatible_model, input_ids, context_length, stride, device)

    result = {
        "meta_path": str(Path(args.config).resolve()),
        "ppl_wikitext2": ppl,
        "context_length": context_length,
        "stride": stride,
        "max_samples": args.max_samples,
        "model_type": type(hmonnx_model).__name__,
        "hf_model_dir": hmonnx_model.hf_model_dir,
    }
    out_json = Path(args.config).resolve().parent / "ppl_wikitext2.json"
    out_json.write_text(json.dumps(result, indent=2))
    logger.info(f"PPL (wikitext-2): {ppl:.4f}")
    logger.info(f"PPL result saved to: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to exported golden_meta_info.json or meta file")
    parser.add_argument("--context-length", type=int, default=0, help="Override evaluation context length")
    parser.add_argument("--stride", type=int, default=512, help="Sliding-window stride (0=context_length//2)")
    parser.add_argument("--max-samples", type=int, default=4096, help="Max token count to evaluate")
    parser.add_argument("--fast", action="store_true", help="Run HMONNX model in fast mode when supported")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    main(args)
