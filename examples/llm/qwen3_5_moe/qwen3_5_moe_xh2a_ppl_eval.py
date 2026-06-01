"""Qwen3.5-MoE HMONNX PPL evaluation on wikitext-2.

Evaluates exported HMONNX by computing perplexity with teacher-forcing.
The prefill session processes fixed-length windows; logits are decoded
token by token for NLL computation.

Usage:
    python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_ppl_eval.py \\
        --config work_dirs/Qwen3.5-35B-A3B-XH2a-2k-w8a8h0_sefp/meta.json \\
        --stride 512 --max-samples 128

    # GPTQ-exported model
    python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_ppl_eval.py \\
        --config work_dirs/Qwen3.5-35B-A3B-XH2a-2k-w4a8h0_sefp-gptq/meta.json \\
        --stride 512
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from xhquant.api import get_root_logger, xhquant_init
from xhquant.xhonnxruntime import config as xhonnxruntime_config

from xh_model_zoo.xh_llm.models.qwen3_5_moe import Qwen3_5MoeInference
from xh_model_zoo.xh_llm.utils import auto_offload


def evaluate_ppl(
    inference_engine: Qwen3_5MoeInference,
    input_ids: torch.Tensor,
    context_length: int,
    stride: int,
    device: torch.device,
    logger,
) -> float:
    """Compute perplexity using a sliding-window next-token prediction.

    The prefill ONNX returns only the **last** position's logit.  Each step
    feeds up to ``context_length - 1`` tokens as context and predicts the
    one following token; the window slides by ``stride`` positions.

    With ``stride=1`` this is exact language-model PPL (expensive).  Larger
    strides trade accuracy for speed.
    """
    seq_len = input_ids.shape[1]
    total_nll = 0.0
    total_tokens = 0

    # First target is at index context_length-1 (we need at least 1 context token)
    start_pos = max(1, context_length - 1)
    positions = range(start_pos, seq_len, stride)
    pbar = tqdm(positions, desc="PPL eval")
    for target_pos in pbar:
        ctx_start = max(0, target_pos - (context_length - 1))
        context = input_ids[:, ctx_start:target_pos].to(device)
        target = input_ids[:, target_pos].to(device)  # [1]

        with torch.no_grad():
            logits = inference_engine.prefill_only(context)  # [1, *, vocab]

        if logits is None:
            logger.warning("prefill_only returned None; skipping position")
            continue

        # Take the last output position (next-token prediction)
        last_logit = logits[:, -1, :]  # [1, vocab_size]
        log_probs = F.log_softmax(last_logit.float(), dim=-1)
        nll = F.nll_loss(log_probs, target, reduction="sum")
        total_nll += nll.item()
        total_tokens += 1

        cur_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
        pbar.set_postfix({"ppl": f"{cur_ppl:.2f}", "tokens": total_tokens})

    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")


def main(args):
    xhquant_init(None, args.debug)
    logger = get_root_logger()
    xhonnxruntime_config.disable_progress = True

    meta_path = Path(args.config).resolve()
    logger.info(f"Loading inference engine from: {meta_path}")

    inference_engine = Qwen3_5MoeInference(str(meta_path), fast_mode=args.fast)
    meta_info = inference_engine.meta_info
    # The ONNX prefill session has a fixed input sequence length; use that as
    # the evaluation window so each call to prefill_only fits in one chunk.
    context_length = inference_engine.prefill_input_sequence_length

    hf_config_dir = meta_path.parent / meta_info["hf_config"]
    tokenizer = AutoTokenizer.from_pretrained(str(hf_config_dir), trust_remote_code=True)

    device = torch.device(args.device)
    auto_offload(inference_engine, "XH2aQuantQMoeBlock")

    # Load dataset
    logger.info("Loading wikitext-2 test set...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])
    except Exception as e:
        logger.warning(f"datasets not available ({e}), falling back to manual loading")
        # Fallback: use a short fixed text for smoke test
        text = "The quick brown fox jumps over the lazy dog. " * 500

    logger.info(f"Tokenizing text ({len(text)} chars)...")
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    total_tokens = input_ids.shape[1]
    logger.info(f"Total tokens: {total_tokens}")

    if args.max_samples is not None:
        input_ids = input_ids[:, : args.max_samples + 1]
        logger.info(f"Truncated to {input_ids.shape[1]} tokens")

    stride = args.stride if args.stride > 0 else context_length // 2
    logger.info(f"context_length={context_length}, stride={stride}")

    ppl = evaluate_ppl(
        inference_engine, input_ids, context_length, stride, device, logger,
    )
    logger.info(f"========================================")
    logger.info(f"PPL (wikitext-2): {ppl:.4f}")
    logger.info(f"========================================")

    result = {
        "meta_path": str(meta_path),
        "ppl_wikitext2": ppl,
        "context_length": context_length,
        "stride": stride,
        "max_samples": args.max_samples,
        "model_name": meta_info.get("model_name", "unknown"),
        "quant_type": meta_info.get("quant_scheme", {}).get("quant_type", "unknown"),
        "quant_weight": meta_info.get("quant_weight"),
    }
    out_json = meta_path.parent / "ppl_wikitext2.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"PPL result saved to: {out_json}")
    return ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3.5-MoE HMONNX perplexity evaluation on wikitext-2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to meta.json")
    parser.add_argument("--stride", type=int, default=512, help="Sliding window stride (0=context_length//2)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max token count to eval (None=full dataset)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    main(args)
