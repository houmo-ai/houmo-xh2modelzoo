"""Verify the sliding-window semantics hypothesis for Unlimited-OCR wrap LLM.

Hypothesis (from official-vs-wrap comparison):
  * Native ``SlidingWindowLlamaAttention`` uses a *ring buffer*: prefill tokens
    (image + prompt) are recorded once and stay fully visible; the sliding
    window (W=128) only recycles decode-generated positions.
  * Wrap ``_SlidingWindowLlamaAttention`` switches ``attention_max_length`` to
    W=128 during decode, which is an *absolute* sliding window over the whole
    KV sequence. Since prefill is already ~278 > 128, the image tokens get
    pushed out of the window during decode -> the model stops "seeing" the
    image and degenerates (repeat / hallucinate short fragments).

This script runs BOTH paths for the *same* number of decode steps (enough to
exceed ``prefill_len + W``) using **native HF visual embeddings** for both, so
visual quant error is isolated out. Both LLMs run **unquantized (wrap FP16)**,
so any divergence is attributable to attention/mask semantics, not w8a8.

It prints, per decode step, the wrap argmax token vs the native HF argmax token
and flags the first divergence. If divergence appears right around when the
cumulative length crosses ``prefill_len + 128``, the hypothesis is confirmed.

Usage:
    python examples_merak/llm/unlimited_ocr/debug_scripts/unlimited_ocr_sliding_window_verify.py \
        --config configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_llm_base_xh2a_32k.py \
        --image-path data/images/unlimited_ocr_pdf_page1.png \
        --generate-tokens 160
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, LLMInferenceContextManager, LLMModelState
from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_unlimitedocr import UnlimitedOCRForCausalLM
from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_unlimitedocr_patch import unlimited_ocr_patch
from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_processor import XHUnlimitedOCRProcessor
from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_visual_model import UnlimitedOCRBaseVisualModel
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, TimeProfiler

if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.unlimited_ocr import XHUnlimitedOCRModel


def _decode(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def run_native_hf(hf_native, inputs_embeds, sliding_window_size, num_tokens, eos_token_id, device):
    """Native HF DeepseekV2 multi-step greedy decode using the ring-buffer attention.

    Mirrors modeling_unlimitedocr.infer(): set config._ring_window, disable
    config.sliding_window (so DynamicCache doesn't truncate prefill), run
    generate-style manual greedy loop through hf_native.model + lm_head.
    """
    from transformers import DynamicCache

    orig_sw = getattr(hf_native.config, "sliding_window", None)
    hf_native.config._ring_window = sliding_window_size
    hf_native.config.sliding_window = None

    tokens: List[int] = []
    past = DynamicCache()
    seq_len = inputs_embeds.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    with torch.no_grad():
        out = hf_native.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        logits = hf_native.lm_head(out.last_hidden_state[:, -1:])
        next_token = int(logits[0, -1].argmax().item())
        tokens.append(next_token)
        cur_len = seq_len
        embed_tokens = hf_native.get_input_embeddings()
        for _ in range(1, num_tokens):
            if next_token == eos_token_id:
                break
            step_embed = embed_tokens(torch.tensor([[next_token]], device=device))
            position_ids = torch.tensor([[cur_len]], device=device)
            out = hf_native.model(
                inputs_embeds=step_embed,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = hf_native.lm_head(out.last_hidden_state[:, -1:])
            next_token = int(logits[0, -1].argmax().item())
            tokens.append(next_token)
            cur_len += 1

    hf_native.config.sliding_window = orig_sw
    return tokens


def _force_full_attention(xh_model, logger):
    """Override every wrap attention module's sliding window to full attention.

    Sets masked_softmax / k_cache / v_cache ``attention_max_length`` to -1 so
    decode attends over the whole KV sequence (matches native ring-buffer
    warmup phase, which is full-causal for the first W decode steps).
    Returns the number of modules touched.
    """
    wrap_root = xh_model._wrap_model
    if hasattr(wrap_root, "layers"):
        layers = wrap_root.layers
    elif hasattr(wrap_root, "model") and hasattr(wrap_root.model, "layers"):
        layers = wrap_root.model.layers
    else:
        raise RuntimeError(f"Cannot locate wrap decoder layers on {type(wrap_root).__name__}")
    touched = 0
    for layer in layers:
        attn = layer.self_attn
        for name in ("masked_softmax", "k_cache", "v_cache"):
            mod = getattr(attn, name, None)
            if mod is not None and hasattr(mod, "attention_max_length"):
                mod.attention_max_length = -1
                touched += 1
    logger.info(f"[force-full-attention] overrode attention_max_length=-1 on {touched} modules")
    return touched


def run_wrap(xh_model, data_processor, input_ids, image_embeds, images_seq_mask,
             seq_length, num_tokens, eos_token_id, device, force_full_attention=False, logger=None):
    """Wrap-state multi-step greedy decode (FP16, unquantized)."""
    tokens: List[int] = []
    processed = data_processor(
        {
            "input_ids": input_ids,
            "image_embeds": image_embeds,
            "images_seq_mask": images_seq_mask,
            "past_seq_length": 0,
        }
    )
    logits = xh_model(*processed)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    last_logits = logits[0, seq_length - 1] if logits.shape[1] >= seq_length else logits[0, -1]
    next_token = int(last_logits.argmax().item())
    tokens.append(next_token)
    for _ in range(1, num_tokens):
        if next_token == eos_token_id:
            break
        xh_model.set_decode()
        xh_model.set_input_sequence_length(1)
        if force_full_attention:
            # set_input_sequence_length(1) re-triggers _update_cfg which sets
            # attention_max_length=128 for decode; override it back to -1.
            _force_full_attention(xh_model, logger)
        decode_processed = data_processor(
            {
                "input_ids": torch.tensor([[tokens[-1]]], dtype=torch.long, device=device),
                "past_seq_length": seq_length + len(tokens) - 1,
            }
        )
        step_logits = xh_model(*decode_processed)
        if isinstance(step_logits, (tuple, list)):
            step_logits = step_logits[0]
        next_token = int(step_logits[0, -1].argmax().item())
        tokens.append(next_token)
    return tokens


def main(args):
    cfg_name = Path(args.config).stem
    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "sliding_window_verify.log"), args.debug)
    set_random_seed(1024)
    logger = get_xhquant_logger()

    cfg = Config.fromfile(args.config)
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    hf_model_dir = args.hf_model or model_cfg.hf_model
    sliding_window_size = int(getattr(model_cfg, "sliding_window_size", 128) or 128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)
    eos_token_id = int(tokenizer.eos_token_id)
    processor = XHUnlimitedOCRProcessor(
        tokenizer,
        image_token_id=model_cfg.image_token_id,
        image_size=model_cfg.visual_config.image_size,
        base_size=model_cfg.visual_config.base_size,
        patch_size=model_cfg.visual_config.patch_size,
        downsample_ratio=model_cfg.visual_config.downsample_ratio,
        crop_mode=False,
    )
    inputs = processor.process(args.prompt, args.image_path, device=device)
    input_ids = inputs["input_ids"]
    images_seq_mask = inputs["images_seq_mask"]
    images_ori = inputs["images_ori"].to(device=device, dtype=dtype)
    seq_length = int(input_ids.shape[1])
    n_image_tokens = int(images_seq_mask.sum().item())
    logger.info(f"input_ids shape: {tuple(input_ids.shape)}  image tokens: {n_image_tokens}")
    logger.info(f"prefill_len={seq_length}  sliding_window(W)={sliding_window_size}  "
                f"decode step where cumulative len crosses prefill+W ~= step {sliding_window_size}")

    # Shared native HF visual embeddings (isolate visual quant error).
    hf_native = UnlimitedOCRForCausalLM.from_pretrained(hf_model_dir, dtype=dtype, trust_remote_code=False)
    hf_visual = UnlimitedOCRBaseVisualModel(unlimited_ocr_patch(hf_native)).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        image_embeds = hf_visual(images_ori)
    image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
    if image_embeds.shape[0] != n_image_tokens:
        raise RuntimeError(f"image tokens ({n_image_tokens}) != image_embeds rows ({image_embeds.shape[0]})")

    # ---- Wrap decode ----
    xh_model: "XHUnlimitedOCRModel" = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.set_state(LLMModelState.WRAP)
    contexts = [TimeProfiler("sw_verify_wrap", logger), LLMInferenceContextManager(xh_model), torch.no_grad()]
    with ContextManagers(contexts):
        xh_model.to(device=device, dtype=dtype)
        xh_model.set_input_sequence_length(max(seq_length, xh_model.wrap_cfg.input_sequence_length))
        data_processor = xh_model.get_data_preprocessor()
        wrap_tokens = run_wrap(
            xh_model, data_processor, input_ids, image_embeds, images_seq_mask,
            seq_length, args.generate_tokens, eos_token_id, device,
            force_full_attention=args.force_full_attention, logger=logger,
        )

    # ---- Native HF decode (ring buffer) ----
    hf_native = hf_native.to(device=device, dtype=dtype).eval()
    embed_tokens = hf_native.get_input_embeddings()
    inputs_embeds = embed_tokens(input_ids.to(device))
    image_mask = images_seq_mask.to(device).unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(dtype))
    native_tokens = run_native_hf(
        hf_native, inputs_embeds, sliding_window_size,
        args.generate_tokens, eos_token_id, device,
    )

    # ---- Compare ----
    logger.info(f"wrap generated {len(wrap_tokens)} tokens; native generated {len(native_tokens)} tokens")
    first_diff = None
    n = min(len(wrap_tokens), len(native_tokens))
    for i in range(n):
        if wrap_tokens[i] != native_tokens[i]:
            first_diff = i
            break
    for i in range(n):
        marker = ""
        if wrap_tokens[i] != native_tokens[i]:
            marker = "  <-- DIFF"
        cum_len = seq_length + i  # KV length seen at decode step i (0-based) query position
        cross = " [past prefill+W]" if cum_len >= seq_length + sliding_window_size else ""
        logger.info(
            f"step {i:03d} (cum_len={cum_len}{cross}): "
            f"wrap={wrap_tokens[i]:>6} {_decode(tokenizer, wrap_tokens[i])!r}  "
            f"native={native_tokens[i]:>6} {_decode(tokenizer, native_tokens[i])!r}{marker}"
        )

    logger.info("=" * 60)
    if first_diff is None:
        logger.info(f"RESULT: wrap and native AGREE on all {n} tokens (no divergence)")
    else:
        cross_step = sliding_window_size
        logger.info(f"RESULT: first divergence at decode step {first_diff} (cum_len={seq_length + first_diff})")
        logger.info(f"        sliding window W={sliding_window_size}; prefill_len={seq_length}")
        if first_diff >= cross_step - 5:
            logger.info(f"        >>> divergence is at/after step ~{cross_step} (W). "
                        f"CONSISTENT with sliding-window semantics bug.")
        else:
            logger.info(f"        >>> divergence BEFORE step {cross_step}. "
                        f"Likely a different root cause (not the W crossover).")

    wrap_text = tokenizer.decode(wrap_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    native_text = tokenizer.decode(native_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    logger.info(f"[wrap  text] {wrap_text!r}")
    logger.info(f"[native text] {native_text!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_llm_base_xh2a_32k.py",
    )
    parser.add_argument("--hf-model", type=str, default="")
    parser.add_argument("--image-path", type=str, default="data/images/unlimited_ocr_pdf_page1.png")
    parser.add_argument("--prompt", type=str, default="<image>\\nFree OCR. ")
    parser.add_argument("--generate-tokens", type=int, default=160)
    parser.add_argument("--force-full-attention", action="store_true",
                        help="Override wrap decode attention_max_length to -1 (full attention) "
                             "to test the sliding-window/reverse-window hypothesis.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
