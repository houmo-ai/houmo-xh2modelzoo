"""Layerwise wrap-vs-native drift diagnosis for Unlimited-OCR LLM.

Runs the wrap LLM and the native HF ``DeepseekV2Model`` on **the same**
``inputs_embeds`` (built from the wrap data pre-processor with HF visual
image embeddings scattered at ``<image>`` positions) and compares hidden
states layer-by-layer. Locates the first layer where wrap starts drifting
from native, which is the root-cause suspect when wrap output drifts from HF.

Usage:
    python examples_merak/llm/unlimited_ocr/debug_scripts/unlimited_ocr_llm_layerwise_diff.py \
        --config configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_llm_base_xh2a_32k.py \
        --image-path data/images/unlimited_ocr_pdf_page1.png
"""

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


def _max_mean_diff(a: torch.Tensor, b: torch.Tensor):
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


def main(args):
    cfg_name = Path(args.config).stem
    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "llm_layerwise_diff.log"), args.debug)
    set_random_seed(1024)
    logger = get_xhquant_logger()

    cfg = Config.fromfile(args.config)
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    hf_model_dir = args.hf_model or model_cfg.hf_model

    xh_model: "XHUnlimitedOCRModel" = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.set_state(LLMModelState.WRAP)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)
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

    # 1) Native HF visual embeds -> shared image_embeds
    hf_native = UnlimitedOCRForCausalLM.from_pretrained(hf_model_dir, dtype=dtype, trust_remote_code=False)
    hf_visual = UnlimitedOCRBaseVisualModel(unlimited_ocr_patch(hf_native)).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        image_embeds = hf_visual(images_ori)
    image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
    if image_embeds.shape[0] != n_image_tokens:
        raise RuntimeError(f"image tokens ({n_image_tokens}) != image_embeds rows ({image_embeds.shape[0]})")

    # 2) Wrap forward with layer hooks capturing per-layer output
    contexts = [TimeProfiler("llm_layerwise", logger), LLMInferenceContextManager(xh_model), torch.no_grad()]
    with ContextManagers(contexts):
        xh_model.to(device=device, dtype=dtype)
        xh_model.set_input_sequence_length(max(seq_length, xh_model.wrap_cfg.input_sequence_length))

        wrap_layer_out: List[torch.Tensor] = []
        wrap_layer_in: List[torch.Tensor] = []
        wrap_sub_out = {}  # keys: "input_ln", "self_attn", "post_attn_ln", "mlp"

        # Locate wrap decoder layers. XH model exposes `_wrap_model` which is
        # the wrap DeepseekV2Model (already .model, not the ForCausalLM).
        wrap_root = xh_model._wrap_model
        if hasattr(wrap_root, "layers"):
            wrap_layers = wrap_root.layers
        elif hasattr(wrap_root, "model") and hasattr(wrap_root.model, "layers"):
            wrap_layers = wrap_root.model.layers
        else:
            raise RuntimeError(f"Cannot locate wrap decoder layers on {type(wrap_root).__name__}")
        logger.info(f"wrap num_layers: {len(wrap_layers)}  wrap_root type: {type(wrap_root).__name__}")

        def _make_out_hook():
            def _hook(module, args, kwargs, output):
                out = output[0] if isinstance(output, tuple) else output
                wrap_layer_out.append(out.detach().float().cpu())
            return _hook

        def _make_in_hook():
            def _hook(module, args, kwargs):
                hs = args[0] if args else kwargs.get("hidden_states")
                wrap_layer_in.append(hs.detach().float().cpu())
            return _hook

        def _make_sub_out_hook(name, store):
            def _hook(module, args, kwargs, output):
                out = output[0] if isinstance(output, tuple) else output
                store[name] = out.detach().float().cpu()
            return _hook

        handles = []
        for i, layer in enumerate(wrap_layers):
            handles.append(layer.register_forward_pre_hook(_make_in_hook(), with_kwargs=True))
            handles.append(layer.register_forward_hook(_make_out_hook(), with_kwargs=True))

        # Layer 0 sub-module hooks (wrap side)
        wrap_layer0 = wrap_layers[0]
        handles.append(wrap_layer0.input_layernorm.register_forward_hook(
            _make_sub_out_hook("input_ln", wrap_sub_out), with_kwargs=True))
        handles.append(wrap_layer0.self_attn.register_forward_hook(
            _make_sub_out_hook("self_attn", wrap_sub_out), with_kwargs=True))
        handles.append(wrap_layer0.post_attention_layernorm.register_forward_hook(
            _make_sub_out_hook("post_attn_ln", wrap_sub_out), with_kwargs=True))
        handles.append(wrap_layer0.mlp.register_forward_hook(
            _make_sub_out_hook("mlp", wrap_sub_out), with_kwargs=True))

        data_processor = xh_model.get_data_preprocessor()
        processed = data_processor(
            {
                "input_ids": input_ids,
                "image_embeds": image_embeds,
                "images_seq_mask": images_seq_mask,
                "past_seq_length": 0,
            }
        )
        wrap_logits = xh_model(*processed)
        if isinstance(wrap_logits, (tuple, list)):
            wrap_logits = wrap_logits[0]
        for h in handles:
            h.remove()

    # 3) Rebuild the same inputs_embeds and run native DeepseekV2Model
    hf_native = hf_native.to(device=device, dtype=dtype).eval()
    embed_tokens = hf_native.get_input_embeddings()
    inputs_embeds = embed_tokens(input_ids.to(device))
    image_mask = images_seq_mask.to(device).unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(dtype))

    # Native Layer 0 sub-module hooks
    native_sub_out = {}
    native_layer0 = hf_native.model.layers[0]
    nh = []
    nh.append(native_layer0.input_layernorm.register_forward_hook(
        _make_sub_out_hook("input_ln", native_sub_out), with_kwargs=True))
    nh.append(native_layer0.self_attn.register_forward_hook(
        _make_sub_out_hook("self_attn", native_sub_out), with_kwargs=True))
    nh.append(native_layer0.post_attention_layernorm.register_forward_hook(
        _make_sub_out_hook("post_attn_ln", native_sub_out), with_kwargs=True))
    nh.append(native_layer0.mlp.register_forward_hook(
        _make_sub_out_hook("mlp", native_sub_out), with_kwargs=True))
    with torch.no_grad():
        native_out = hf_native.model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    for h in nh:
        h.remove()
    # tuple length N+1: [inputs_embeds, layer_0_out, layer_1_out, ..., layer_{N-1}_out]
    native_hidden = [h.detach().float().cpu() for h in native_out.hidden_states]
    native_final = native_out.last_hidden_state.detach().float().cpu()
    with torch.no_grad():
        native_logits = hf_native.lm_head(native_out.last_hidden_state).detach().float().cpu()

    # 4) Compare per layer
    logger.info(f"native hidden_states count: {len(native_hidden)}  wrap captured out: {len(wrap_layer_out)}")
    # Compare inputs_embeds vs wrap layer0 input
    if len(wrap_layer_in) > 0:
        mx, mn = _max_mean_diff(native_hidden[0], wrap_layer_in[0])
        logger.info(f"[embeds ]  max={mx:.4e}  mean={mn:.4e}")

    n_layers = min(len(wrap_layer_out), len(native_hidden) - 1)
    # NOTE: native DeepseekV2Model appends final-normed hidden as the last entry
    # of all_hidden_states. So native_hidden[i+1] for i in [0..N-2] is the raw
    # layer output, but native_hidden[N] is norm(layer_{N-1}_output). To keep
    # the per-layer comparison apples-to-apples, we only iterate up to the
    # second-to-last layer for raw output diff and handle the last layer via
    # post-final-norm alignment below.
    first_bad = None
    for i in range(n_layers - 1):
        mx, mn = _max_mean_diff(native_hidden[i + 1], wrap_layer_out[i])
        marker = ""
        if mx > 1e-1 and first_bad is None:
            first_bad = i
            marker = "  <-- first > 1e-1"
        logger.info(f"[layer {i:>2}]  max={mx:.4e}  mean={mn:.4e}{marker}")

    # 4b) Last layer: apply native final norm to wrap output and compare with
    #     native's normed last hidden state.
    last_idx = n_layers - 1
    if last_idx >= 0:
        w_last_raw = wrap_layer_out[last_idx]
        native_last_normed = native_hidden[last_idx + 1]  # = norm(layer_last_output)
        native_norm = hf_native.model.norm
        with torch.no_grad():
            w_last_normed = native_norm(w_last_raw.to(device=device, dtype=dtype)).float().cpu()
        mx, mn = _max_mean_diff(native_last_normed, w_last_normed)
        marker = "  <-- first > 1e-1" if (mx > 1e-1 and first_bad is None) else ""
        if marker:
            first_bad = last_idx
        logger.info(f"[layer {last_idx:>2}* normed]  max={mx:.4e}  mean={mn:.4e}{marker}   (compared after final RMSNorm)")

        # Also show raw layer_last outputs stats (native_final IS the normed last hidden state)
        logger.info(f"[layer {last_idx:>2}  raw stats] wrap range=[{w_last_raw.min():+.3e},{w_last_raw.max():+.3e}]  wrap std={w_last_raw.float().std().item():.3e}")

    # 5) Compare final norm output and logits
    # wrap final logits shape: (1, num_logits_to_keep, vocab). Match native last-token logits.
    wrap_last = wrap_logits[0, -1].detach().float().cpu()
    native_last = native_logits[0, -1]
    mx, mn = _max_mean_diff(native_last, wrap_last)
    logger.info(f"[logits (last tok)]  max={mx:.4e}  mean={mn:.4e}")
    logger.info(f"wrap argmax: {int(wrap_last.argmax())}   native argmax: {int(native_last.argmax())}")

    if first_bad is not None:
        logger.info(f"FIRST DRIFT LAYER: {first_bad} (max diff > 1e-1)")
    else:
        logger.info("No layer exceeded 1e-1 threshold; drift may accumulate slowly.")

    # 6) Layer 0 sub-module diagnosis
    logger.info("=" * 60)
    logger.info("Layer 0 sub-module diff (wrap vs native, same inputs_embeds):")
    for name in ("input_ln", "self_attn", "post_attn_ln", "mlp"):
        if name in wrap_sub_out and name in native_sub_out:
            w = wrap_sub_out[name]
            n = native_sub_out[name]
            if w.shape != n.shape:
                logger.info(f"[L0 {name:12s}] shape mismatch wrap={tuple(w.shape)} native={tuple(n.shape)}")
                continue
            mx, mn = _max_mean_diff(n, w)
            logger.info(f"[L0 {name:12s}] max={mx:.4e}  mean={mn:.4e}  wrap_range=[{w.min():+.3e},{w.max():+.3e}]  native_range=[{n.min():+.3e},{n.max():+.3e}]")
        else:
            logger.info(f"[L0 {name:12s}] MISSING wrap={name in wrap_sub_out} native={name in native_sub_out}")

    # 7) Attention internals: independently compute q/k/v/rope/softmax/o with
    #    identical hidden_states input (native L0 input_ln output) using both
    #    wrap and native attention weights. This pinpoints which internal op diverges.
    logger.info("=" * 60)
    logger.info("Layer 0 self_attn internals (independent recompute):")
    try:
        import math
        from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_deepseekv2 import (
            _llama_apply_rotary_pos_emb, _llama_repeat_kv,
        )

        wrap_attn = wrap_layer0.self_attn
        native_attn = native_layer0.self_attn

        # Shared input = LN output (already aligned)
        h = wrap_sub_out["input_ln"].to(device=device, dtype=dtype)
        bsz, q_len, _ = h.shape
        num_heads = native_attn.config.num_attention_heads
        num_kv_heads = native_attn.config.num_key_value_heads
        head_dim = native_attn.head_dim
        num_kv_groups = num_heads // num_kv_heads

        # -- Q/K/V projections (both sides use identical weights) --
        with torch.no_grad():
            wq = wrap_attn.q_proj(h)
            wk = wrap_attn.k_proj(h)
            wv = wrap_attn.v_proj(h)
            nq = native_attn.q_proj(h)
            nk = native_attn.k_proj(h)
            nv = native_attn.v_proj(h)
        logger.info(f"[L0 q_proj      ] max={(wq.float()-nq.float()).abs().max().item():.4e}")
        logger.info(f"[L0 k_proj      ] max={(wk.float()-nk.float()).abs().max().item():.4e}")
        logger.info(f"[L0 v_proj      ] max={(wv.float()-nv.float()).abs().max().item():.4e}")

        # -- RoPE: native reshape then RoPE on (b, H, q_len, D) --
        with torch.no_grad():
            nq_ = nq.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            nk_ = nk.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            nv_ = nv.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            pos_ids = torch.arange(q_len, device=device).unsqueeze(0)
            cos_n, sin_n = native_attn.rotary_emb(nv_, pos_ids)
            nq_rope, nk_rope = _llama_apply_rotary_pos_emb(nq_, nk_, cos_n, sin_n)

            # wrap: reshape without transpose, apply RoPE on (b, q_len, H, D), then transpose
            wq_ = wq.view(bsz, q_len, num_heads, head_dim)
            wk_ = wk.view(bsz, q_len, num_kv_heads, head_dim)
            wv_ = wv.view(bsz, q_len, num_kv_heads, head_dim)
            # wrap position_embeddings comes from PDeepseekV2Model; rebuild same cos/sin
            # match wrap's precomputed cos/sin passed into layer: use native rotary_emb output shape (b,q_len,D)
            # native cos_n shape: (b, q_len, head_dim). Need to broadcast to (b, q_len, 1, D) for wrap's (b,q,H,D) input.
            cos_w = cos_n.unsqueeze(2)  # (b, q_len, 1, D)
            sin_w = sin_n.unsqueeze(2)
            # wrap RoPE: uses xhnn.Rope which does q*cos + rotate_half(q)*sin. Reproduce manually:
            def _rotate_half(x):
                d = x.shape[-1] // 2
                return torch.cat((-x[..., d:], x[..., :d]), dim=-1)
            wq_rope_pre = wq_ * cos_w + _rotate_half(wq_) * sin_w
            wk_rope_pre = wk_ * cos_w + _rotate_half(wk_) * sin_w
            wq_rope = wq_rope_pre.transpose(1, 2)
            wk_rope = wk_rope_pre.transpose(1, 2)

        logger.info(f"[L0 rope(q)     ] max={(wq_rope.float()-nq_rope.float()).abs().max().item():.4e}")
        logger.info(f"[L0 rope(k)     ] max={(wk_rope.float()-nk_rope.float()).abs().max().item():.4e}")

        # -- KV repeat --
        with torch.no_grad():
            nk_full = _llama_repeat_kv(nk_rope, num_kv_groups)
            nv_full = _llama_repeat_kv(nv_, num_kv_groups)

        # -- Attention weights (native path) --
        with torch.no_grad():
            # Causal mask for prefill
            causal = torch.triu(torch.full((q_len, q_len), float('-inf'), device=device, dtype=torch.float32), diagonal=1)
            attn_n = torch.matmul(nq_rope.float(), nk_full.float().transpose(2, 3)) / math.sqrt(head_dim)
            attn_n = attn_n + causal[None, None, :, :]
            attn_n_sm = torch.nn.functional.softmax(attn_n, dim=-1, dtype=torch.float32).to(dtype)
            out_n = torch.matmul(attn_n_sm, nv_full).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            out_n_final = native_attn.o_proj(out_n)

            # Wrap path: repeat_interleave K.transpose(2,3), scale pre-mul, then softmax via masked_softmax
            wk_ri = torch.repeat_interleave(nk_rope.transpose(2, 3), num_kv_groups, dim=1)  # use nk_rope (rope aligned)
            wv_ri = torch.repeat_interleave(nv_, num_kv_groups, dim=1)
            kv_scale = 1.0 / math.sqrt(head_dim)
            # Simulate wrap: use nq_rope but with pre-multiplied scale in fp16, then softmax in fp16 without fp32 upcast
            attn_w = torch.matmul(nq_rope * kv_scale, wk_ri)   # fp16 matmul, scale pre-applied
            attn_w = attn_w + causal[None, None, :, :].to(dtype)
            attn_w_sm_fp16 = torch.nn.functional.softmax(attn_w, dim=-1)  # fp16 softmax (no fp32 upcast)
            out_w_fp16sm = torch.matmul(attn_w_sm_fp16, wv_ri).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            out_w_fp16sm_final = native_attn.o_proj(out_w_fp16sm)

            # Same but with fp32 softmax to isolate softmax-precision effect
            attn_w_sm_fp32 = torch.nn.functional.softmax(attn_w.float(), dim=-1, dtype=torch.float32).to(dtype)
            out_w_fp32sm = torch.matmul(attn_w_sm_fp32, wv_ri).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            out_w_fp32sm_final = native_attn.o_proj(out_w_fp32sm)

        logger.info(f"[L0 attn_w(fp16sm)] max={(out_w_fp16sm_final.float()-out_n_final.float()).abs().max().item():.4e}")
        logger.info(f"[L0 attn_w(fp32sm)] max={(out_w_fp32sm_final.float()-out_n_final.float()).abs().max().item():.4e}")
        logger.info(f"native o_proj out range=[{out_n_final.min():+.3e},{out_n_final.max():+.3e}]")

        # -- Try wrap's actual masked_softmax (with sliding_window) --
        with torch.no_grad():
            past_seq_length_zero = torch.tensor([0], device=device, dtype=torch.int32)
            attn_pre = torch.matmul(nq_rope * kv_scale, wk_ri)  # fp16 raw scores, no explicit causal mask
            # Wrap forward feeds attention_mask separately; try both with & without it
            try:
                attn_sm_ms_nomask = wrap_attn.masked_softmax(attn_pre, past_seq_length_zero)
                out_ms_nomask = torch.matmul(attn_sm_ms_nomask, wv_ri).transpose(1,2).contiguous().reshape(bsz, q_len, -1)
                out_ms_nomask = native_attn.o_proj(out_ms_nomask)
                logger.info(f"[L0 wrap_masked_sm(nomask)] max={(out_ms_nomask.float()-out_n_final.float()).abs().max().item():.4e}  slot={wrap_attn.masked_softmax.attention_max_length}")
            except Exception as e:
                logger.info(f"masked_softmax(nomask) failed: {e!r}")
            # With explicit causal mask added first (as wrap forward does when attention_mask is not None)
            try:
                attn_pre_masked = attn_pre + causal[None, None, :, :].to(dtype)
                attn_sm_ms_mask = wrap_attn.masked_softmax(attn_pre_masked, past_seq_length_zero)
                out_ms_mask = torch.matmul(attn_sm_ms_mask, wv_ri).transpose(1,2).contiguous().reshape(bsz, q_len, -1)
                out_ms_mask = native_attn.o_proj(out_ms_mask)
                logger.info(f"[L0 wrap_masked_sm(w/mask)] max={(out_ms_mask.float()-out_n_final.float()).abs().max().item():.4e}")
            except Exception as e:
                logger.info(f"masked_softmax(w/mask) failed: {e!r}")
    except Exception as e:
        logger.info(f"attention internals probe failed: {e!r}")


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
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
