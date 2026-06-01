"""Debug: text frontend decode verification (prefill + decode path via ModelSwitcher).

Tests that the frontend model (with ModelSwitcher for prefill/decode) produces
matching logits vs the wrap model for both prefill and single-token decode.

Usage:
    CUDA_VISIBLE_DEVICES=7 conda run -n gemma4 bash -c \
      'PYTHONPATH=/data01/home/yujy/work/xh2modelzoo:$PYTHONPATH python \
       examples_merak/llm/gemma4/debug_scripts/debug_gemma4_text_frontend_decode.py'
"""
import torch
from pathlib import Path
from transformers import AutoModelForImageTextToText

from xhquant.api import Config, xhquant_init, get_xhquant_logger, set_random_seed
from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
from xhmodel_merak.xh_llm.utils import unfold_args, is_graph_module

logger = get_xhquant_logger()


def main():
    device = "cuda:0"
    set_random_seed(1024)
    work_dir = Path("work_dirs/gemma4_text_frontend_decode_debug")
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "text_frontend_decode.log"), debug=True)

    cfg = Config.fromfile("configs_merak/xh2a/llm_models/gemma4/31b/gemma4_31b_it_xh2a_2k.py")
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    xh_model = AutoLLMModel.from_pretrained(model_cfg)
    xh_model.work_dir = str(work_dir)

    # Load HF model
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_cfg.hf_model, dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True
    ).eval()

    # Init wrap
    xh_model.init_wrap_model(hf_model)
    xh_model._device = device
    xh_model._dtype = torch.bfloat16
    from xhmodel_merak.xh_llm.types import LLMModelState
    xh_model._state = LLMModelState.WRAP

    input_ids = torch.tensor([[2, 106, 1645, 108, 7725, 2134, 107, 108, 106, 2516, 108]], dtype=torch.long)
    seq_len = input_ids.shape[1]

    # === Step 1: wrap prefill reference ===
    logger.info("=== Wrap prefill reference ===")
    xh_model.get_kvcache_mixin().prepare_kv_cache(dtype=torch.bfloat16)
    data_processor = xh_model.get_data_preprocessor()
    prefill_data = {"input_ids": input_ids, "past_seq_length": 0}
    wrap_prefill_inputs = data_processor(prefill_data)
    with torch.no_grad():
        wrap_prefill_logits = xh_model.wrap_model(*wrap_prefill_inputs)
    wrap_prefill_last = wrap_prefill_logits[:, seq_len - 1, :].detach().cpu().float()
    next_token = wrap_prefill_logits[:, seq_len - 1, :].argmax(dim=-1)
    logger.info(f"[wrap prefill] logits shape: {tuple(wrap_prefill_logits.shape)}, next_token: {next_token.item()}")

    # === Step 2: wrap decode reference ===
    logger.info("=== Wrap decode reference ===")
    # Manually set decode mode (input_sequence_length=1) without calling set_decode()
    data_processor.input_sequence_length = 1
    xh_model.wrap_cfg.input_sequence_length = 1
    decode_data = {"input_ids": next_token.unsqueeze(0), "past_seq_length": seq_len}
    wrap_decode_inputs = data_processor(decode_data)
    with torch.no_grad():
        wrap_decode_logits = xh_model.wrap_model(*wrap_decode_inputs)
    wrap_decode_last = wrap_decode_logits[:, 0, :].detach().cpu().float()
    decode_next_token = wrap_decode_logits[:, 0, :].argmax(dim=-1)
    logger.info(f"[wrap decode] logits shape: {tuple(wrap_decode_logits.shape)}, next_token: {decode_next_token.item()}")
    # Restore prefill mode
    data_processor.input_sequence_length = xh_model.config.prefill_chunk_length
    xh_model.wrap_cfg.input_sequence_length = xh_model.config.prefill_chunk_length
    del wrap_prefill_logits, wrap_decode_logits
    torch.cuda.empty_cache()

    # === Step 3: frontend conversion (prefill + decode via ModelSwitcher) ===
    logger.info("=== Converting wrap → frontend (ModelSwitcher: prefill + decode) ===")
    xh_model.get_kvcache_mixin().prepare_kv_cache(dtype=torch.bfloat16)
    try:
        fe_model = xh_model._to_fronted(xh_model._wrap_model)
        logger.info(f"Frontend conversion OK. Type: {type(fe_model).__name__}")
    except Exception as e:
        logger.error(f"Frontend conversion FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # === Step 4: frontend prefill ===
    logger.info("=== Frontend prefill ===")
    xh_model.set_prefill()
    xh_model.get_kvcache_mixin().prepare_kv_cache(dtype=torch.bfloat16)
    data_processor = xh_model.get_data_preprocessor()
    fe_prefill_inputs = data_processor(prefill_data)
    prefill_fe = fe_model.prefill
    prefill_fe.to(device=device)
    fe_prefill_flat = unfold_args(fe_prefill_inputs)
    with torch.no_grad():
        fe_prefill_logits = prefill_fe(*fe_prefill_flat)
    if isinstance(fe_prefill_logits, (tuple, list)) and len(fe_prefill_logits) == 1:
        fe_prefill_logits = fe_prefill_logits[0]
    fe_prefill_last = fe_prefill_logits[:, seq_len - 1, :].detach().cpu().float()
    fe_next_token = fe_prefill_logits[:, seq_len - 1, :].argmax(dim=-1)
    logger.info(f"[frontend prefill] logits shape: {tuple(fe_prefill_logits.shape)}, next_token: {fe_next_token.item()}")

    cosine_prefill = torch.nn.functional.cosine_similarity(
        wrap_prefill_last.flatten(), fe_prefill_last.flatten(), dim=0
    ).item()
    logger.info(f"[prefill] cosine: {cosine_prefill:.6f}")

    # === Step 5: frontend decode ===
    logger.info("=== Frontend decode ===")
    xh_model.set_decode()
    data_processor = xh_model.get_data_preprocessor()
    fe_decode_inputs = data_processor(decode_data)
    decode_fe = fe_model.decode
    decode_fe.to(device=device)
    fe_decode_flat = unfold_args(fe_decode_inputs)
    with torch.no_grad():
        fe_decode_logits = decode_fe(*fe_decode_flat)
    if isinstance(fe_decode_logits, (tuple, list)) and len(fe_decode_logits) == 1:
        fe_decode_logits = fe_decode_logits[0]
    fe_decode_last = fe_decode_logits[:, 0, :].detach().cpu().float()
    fe_decode_next = fe_decode_logits[:, 0, :].argmax(dim=-1)
    logger.info(f"[frontend decode] logits shape: {tuple(fe_decode_logits.shape)}, next_token: {fe_decode_next.item()}")

    cosine_decode = torch.nn.functional.cosine_similarity(
        wrap_decode_last.flatten(), fe_decode_last.flatten(), dim=0
    ).item()
    logger.info(f"[decode] cosine: {cosine_decode:.6f}")
    if cosine_decode >= 0.999:
        logger.info("✅ PASS: frontend decode precision OK")
    elif cosine_decode >= 0.99:
        logger.info("⚠️  MARGINAL: decode cosine < 0.999 but >= 0.99")
    else:
        logger.info(f"❌ FAIL: frontend decode cosine {cosine_decode:.6f}")

    # Summary
    logger.info("=" * 60)
    logger.info(f"Prefill cosine: {cosine_prefill:.6f} | Decode cosine: {cosine_decode:.6f}")
    tokens_match = (fe_next_token.item() == next_token.item()) and (fe_decode_next.item() == decode_next_token.item())
    logger.info(f"Token match: prefill={'✓' if fe_next_token.item() == next_token.item() else '✗'} "
                f"decode={'✓' if fe_decode_next.item() == decode_next_token.item() else '✗'}")


if __name__ == "__main__":
    main()
