"""Debug: text wrap → frontend conversion + precision test.

Usage:
    CUDA_VISIBLE_DEVICES=7 conda run -n gemma4 bash -c \
      'PYTHONPATH=/data01/home/yujy/work/xh2modelzoo:$PYTHONPATH python \
       examples_merak/llm/gemma4/debug_scripts/debug_gemma4_text_frontend.py'
"""
import torch
from pathlib import Path
from transformers import AutoModelForImageTextToText

from xhquant.api import Config, xhquant_init, get_xhquant_logger, set_random_seed
from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel

logger = get_xhquant_logger()


def main():
    device = "cuda:0"
    set_random_seed(1024)
    work_dir = Path("work_dirs/gemma4_text_frontend_debug")
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "text_frontend.log"), debug=True)

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

    # === Step 1: wrap reference output ===
    xh_model.get_kvcache_mixin().prepare_kv_cache(dtype=torch.bfloat16)
    data = {
        "input_ids": torch.tensor([[2, 106, 1645, 108, 7725, 2134, 107, 108, 106, 2516, 108]], dtype=torch.long),
        "past_seq_length": 0,
    }
    data_processor = xh_model.get_data_preprocessor()
    wrap_inputs = data_processor(data)
    with torch.no_grad():
        wrap_logits = xh_model.wrap_model(*wrap_inputs)
    seq_len = data["input_ids"].shape[1]
    wrap_last = wrap_logits[:, seq_len - 1, :].detach().cpu().float()
    logger.info(f"[wrap] logits shape: {tuple(wrap_logits.shape)}, last-token sample: {wrap_last[0,:5]}")
    del wrap_logits
    torch.cuda.empty_cache()

    # === Step 2: frontend conversion ===
    logger.info("Converting wrap → frontend (TorchFX)...")
    try:
        fe_model = xh_model._to_fronted(xh_model._wrap_model)
        logger.info(f"Frontend conversion OK. Type: {type(fe_model).__name__}")
    except Exception as e:
        logger.error(f"Frontend conversion FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # === Step 3: run frontend model ===
    xh_model.get_kvcache_mixin().prepare_kv_cache(dtype=torch.bfloat16)
    fe_inputs = data_processor(data)
    # Frontend model needs flattened inputs (FX tracing flattens list args into positional args)
    from xhmodel_merak.xh_llm.utils import unfold_args
    fe_inputs_flat = unfold_args(fe_inputs)
    fe_model.to(device=device)
    with torch.no_grad():
        fe_logits = fe_model(*fe_inputs_flat)
    fe_last = fe_logits[:, seq_len - 1, :].detach().cpu().float()
    logger.info(f"[frontend] logits shape: {tuple(fe_logits.shape)}, last-token sample: {fe_last[0,:5]}")

    # === Step 4: compare ===
    cosine = torch.nn.functional.cosine_similarity(wrap_last.flatten(), fe_last.flatten(), dim=0).item()
    max_diff = (wrap_last - fe_last).abs().max().item()
    mean_diff = (wrap_last - fe_last).abs().mean().item()
    logger.info(f"cosine similarity: {cosine:.6f}")
    logger.info(f"max abs diff: {max_diff:.6e}, mean abs diff: {mean_diff:.6e}")
    if cosine >= 0.999:
        logger.info("✅ PASS: wrap → frontend text precision OK")
    elif cosine >= 0.99:
        logger.info("⚠️  MARGINAL: cosine < 0.999 but >= 0.99 (bf16→fp32 gap likely)")
    else:
        logger.info("❌ FAIL: cosine too low")


if __name__ == "__main__":
    main()
