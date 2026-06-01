"""Vision wrap → frontend precision verification.

Loads the Gemma4 vision model, runs wrap inference and frontend inference on the
same dummy image, then compares the cosine similarity of the two outputs.

Usage:
    CUDA_VISIBLE_DEVICES=6 conda run -n gemma4 bash -c \
        'PYTHONPATH=/data01/home/yujy/work/xh2modelzoo:$PYTHONPATH \
         python examples_merak/llm/gemma4/debug_scripts/debug_gemma4_vision_frontend.py'
"""

import copy

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, LLMModelState
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init

CONFIG = "configs_merak/xh2a/llm_models/gemma4/31b/gemma4_31b_it_xh2a_2k.py"
WORK_DIR = "./work_dirs/gemma4_31b_it_xh2a_2k"


def main():
    import pathlib

    pathlib.Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    xhquant_init(f"{WORK_DIR}/vision_frontend_verify.log", True)
    set_random_seed(1024)
    logger = get_xhquant_logger()

    cfg = Config.fromfile(CONFIG)
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)

    device = "cuda"

    # --- Load HF model (bfloat16 to fit in GPU) and build wrap ---
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_cfg.hf_model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()

    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.visual.init_wrap_model(hf_model)
    xh_model.visual.config.work_dir = WORK_DIR

    from xhmodel_merak.xh_llm.models.gemma4.gemma4_processor import XHGemma4Processor

    processor = XHGemma4Processor.from_pretrained(model_cfg.hf_model)
    inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (224, 224), color="white")},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
    )

    # --- wrap inference (convert to float32 for fair comparison) ---
    wrap_model = xh_model.visual.wrap_model
    # Free non-vision parts: adapter holds refs to vision_tower + embed_vision
    del hf_model
    torch.cuda.empty_cache()
    wrap_model.to(device=device, dtype=torch.float32)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float32)
    image_position_ids = inputs["image_position_ids"].to(device)

    with torch.no_grad():
        wrap_out = wrap_model(pixel_values, image_position_ids)
    logger.info(f"[wrap] output shape: {tuple(wrap_out.shape)}, dtype: {wrap_out.dtype}")
    wrap_out_cpu = wrap_out.detach().cpu().float()

    # Move wrap model to CPU for ONNX export (done in float32 by _to_fronted)
    wrap_model.cpu()
    torch.cuda.empty_cache()

    # --- frontend conversion ---
    logger.info("Converting wrap → frontend (ONNX export + to_frontend_graph)...")
    fe_model = xh_model.visual._to_fronted(wrap_model)
    # Keep frontend in float32 — pooler's one_hot requires integer dtypes
    fe_model.to(device=device, dtype=torch.float32)
    logger.info("Frontend conversion done.")

    # --- frontend inference ---
    with torch.no_grad():
        fe_out = fe_model(
            inputs["pixel_values"].to(device=device, dtype=torch.float32),
            inputs["image_position_ids"].to(device),
        )
    if isinstance(fe_out, (tuple, list)):
        fe_out = fe_out[0]
    logger.info(f"[frontend] output shape: {tuple(fe_out.shape)}, dtype: {fe_out.dtype}")
    fe_out_cpu = fe_out.detach().cpu().float()

    # --- compare ---
    cos = F.cosine_similarity(wrap_out_cpu.flatten(), fe_out_cpu.flatten(), dim=0).item()
    abs_diff = (wrap_out_cpu - fe_out_cpu).abs()
    logger.info(f"cosine similarity: {cos:.6f}")
    logger.info(f"max abs diff: {abs_diff.max().item():.6e}, mean abs diff: {abs_diff.mean().item():.6e}")

    if cos > 0.999:
        logger.info("✅ PASS: wrap → frontend vision precision OK")
    else:
        logger.warning(f"⚠️ FAIL: cosine similarity {cos:.6f} < 0.999")


if __name__ == "__main__":
    main()
