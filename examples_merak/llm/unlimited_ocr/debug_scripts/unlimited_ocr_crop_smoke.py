"""Unlimited-OCR crop/gundam smoke (eager wrap model).

Validates the crop/gundam path end-to-end without HMONNX: builds crop inputs
(global view + dynamic local crops), runs the eager visual ``forward_crop`` +
LLM wrap prefill, and checks token/feature alignment for small (<=640, no crop)
and large (dynamic crop) images.

HMONNX export of crop mode is intentionally out of scope: the dynamic crop count
produces variable-length visual sequences that break static-shape export.

Usage:
    python examples_merak/llm/unlimited_ocr/debug_scripts/unlimited_ocr_crop_smoke.py \
        --config configs_merak/xh2a/llm_models/unlimited_ocr/gundam/unlimited_ocr_llm_gundam_xh2a_32k.py \
        --image-path data/images/qwen2_vl_demo.jpeg
"""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, LLMInferenceContextManager, LLMModelState
from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_processor import XHUnlimitedOCRProcessor
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, TimeProfiler

if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.unlimited_ocr import XHUnlimitedOCRModel


def main(args):
    cfg_name = Path(args.config).stem
    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "crop_smoke.log"), args.debug)
    set_random_seed(1024)
    logger = get_xhquant_logger()

    cfg = Config.fromfile(args.config)
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    hf_model_dir = args.hf_model or model_cfg.hf_model
    vc = model_cfg.visual_config
    if not vc.crop_mode:
        raise ValueError("This smoke expects a crop/gundam config (crop_mode=True).")

    xh_model: "XHUnlimitedOCRModel" = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.set_state(LLMModelState.from_string(args.eval_type))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)
    processor = XHUnlimitedOCRProcessor(
        tokenizer,
        image_token_id=model_cfg.image_token_id,
        image_size=vc.image_size,
        base_size=vc.base_size,
        patch_size=vc.patch_size,
        downsample_ratio=vc.downsample_ratio,
        crop_mode=True,
        max_crop_num=getattr(vc, "max_crop_num", 32),
    )
    inputs = processor.process(args.prompt, args.image_path, device=device)
    n_tokens = int(inputs["images_seq_mask"].sum().item())
    wc, hc = (int(x) for x in inputs["images_spatial_crop"][0])
    logger.info(
        f"crop_ratio=({wc},{hc}) image tokens={n_tokens} "
        f"images_ori={tuple(inputs['images_ori'].shape)} images_crop={tuple(inputs['images_crop'].shape)}"
    )

    contexts = [TimeProfiler("crop_smoke", logger), LLMInferenceContextManager(xh_model), torch.no_grad()]
    with ContextManagers(contexts):
        xh_model.to(device=device, dtype=dtype)
        # visual crop features
        image_embeds = xh_model.visual.forward_crop(
            inputs["images_ori"][0].unsqueeze(0).to(dtype),
            inputs["images_crop"].to(dtype),
            wc,
            hc,
        )
        n_feat = int(image_embeds.shape[1])
        logger.info(f"visual crop features={n_feat} match={n_tokens == n_feat}")
        assert n_tokens == n_feat, f"token/feature mismatch: {n_tokens} vs {n_feat}"

        # LLM prefill through the wrap model with scattered crop embeds
        seq_length = int(inputs["input_ids"].shape[1])
        xh_model.set_input_sequence_length(max(seq_length, xh_model.wrap_cfg.input_sequence_length))
        data_processor = xh_model.get_data_preprocessor()
        processed = data_processor(
            {
                "input_ids": inputs["input_ids"],
                "image_embeds": image_embeds.reshape(-1, image_embeds.shape[-1]),
                "images_seq_mask": inputs["images_seq_mask"],
                "past_seq_length": 0,
            }
        )
        logits = xh_model(*processed)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    last = logits[0, seq_length - 1] if logits.shape[1] >= seq_length else logits[0, -1]
    next_token = int(last.argmax().item())
    logger.info(f"wrap crop prefill logits shape: {tuple(logits.shape)}")
    logger.info(f"wrap crop last-token argmax: {next_token} -> {tokenizer.decode([next_token])!r}")
    logger.info("crop smoke PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/unlimited_ocr/gundam/unlimited_ocr_llm_gundam_xh2a_32k.py",
    )
    parser.add_argument("--eval-type", type=str, default="wrap", choices=LLMModelState.get_all_values())
    parser.add_argument("--hf-model", type=str, default="")
    parser.add_argument("--image-path", type=str, default="data/images/qwen2_vl_demo.jpeg")
    parser.add_argument("--prompt", type=str, default="<image>\\nFree OCR. ")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
