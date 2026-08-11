"""Unlimited-OCR visual wrap embedding golden-alignment.

Runs the wrap-state ``XHUnlimitedOCRVisualModel`` on a base/no-crop global-view
image and compares the image embeddings against the native HF visual branch
(``sam_model`` + ``vision_model`` + ``projector`` + newline/separator).

Base/no-crop expects output shape ``(1, 273, 1280)``.

Usage:
    python examples_merak/llm/unlimited_ocr/debug_scripts/unlimited_ocr_visual_xh_debug.py \
        --config configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_visual_base_xh2a_32k.py \
        --image-path data/images/qwen2_vl_demo.jpeg
"""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, LLMInferenceContextManager, LLMModelState
from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_unlimitedocr import UnlimitedOCRForCausalLM
from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_visual_model import UnlimitedOCRBaseVisualModel
from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_unlimitedocr_patch import unlimited_ocr_patch
from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_processor import XHUnlimitedOCRProcessor
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, TimeProfiler

if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.unlimited_ocr import XHUnlimitedOCRVisualModel


def _build_image(args, device, dtype) -> torch.Tensor:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    processor = XHUnlimitedOCRProcessor(
        tokenizer,
        image_size=args.image_size,
        base_size=args.image_size,
        patch_size=args.patch_size,
        downsample_ratio=args.downsample_ratio,
        crop_mode=False,
    )
    inputs = processor.process("<image>\n", args.image_path, device=device)
    return inputs["images_ori"].to(dtype)


def main(args):
    cfg_name = Path(args.config).stem
    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "visual_debug.log"), args.debug)
    set_random_seed(1024)
    logger = get_xhquant_logger()

    cfg = Config.fromfile(args.config)
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    model_cfg.work_dir = str(work_dir)
    if not args.hf_model:
        args.hf_model = model_cfg.hf_model

    xh_model: "XHUnlimitedOCRVisualModel" = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.set_state(LLMModelState.from_string(args.eval_type))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.fp32 else torch.float16
    image = _build_image(args, device, dtype)

    contexts = [TimeProfiler("visual_debug", logger), LLMInferenceContextManager(xh_model), torch.no_grad()]
    with ContextManagers(contexts):
        xh_model.to(device=device, dtype=dtype)
        wrap_embeds = xh_model(image)
    if isinstance(wrap_embeds, (tuple, list)):
        wrap_embeds = wrap_embeds[0]
    logger.info(f"wrap image_embeds shape: {tuple(wrap_embeds.shape)} dtype: {wrap_embeds.dtype}")

    if args.compare_hf:
        hf_native = UnlimitedOCRForCausalLM.from_pretrained(
            args.hf_model, dtype=dtype, trust_remote_code=False
        )
        hf_visual = UnlimitedOCRBaseVisualModel(unlimited_ocr_patch(hf_native)).to(device=device, dtype=dtype).eval()
        with torch.no_grad():
            hf_embeds = hf_visual(image)
        logger.info(f"hf image_embeds shape: {tuple(hf_embeds.shape)}")
        if hf_embeds.shape == wrap_embeds.shape:
            diff = (hf_embeds.float() - wrap_embeds.float()).abs()
            logger.info(f"visual max abs diff: {diff.max().item():.6f}  mean abs diff: {diff.mean().item():.6f}")
            passed = diff.max().item() <= args.atol
            logger.info(f"visual align {'PASS' if passed else 'FAIL'} (atol={args.atol})")
        else:
            logger.warning(f"shape mismatch wrap={tuple(wrap_embeds.shape)} hf={tuple(hf_embeds.shape)}")

    if args.dump:
        dump_path = Path(args.dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"wrap_image_embeds": wrap_embeds.float().cpu()}, dump_path)
        logger.info(f"dumped visual embeds to {dump_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_visual_base_xh2a_32k.py",
    )
    parser.add_argument("--eval-type", type=str, default="wrap", choices=LLMModelState.get_all_values())
    parser.add_argument("--hf-model", type=str, default="")
    parser.add_argument("--image-path", type=str, default="data/images/qwen2_vl_demo.jpeg")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--downsample-ratio", type=int, default=4)
    parser.add_argument("--compare-hf", action="store_true", help="compare wrap visual against native HF visual")
    parser.add_argument("--fp32", action="store_true", help="run in fp32 (recommended for numeric alignment)")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--dump", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
