"""Unlimited-OCR multimodal prefill logits golden-alignment (wrap state).

Builds base/no-crop inputs, obtains image embeddings (native HF visual by
default), scatters them via ``UnlimitedOCRDataPreprocess``, runs the wrap-state
LLM, and compares the last-token argmax against the native golden dump produced
by ``native_unlimited_ocr_forward.py``.

Usage:
    python examples_merak/llm/unlimited_ocr/debug_scripts/unlimited_ocr_llm_xh_debug.py \
        --config configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_llm_base_xh2a_32k.py \
        --image-path data/images/qwen2_vl_demo.jpeg \
        --golden work_dirs/unlimited_ocr_debug/native_prefill.pt
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
from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_unlimitedocr import UnlimitedOCRForCausalLM
from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_unlimitedocr_patch import unlimited_ocr_patch
from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_visual_model import UnlimitedOCRBaseVisualModel
from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_processor import XHUnlimitedOCRProcessor
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, TimeProfiler

if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.unlimited_ocr import XHUnlimitedOCRModel


def main(args):
    cfg_name = Path(args.config).stem
    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "llm_debug.log"), args.debug)
    set_random_seed(1024)
    logger = get_xhquant_logger()

    cfg = Config.fromfile(args.config)
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    hf_model_dir = args.hf_model or model_cfg.hf_model

    xh_model: "XHUnlimitedOCRModel" = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.set_state(LLMModelState.from_string(args.eval_type))

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

    # Native HF visual produces the reference image embeddings to isolate the
    # LLM prefill path from visual quant error.
    hf_native = UnlimitedOCRForCausalLM.from_pretrained(hf_model_dir, dtype=dtype, trust_remote_code=False)
    hf_visual = UnlimitedOCRBaseVisualModel(unlimited_ocr_patch(hf_native)).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        image_embeds = hf_visual(images_ori)
    image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
    logger.info(f"image_embeds shape: {tuple(image_embeds.shape)}")

    generated = []
    contexts = [TimeProfiler("llm_debug", logger), LLMInferenceContextManager(xh_model), torch.no_grad()]
    with ContextManagers(contexts):
        xh_model.to(device=device, dtype=dtype)
        # base single-image prompt (~278) exceeds prefill_chunk_length=256; widen
        # the wrap input sequence length so this debug can run a single forward.
        xh_model.set_input_sequence_length(max(seq_length, xh_model.wrap_cfg.input_sequence_length))
        data_processor = xh_model.get_data_preprocessor()
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
        if args.generate_tokens > 0:
            generated.append(next_token)
            for _ in range(1, args.generate_tokens):
                xh_model.set_decode()
                xh_model.set_input_sequence_length(1)
                decode_processed = data_processor(
                    {
                        "input_ids": torch.tensor([[generated[-1]]], dtype=torch.long, device=device),
                        "past_seq_length": seq_length + len(generated) - 1,
                    }
                )
                step_logits = xh_model(*decode_processed)
                if isinstance(step_logits, (tuple, list)):
                    step_logits = step_logits[0]
                generated.append(int(step_logits[0, -1].argmax().item()))
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    last_logits = logits[0, seq_length - 1] if logits.shape[1] >= seq_length else logits[0, -1]
    next_token = int(last_logits.argmax().item())
    logger.info(f"input_ids shape: {tuple(input_ids.shape)} image tokens: {int(images_seq_mask.sum().item())}")
    logger.info(f"wrap logits shape: {tuple(logits.shape)}")
    logger.info(f"wrap last-token argmax: {next_token} -> {tokenizer.decode([next_token])!r}")

    if args.generate_tokens > 0:
        logger.info(f"manual generate tokens: {generated}")
        logger.info(
            "manual generate text: "
            f"{tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)!r}"
        )

    if args.golden and Path(args.golden).exists():
        golden = torch.load(args.golden, map_location="cpu")
        g_token = int(golden["next_token"])
        logger.info(f"golden next_token: {g_token} -> {tokenizer.decode([g_token])!r}")
        logger.info(f"argmax match: {'PASS' if g_token == next_token else 'FAIL'}")
        if "last_logits" in golden and golden["last_logits"].shape == last_logits.shape:
            diff = (golden["last_logits"].float() - last_logits.float().cpu()).abs()
            logger.info(f"last-logits max abs diff: {diff.max().item():.4f}  mean: {diff.mean().item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_llm_base_xh2a_32k.py",
    )
    parser.add_argument("--eval-type", type=str, default="wrap", choices=LLMModelState.get_all_values())
    parser.add_argument("--hf-model", type=str, default="")
    parser.add_argument("--image-path", type=str, default="data/images/qwen2_vl_demo.jpeg")
    parser.add_argument("--prompt", type=str, default="<image>\\nFree OCR. ")
    parser.add_argument("--golden", type=str, default="")
    parser.add_argument("--generate-tokens", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
