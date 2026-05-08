"""Gemma4 MoE 26B-A4B-it with-mask LLM HMONNX export (xh2modelzoo idiomatic).

Mirrors the structure of ``gpt_oss_xh_export_hmonnx.py``: build / load the
config, materialise the merak model via ``AutoLLMConfig`` /
``AutoLLMModel`` and let ``xh_model.export_hmonnx(work_dir)`` drive the
WRAPED → FRONTEND → QUANTED_ALIGNED → EXPORTED → HMONNX pipeline plus
``golden_meta_info.json``.

Notes
-----
* Heterogeneous KV-cache shapes per layer (Gemma4 alternates local / global
  attention) and the extra ``local_attention_mask`` / ``global_attention_mask``
  inputs are handled by ``XHGemma4MoeWithMaskModel`` (see
  ``xhmodel_merak/xh_llm/models/gemma4_moe/``).
* The visual encoder is exported separately via
  ``gemma4_moe_visual_xh_export_onnx.py``.
* Source-side golden generation (`xh_model_zoo`-registered with-mask ONNX
  model) is intentionally not wired here because the zoo pattern keeps that
  in a separate script when the model is registered on the source side.
"""

from __future__ import annotations

import argparse
import os.path as osp
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name, support_llm_model_types
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.gemma4_moe import XHGemma4MoeWithMaskConfig, XHGemma4MoeWithMaskModel


def _build_cfg_from_model(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = args.chip_arch
    quant_type = args.quant_type
    prefill_chunk_length = args.prefill_chunk_length
    context_length = args.context_length

    cfg_name = (
        f"{target_device}_{model_name}_with_mask_{quant_type}_{prefill_chunk_length}_{context_length // 1024}k_cli"
    )
    cfg = dict(
        chip_arch=target_device,
        model=dict(
            model_type=args.model_type,
            hf_model=hf_model_path,
            fallback_hf_model=hf_model_path,
            model_name=model_name,
            context_max_length=context_length,
            prefill_chunk_length=prefill_chunk_length,
            use_cache=True,
            num_logits_to_keep=1,
            quant_scheme=dict(
                quant_type=quant_type,
            ),
            only_first_block=False,
            quant_weight=args.quant_weight,
        ),
    )
    cfg = format_model_name(cfg)
    return cfg_name.lower(), Config(cfg)


def main(args):
    config_file = args.config
    model_dir = args.model
    if config_file and model_dir:
        raise ValueError("Cannot specify both --config and --model at the same time. Please choose one.")
    if config_file:
        cfg_name = Path(config_file).stem
        cfg = Config.fromfile(args.config)
    elif model_dir:
        cfg_name, cfg = _build_cfg_from_model(args)
    else:
        raise ValueError("Either --config or --model must be specified.")

    if args.debug:
        cfg_name += "_debug"
    if args.valid:
        cfg_name += "_valid"
        # Single decoder block — fast smoke path.
        cfg.model.only_first_block = True

    args.work_dir = str(Path("./work_dirs") / cfg_name)
    work_dir = Path(args.work_dir)
    if work_dir.exists():
        if args.force:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            from loguru import logger

            logger.info(f"Exported model already exists at {work_dir}, use --force to overwrite.")
            return -1

    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(work_dir / "export_hmonnx.log")

    xhquant_init(log_file, args.debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()

    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    dumped_config_file = work_dir / f"{cfg_name}.py"
    cfg.dump(dumped_config_file)

    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")

    model_cfg: XHGemma4MoeWithMaskConfig = AutoLLMConfig.from_pretrained(cfg.model)
    assert type(model_cfg).__name__ == "XHGemma4MoeWithMaskConfig", (
        f"Expected model config type XHGemma4MoeWithMaskConfig, but got {type(model_cfg).__name__}"
    )
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")

    xh_model: XHGemma4MoeWithMaskModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHGemma4MoeWithMaskModel", (
        f"Expected model type XHGemma4MoeWithMaskModel, but got {type(xh_model).__name__}"
    )

    # The visual sub-model is exported separately by
    # ``gemma4_moe_visual_xh_export_onnx.py``; drop it here so the LLM export
    # path doesn't drag the vision tower through wrap / quant.
    if hasattr(xh_model, "visual"):
        try:
            delattr(xh_model, "visual")
        except AttributeError:
            pass
    if hasattr(xh_model, "_models") and isinstance(xh_model._models, dict):
        xh_model._models.pop("visual", None)

    with TimeProfiler("convert", logger), MemoryTracker(device, "convert2hmonnx", logger):
        xh_model.export_hmonnx(str(work_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/gemma4_moe/26b_a4b_it/gemma4_moe_with_mask_26b_a4b_it_xh2a_w8a8_256_2k.py",
        help="model config file, for development and debugging.",
    )
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--force", default=True, action="store_true", help="Whether to force export even if the model exists.")
    parser.add_argument(
        "--valid",
        default=False,
        action="store_true",
        help="Wrap only the first decoder block for a fast smoke export.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="Gemma4ForConditionalGeneration_with_mask",
        choices=support_llm_model_types,
    )
    parser.add_argument(
        "--chip-arch",
        type=str,
        default="XH2a",
        choices=["XH2a", "YueHui"],
    )
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--context-length", type=int, default=2048, help="max context sequence length")
    parser.add_argument("--prefill-chunk-length", type=int, default=256, help="prefill chunk length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type")
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    main(parser.parse_args())
