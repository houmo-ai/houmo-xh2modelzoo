import argparse
import os.path as osp
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name, support_llm_model_types
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen2_vl import XHQwen2VLModel, XHQwen2VLModelConfig


def _build_cfg_from_model(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    cfg_name = (
        f"{args.chip_arch}_{model_name}_{args.quant_type}_{args.prefill_chunk_length}_"
        f"{args.context_length // 1024}k_cli"
    )
    cfg = dict(
        chip_arch=args.chip_arch,
        model=dict(
            model_type=args.model_type,
            hf_model=hf_model_path,
            model_name=model_name,
            context_max_length=args.context_length,
            prefill_chunk_length=args.prefill_chunk_length,
            use_cache=True,
            num_logits_to_keep=1,
            quant_scheme=dict(
                quant_type=args.quant_type,
            ),
            only_first_block=False,
            quant_weight=args.quant_weight,
            visual_config=dict(
                max_size_w=args.image_size_w,
                max_size_h=args.image_size_h,
                quant_scheme=dict(
                    quant_type=args.quant_type,
                    ops={},
                ),
            ),
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

    work_dir = Path("./work_dirs") / cfg_name
    if work_dir.exists():
        if args.force:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            from loguru import logger

            logger.info(f"Exported model already exists at {work_dir}, use --force to overwrite.")
            answer = input("Continue and overwrite? [y/N]: ").strip().lower()
            if answer != "y":
                return -1

    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "export_hmonnx.log"), args.debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()
    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    cfg.dump(work_dir / f"{cfg_name}.py")

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")
    model_cfg: XHQwen2VLModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    assert type(model_cfg).__name__ == "XHQwen2VLModelConfig", (
        f"Expected model config type XHQwen2VLModelConfig, but got {type(model_cfg).__name__}"
    )
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model: XHQwen2VLModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHQwen2VLModel", (
        f"Expected model type XHQwen2VLModel, but got {type(xh_model).__name__}"
    )
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert2hmonnx", logger):
        xh_model.export_hmonnx(str(work_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--force", action="store_true", help="Whether to force export even if the model exists.")
    parser.add_argument(
        "--model-type", type=str, default="Qwen2VLForConditionalGeneration", help="", choices=support_llm_model_types
    )
    parser.add_argument(
        "--chip-arch", type=str, default="XH2a", help="chip architecture, default is XH2a", choices=["XH2a", "YueHui"]
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--context-length", type=int, default=4096, help="max context sequence length")
    parser.add_argument("--prefill-chunk-length", type=int, default=256, help="prefill chunk length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type")
    parser.add_argument("--quant-weight", type=str, default=None, help="quant weight path")
    parser.add_argument("--image-size-w", type=int, default=448, help="image width for vision input")
    parser.add_argument("--image-size-h", type=int, default=448, help="image height for vision input")
    args = parser.parse_args()
    main(args)
