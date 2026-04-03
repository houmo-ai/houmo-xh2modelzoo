import argparse
import os.path as osp
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name, support_llm_model_types
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.gpt_oss import XHGptOssModel, XHGptOssModelConfig


def _build_cfg_from_model(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = args.chip_arch
    quant_type = args.quant_type
    prefill_chunk_length = args.prefill_chunk_length
    context_length = args.context_length

    cfg_name = f"{target_device}_{model_name}_{quant_type}_{prefill_chunk_length}_{context_length // 1024}k_cli"
    cfg = dict(
        chip_arch=target_device,
        model=dict(
            model_type=args.model_type,
            hf_model=hf_model_path,
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

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")
    model_cfg: XHGptOssModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    assert type(model_cfg).__name__ == "XHGptOssModelConfig", (
        f"Expected model config type XHGptOssModelConfig, but got {type(model_cfg).__name__}"
    )

    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model: XHGptOssModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHGptOssModel", (
        f"Expected model type XHGptOssModel, but got {type(xh_model).__name__}"
    )

    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert2hmonnx", logger):
        xh_model.export_hmonnx(str(work_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/gpt_oss/20b/gpt_oss_20b_xh2a_2k.py",
        help="model config file, for development and debugging.",
    )
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--force", action="store_true", help="Whether to force export even if the model exists.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="GptOssForCausalLM",
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
