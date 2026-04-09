import argparse
import os.path as osp
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name, support_llm_model_types
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import MemoryTracker, TimeProfiler

if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.gemma4 import XHGemma4Model, XHGemma4ModelConfig


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
            max_pe_length=max(context_length, 32768),
            use_cache=True,
            num_logits_to_keep=1,
            quant_scheme=dict(
                quant_type=quant_type,
            ),
            quant_weight=args.quant_weight,
            only_first_block=False,
            visual_config=dict(
                image_seq_length=280,
                patch_size=16,
                pooling_kernel_size=3,
                quant_scheme=dict(quant_type=quant_type),
            ),
        ),
    )
    cfg = format_model_name(cfg)
    return cfg_name.lower(), Config(cfg)


def main(args):
    config_file = args.config
    model_dir = args.model
    if config_file and model_dir:
        raise ValueError("Cannot specify both --config and --model at the same time.")
    if config_file:
        cfg_name = Path(config_file).stem
        cfg = Config.fromfile(config_file)
    elif model_dir:
        cfg_name, cfg = _build_cfg_from_model(args)
    else:
        raise ValueError("Either --config or --model must be specified.")

    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "export_hmonnx.log"), args.debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()
    cfg.seed = seed
    cfg.dump(work_dir / f"{cfg_name}.py")

    model_cfg: XHGemma4ModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    xh_model: XHGemma4Model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.work_dir = str(work_dir)
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert2hmonnx", logger):
        xh_model.export_hmonnx(str(work_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/gemma4/31b/gemma4_31b_it_xh2a_2k.py",
        help="model config file for development and debugging",
    )
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--model-type", type=str, default="Gemma4ForConditionalGeneration", choices=support_llm_model_types)
    parser.add_argument("--chip-arch", type=str, default="XH2a", choices=["XH2a", "YueHui"])
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--context-length", type=int, default=2048, help="max context sequence length")
    parser.add_argument("--prefill-chunk-length", type=int, default=256, help="prefill chunk length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type")
    parser.add_argument("--quant-weight", type=str, default=None, help="optional quant weight path")
    args = parser.parse_args()
    main(args)
