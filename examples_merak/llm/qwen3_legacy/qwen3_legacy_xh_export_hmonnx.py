import argparse
import os.path as osp
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name, support_llm_model_types
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen3_legacy import XHQwen3LegacyModel, XHQwen3LegacyModelConfig


def _override_hf_model_dir(cfg, hf_model_dir):
    if hf_model_dir is None or len(hf_model_dir) == 0:
        return
    cfg.hf_model_dir = hf_model_dir
    cfg.model.hf_model = hf_model_dir


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
                quant_type=quant_type,  # Node默认量化类型
            ),
            # 内部调试参数
            only_first_block=False,  # 仅包裹第一层，调试用
            quant_weight=args.quant_weight,
        ),
    )
    cfg = format_model_name(cfg)
    return cfg_name.lower(), Config(cfg)


def main(args):
    config_file = args.config
    model_dir = args.model
    if config_file is not None and len(config_file) > 0 and model_dir is not None and len(model_dir) > 0:
        raise ValueError("Cannot specify both --config and --model at the same time. Please choose one.")
    if args.config_hf_model_dir and model_dir is not None and len(model_dir) > 0:
        raise ValueError("--config-hf-model-dir can only be used with --config, not --model.")
    if config_file is not None and len(config_file) > 0:
        cfg_name = Path(config_file).stem
        cfg = Config.fromfile(args.config)
        _override_hf_model_dir(cfg, args.config_hf_model_dir)
    elif model_dir is not None and len(model_dir) > 0:
        cfg_name, cfg = _build_cfg_from_model(args)
    else:
        raise ValueError("Either --config or --model must be specified.")

    debug = args.debug
    if debug:
        cfg_name += "_debug"

    args.work_dir = str(Path("./work_dirs") / cfg_name)
    work_dir = Path(args.work_dir)
    if work_dir.exists():
        if args.force:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            from loguru import logger

            logger.info(f"Exported model already exists at {work_dir}, use --force to overwrite. ")

            return -1
    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(work_dir / "export_hmonnx.log")

    xhquant_init(log_file, debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()
    # 使用config文件,代替命令行参数,方便调试不同的配置

    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = work_dir / f"{cfg_name}.py"
    cfg.dump(config_file)

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")
    model_cfg: XHQwen3LegacyModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    assert type(model_cfg).__name__ == "XHQwen3LegacyModelConfig", (
        f"Expected model config type XHQwen3LegacyModelConfig, but got {type(model_cfg).__name__}"
    )
    # model_cfg = XHQwen3LegacyModelConfig.from_dict(cfg.model)
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model: XHQwen3LegacyModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHQwen3LegacyModel", (
        f"Expected model type XHQwen3LegacyModel, but got {type(xh_model).__name__}"
    )
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert2hmonnx", logger):
        xh_model.export_hmonnx(str(work_dir))


if __name__ == "__main__":
    """
    --config 提供给开发使用，指定模型配置文件，在配置文件中可以设置复杂的参数组合，方便调试和开发。
    对于正式使用和部署，使用--context-length，--prefill-chunk-length，--quant-type，--quant-weight等参数来指定模型配置，更加简洁和灵活。
     - --context-length: 模型的最大上下文长度，默认为2048。
     - --prefill-chunk-length: 预填充块的长度，默认为256。
     - --quant-type: 量化类型，默认为w4a8h0_ssfp。
     - --quant-weight: 使用后摩的GPTQ量化后的权重文件。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="model config file, for development and debugging, use configs in examples_merak/llm/qwen3_legacy/configs.",
    )
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--force", action="store_true", help="Whether to force export even if the model is exist.")
    parser.add_argument(
        "--model-type", type=str, default="Qwen3ForCausalLM_legacy", help="", choices=support_llm_model_types
    )
    parser.add_argument(
        "--chip-arch", type=str, default="XH2a", help="chip architecture, default is XH2a", choices=["XH2a", "YueHui"]
    )
    parser.add_argument("--model", type=str, default="")
    parser.add_argument(
        "--config-hf-model-dir",
        type=str,
        default="",
        help="Override hf_model_dir/hf_model in the config file.",
    )
    parser.add_argument("--context-length", type=int, default=2048, help="max context sequence length")
    parser.add_argument("--prefill-chunk-length", type=int, default=256, help="prefill chunk length")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8_sefp")
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    args = parser.parse_args()
    main(args)
