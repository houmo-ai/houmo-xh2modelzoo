import argparse
import os.path as osp
from pathlib import Path

import torch

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import MemoryTracker, TimeProfiler


class LLMExportCommand:
    doc_string: str = "convert transformers model to hmonnx model."

    def __init__(self, args):
        self.onnx_file = None
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            "--config",
            type=str,
            default="",
            help=(
                "model config file, for development and debugging, "
                "use configs in examples_merak/llm/qwen3_legacy/configs."
            ),
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Whether to force export even if the model is exist.",
        )
        parser.add_argument(
            "--chip-arch",
            type=str,
            default="XH2a",
            help="chip architecture, default is XH2a",
        )
        parser.add_argument("--model", type=str, default="")
        parser.add_argument("--context-length", type=int, default=2048, help="max sequence length")
        parser.add_argument(
            "--prefill-chunk-length",
            type=int,
            default=256,
            help="prefill chunk length",
        )
        parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8")
        parser.add_argument(
            "--quant-weight",
            type=str,
            default=None,
            help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
        )
        parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
        args = parser.parse_args(args)
        self.args = args

    def run(self):
        args = self.args
        config_file = args.config
        model_dir = args.model
        if config_file is not None and len(config_file) > 0 and model_dir is not None and len(model_dir) > 0:
            raise ValueError("Cannot specify both --config and --model at the same time. Please choose one.")
        if config_file is not None and len(config_file) > 0:
            cfg_name = Path(config_file).stem
            cfg = Config.fromfile(args.config)
        elif model_dir is not None and len(model_dir) > 0:
            hf_model_path = osp.normpath(osp.abspath(args.model))
            model_name = Path(hf_model_path).name
            target_device = args.chip_arch
            quant_type = args.quant_type
            prefill_chunk_length = args.prefill_chunk_length
            context_length = args.context_length

            cfg_name = (
                f"{model_name}_{target_device}_{quant_type}_{prefill_chunk_length}_{args.context_length // 1024}k_cli"
            )
            cfg_name = cfg_name.lower()
            cfg = dict(
                chip_arch=target_device,
                model=dict(
                    model_type="Qwen3ForCausalLM_legacy",
                    chip_arch=target_device,
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
            cfg = Config(cfg)

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

                logger.info(f"Exported model already exists at {work_dir}, use --force to overwrite.")

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
        model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
        logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
        xh_model = AutoLLMModel.from_pretrained(config=model_cfg)

        with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert2hmonnx", logger):
            xh_model.export_hmonnx(str(work_dir))
