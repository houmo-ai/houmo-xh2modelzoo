import argparse
import os.path as osp
from pathlib import Path
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger, Config  # isort:skip

from xh_model_zoo_new.utils import MemoryTracker, TimeProfiler
from xh_model_zoo_new.xh_llm.models.qwen2 import Qwen2ConverterConfig, Qwen2Converter
from transformers import Qwen25ForCausalLM

def main(args):
    # 1. Parse Config
    cfg = Config.fromfile(args.extra_config_file) if args.extra_config_file is not None else Config()
    cfg.merge_from_dict(
        dict(
            context_length=args.context_length,
            input_sequence_length=args.input_sequence_length,
            quant_weight=args.quant_weight,
            quant_scheme=dict(
                target_device=DeviceType.XH2a, quant_type=args.quant_type, nodes=dict(lm_head="w8a8h1_sefp")
            ),
        )
    )
    config = Qwen2ConverterConfig.from_dict_or_other(cfg.to_dict())

    # 2. Convert
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    prefix = f"{model_name}-{target_device}-{args.context_length//1024}k-{args.quant_type}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        Qwen2Converter.convert_and_export(hf_model_path, config, work_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default="/data01/datasets/Qwen2-7B")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--context-length", type=int, default=2048, help="max sequence length")
    parser.add_argument("--input-sequence-length", type=int, default=256, help="input sequence length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    parser.add_argument("--extra-config-file", type=str, default=None, help="extra config file")
    args = parser.parse_args()
    main(args)
