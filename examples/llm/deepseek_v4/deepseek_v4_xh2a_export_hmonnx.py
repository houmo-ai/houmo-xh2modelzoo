# ================================================================== #
#  File: deepseek_v4_xh2a_export_hmonnx.py                           #
#  Description:                                                       #
#    Export script: DeepSeek-V4 -> prefill/decode HMONNX.            #
#                                                                     #
#  Usage:                                                             #
#    python examples/llm/deepseek_v4/deepseek_v4_xh2a_export_hmonnx.py \
#        --model /data01/nfs_shared/llm/DeepSeek-V4-Flash-slim5l \
#        --context-length 2048 \
#        --input-sequence-length 256 \
#        --quant-type w8a8h0_ssfp                                     #
# ================================================================== #

import argparse
import os.path as osp
from pathlib import Path

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.deepseek_v4 import DeepseekV4ConvertConfig
from xhquant.api import (
    DeviceType,
    QuantScheme,
    get_root_logger,
    xhquant_init,
)  # isort:skip


from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)

    config = DeepseekV4ConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        num_logits_to_keep=args.num_logits_to_keep,
    )

    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = Path("work_dirs") / f"{model_name}-{target_device}-{args.context_length // 1024}k-{quant_type}"
    work_dir.mkdir(exist_ok=True, parents=True)

    log_file = work_dir / "convert.log"
    xhquant_init(str(log_file), debug=args.debug)
    logger = get_root_logger()
    logger.info(f"model: {hf_model_path}")
    logger.info(f"output: {work_dir}")

    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, args.architecture, config, str(work_dir))

    logger.info(f"Done. Artifacts in: {work_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export DeepSeek-V4 to prefill/decode HMONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="/data01/nfs_shared/llm/DeepSeek-V4-Flash-slim5l")
    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--input-sequence-length", type=int, default=256)
    parser.add_argument("--quant-type", type=str, default="w8a8h1_sefp")
    parser.add_argument("--num-logits-to-keep", type=int, default=1)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())
