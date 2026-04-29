# Copyright 2025 HOUMO AI
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os.path as osp
from pathlib import Path

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.stella_mrl import StellaMRLConvertConfig

from xhquant.api import DeviceType, QuantScheme, get_root_logger, xhquant_init  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a

    quant_scheme = QuantScheme(target_device=target_device, quant_type=args.quant_type)
    config = StellaMRLConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        quant_scheme=quant_scheme,
        output_normalized=args.output_normalized,
    )

    prefix = f"{model_name}-{target_device}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)

    xhquant_init(str(work_dir / "convert.log"), debug=args.debug)
    logger = get_root_logger()

    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, "BertModel_StellaMRL", config, str(work_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--model",
        type=str,
        default="/data02/datasets/stella-mrl/t3lsss/stella-mrl-large-zh-v3___5-1792d",
        help="HF model directory",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--context-length", type=int, default=512, help="max sequence length")
    parser.add_argument("--quant-type", type=str, default="w8a8_sefp", help="quant type")
    parser.add_argument(
        "--output-normalized",
        action="store_true",
        help="L2 normalize embeddings in graph output. Default keeps raw stella output.",
    )
    args = parser.parse_args()
    main(args)
