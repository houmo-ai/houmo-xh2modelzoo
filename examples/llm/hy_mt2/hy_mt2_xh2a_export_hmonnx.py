# Copyright 2025 HOUMO AI
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os.path as osp
from pathlib import Path

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.hy_mt2 import HyMT2ConvertConfig

from xhquant.api import DeviceType, QuantScheme, get_root_logger, xhquant_init  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)
    config = HyMT2ConvertConfig(
        batch_size=1,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        mix_search=args.mix_search,
        num_logits_to_keep=args.num_logits_to_keep,
    )

    prefix = f"{model_name}-{target_device}-{args.context_length // 1024}k-{quant_type}"
    work_dir = Path(args.work_dir) if args.work_dir else Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, "HunYuanDenseV1ForCausalLM", config, str(work_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default="/data01/datasets/Hy-MT2-7B")
    parser.add_argument("--work-dir", type=str, default=None, help="output work dir")
    parser.add_argument("--context-length", type=int, default=4096, help="max sequence length")
    parser.add_argument("--input-sequence-length", type=int, default=256, help="input sequence length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type")
    parser.add_argument("--mix_search", type=str, default=None, help="mix search settings")
    parser.add_argument("--num_logits_to_keep", type=int, default=1, help="use 0 for logits eval, 1 for generation")
    parser.add_argument("--quant-weight", type=str, default=None, help="quant weight path")
    args = parser.parse_args()
    main(args)
