import argparse
import os.path as osp
import re
import time
from pathlib import Path

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen3_embedding import Qwen3EmbeddingConvertConfig

from xhquant.api import (  # isort:skip
    DeviceType,
    QuantScheme,
    get_root_logger,
    register_none_quanted_module,
    xhquant_init,
)
from xhquant import nn as xhnn  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    is_0_6b = "0.6b" in model_name.lower()
    quant_type = args.quant_type
    if is_0_6b and args.quant_weight is None and quant_type == "w4a8h0_ssfp":
        quant_type = "w8a8h1_sefp"
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    config = Qwen3EmbeddingConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
    )
    # PTQ-only path for 0.6B (no Quarot/GPTQ weights)
    config.ptq = bool(args.ptq or (is_0_6b and args.quant_weight is None))

    model_tag = model_name.lower().replace("-", "_")
    model_tag = re.sub(r"_\\d+(?:\\.\\d+)?b", "", model_tag)
    size_match = re.search(r"(\\d+(?:\\.\\d+)?)b", model_name.lower())
    size_tag = size_match.group(0).lower() if size_match else ""
    quant_tag = quant_type.split("_")[0]
    quant_tag = re.sub(r"h\\d+$", "", quant_tag)
    context_k = f"{args.context_length // 1024}k"
    export_date = args.export_date or time.strftime("%Y%m%d")
    parts = ["hmquant_xh2", model_tag]
    if size_tag:
        parts.append(size_tag)
    parts += [quant_tag, str(args.input_sequence_length), context_k, export_date]
    prefix = "_".join(parts)
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    register_none_quanted_module("xh2a", xhnn.DType)
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(
            hf_model_path, "Qwen3Embedding", config, str(work_dir)
        )


if __name__ == "__main__":
    # import debugpy

    # debugpy.listen(("0.0.0.0", 1160))
    # print("✅ debugpy listening on 0.0.0.0:5678, waiting for VSCode attach...")
    # debugpy.wait_for_client()
    # print("✅ VSCode attached, continue running.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--model",
        type=str,
        default="/data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-4B",
        help="HF model directory",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--context-length", type=int, default=2048, help="max sequence length"
    )
    parser.add_argument(
        "--input-sequence-length", type=int, default=256, help="input sequence length"
    )
    parser.add_argument(
        "--quant-type", default="w4a8h0_ssfp", help="quant type"
    )  # w8a8_sefp
    parser.add_argument(
        "--ptq",
        action="store_true",
        help="enable PTQ calibration (recommended for 0.6B when no quant-weight)",
    )
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="gptq+quarot state dict path (safetensors). If empty, use FP weights.",
    )
    parser.add_argument(
        "--export-date",
        type=str,
        default=None,
        help="override export date in folder/file name, format: YYYYMMDD",
    )
    args = parser.parse_args()
    main(args)
