import argparse
import os
import os.path as osp
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoProcessor

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen2_5_vl import Qwen2_5_VLConvertConfig, VisualConfig
from xh_model_zoo.xh_llm.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default="weights/UI-TARS-1.5-7B")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--context-length", type=int, default=4096, help="max sequence length")
    parser.add_argument("--max_pe_length", type=int, default=32768, help="max pe length")
    parser.add_argument("--quant-type", default="w4a8h0_ssfp", help="quant type, default is w4a8")
    
    # UI-TARS requires larger resolution for detail perception
    parser.add_argument("--image_max_size_h", type=int, default=1008, help="image max size height")
    parser.add_argument("--image_max_size_w", type=int, default=1008, help="image max size width")
    
    parser.add_argument("--image_max_size_t", type=int, default=2, help="if image, temporal max size is 2, if video, temporal max size is fps")
    parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="temporal patch size")
    parser.add_argument("--sample_image_path", type=str, default="data/test/example.png", help="sample image path for generate golden")
    parser.add_argument("--use_gptqmodel", action="store_true", help="use gptqmodel quanted model")
    parser.add_argument(
        "--quant_weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()
    logger = get_root_logger()

    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type

    # Default Quant Scheme for XH2a
    ops = dict(
        MatMul=dict(
            act_scheme=dict(
                bits=8,
                fp_mode="sefp",
            ),
            act_schema_2=dict(
                bits=16,
                fp_mode="sefp",
            ),
        )
    )

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type, ops=ops)

    hardcoded_path = Path("data/images/qwen2_vl_demo.jpeg")
    sample_image_path = Path(args.sample_image_path).resolve()
    
    hardcoded_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(hardcoded_path):
        if hardcoded_path.is_dir():
            raise RuntimeError(f"Hardcoded demo image path is a directory: {hardcoded_path}")
        if hardcoded_path.is_symlink():
            hardcoded_path.unlink()
            os.symlink(sample_image_path, hardcoded_path)
    else:
        os.symlink(sample_image_path, hardcoded_path)

    # 1. Load Native Model (CPU) to prepare for conversion
    native_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if native_model.config.tie_word_embeddings:
        old_torchscript = native_model.config.torchscript
        native_model.config.torchscript = True
        native_model.tie_weights()
        native_model.config.tie_word_embeddings = False
        native_model.config.torchscript = old_torchscript

    native_model.eval()
    native_model.to(torch.bfloat16)

    # 2. Configure Converter
    config = Qwen2_5_VLConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        gptqmodel_cfg=args.use_gptqmodel,
        max_pe_length=args.max_pe_length,
        visual_config=VisualConfig(
            image_max_size_h=args.image_max_size_h,
            image_max_size_w=args.image_max_size_w,
            image_max_size_t=args.image_max_size_t,
            temporal_patch_size=args.temporal_patch_size,
            patch_size=args.patch_size,
            sample_image_path=args.sample_image_path,
        ),
    )

    prefix = f"{model_name}-{target_device}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    
    xhquant_init(log_file, debug=args.debug)
    
    logger.info(f"Starting Export with Config: {config}")
    
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, "Qwen2_5_VLForConditionalGeneration", config, work_dir)


if __name__ == "__main__":
    main()
