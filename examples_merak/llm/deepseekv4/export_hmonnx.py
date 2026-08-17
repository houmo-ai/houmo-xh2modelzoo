#!/usr/bin/env python3
"""Export fixed-shape DeepSeek-V4-Flash-0731 prefill/decode HMONNX graphs."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from xhmodel_merak.xh_llm import AutoLLMModel
from xhmodel_merak.xh_llm.models.deepseek_v4 import XHDeepSeekV4ModelConfig
from xhmodel_merak.xh_llm.utils import configure_huge_model_export
from xhquant.api import get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import MemoryTracker, TimeProfiler


def build_config(args: argparse.Namespace) -> XHDeepSeekV4ModelConfig:
    return XHDeepSeekV4ModelConfig(
        model_name=args.model_name,
        hf_model=str(Path(args.model).resolve()),
        quant_weight=args.quant_weight,
        chip_arch=args.chip_arch,
        context_max_length=args.context_length,
        prefill_chunk_length=256,
        max_layers=args.max_layers,
        enable_auto_offload=args.enable_auto_offload,
        packed_weight_only=args.packed_weight_only,
        quant_scheme={"quant_type": args.quant_type},
    )


def main(args: argparse.Namespace) -> None:
    configure_huge_model_export(args.low_memory)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(output_dir / "export_hmonnx.log"), args.debug)
    set_random_seed(args.seed)
    logger = get_xhquant_logger()
    config = build_config(args)
    logger.info(f"DeepSeek-V4 Merak config:\n{config.to_json_string()}")
    logger.info("low-memory streamed export is %s", "enabled" if args.low_memory else "disabled")
    if config.enable_auto_offload:
        logger.warning("auto offload was explicitly enabled by the command line")
    else:
        logger.info("auto offload is disabled")

    model = AutoLLMModel.from_pretrained(config=config)
    with (
        TimeProfiler("deepseek_v4_export", logger),
        MemoryTracker([0], "deepseek_v4_export", logger),
    ):
        meta = model.export_hmonnx(str(output_dir))
    logger.info(f"golden_meta_info: {meta}")

    if args.dump_golden:
        # The export model owns the quantized PyTorch graph. Release it before
        # loading the HMONNX runtime so large-model golden generation does not
        # retain both representations at once.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dump_export_golden(output_dir, args)


def dump_export_golden(output_dir: Path, args: argparse.Namespace) -> None:
    """Dump one formal prefill/decode golden step for an exported package."""

    from examples_merak.llm.deepseekv4.deepseek_v4_xh_hmonnx_generate import run

    device_map = args.golden_device_map or ["0"]
    run(
        argparse.Namespace(
            config=str(output_dir),
            prompt=args.golden_prompt,
            context_file=None,
            raw_prompt=False,
            max_new_tokens=2,
            device=",".join(str(device) for device in device_map),
            pack_w4=True,
            cuda_graph=False,
            golden=True,
            stream=False,
            load_only=False,
            debug=args.debug,
            log_file=str(output_dir / "dump_golden.log"),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF/GPTQModel/AutoRound checkpoint directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="deepseek-v4-flash-0731")
    parser.add_argument("--chip-arch", default="XH2a", choices=("XH2a", "YueHui"))
    parser.add_argument("--context-length", type=int, default=256 * 1024)
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Optional diagnostic prefix; omit to export all 43 blocks.",
    )
    parser.add_argument("--quant-type", default="w8a16h1_sefp")
    parser.add_argument(
        "--quant-weight",
        default=None,
        help="Optional standalone quant_weight.pt; omit for a GPTQModel/AutoRound checkpoint.",
    )
    parser.add_argument(
        "--enable-auto-offload",
        action="store_true",
        help="Explicit opt-in. Offload is disabled by default.",
    )
    parser.add_argument(
        "--packed-weight-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Consume GPTQ/AutoRound integers and FP16 scales directly while keeping "
            "the dense FP weights unmaterialized; enabled by default for full-network export."
        ),
    )
    parser.add_argument(
        "--low-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream MoE layers during export; use --no-low-memory to materialize the packed model normally.",
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        help="After export, dump formal prefill/decode step_0 golden data.",
    )
    parser.add_argument(
        "--golden-device-map",
        nargs="+",
        default=None,
        help="Explicit CUDA device map used only by --dump-golden; defaults to device 0.",
    )
    parser.add_argument(
        "--golden-prompt",
        default="17乘以3等于多少？只回答结果。",
        help="Prompt used for the two-token golden generation pass.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
