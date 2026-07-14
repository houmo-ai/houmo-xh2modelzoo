"""Export a text-only Qwen3-Next target and optional MTP draft graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

from xhmodel_merak.xh_llm import AutoLLMModel
from xhmodel_merak.xh_llm.models.qwen3_next import XHQwen3NextModelConfig
from xhquant.api import get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import MemoryTracker, TimeProfiler


def build_config(args: argparse.Namespace) -> XHQwen3NextModelConfig:
    mtp_config = None
    if args.mtp:
        mtp_config = {
            "model_name": f"{args.model_name}_mtp",
            "hf_model": args.model,
            "context_max_length": args.context_length,
            "draft_head_weight_bits": args.mtp_head_weight_bits,
        }
    return XHQwen3NextModelConfig(
        model_name=args.model_name,
        hf_model=args.model,
        chip_arch=args.chip_arch,
        context_max_length=args.context_length,
        prefill_chunk_length=args.prefill_chunk_length,
        max_layers=args.max_layers,
        split_conv_cache=args.split_conv_cache,
        flash_attention={
            "enable": args.flash_attention,
            "q_bits": args.flash_attention_bits,
            "k_bits": args.flash_attention_bits,
            "v_bits": args.flash_attention_bits,
            "s_bits": args.flash_attention_bits,
            "p_bits": args.flash_attention_bits,
        },
        fuse_gdr_ops=args.fuse_gdr_ops,
        fuse_gdr_block_recurrent_ops=args.fuse_gdr_block_recurrent_ops,
        use_manual_depthwise_conv1d=args.manual_depthwise_conv1d,
        normalize_force_fp32=args.normalize_force_fp32,
        quant_scheme={"quant_type": args.quant_type},
        spec_decode_mode="mtp" if args.mtp else None,
        mtp_config=mtp_config,
        num_draft_tokens=args.num_draft_tokens,
    )


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(output_dir / "export_hmonnx.log"), args.debug)
    set_random_seed(1024)
    logger = get_xhquant_logger()
    config = build_config(args)
    logger.info(f"Qwen3-Next Merak config:\n{config.to_json_string()}")
    model = AutoLLMModel.from_pretrained(config=config)
    with (
        TimeProfiler("qwen3_next_export", logger),
        MemoryTracker([0, 1], "qwen3_next_export", logger),
    ):
        meta = model.export_hmonnx(str(output_dir))
    logger.info(f"golden_meta_info: {meta}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF/GPTQModel checkpoint directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="qwen3-next-80b-a3b-instruct")
    parser.add_argument("--chip-arch", default="XH2a", choices=["XH2a", "YueHui"])
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--prefill-chunk-length", type=int, default=256)
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Export only the first N target layers; use 4 for the reduced validation gate.",
    )
    parser.add_argument("--quant-type", default="w8a8h1_sefp")
    parser.add_argument("--fuse-gdr-ops", action="store_true")
    parser.add_argument("--fuse-gdr-block-recurrent-ops", action="store_true")
    parser.add_argument(
        "--split-conv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use independent q/k/v convolution cache tensors.",
    )
    parser.add_argument("--manual-depthwise-conv1d", action="store_true")
    parser.add_argument(
        "--flash-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export ordinary FlashAttention nodes; required for runtime page-attention conversion.",
    )
    parser.add_argument(
        "--flash-attention-bits",
        type=int,
        choices=[8, 16],
        default=8,
        help="Shared q/k/v/score/probability precision for ordinary FlashAttention export.",
    )
    parser.add_argument("--normalize-force-fp32", action="store_true")
    parser.add_argument("--mtp", action="store_true", help="Export Qwen3NextMTP draft graphs")
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    parser.add_argument("--mtp-head-weight-bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
