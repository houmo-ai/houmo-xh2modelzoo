# Copyright 2025 HOUMO AI
#
# File: qwen3_5_moe_xh2a_export_hmonnx.py
# Description:
#   Export script: Qwen3.5-MoE LLM -> prefill/decode HMONNX via LLMConverter.
#
# Usage (float weights):
#   python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
#       --model /data01/nfs_shared/Qwen3.5-35B-A3B \
#       --context-length 2048 --input-sequence-length 256 \
#       --quant-type w8a8h0_ssfp
#
# Usage (GPTQModel weights):
#   python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
#       --model /data01/nfs_shared/Qwen3.5-35B-A3B \
#       --quant-weight /data01/home/huxing/gptqmodel/work_dirs/Qwen35_35B_A3B_attn4_e4_se4_0324 \
#       --quant-type w4a8h0_ssfp \
#       --context-length 2048 --input-sequence-length 256
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os.path as osp
import time
from pathlib import Path

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen3_5_moe import Qwen3_5MoeConvertConfig
from xh_model_zoo.xh_llm.models.qwen3_5_moe.qwen3_5_moe_converter import Qwen3_5MoeConverterXH2a


from xhquant.api import DeviceType, QuantScheme, get_root_logger, xhquant_init  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def _build_draft_only_default_work_dir(
    existing_work_dir: Path, spec_decode_mode: str
) -> Path:
    existing_work_dir = existing_work_dir.resolve()
    suffix = f"draft_{spec_decode_mode}"
    base = existing_work_dir.with_name(f"{existing_work_dir.name}-{suffix}")
    if not base.exists() and base.resolve() != existing_work_dir:
        return base
    return existing_work_dir.with_name(f"{existing_work_dir.name}-{suffix}-{time.strftime('%Y%m%d%H%M%S')}")


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)

    spec_decode_mode = args.spec_decode_mode or None
    if spec_decode_mode == "none":
        spec_decode_mode = None
        args.spec_decode_mode = None
    num_draft_tokens = args.num_draft_tokens

    config = Qwen3_5MoeConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        num_logits_to_keep=args.num_logits_to_keep,
        linear_attention_mode=args.linear_attention_mode,
        linear_chunk_size=args.linear_chunk_size,
        spec_decode_mode=spec_decode_mode,
        num_draft_tokens=num_draft_tokens,
        dflash_model_dir=args.dflash_model_dir,
    )

    if args.draft_only:
        if args.existing_work_dir is None:
            raise ValueError("--draft-only requires --existing-work-dir")
        if spec_decode_mode not in {"mtp", "dflash"}:
            raise ValueError("--draft-only requires --spec-decode-mode to be one of {'mtp', 'dflash'}")
        if args.work_dir:
            work_dir = Path(args.work_dir)
        else:
            work_dir = _build_draft_only_default_work_dir(
                Path(args.existing_work_dir),
                spec_decode_mode,
            )
        if work_dir.resolve() == Path(args.existing_work_dir).resolve():
            raise ValueError("--draft-only --work-dir must not overwrite --existing-work-dir")
    elif args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        prefix = f"{model_name}-{target_device}-{args.context_length // 1024}k-{quant_type}"
        if args.quant_weight:
            prefix += "-gptq"
        if spec_decode_mode:
            prefix += f"-spec_{spec_decode_mode}"
        work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    logger.info(f"model: {hf_model_path}")
    logger.info(f"quant_weight: {args.quant_weight}")
    logger.info(f"spec_decode_mode: {spec_decode_mode}")
    logger.info(f"output: {work_dir}")

    if args.draft_only:
        with TimeProfiler("draft-only convert", logger), MemoryTracker("cuda:0", "draft-only convert", logger):
            Qwen3_5MoeConverterXH2a(config).export_draft_only(
                hf_model_path,
                args.existing_work_dir,
                str(work_dir),
            )
        logger.info(f"Done. Draft-only artifacts in: {work_dir}")
        return

    # Detect architecture from config.json automatically
    architecture = args.architecture  # may be None → auto-detect

    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, architecture, config, str(work_dir))

    logger.info(f"Done. Artifacts in: {work_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Qwen3.5-MoE to prefill/decode HMONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="/data01/nfs_shared/Qwen3.5-35B-A3B", help="HuggingFace model directory"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help="Architecture string (auto-detected if None). "
        "Use 'Qwen3_5MoeForConditionalGeneration' or 'Qwen3_5MoeForCausalLM'",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export model inputs")
    parser.add_argument("--context-length", type=int, default=2048, help="Maximum context length (kv cache size)")
    parser.add_argument("--input-sequence-length", type=int, default=256, help="Prefill chunk size")
    parser.add_argument("--quant-type", type=str, default="w8a8h0_sefp", help="Quantisation type string")
    parser.add_argument("--quant-weight", type=str, default=None, help="Path to GPTQModel quantised weights (optional)")
    parser.add_argument(
        "--work-dir",
        "--work_dir",
        dest="work_dir",
        type=str,
        default=None,
        help="Output work directory. Defaults to the standard export prefix, or a non-overwriting draft-only sibling.",
    )
    parser.add_argument(
        "--num-logits-to-keep", type=int, default=1, help="How many final logit positions to keep (1 = last only)"
    )
    parser.add_argument(
        "--linear-attention-mode",
        type=str,
        default="auto",
        choices=["auto", "chunk", "recurrent"],
        help="Linear attention computation mode",
    )
    parser.add_argument("--linear-chunk-size", type=int, default=64, help="Chunk size for linear attention")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--draft-only",
        "--draft_only",
        dest="draft_only",
        action="store_true",
        help="Reuse target artifacts from --existing-work-dir and export only MTP/DFlash draft ONNX/meta.",
    )
    parser.add_argument(
        "--existing-work-dir",
        "--existing_work_dir",
        dest="existing_work_dir",
        type=str,
        default=None,
        help="Existing target export work_dir containing meta.json for --draft-only.",
    )
    # Speculative decoding
    parser.add_argument(
        "--spec-decode-mode",
        "--spec_decode_mode",
        dest="spec_decode_mode",
        type=str,
        default=None,
        choices=["none", "mtp", "dflash"],
        help="Speculative decoding mode.  'mtp' exports MTP draft graphs; "
        "'dflash' exports DFlash context/decode draft graphs.",
    )
    parser.add_argument(
        "--dflash-model-dir",
        "--dflash_model_dir",
        dest="dflash_model_dir",
        type=str,
        default=None,
        help="Path to DFlash draft model dir (required for --spec-decode-mode dflash)",
    )
    parser.add_argument(
        "--num-draft-tokens",
        "--num_draft_tokens",
        dest="num_draft_tokens",
        type=int,
        default=4,
        help=(
            "Number of draft tokens per spec-decode round (verify_length = N + 1). "
            "For DFlash the draft decode input length is also verify_length."
        ),
    )
    args = parser.parse_args()
    if args.spec_decode_mode == "none":
        args.spec_decode_mode = None
    main(args)
