# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: qwen3_vl_embedding_xh2a_export_hmonnx.py
# Description:
#   Export Qwen3-VL-Embedding backbone (Qwen3VLModel: visual + language) to
#   XH2a / HMONNX. Pooling (last-token) and L2 normalize are kept outside of
#   the ONNX graph (done in Python at inference time).
#
#   The model's config.architectures is Qwen3VLForConditionalGeneration, but
#   the embedding checkpoint only carries the backbone weights and ties
#   lm_head to embed_tokens. We reuse xh_model_zoo.xh_llm.models.qwen3_vl
#   verbatim and let LLMConverter handle the lm_head tie path the same way
#   as the generative qwen3-vl export does.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from pathlib import Path

from xh_model_zoo.xh_llm.models.qwen3_vl import (
    Qwen3_VLConvertConfig,
    Qwen3_VLEmbeddingConverterXH2a,
    VisualConfig,
)

from xhquant.api import DeviceType, QuantScheme, get_root_logger, xhquant_init  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


DEFAULT_LOCAL_2B = (
    "/data01/home/she.gao/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-Embedding-2B/snapshots/"
    "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
)


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type

    ops = dict(
        MatMul=dict(
            act_scheme=dict(bits=8, fp_mode="sefp"),
            act_schema_2=dict(bits=16, fp_mode="sefp"),
        )
    )
    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type, ops=ops)

    config = Qwen3_VLConvertConfig(
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
            sample_video_path=args.sample_video_path,
        ),
    )

    folder = args.tag or f"{model_name}-{target_device}-{args.context_length // 1024}k-{quant_type}"
    work_dir = Path("work_dirs") / folder
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    logger.info(f"Exporting Qwen3-VL-Embedding backbone from {hf_model_path}")
    logger.info(f"Output directory: {work_dir}")

    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        Qwen3_VLEmbeddingConverterXH2a.convert(hf_model_path, config, work_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LOCAL_2B,
        help="HF Qwen3-VL-Embedding model directory",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--max_pe_length", type=int, default=32768)
    parser.add_argument(
        "--quant-type",
        default="w8a8h1_sefp",
        help="quant type (default w8a8h1_sefp; switch to w4a8 with quant-weight)",
    )
    parser.add_argument("--image_max_size_h", type=int, default=448)
    parser.add_argument("--image_max_size_w", type=int, default=448)
    parser.add_argument(
        "--image_max_size_t",
        type=int,
        default=2,
        help="image: 2 temporal patches; video: fps-driven",
    )
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--temporal_patch_size", type=int, default=2)
    parser.add_argument(
        "--sample_image_path",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo_new/data/images/qwen2_vl_demo.jpeg",
        help="sample image for generating the golden tensor",
    )
    parser.add_argument(
        "--sample_video_path",
        type=str,
        default="",
        help="sample video; leave empty for image-only export",
    )
    parser.add_argument("--use_gptqmodel", action="store_true")
    parser.add_argument(
        "--quant_weight",
        type=str,
        default=None,
        help="optional gptq/quarot state dict; if empty, FP weights are used",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="optional output folder name; overrides the default <name>-<device>-<ctx>k-<qt>",
    )
    args = parser.parse_args()
    main(args)
