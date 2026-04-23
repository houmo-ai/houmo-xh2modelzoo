"""
Qwen3.5-4B VL hmonnx 量化模型推理测试。

使用重新量化的模型，通过 hmonnx 推理引擎测试 VL 多模态推理结果。
用于 CMS-477 排查：验证量化模型结果是否正确。

CMS-477 复现场景：
  - 图片: data/images/cms477_0.jpg (3840x2160 监控/航拍图)
  - Prompt: "图中有什么？"
  - repetition_penalty: 1.1 / 1.2

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python examples/llm/qwen3_5/qwen3_5_4b_vl_hmonnx_test.py

    # CMS-477 复现
    CUDA_VISIBLE_DEVICES=0,1 python examples/llm/qwen3_5/qwen3_5_4b_vl_hmonnx_test.py \
        --image-path data/images/cms477_0.jpg --prompt "图中有什么？" \
        --repetition-penalty 1.2
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import xhquant.utils.suppress_printing
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoTokenizer
from transformers.image_processing_utils import BatchFeature
from transformers.video_processing_utils import BaseVideoProcessor

from xh_model_zoo.api import ConfigDict, get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.qwen3_5 import Qwen3_5Processor
from xh_model_zoo.xh_llm.models.qwen3_5.image_processing_qwen3_5 import Qwen3_5ImageProcessor

# Import VL model from demo
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen3_5_vl_hmonnx_demo import (
    Qwen35VLHMONNXModel,
    _build_processor,
    _load_token_embedding,
    _parse_auto_offload_max_memory,
    _parse_dtype,
    _prepare_multimodal_inputs,
    _resolve_path,
    _resolve_processor_source,
    parse_arguments as _vl_parse_arguments,
)


def parse_arguments():
    parser = _vl_parse_arguments()
    parser.description = "Qwen3.5-4B VL hmonnx quantized model test (CMS-477)"
    # Override defaults for testing
    parser.set_defaults(
        model_config=(
            "work_dirs/qwen3_5_27b/qwen3_5_4b_xh2a_Qwen3/export_meta_info.json"
        ),
        vision_onnx=(
            "work_dirs/qwen3_5_4B/"
            "qwen3_5_instruct_vision_config_1_2_448_448_use_gptq_model_False_Qwen3/"
            "vision/qwen3_5_instruct_vision_config.onnx"
        ),
        image_path="data/images/cms477_0.jpg",
        prompt="图中有什么？",
        max_new_tokens=512,
        stream_output=True,
        do_sample=False,
    )
    return parser


def main():
    args = parse_arguments().parse_args()

    model_meta_file_path = Path(args.model_config).resolve()
    model_dir = model_meta_file_path.parent
    meta_info = json.load(open(model_meta_file_path, "r", encoding="utf-8"))

    prefill_onnx = _resolve_path(model_dir, meta_info["prefill_onnx_file"])
    decode_onnx = _resolve_path(model_dir, meta_info["decode_onnx_file"])
    hf_model_config_dir = _resolve_path(model_dir, meta_info["hf_config"])
    embed_tokens_file = _resolve_path(model_dir, meta_info["token_embedding_file"])
    vision_onnx = Path(args.vision_onnx).resolve()
    hf_model_dir = meta_info.get("hf_model", None)
    processor_source = _resolve_processor_source(hf_model_config_dir, hf_model_dir)

    max_context_tokens = args.max_context_tokens
    if max_context_tokens is None:
        kv_cache_shape = meta_info.get("kv_cache_shape", None)
        if isinstance(kv_cache_shape, list) and len(kv_cache_shape) >= 3:
            max_context_tokens = int(kv_cache_shape[2])

    cfg_name = "qwen3_5_4b_vl_hmonnx_test"
    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / f"{cfg_name}.log"
    xhquant_llm_init(log_file, args.debug)
    logger = get_root_logger()

    xhquant.utils.suppress_printing.disable_printing = True

    dtype = _parse_dtype(args.dtype)
    auto_offload_max_memory = _parse_auto_offload_max_memory(args.auto_offload_max_memory)
    prefill_auto_offload_max_memory = _parse_auto_offload_max_memory(args.prefill_auto_offload_max_memory)
    decode_auto_offload_max_memory = _parse_auto_offload_max_memory(args.decode_auto_offload_max_memory)

    processor = _build_processor(processor_source, args)
    model_config = AutoConfig.from_pretrained(str(hf_model_config_dir), trust_remote_code=True)
    token_embedding = _load_token_embedding(embed_tokens_file).to(dtype=dtype)

    xh_model = Qwen35VLHMONNXModel(
        vision_onnx=str(vision_onnx),
        image_token_id=getattr(processor, "image_token_id", getattr(model_config, "image_token_id", 248056)),
        video_token_id=getattr(processor, "video_token_id", getattr(model_config, "video_token_id", 248057)),
        vision_start_token_id=getattr(
            processor,
            "vision_start_token_id",
            getattr(model_config, "vision_start_token_id", 248053),
        ),
        spatial_merge_size=getattr(getattr(model_config, "vision_config", None), "spatial_merge_size", 2),
        prefill=ConfigDict(onnx=str(prefill_onnx)),
        decode=ConfigDict(onnx=str(decode_onnx)),
        max_context_tokens=max_context_tokens,
        auto_offload=not args.disable_auto_offload,
        auto_offload_max_memory=auto_offload_max_memory,
        prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
        decode_auto_offload_max_memory=decode_auto_offload_max_memory,
        resource_tight_mode=args.resource_tight_mode,
        pad_token_id=(
            processor.tokenizer.pad_token_id
            if processor.tokenizer.pad_token_id is not None
            else processor.tokenizer.eos_token_id
        ),
    )
    xh_model.set_input_embeddings(token_embedding)

    if args.disable_auto_offload:
        xh_model.to(torch.device(args.device))
    else:
        target_device = torch.device(args.device)
        xh_model._device = target_device
        if xh_model.token_embedding is not None:
            xh_model.token_embedding.to(target_device)

    xh_model.set_exec_device(torch.device(args.exec_device))
    if args.disable_auto_offload:
        xh_model.to(dtype)
    else:
        if xh_model.token_embedding is not None:
            xh_model.token_embedding.to(dtype)
        xh_model._dtype = dtype

    print("=" * 60)
    print(f"[Config] vision={vision_onnx}")
    print(f"[Config] prefill={prefill_onnx}")
    print(f"[Config] decode={decode_onnx}")
    print(f"[Config] image={Path(args.image_path).resolve()}")
    print(f"[Config] prompt={args.prompt!r}")
    print(f"[Config] max_new_tokens={args.max_new_tokens}, stream={args.stream_output}")
    print(f"[Config] do_sample={args.do_sample}, temp={args.temperature}")
    print("=" * 60, flush=True)

    # Run inference
    t0 = time.perf_counter()
    if args.stream_output:
        print("[Stream Output]:", flush=True)
    out = xh_model.chat(
        prompt=args.prompt,
        image_path=args.image_path,
        processor=processor,
        args=args,
        history=None,
        system_prompt=args.system_prompt,
    )
    elapsed = time.perf_counter() - t0

    if args.stream_output:
        print("", flush=True)

    num_tokens = len(processor.tokenizer.encode(out, add_special_tokens=False)) if out else 0
    tps = num_tokens / elapsed if elapsed > 0 else 0

    print("=" * 60)
    print(f"[Result] hmonnx VL output ({num_tokens} tokens, {elapsed:.2f}s, {tps:.1f} tok/s):")
    print(out)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
