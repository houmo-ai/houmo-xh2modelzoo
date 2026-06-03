#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for Qwen3-VL-Embedding HMONNX model.
Tests text embedding extraction using Qwen3VLONNXModel.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

import xhquant.utils.suppress_printing
from xh_model_zoo.xh_llm.models.qwen3_vl import Qwen3VLONNXModel, Qwen3VLProcessor
from xhquant.api import get_root_logger


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_dir", type=str, required=True, help="Path to ONNX model directory")
    parser.add_argument("--hf_model", type=str, required=True, help="Path to HuggingFace model for processor")
    parser.add_argument("--text", type=str, nargs="+", default=["A dog playing in the park"], help="Text(s) to embed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--use_fast", action="store_true", help="Use fast mode")
    parser.add_argument("--model_type", type=str, default="2B", choices=["2B", "4B", "8B"])
    parser.add_argument("--input_sequence_length", type=int, default=256, help="prefill input sequence length")
    parser.add_argument("--cache_len", type=int, default=2048, help="kv cache length")
    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    logger = get_root_logger()
    xhquant.utils.suppress_printing.disable_printing = True

    device = torch.device(args.device)
    model_dir = Path(args.model_dir)

    logger.info(f"Loading model from: {model_dir}")
    logger.info(f"HF model: {args.hf_model}")
    logger.info(f"Device: {device}")

    # Load meta.json
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {model_dir}")

    meta = json.load(open(meta_path, "r"))

    # Model type to num blocks
    MODEL_TYPE_TO_NUM_BLOCKS = {"2B": 28, "4B": 36, "8B": 36}
    blocks_num = MODEL_TYPE_TO_NUM_BLOCKS[args.model_type]

    # Build config
    visual_onnx = str(model_dir / meta.get("vision_onnx", "vision.onnx"))
    prefill_onnx = str(model_dir / meta["prefill_onnx"])
    decode_onnx = str(model_dir / meta.get("decode_onnx", "decode.onnx"))

    image_feature_cfg = SimpleNamespace(
        onnx=visual_onnx,
        patch_size=16,
        image_size_w=448,
        image_size_h=448,
        max_size_t=2,
        temporal_patch_size=2,
    )

    prefill_cfg = SimpleNamespace(
        onnx=prefill_onnx,
        input_sequence_length=args.input_sequence_length,
    )

    decode_cfg = SimpleNamespace(
        onnx=decode_onnx,
    )

    kv_cache_cfg = SimpleNamespace(
        num_decoder_layers=blocks_num,
        num_hidden_layers=blocks_num,
        shape=[1, 8, args.cache_len, 128],
    )

    logger.info(f"Initializing Qwen3VLONNXModel...")

    # Initialize model
    xh_model = Qwen3VLONNXModel(
        image_feature=image_feature_cfg,
        prefill=prefill_cfg,
        decode=decode_cfg,
        kv_cache=kv_cache_cfg,
        image_size_w=448,
        image_size_h=448,
        max_size_t=2,
        resize_v1=True,
        presence_penalty=0.0
    )

    # Load token embedding
    torch.serialization.add_safe_globals([nn.Embedding])
    embedding_key = "quant_embedding_file" if "quant_embedding_file" in meta else "token_embedding_file"
    token_embedding = torch.load(model_dir / meta[embedding_key], weights_only=False, map_location="cpu")
    torch.serialization.clear_safe_globals()

    xh_model.set_input_embeddings(token_embedding)
    xh_model.set_exec_device(args.device)

    # Load processor from HF model path
    processor = Qwen3VLProcessor.from_pretrained(args.hf_model)

    logger.info("Model loaded successfully!")
    logger.info("")

    # Embed texts
    texts = args.text
    logger.info(f"=== Embedding {len(texts)} text(s) ===")

    embeddings = xh_model.embed_texts(texts, processor, use_fast=args.use_fast)

    for i, text in enumerate(texts):
        logger.info(f"\n[{i+1}/{len(texts)}] Text: '{text}'")
        logger.info(f"  Embedding shape: {embeddings[i].shape}")
        logger.info(f"  Embedding norm: {embeddings[i].norm().item():.6f}")
        logger.info(f"  First 10 dims: {embeddings[i, :10].tolist()}")

    # Compute similarity matrix if multiple texts
    if len(texts) > 1:
        logger.info("\n=== Similarity Matrix ===")
        similarity_matrix = torch.mm(embeddings, embeddings.t())

        logger.info("\nSimilarity matrix:")
        for i in range(len(texts)):
            row_str = f"  [{i}] "
            for j in range(len(texts)):
                row_str += f"{similarity_matrix[i, j].item():.4f} "
            logger.info(row_str)

        logger.info("\nText labels:")
        for i, text in enumerate(texts):
            logger.info(f"  [{i}] {text}")

    logger.info("\n=== Demo completed ===")


if __name__ == "__main__":
    main()
