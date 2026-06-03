# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: qwen3_vl_embedding_eval_flickr30k_hmonnx_aligned.py
# Description:
#   Flickr30K retrieval evaluation for HMONNX model using the
#   Qwen3VLONNXModel embedding API (embed_texts / embed_images).
#   Tests both text-to-image and image-to-text retrieval.
#   Reports Recall@1/5/10, MRR@10, nDCG@10 for both directions.
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
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import torch
import torch.nn as nn

import xhquant.utils.suppress_printing
from xh_model_zoo.xh_llm.models.qwen3_vl import Qwen3VLONNXModel, Qwen3VLProcessor
from xhquant.api import get_root_logger, xhquant_init

MODEL_TYPE_TO_NUM_BLOCKS = {"2B": 28, "4B": 36, "8B": 36}


def _read_local_flickr30k(dataset_dir: Path) -> List[Tuple[str, List[str]]]:
    """Return [(image_abs_path, [cap1..cap5]), ...] from a local Flickr30K dump."""
    img_dirs = [
        dataset_dir / "flickr30k-images",
        dataset_dir / "flickr30k_images",
        dataset_dir / "images",
    ]
    img_dir = next((d for d in img_dirs if d.is_dir()), None)
    if img_dir is None:
        return []

    # Try flickr_annotations_30k.csv format (HF nlphuji/flickr30k format)
    flickr_annotations = dataset_dir / "flickr_annotations_30k.csv"
    if flickr_annotations.is_file():
        out = []
        with open(flickr_annotations, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row.get("split", "").strip()
                if split != "test":
                    continue
                filename = row.get("filename", "").strip()
                raw = row.get("raw", "").strip()
                if not filename or not raw:
                    continue
                p = img_dir / filename
                if not p.is_file():
                    continue
                try:
                    caps = json.loads(raw)
                    if isinstance(caps, list):
                        caps = [c for c in caps if isinstance(c, str) and c]
                        if caps:
                            out.append((str(p), caps))
                except Exception:
                    continue
        return out

    return []


def build_xh_model(meta_path: Path, args, logger) -> Tuple[Qwen3VLONNXModel, Qwen3VLProcessor, str]:
    """Build Qwen3VLONNXModel from meta.json (same as the demo)."""
    model_dir = meta_path.parent
    meta = json.load(open(meta_path, "r"))

    hf_model_path = args.hf_model or meta.get("hf_model_path")
    if not hf_model_path:
        raise ValueError("hf_model not provided and hf_model_path missing in meta.json")

    blocks_num = MODEL_TYPE_TO_NUM_BLOCKS[args.model_type]

    visual_onnx = str(model_dir / meta.get("vision_onnx", "vision.onnx"))
    prefill_onnx = str(model_dir / meta["prefill_onnx"])
    decode_onnx = str(model_dir / meta.get("decode_onnx", "decode.onnx"))

    input_seq_len = meta["wrap_cfg"]["input_sequence_length"]
    cache_len = meta["wrap_cfg"]["max_sequence_length"]
    visual_cfg = meta["wrap_cfg"].get("visual", {})
    image_size_h = visual_cfg.get("image_max_size_h", 448)
    image_size_w = visual_cfg.get("image_max_size_w", 448)
    max_size_t = visual_cfg.get("image_max_size_t", 2)

    image_feature_cfg = SimpleNamespace(
        onnx=visual_onnx,
        patch_size=visual_cfg.get("patch_size", 16),
        image_size_w=image_size_w,
        image_size_h=image_size_h,
        max_size_t=max_size_t,
        temporal_patch_size=visual_cfg.get("temporal_patch_size", 2),
    )
    prefill_cfg = SimpleNamespace(onnx=prefill_onnx, input_sequence_length=input_seq_len)
    decode_cfg = SimpleNamespace(onnx=decode_onnx)
    kv_cache_cfg = SimpleNamespace(
        num_decoder_layers=blocks_num,
        num_hidden_layers=blocks_num,
        shape=[1, 8, cache_len, 128],
    )

    logger.info("Initializing Qwen3VLONNXModel...")
    xh_model = Qwen3VLONNXModel(
        image_feature=image_feature_cfg,
        prefill=prefill_cfg,
        decode=decode_cfg,
        kv_cache=kv_cache_cfg,
        image_size_w=image_size_w,
        image_size_h=image_size_h,
        max_size_t=max_size_t,
        resize_v1=True,
        presence_penalty=0.0,
    )

    # Load token embedding
    torch.serialization.add_safe_globals([nn.Embedding])
    embedding_key = "quant_embedding_file" if "quant_embedding_file" in meta else "token_embedding_file"
    token_embedding = torch.load(model_dir / meta[embedding_key], weights_only=False, map_location="cpu")
    torch.serialization.clear_safe_globals()

    xh_model.set_input_embeddings(token_embedding)
    xh_model.set_exec_device(args.device)

    processor = Qwen3VLProcessor.from_pretrained(hf_model_path)
    return xh_model, processor, hf_model_path


def retrieval_metrics(query: torch.Tensor, doc: torch.Tensor,
                      gt_doc_idx_list: List[List[int]],
                      recall_ks=(1, 5, 10), mrr_k=10, ndcg_k=10) -> dict:
    """Compute Recall@K, MRR@K, nDCG@K with binary relevance."""
    scores = query @ doc.T
    max_k = max(max(recall_ks), mrr_k, ndcg_k)
    topk_idx = scores.topk(min(max_k, scores.shape[1]), dim=-1).indices.tolist()

    n = scores.shape[0]
    recall = {f"@{k}": 0.0 for k in recall_ks}
    mrr_sum = 0.0
    ndcg_sum = 0.0

    log2 = torch.log2(torch.arange(2, max_k + 2, dtype=torch.float64))

    for i in range(n):
        gt = set(gt_doc_idx_list[i])
        row = topk_idx[i]

        for k in recall_ks:
            if any(r in gt for r in row[:k]):
                recall[f"@{k}"] += 1.0

        rr = 0.0
        for rank, doc_id in enumerate(row[:mrr_k], start=1):
            if doc_id in gt:
                rr = 1.0 / rank
                break
        mrr_sum += rr

        dcg = 0.0
        for rank_idx, doc_id in enumerate(row[:ndcg_k]):
            if doc_id in gt:
                dcg += 1.0 / float(log2[rank_idx])
        ideal_hits = min(len(gt), ndcg_k)
        idcg = sum(1.0 / float(log2[r]) for r in range(ideal_hits)) if ideal_hits > 0 else 1.0
        ndcg_sum += dcg / idcg if idcg > 0 else 0.0

    out = {f"recall@{k.strip('@')}": v / max(n, 1) for k, v in recall.items()}
    out[f"mrr@{mrr_k}"] = mrr_sum / max(n, 1)
    out[f"ndcg@{ndcg_k}"] = ndcg_sum / max(n, 1)
    return out


def main(args):
    xhquant_init(None, args.debug)
    logger = get_root_logger()
    xhquant.utils.suppress_printing.disable_printing = True

    meta_path = Path(args.hmonnx_config)
    assert meta_path.exists(), f"HMONNX config not found: {meta_path}"

    dataset_dir = Path(args.dataset_dir)
    assert dataset_dir.exists(), f"Dataset directory not found: {dataset_dir}"

    logger.info(f"Loading Flickr30K from {dataset_dir} ...")
    rows = _read_local_flickr30k(dataset_dir)
    logger.info(f"Loaded {len(rows)} images with captions")

    if not rows:
        raise RuntimeError(f"No Flickr30K data found in {dataset_dir}")

    if args.max_images:
        rows = rows[: args.max_images]
        logger.warning(f"--max-images={args.max_images} active; eval is NOT full dataset")

    # Extract images and captions
    images: List[str] = []
    captions_flat: List[str] = []
    cap_owner: List[int] = []
    image_to_caps: List[List[int]] = []

    for img_idx, (img_path, caps) in enumerate(rows):
        images.append(img_path)
        own_caps = []
        for c in caps:
            cap_id = len(captions_flat)
            captions_flat.append(c)
            cap_owner.append(img_idx)
            own_caps.append(cap_id)
        image_to_caps.append(own_caps)

    logger.info(
        f"#images={len(images)}  #captions={len(captions_flat)}  "
        f"(avg {len(captions_flat) / max(len(images), 1):.2f} cap/img)"
    )

    # Build model with the demo interface
    xh_model, processor, hf_model_path = build_xh_model(meta_path, args, logger)

    # Embed texts with HMONNX via embed_text() (keep_session=True reuses the
    # prefill ONNX across all captions for speed)
    logger.info("Embedding texts with HMONNX model...")
    t0 = time.time()
    text_emb_list = []
    for i in range(0, len(captions_flat)):
        emb = xh_model.embed_text(captions_flat[i], processor, use_fast=args.fast, keep_session=True)
        text_emb_list.append(emb.cpu())
        if (i + 1) % 100 == 0 or (i + 1) == len(captions_flat):
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            logger.info(f"[HMONNX text] {i + 1}/{len(captions_flat)}  ({rate:.1f} samples/s)")
    xh_model.release_prefill_session()
    text_emb = torch.cat(text_emb_list, dim=0)

    # Embed images with HMONNX via embed_image() (keep_session=True reuses the
    # vision + prefill ONNX across all images for speed)
    logger.info("Embedding images with HMONNX model...")
    t0 = time.time()
    image_emb_list = []
    for i in range(0, len(images)):
        emb = xh_model.embed_image(images[i], processor, use_fast=args.fast, keep_session=True)
        image_emb_list.append(emb.cpu())
        if (i + 1) % 20 == 0 or (i + 1) == len(images):
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            logger.info(f"[HMONNX image] {i + 1}/{len(images)}  ({rate:.1f} samples/s)")
    xh_model.release_image_feature()
    xh_model.release_prefill_session()
    image_emb = torch.cat(image_emb_list, dim=0)

    # Text-to-Image retrieval: each caption queries for its image (1 GT per query)
    cap_owner_t = torch.tensor(cap_owner, dtype=torch.long)
    text_gt = [[int(o)] for o in cap_owner_t.tolist()]

    logger.info("Computing text->image metrics (HMONNX)...")
    t2i = retrieval_metrics(text_emb, image_emb, text_gt)

    # Image-to-Text retrieval: each image queries for its captions (5 GT per query)
    logger.info("Computing image->text metrics (HMONNX)...")
    i2t = retrieval_metrics(image_emb, text_emb, image_to_caps)

    report = {
        "hmonnx_config": str(meta_path),
        "hf_model": hf_model_path,
        "dataset": f"Flickr30K ({dataset_dir})",
        "num_images": len(images),
        "num_captions": len(captions_flat),
        "text_to_image": t2i,
        "image_to_text": i2t,
    }

    out_path = Path(args.report) if args.report else Path("work_dirs") / "flickr30k_hmonnx_aligned_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("===== Flickr30K Retrieval (HMONNX) =====")
    fmt = lambda d: "  ".join(f"{k}={v:.4f}" for k, v in d.items())
    logger.info(f"text -> image  {fmt(t2i)}")
    logger.info(f"image -> text  {fmt(i2t)}")
    logger.info(f"Wrote report to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hmonnx-config",
        type=str,
        required=True,
        help="Path to HMONNX meta.json",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="Path to HF model (for processor)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Local Flickr30K root containing flickr30k-images/ and flickr_annotations_30k.csv",
    )
    parser.add_argument("--model-type", type=str, default="2B", choices=["2B", "4B", "8B"])
    parser.add_argument("--max-images", type=int, default=0, help="0 = use all images")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()
    main(args)
