# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: qwen3_vl_embedding_eval_flickr30k.py
# Description:
#   Standard Flickr30K retrieval evaluation for Qwen3-VL-Embedding (HF BF16).
#   Loads the 1K test split (5 captions per image) and reports
#   Recall@1/5/10, MRR@10, nDCG@10 for both directions:
#     - text->image  (5000 queries, 1000 candidates, 1 GT per query)
#     - image->text  (1000 queries, 5000 candidates, 5 GT per query)
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
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from loguru import logger
from PIL import Image


DEFAULT_LOCAL_2B = (
    "/data01/home/she.gao/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-Embedding-2B/snapshots/"
    "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
)


def load_qwen3_vl_embedder(model_dir: str):
    script_path = Path(model_dir) / "scripts" / "qwen3_vl_embedding.py"
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qwen3_vl_embedding"] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder


def _read_local_flickr30k(dataset_dir: Path) -> List[Tuple[str, List[str]]]:
    """Return [(image_abs_path, [cap1..cap5]), ...] from a local Flickr30K dump.

    Layout A (Kaggle "hsankesara/flickr-image-dataset"):
        <dataset_dir>/flickr30k_images/*.jpg
        <dataset_dir>/results.csv  (image_name| comment_number| comment)

    Layout B (HF nlphuji/flickr30k):
        <dataset_dir>/flickr30k-images/*.jpg
        <dataset_dir>/flickr_annotations_30k.csv  (raw, sentids, split, filename, img_id)

    Layout C:
        <dataset_dir>/flickr30k-images/*.jpg
        <dataset_dir>/captions.json   {"image_name.jpg": ["cap1", ...]}
    """
    img_dirs = [
        dataset_dir / "flickr30k-images",
        dataset_dir / "flickr30k_images",
        dataset_dir / "images",
    ]
    img_dir = next((d for d in img_dirs if d.is_dir()), None)
    if img_dir is None:
        return []

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

    results_csv = dataset_dir / "results.csv"
    if results_csv.is_file():
        by_image: dict = {}
        with open(results_csv, "r", encoding="utf-8") as f:
            sample_line = f.readline()
            f.seek(0)
            delimiter = "|" if "|" in sample_line else ","
            reader = csv.reader(f, delimiter=delimiter)
            header = [c.strip() for c in next(reader)]
            try:
                name_idx = header.index("image_name")
            except ValueError:
                name_idx = 0
            try:
                cap_idx = header.index("comment")
            except ValueError:
                cap_idx = len(header) - 1
            for row in reader:
                if len(row) <= max(name_idx, cap_idx):
                    continue
                name = row[name_idx].strip()
                cap = row[cap_idx].strip()
                if not cap:
                    continue
                p = img_dir / name
                if p.is_file():
                    by_image.setdefault(str(p), []).append(cap)
        return list(by_image.items())

    captions_json = dataset_dir / "captions.json"
    if captions_json.is_file():
        data = json.load(open(captions_json, "r", encoding="utf-8"))
        out = []
        for name, caps in data.items():
            p = img_dir / name
            caps = [c for c in caps if isinstance(c, str) and c]
            if p.is_file() and caps:
                out.append((str(p), caps))
        return out

    return []


def _read_hf_flickr30k_test() -> List[Tuple[Image.Image, List[str]]]:
    """Load the standard 1K test split of nlphuji/flickr30k."""
    from datasets import load_dataset

    ds = load_dataset("nlphuji/flickr30k", split="test")
    out: List[Tuple[Image.Image, List[str]]] = []
    for row in ds:
        split = row.get("split", "test")
        if split != "test":
            continue
        img = row["image"]
        caps = row.get("caption") or row.get("captions") or []
        if isinstance(caps, str):
            caps = [caps]
        caps = [c for c in caps if isinstance(c, str) and c]
        if img is not None and caps:
            out.append((img, caps))
    return out


def load_flickr30k_test(dataset_dir: str):
    """Return [(image, [cap1..capN]), ...] for the full 1K test split."""
    if dataset_dir:
        rows = _read_local_flickr30k(Path(dataset_dir))
        if rows:
            return rows
        logger.warning(
            f"--dataset-dir={dataset_dir!r} did not yield Flickr30K rows; "
            "falling back to HF datasets cache."
        )
    try:
        return _read_hf_flickr30k_test()
    except Exception as exc:
        raise RuntimeError(
            "Flickr30K not found. Either point --dataset-dir to a local dump "
            "containing 'flickr30k_images/' + 'results.csv', or pre-cache "
            f"'nlphuji/flickr30k' via `datasets.load_dataset`. Cause: {exc}"
        ) from exc


def _image_to_arg(img):
    if isinstance(img, str):
        return img
    if isinstance(img, Image.Image):
        return img
    raise TypeError(f"Unsupported image type for Qwen3VLEmbedder: {type(img)}")


def embed_in_batches(embedder, items, batch_size: int, tag: str) -> torch.Tensor:
    parts = []
    t0 = time.time()
    n = len(items)
    for s in range(0, n, batch_size):
        chunk = items[s : s + batch_size]
        with torch.no_grad():
            emb = embedder.process(chunk, normalize=True)
        parts.append(emb.float().cpu())
        done = s + len(chunk)
        if done == n or done % (batch_size * 20) == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            logger.info(f"[{tag}] {done}/{n}  ({rate:.1f} samples/s)")
    return torch.cat(parts, dim=0)


def retrieval_metrics(query: torch.Tensor, doc: torch.Tensor,
                      gt_doc_idx_list: List[List[int]],
                      recall_ks=(1, 5, 10), mrr_k=10, ndcg_k=10) -> dict:
    """Compute Recall@K, MRR@K, nDCG@K with binary relevance.

    `gt_doc_idx_list[i]` is the set of ground-truth doc indices for query i
    (one element for text->image; up to 5 for image->text on Flickr30K).
    """
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
    model_path = args.model_dir
    assert Path(model_path).exists(), f"HF model not found: {model_path}"

    logger.info(f"Loading Flickr30K test split (dataset_dir={args.dataset_dir!r}) ...")
    rows = load_flickr30k_test(args.dataset_dir)
    logger.info(f"Loaded {len(rows)} images with captions")

    if args.max_images:
        rows = rows[: args.max_images]
        logger.warning(f"--max-images={args.max_images} active; eval is NOT full 1K standard")

    images = []
    captions_flat: List[str] = []
    cap_owner: List[int] = []
    image_to_caps: List[List[int]] = []
    for img_idx, (img, caps) in enumerate(rows):
        images.append(_image_to_arg(img))
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

    Qwen3VLEmbedder = load_qwen3_vl_embedder(model_path)
    embedder = Qwen3VLEmbedder(model_name_or_path=model_path, torch_dtype=torch.bfloat16)
    embedder.model.eval()
    logger.info(f"Loaded Qwen3VLEmbedder on {embedder.model.device}")

    text_items = [{"text": c} for c in captions_flat]
    image_items = [{"image": img} for img in images]

    text_emb = embed_in_batches(embedder, text_items, args.text_batch_size, "text")
    image_emb = embed_in_batches(embedder, image_items, args.image_batch_size, "image")

    cap_owner_t = torch.tensor(cap_owner, dtype=torch.long)
    text_gt = [[int(o)] for o in cap_owner_t.tolist()]

    logger.info("Computing text->image metrics ...")
    t2i = retrieval_metrics(text_emb, image_emb, text_gt)

    logger.info("Computing image->text metrics ...")
    i2t = retrieval_metrics(image_emb, text_emb, image_to_caps)

    report = {
        "model_dir": model_path,
        "dataset": "Flickr30K (test split)",
        "num_images": len(images),
        "num_captions": len(captions_flat),
        "text_to_image": t2i,
        "image_to_text": i2t,
    }

    out_path = Path(args.report) if args.report else Path("work_dirs") / "flickr30k_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("===== Flickr30K Retrieval =====")
    fmt = lambda d: "  ".join(f"{k}={v:.4f}" for k, v in d.items())
    logger.info(f"text -> image  {fmt(t2i)}")
    logger.info(f"image -> text  {fmt(i2t)}")
    logger.info(f"Wrote report to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_LOCAL_2B,
        help="Path to Qwen3-VL-Embedding HF snapshot",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Local Flickr30K root; if absent, falls back to nlphuji/flickr30k HF cache",
    )
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="0 = full 1K test (standard)")
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()
    main(args)
