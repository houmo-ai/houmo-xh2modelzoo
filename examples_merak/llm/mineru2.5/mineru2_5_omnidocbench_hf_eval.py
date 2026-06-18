#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate native HF MinerU2.5-Pro on a deterministic OmniDocBench subset.

This is a MinerU-oriented extraction evaluation:
image -> MinerUClient.two_step_extract -> concatenate extracted contents,
then compare the page-level extracted string against OmniDocBench annotations.

The metrics here are lightweight page-level similarity proxies for HF/HMONNX
regression. They are not the official OmniDocBench metrics.
"""

import argparse
import copy
import gc
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from static_vit_utils import (
    DEFAULT_VISUAL_BUCKETS_CONFIG,
    StaticBucketProcessorAdapter,
    describe_static_vit,
    load_python_config,
    parse_bucket,
    validate_static_buckets,
)

Image.MAX_IMAGE_PIXELS = None


DEFAULT_CONTENT_CATEGORIES = [
    "title",
    "text_block",
    "list_group",
    "reference",
    "figure_caption",
    "figure_footnote",
    "table",
    "table_caption",
    "table_footnote",
    "equation_isolated",
    "equation_caption",
    "equation_semantic",
    "equation_explanation",
    "header",
    "footer",
    "page_number",
    "page_footnote",
    "code_txt",
    "code_txt_caption",
]

PAGE_ATTR_KEYS = ["data_source", "language", "layout"]

def load_omnidocbench(annotation_path: Path) -> list[dict]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected OmniDocBench annotation list, got {type(data)!r}")
    return data


def stable_sample(data: list[dict], sample_size: int | None, strategy: str) -> list[tuple[int, dict]]:
    indexed = list(enumerate(data))
    if strategy == "image_path_evenly_spaced":
        indexed.sort(key=lambda item: item[1].get("page_info", {}).get("image_path", ""))
    elif strategy == "json_order_evenly_spaced":
        pass
    else:
        raise ValueError(f"Unsupported sample strategy: {strategy}")

    if sample_size is None or sample_size >= len(indexed):
        return indexed
    if sample_size <= 0:
        return []
    if sample_size == 1:
        return [indexed[0]]

    positions = [math.floor(i * (len(indexed) - 1) / (sample_size - 1)) for i in range(sample_size)]
    return [indexed[pos] for pos in positions]


def load_manifest_sample(data: list[dict], manifest_path: Path, sample_size: int | None) -> list[tuple[int, dict]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest = sorted(manifest, key=lambda item: int(item.get("sample_rank", 0)))
    if sample_size is not None:
        manifest = manifest[:sample_size]
    return [(int(item["json_index"]), data[int(item["json_index"])]) for item in manifest]


def image_path_for(record: dict) -> str:
    path = record.get("page_info", {}).get("image_path")
    if not path:
        raise ValueError(f"Missing page_info.image_path in record: {record.keys()}")
    path = str(path)
    if "/" not in path:
        return f"images/{path}"
    return path


def download_missing_images(samples: list[tuple[int, dict]], dataset_dir: Path, repo_id: str) -> None:
    from huggingface_hub import hf_hub_download

    missing = [image_path_for(record) for _, record in samples if not (dataset_dir / image_path_for(record)).exists()]
    if not missing:
        print("All sampled OmniDocBench images already exist.")
        return

    print(f"Downloading {len(missing)} sampled OmniDocBench images into {dataset_dir}")
    for idx, filename in enumerate(tqdm(missing, desc="Download images"), 1):
        hf_hub_download(repo_id, repo_type="dataset", filename=filename, local_dir=str(dataset_dir))
        if idx % 10 == 0:
            print(f"Downloaded {idx}/{len(missing)} images.")


def normalize_text(text: str, remove_spaces: bool = False) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    if remove_spaces:
        text = re.sub(r"\s+", "", text)
    return text


def char_prf(prediction: str, target: str) -> dict:
    pred = normalize_text(prediction, remove_spaces=True)
    gt = normalize_text(target, remove_spaces=True)
    pred_counter = Counter(pred)
    gt_counter = Counter(gt)
    overlap = sum((pred_counter & gt_counter).values())
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"char_precision": precision, "char_recall": recall, "char_f1": f1}


def edit_similarity(prediction: str, target: str) -> float:
    pred = normalize_text(prediction)
    gt = normalize_text(target)
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    return SequenceMatcher(None, pred, gt).ratio()


def extract_gt_blocks(record: dict, categories: set[str], include_ignored: bool) -> list[dict]:
    blocks = []
    for det in record.get("layout_dets", []):
        if det.get("ignore", False) and not include_ignored:
            continue
        category = det.get("category_type", "")
        if category not in categories:
            continue
        content = det.get("text")
        if content is None:
            content = det.get("latex")
        if content is None:
            content = det.get("html")
        if content is None or not str(content).strip():
            continue
        blocks.append(
            {
                "category": category,
                "order": det.get("order") if det.get("order") is not None else 10**9,
                "content": str(content).strip(),
            }
        )
    blocks.sort(key=lambda item: (item["order"], item["category"]))
    return blocks


def stringify_gt(record: dict, categories: set[str], include_ignored: bool) -> str:
    return "\n".join(block["content"] for block in extract_gt_blocks(record, categories, include_ignored))


def extract_texts_from_mineru_result(result: Any) -> list[str]:
    texts: list[str] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, str):
            if node.strip():
                texts.append(node.strip())
            return
        if isinstance(node, dict):
            content = node.get("content")
            if content is not None:
                visit(content)
            for key in ("text", "html", "latex", "md"):
                if key in node and key != "content":
                    visit(node[key])
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(result)
    return texts


def stringify_mineru_result(result: Any) -> str:
    return "\n".join(extract_texts_from_mineru_result(result))


def mineru_type_counts(result: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type:
                counts[str(node_type)] += 1
            for value in node.values():
                visit(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(result)
    return dict(counts)


def resolve_torch_dtype(dtype: str):
    import torch

    if dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


def build_sampling_params(max_new_tokens: int | None):
    from mineru_vl_utils.mineru_client import DEFAULT_SAMPLING_PARAMS

    sampling_params = copy.deepcopy(DEFAULT_SAMPLING_PARAMS)
    if max_new_tokens is not None:
        for params in sampling_params.values():
            params.max_new_tokens = max_new_tokens
    return sampling_params


def load_native_model(args):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    dtype = resolve_torch_dtype(args.dtype)
    model_kwargs = {"device_map": args.device_map}
    if dtype == "auto":
        model_kwargs["dtype"] = "auto"
    else:
        model_kwargs["torch_dtype"] = dtype
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, **model_kwargs).eval()
    except TypeError:
        if "torch_dtype" in model_kwargs:
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, **model_kwargs).eval()
    processor = AutoProcessor.from_pretrained(args.model, use_fast=args.use_fast_processor)
    return model, processor


def build_static_bucket_processor(args, processor, model) -> StaticBucketProcessorAdapter:
    cfg = load_python_config(Path(args.static_visual_buckets_config))
    if args.static_vit_max_upscale <= 0:
        raise ValueError("--static-vit-max-upscale must be positive.")
    if args.static_vit_ref_area <= 0:
        raise ValueError("--static-vit-ref-area must be positive.")
    fallback_bucket = (int(args.layout_image_size), int(args.layout_image_size))
    buckets = [parse_bucket(bucket) for bucket in getattr(cfg, "visual_buckets")]
    if fallback_bucket not in buckets:
        buckets.append(fallback_bucket)
    patch_size = int(getattr(getattr(model.config, "vision_config", None), "patch_size", 14))
    validate_static_buckets(buckets, patch_size)
    print(f"Static ViT simulation buckets: {sorted(set(buckets), key=lambda x: (x[0] * x[1], x[0], x[1]))}")
    print(f"Static ViT simulation fallback bucket: {fallback_bucket}")
    adapter = StaticBucketProcessorAdapter(
        processor,
        buckets,
        fallback_bucket,
        patch_size,
        max_upscale=args.static_vit_max_upscale,
        score_mode=args.static_vit_score_mode,
        alpha_down=args.static_vit_alpha_down,
        beta_up=args.static_vit_beta_up,
        gamma_pad=args.static_vit_gamma_pad,
        ref_area=args.static_vit_ref_area,
        allow_content_fallback_bucket=args.allow_content_fallback_bucket,
    )
    print(describe_static_vit(adapter))
    return adapter


def compute_scores(results: list[dict]) -> dict:
    metric_keys = ["edit_similarity", "char_precision", "char_recall", "char_f1"]

    def aggregate(items: list[dict]) -> dict:
        if not items:
            return {"total": 0, **{key: 0.0 for key in metric_keys}}
        out = {"total": len(items)}
        for key in metric_keys:
            out[key] = sum(float(item[key]) for item in items) / len(items)
        out["avg_time"] = sum(float(item.get("time", 0.0)) for item in items) / len(items)
        return out

    scores = {"overall": aggregate(results), "by_attribute": {}}
    for attr_key in PAGE_ATTR_KEYS:
        groups = defaultdict(list)
        for item in results:
            groups[item.get("page_attribute", {}).get(attr_key, "unknown")].append(item)
        scores["by_attribute"][attr_key] = {
            key: aggregate(value)
            for key, value in sorted(groups.items(), key=lambda item: item[0])
        }
    return scores


def print_scores(scores: dict) -> None:
    overall = scores["overall"]
    print("\n" + "=" * 96)
    print("OmniDocBench HF Extract Similarity (not official OmniDocBench metrics)")
    print("=" * 96)
    print(
        f"OVERALL total={overall['total']} "
        f"edit={overall['edit_similarity']:.4f} "
        f"char_p={overall['char_precision']:.4f} "
        f"char_r={overall['char_recall']:.4f} "
        f"char_f1={overall['char_f1']:.4f} "
        f"avg_time={overall['avg_time']:.2f}s"
    )
    for attr_key, groups in scores["by_attribute"].items():
        print(f"\nBy {attr_key}:")
        for name, item in groups.items():
            print(
                f"  {name:<28} n={item['total']:>3} "
                f"edit={item['edit_similarity']:.4f} "
                f"char_f1={item['char_f1']:.4f}"
            )
    print("=" * 96)


def write_sample_manifest(samples: list[tuple[int, dict]], output_path: Path) -> None:
    manifest = []
    for rank, (json_index, record) in enumerate(samples):
        page_info = record.get("page_info", {})
        manifest.append(
            {
                "sample_rank": rank,
                "json_index": json_index,
                "image_path": page_info.get("image_path"),
                "page_no": page_info.get("page_no"),
                "height": page_info.get("height"),
                "width": page_info.get("width"),
                "page_attribute": page_info.get("page_attribute", {}),
            }
        )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MinerU2.5-Pro native HF OmniDocBench deterministic subset evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="/data02/datasets/MinerU2.5-Pro-2604-1.2B")
    parser.add_argument("--dataset-dir", type=str, default="work_dirs/mineru2_5_omnidocbench_data")
    parser.add_argument("--annotation", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default="opendatalab/OmniDocBench")
    parser.add_argument("--download-missing", action="store_true", default=True)
    parser.add_argument("--no-download-missing", dest="download_missing", action="store_false")
    parser.add_argument("--output-dir", type=str, default="work_dirs/mineru2_5_omnidocbench_hf_extract")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument(
        "--sample-manifest",
        type=str,
        default="",
        help="Use an existing OmniDocBench sample manifest instead of recomputing the deterministic sample.",
    )
    parser.add_argument(
        "--sample-strategy",
        type=str,
        default="image_path_evenly_spaced",
        choices=["image_path_evenly_spaced", "json_order_evenly_spaced"],
    )
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CONTENT_CATEGORIES)
    parser.add_argument("--include-ignored", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--use-fast-processor", action="store_true", default=True)
    parser.add_argument("--no-fast-processor", dest="use_fast_processor", action="store_false")
    parser.add_argument("--image-analysis", action="store_true")
    parser.add_argument("--mineru-batch-size", type=int, default=1)
    parser.add_argument("--layout-image-size", type=int, default=1036)
    parser.add_argument(
        "--simulate-static-vit",
        action="store_true",
        help="Letterbox inputs to static visual buckets before running the native HF model.",
    )
    parser.add_argument("--static-visual-buckets-config", type=str, default=DEFAULT_VISUAL_BUCKETS_CONFIG)
    parser.add_argument(
        "--static-vit-max-upscale",
        type=float,
        default=2.5,
        help="Maximum content upscale ratio before centering on a static visual bucket.",
    )
    parser.add_argument(
        "--static-vit-score-mode",
        type=str,
        default="fit_padding",
        choices=["fit_padding", "ratio"],
        help="Static visual bucket routing score. ratio is disabled because tests showed poor stability; use fit_padding.",
    )
    parser.add_argument(
        "--static-vit-alpha-down",
        type=float,
        default=10.0,
        help="Downscale penalty weight for --static-vit-score-mode fit_padding.",
    )
    parser.add_argument(
        "--static-vit-beta-up",
        type=float,
        default=1.0,
        help="Upscale penalty weight for --static-vit-score-mode fit_padding.",
    )
    parser.add_argument(
        "--static-vit-gamma-pad",
        type=float,
        default=3.0,
        help="Padding penalty weight for --static-vit-score-mode fit_padding.",
    )
    parser.add_argument(
        "--static-vit-ref-area",
        type=float,
        default=448 * 448,
        help="Reference area used to normalize padding pixels for --static-vit-score-mode fit_padding.",
    )
    parser.add_argument(
        "--allow-content-fallback-bucket",
        action="store_true",
        help="Allow non-layout content crops to route to the square fallback bucket.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--mineru-log-level", type=str, default="WARNING")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from loguru import logger
    from mineru_vl_utils import MinerUClient

    logger.remove()
    logger.add(sys.stderr, level=args.mineru_log_level)

    dataset_dir = Path(args.dataset_dir)
    annotation_path = Path(args.annotation) if args.annotation else dataset_dir / "OmniDocBench.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading OmniDocBench annotation from {annotation_path}")
    data = load_omnidocbench(annotation_path)
    if args.sample_manifest:
        samples = load_manifest_sample(data, Path(args.sample_manifest), args.sample_size)
        print(f"Will evaluate {len(samples)} / {len(data)} pages from manifest {args.sample_manifest}.")
    else:
        samples = stable_sample(data, args.sample_size, args.sample_strategy)
        print(f"Will evaluate {len(samples)} / {len(data)} pages by {args.sample_strategy}.")
    write_sample_manifest(samples, output_dir / "omnidocbench_sample_manifest.json")

    if args.download_missing:
        download_missing_images(samples, dataset_dir, args.repo_id)

    print(f"Loading MinerU2.5-Pro native HF model from {args.model}")
    model, processor = load_native_model(args)
    if args.simulate_static_vit:
        processor = build_static_bucket_processor(args, processor, model)
        args.mineru_batch_size = 1
    client = MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
        image_analysis=args.image_analysis,
        sampling_params=build_sampling_params(args.max_new_tokens),
        layout_image_size=(args.layout_image_size, args.layout_image_size),
        batch_size=args.mineru_batch_size,
        use_tqdm=not args.no_tqdm,
    )
    print("Model loaded.")

    categories = set(args.categories)
    results_path = output_dir / "omnidocbench_results.json"
    existing_results = {}
    if args.resume and results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                existing_results[int(item["json_index"])] = item
        print(f"Resumed {len(existing_results)} existing results.")

    results = []
    new_times = []
    iterator = samples if args.no_tqdm else tqdm(samples, desc="MinerU2.5 OmniDocBench HF extract")
    for sample_rank, (json_index, record) in enumerate(iterator):
        if json_index in existing_results:
            results.append(existing_results[json_index])
            continue

        page_info = record.get("page_info", {})
        rel_image_path = image_path_for(record)
        image_path = dataset_dir / rel_image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing sampled image: {image_path}")

        gt_text = stringify_gt(record, categories, args.include_ignored)
        gt_blocks = extract_gt_blocks(record, categories, args.include_ignored)

        image = Image.open(image_path).convert("RGB")
        t0 = time.perf_counter()
        with torch.no_grad():
            mineru_result = client.two_step_extract(image, image_analysis=args.image_analysis)
        elapsed = time.perf_counter() - t0
        new_times.append(elapsed)

        prediction = stringify_mineru_result(mineru_result)
        prf = char_prf(prediction, gt_text)
        result = {
            "sample_rank": sample_rank,
            "json_index": json_index,
            "image_path": rel_image_path,
            "page_attribute": page_info.get("page_attribute", {}),
            "gt_block_count": len(gt_blocks),
            "gt_type_counts": dict(Counter(block["category"] for block in gt_blocks)),
            "pred_type_counts": mineru_type_counts(mineru_result),
            "gt_text": gt_text,
            "prediction": prediction,
            "edit_similarity": edit_similarity(prediction, gt_text),
            **prf,
            "time": elapsed,
            "mineru_result": mineru_result,
        }
        results.append(result)

        if args.save_every > 0 and len(results) % args.save_every == 0:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        del image, mineru_result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_new_time = sum(new_times)
    avg_new_time = total_new_time / len(new_times) if new_times else 0.0
    print(f"\nInference done. New sample total time: {total_new_time:.1f}s, avg: {avg_new_time:.2f}s/page")

    scores = compute_scores(results)
    print_scores(scores)
    scores_path = output_dir / "omnidocbench_scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    if isinstance(processor, StaticBucketProcessorAdapter):
        bucket_stats = {
            f"{bucket[0]}x{bucket[1]}": count
            for bucket, count in sorted(processor.bucket_counts.items(), key=lambda item: (item[0][0] * item[0][1], item[0]))
        }
        bucket_stats_path = output_dir / "static_vit_bucket_stats.json"
        with open(bucket_stats_path, "w", encoding="utf-8") as f:
            json.dump({"bucket_counts": bucket_stats, "fallback_count": processor.fallback_count}, f, ensure_ascii=False, indent=2)
        print(f"Static ViT bucket stats saved to {bucket_stats_path}")
    print(f"Sample manifest saved to {output_dir / 'omnidocbench_sample_manifest.json'}")
    print(f"Results saved to {results_path}")
    print(f"Scores saved to {scores_path}")


if __name__ == "__main__":
    main()
