#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate MinerU2.5 HMONNX on 10 pages from the HF OmniDocBench 100-page set.

This script reuses the HMONNX inference adapters from
mineru2_5_xh_hmonnx_generate.py and only adds evaluation/data plumbing.
"""

import argparse
import gc
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HMONNX_DIR = (
    "work_dirs/mineru2_5_llm_1_2b_xh2a_4k/"
    "hmquant_xh2_mineru2_5_pro_1_2b_w8a8_256_4k_1036x1036_20260602"
)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


hf_eval = load_module("mineru2_5_omnidocbench_hf_eval_reuse", SCRIPT_DIR / "mineru2_5_omnidocbench_hf_eval.py")
hmonnx_generate = load_module("mineru2_5_xh_hmonnx_generate_reuse", SCRIPT_DIR / "mineru2_5_xh_hmonnx_generate.py")


def load_hf_manifest_subset(manifest_path: Path, sample_count: int) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest = sorted(manifest, key=lambda item: int(item["sample_rank"]))
    if sample_count is not None:
        manifest = manifest[:sample_count]
    return manifest


def build_hmonnx_client(args):
    import torch
    from mineru_vl_utils import MinerUClient
    from xhmodel_merak.xh_llm import AutoLLMHONNXModel
    from xhquant.api import get_xhquant_logger, xhquant_init

    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()

    hmonnx_model = AutoLLMHONNXModel.from_pretrained(args.config)
    assert type(hmonnx_model).__name__ == "XHQwen2VLHMONNXModel", (
        f"Expected XHQwen2VLHMONNXModel, got {type(hmonnx_model).__name__}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    logger.info(f"Using HMONNX device={device}, dtype={dtype}")
    hmonnx_model.to(device)
    hmonnx_generate._set_hmonnx_model_dtype(hmonnx_model, dtype)
    if args.auto_offload:
        hmonnx_model.enable_auto_offload = True
    if args.golden:
        hmonnx_model.enable_golden = True
    if args.fast and not args.golden:
        hmonnx_model.to_fast()

    manifest_args = SimpleNamespace(
        config=args.config,
        visual_buckets_manifest=args.visual_buckets_manifest,
    )
    manifest, manifest_dir = hmonnx_generate._load_visual_bucket_manifest(manifest_args)
    visual_router = hmonnx_generate.StaticHMONNXVisualRouter(
        hmonnx_model=hmonnx_model,
        manifest=manifest,
        manifest_dir=manifest_dir,
        device=device,
        dtype=dtype,
        fast=args.fast,
        golden=args.golden,
        auto_offload=args.auto_offload,
        logger=logger,
    )
    processor = hmonnx_model.get_tf_processor()
    max_new_tokens = 2 if args.golden else args.max_new_tokens
    client = MinerUClient(
        backend="transformers",
        model=hmonnx_generate.MinerUHMONNXModelAdapter(hmonnx_model, visual_router, device, dtype),
        processor=hmonnx_generate.StaticBucketProcessorAdapter(
            processor=processor,
            buckets=visual_router.buckets,
            fallback_bucket=visual_router.fallback_bucket,
            patch_size=visual_router.patch_size,
            spatial_merge_size=int(manifest["spatial_merge_size"]),
            logger=logger,
            max_upscale=args.static_vit_max_upscale,
        ),
        image_analysis=args.image_analysis,
        sampling_params=hmonnx_generate._build_sampling_params(max_new_tokens),
        layout_image_size=(args.layout_image_size, args.layout_image_size),
        batch_size=1,
        use_tqdm=not args.no_tqdm,
        debug=args.debug,
    )
    return client


def print_hmonnx_scores(scores: dict) -> None:
    overall = scores["overall"]
    print("\n" + "=" * 96)
    print("OmniDocBench HMONNX Extract Similarity (not official OmniDocBench metrics)")
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


def evaluate(args):
    import torch

    dataset_dir = Path(args.dataset_dir)
    annotation_path = Path(args.annotation) if args.annotation else dataset_dir / "OmniDocBench.json"
    hf_manifest_path = Path(args.hf_sample_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = hf_eval.load_omnidocbench(annotation_path)
    selected_manifest = load_hf_manifest_subset(hf_manifest_path, args.sample_count)
    categories = set(args.categories)
    with open(output_dir / "omnidocbench_hmonnx_sample_manifest.json", "w", encoding="utf-8") as f:
        json.dump(selected_manifest, f, ensure_ascii=False, indent=2)

    client = build_hmonnx_client(args)
    results_path = output_dir / "omnidocbench_hmonnx_results.json"
    existing_results = {}
    if args.resume and results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                existing_results[int(item["json_index"])] = item
        print(f"Resumed {len(existing_results)} existing HMONNX results.")

    results = []
    new_times = []
    iterator = selected_manifest if args.no_tqdm else tqdm(selected_manifest, desc="MinerU2.5 OmniDocBench HMONNX")
    for manifest_item in iterator:
        sample_rank = int(manifest_item["sample_rank"])
        json_index = int(manifest_item["json_index"])
        if json_index in existing_results:
            results.append(existing_results[json_index])
            continue

        record = data[json_index]
        rel_image_path = hf_eval.image_path_for(record)
        image_path = dataset_dir / rel_image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing sampled image: {image_path}")

        gt_text = hf_eval.stringify_gt(record, categories, args.include_ignored)
        gt_blocks = hf_eval.extract_gt_blocks(record, categories, args.include_ignored)

        image = Image.open(image_path).convert("RGB")
        t0 = time.perf_counter()
        with torch.no_grad():
            mineru_result = client.two_step_extract(image, image_analysis=args.image_analysis)
        elapsed = time.perf_counter() - t0
        new_times.append(elapsed)

        prediction = hf_eval.stringify_mineru_result(mineru_result)
        prf = hf_eval.char_prf(prediction, gt_text)
        result = {
            "sample_rank": sample_rank,
            "json_index": json_index,
            "image_path": rel_image_path,
            "page_attribute": record.get("page_info", {}).get("page_attribute", {}),
            "gt_block_count": len(gt_blocks),
            "gt_type_counts": dict(Counter(block["category"] for block in gt_blocks)),
            "pred_type_counts": hf_eval.mineru_type_counts(mineru_result),
            "gt_text": gt_text,
            "prediction": prediction,
            "edit_similarity": hf_eval.edit_similarity(prediction, gt_text),
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
    print(f"\nHMONNX inference done. New sample total time: {total_new_time:.1f}s, avg: {avg_new_time:.2f}s/page")

    scores = hf_eval.compute_scores(results)
    print_hmonnx_scores(scores)
    scores_path = output_dir / "omnidocbench_hmonnx_scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print(f"HMONNX sample manifest saved to {output_dir / 'omnidocbench_hmonnx_sample_manifest.json'}")
    print(f"HMONNX results saved to {results_path}")
    print(f"HMONNX scores saved to {scores_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MinerU2.5 HMONNX OmniDocBench 10-page evaluation from HF 100-page manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=f"{DEFAULT_HMONNX_DIR}/golden_meta_info.json")
    parser.add_argument("--visual-buckets-manifest", type=str, default=f"{DEFAULT_HMONNX_DIR}/mineru_visual_buckets.json")
    parser.add_argument("--dataset-dir", type=str, default="work_dirs/mineru2_5_omnidocbench_data")
    parser.add_argument("--annotation", type=str, default=None)
    parser.add_argument(
        "--hf-sample-manifest",
        type=str,
        default="work_dirs/mineru2_5_omnidocbench_hf_extract/omnidocbench_sample_manifest.json",
    )
    parser.add_argument("--output-dir", type=str, default="work_dirs/mineru2_5_omnidocbench_hmonnx_extract")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--categories", nargs="+", default=hf_eval.DEFAULT_CONTENT_CATEGORIES)
    parser.add_argument("--include-ignored", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--layout-image-size", type=int, default=1036)
    parser.add_argument("--static-vit-max-upscale", type=float, default=2.0)
    parser.add_argument("--image-analysis", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--auto-offload", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
