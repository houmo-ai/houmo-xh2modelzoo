#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-GPU Flickr30K retrieval evaluation for Qwen3-VL-Embedding HMONNX.

This is inference data parallelism, not training DDP: each worker process owns
one GPU, builds an independent Qwen3VLONNXModel, embeds a shard of captions and
images, and the parent process merges embeddings before computing metrics.
"""

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import torch

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "qwen3_vl_embedding_eval_flickr30k_hmonnx_aligned.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("hmonnx_eval_base", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hmonnx_eval_base"] = module
    spec.loader.exec_module(module)
    return module


def read_local_flickr30k(dataset_dir: Path) -> List[Tuple[str, List[str]]]:
    img_dirs = [
        dataset_dir / "flickr30k-images",
        dataset_dir / "flickr30k_images",
        dataset_dir / "images",
    ]
    img_dir = next((d for d in img_dirs if d.is_dir()), None)
    if img_dir is None:
        return []

    flickr_annotations = dataset_dir / "flickr_annotations_30k.csv"
    if not flickr_annotations.is_file():
        return []

    out = []
    with open(flickr_annotations, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split", "").strip() != "test":
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
            except Exception:
                continue
            if isinstance(caps, list):
                caps = [c for c in caps if isinstance(c, str) and c]
                if caps:
                    out.append((str(p), caps))
    return out


def flatten_rows(rows):
    images = []
    captions_flat = []
    cap_owner = []
    image_to_caps = []
    for img_idx, (img_path, caps) in enumerate(rows):
        images.append(img_path)
        own_caps = []
        for c in caps:
            cap_id = len(captions_flat)
            captions_flat.append(c)
            cap_owner.append(img_idx)
            own_caps.append(cap_id)
        image_to_caps.append(own_caps)
    return images, captions_flat, cap_owner, image_to_caps


def split_ranges(n: int, parts: int):
    ranges = []
    for rank in range(parts):
        start = (n * rank) // parts
        end = (n * (rank + 1)) // parts
        ranges.append((start, end))
    return ranges


def retrieval_metrics(query: torch.Tensor, doc: torch.Tensor, gt_doc_idx_list: List[List[int]],
                      recall_ks=(1, 5, 10), mrr_k=10, ndcg_k=10) -> dict:
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


def worker_main(args):
    base = load_base_module()
    from xhquant.api import get_root_logger, xhquant_init
    import xhquant.utils.suppress_printing

    xhquant_init(None, args.debug)
    xhquant.utils.suppress_printing.disable_printing = True
    logger = get_root_logger()

    rows = read_local_flickr30k(Path(args.dataset_dir))
    if args.max_images:
        rows = rows[: args.max_images]
    if not rows:
        raise RuntimeError(f"No Flickr30K data found in {args.dataset_dir}")

    images, captions_flat, _, _ = flatten_rows(rows)
    text_start, text_end = args.text_start, args.text_end
    image_start, image_end = args.image_start, args.image_end

    logger.info(
        f"Worker rank={args.rank} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"text={text_start}:{text_end} image={image_start}:{image_end}"
    )

    model_args = SimpleNamespace(
        hf_model=args.hf_model,
        model_type=args.model_type,
        device=args.device,
    )
    xh_model, processor, _ = base.build_xh_model(Path(args.hmonnx_config), model_args, logger)

    text_parts = []
    t0 = time.time()
    for pos, i in enumerate(range(text_start, text_end), start=1):
        emb = xh_model.embed_text(captions_flat[i], processor, use_fast=args.fast, keep_session=True)
        text_parts.append(emb.cpu())
        if pos % args.text_log_every == 0 or i + 1 == text_end:
            rate = pos / max(time.time() - t0, 1e-6)
            logger.info(f"[rank {args.rank} text] {pos}/{text_end - text_start} ({rate:.3f} samples/s)")
    xh_model.release_prefill_session()

    image_parts = []
    t0 = time.time()
    for pos, i in enumerate(range(image_start, image_end), start=1):
        emb = xh_model.embed_image(images[i], processor, use_fast=args.fast, keep_session=True)
        image_parts.append(emb.cpu())
        if pos % args.image_log_every == 0 or i + 1 == image_end:
            rate = pos / max(time.time() - t0, 1e-6)
            logger.info(f"[rank {args.rank} image] {pos}/{image_end - image_start} ({rate:.3f} samples/s)")
    xh_model.release_image_feature()
    xh_model.release_prefill_session()

    payload = {
        "rank": args.rank,
        "text_start": text_start,
        "text_end": text_end,
        "image_start": image_start,
        "image_end": image_end,
        "text_emb": torch.cat(text_parts, dim=0) if text_parts else None,
        "image_emb": torch.cat(image_parts, dim=0) if image_parts else None,
    }
    Path(args.shard_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.shard_out)
    logger.info(f"Worker rank={args.rank} wrote {args.shard_out}")


def parent_main(args):
    rows = read_local_flickr30k(Path(args.dataset_dir))
    if not rows:
        raise RuntimeError(f"No Flickr30K data found in {args.dataset_dir}")
    if args.max_images:
        rows = rows[: args.max_images]
        print(f"--max-images={args.max_images} active; eval is NOT full dataset", flush=True)

    images, captions_flat, cap_owner, image_to_caps = flatten_rows(rows)
    print(f"#images={len(images)} #captions={len(captions_flat)}", flush=True)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("--gpus must not be empty")

    run_dir = Path(args.run_dir or (Path(args.report).with_suffix("") if args.report else Path("work_dirs/hmonnx_multigpu_run")))
    run_dir.mkdir(parents=True, exist_ok=True)

    text_ranges = split_ranges(len(captions_flat), len(gpus))
    image_ranges = split_ranges(len(images), len(gpus))

    procs = []
    for rank, gpu in enumerate(gpus):
        shard_out = run_dir / f"worker_{rank}.pt"
        log_path = run_dir / f"worker_{rank}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONPATH"] = args.pythonpath or env.get("PYTHONPATH", "")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--rank", str(rank),
            "--hmonnx-config", args.hmonnx_config,
            "--hf-model", args.hf_model,
            "--dataset-dir", args.dataset_dir,
            "--model-type", args.model_type,
            "--device", "cuda:0",
            "--text-start", str(text_ranges[rank][0]),
            "--text-end", str(text_ranges[rank][1]),
            "--image-start", str(image_ranges[rank][0]),
            "--image-end", str(image_ranges[rank][1]),
            "--shard-out", str(shard_out),
            "--text-log-every", str(args.text_log_every),
            "--image-log-every", str(args.image_log_every),
        ]
        if args.max_images:
            cmd.extend(["--max-images", str(args.max_images)])
        if args.fast:
            cmd.append("--fast")
        if args.debug:
            cmd.append("--debug")
        log_f = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(HERE), env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((rank, gpu, proc, log_f, shard_out, log_path))
        print(f"Launched rank={rank} gpu={gpu} pid={proc.pid} log={log_path}", flush=True)

    last_report = 0.0
    while True:
        alive = [p for _, _, p, _, _, _ in procs if p.poll() is None]
        now = time.time()
        if now - last_report >= args.parent_log_every:
            last_report = now
            for rank, gpu, proc, _, _, log_path in procs:
                status = "running" if proc.poll() is None else f"exit={proc.returncode}"
                tail = ""
                try:
                    lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
                    tail = lines[-1] if lines else ""
                except Exception:
                    pass
                print(f"rank={rank} gpu={gpu} pid={proc.pid} {status} :: {tail}", flush=True)
        if not alive:
            break
        time.sleep(5)

    failed = []
    for rank, _, proc, log_f, _, _ in procs:
        log_f.close()
        if proc.returncode != 0:
            failed.append((rank, proc.returncode))
    if failed:
        raise RuntimeError(f"Worker failure(s): {failed}; logs in {run_dir}")

    shards = [torch.load(shard_out, map_location="cpu", weights_only=False) for _, _, _, _, shard_out, _ in procs]
    text_dim = next(s["text_emb"].shape[1] for s in shards if s["text_emb"] is not None)
    image_dim = next(s["image_emb"].shape[1] for s in shards if s["image_emb"] is not None)
    text_emb = torch.empty((len(captions_flat), text_dim), dtype=torch.float32)
    image_emb = torch.empty((len(images), image_dim), dtype=torch.float32)

    for s in shards:
        if s["text_emb"] is not None:
            text_emb[s["text_start"]:s["text_end"]] = s["text_emb"].float()
        if s["image_emb"] is not None:
            image_emb[s["image_start"]:s["image_end"]] = s["image_emb"].float()

    text_gt = [[int(o)] for o in cap_owner]
    t2i = retrieval_metrics(text_emb, image_emb, text_gt)
    i2t = retrieval_metrics(image_emb, text_emb, image_to_caps)

    report = {
        "hmonnx_config": args.hmonnx_config,
        "hf_model": args.hf_model,
        "dataset": f"Flickr30K ({args.dataset_dir})",
        "num_images": len(images),
        "num_captions": len(captions_flat),
        "gpus": gpus,
        "run_dir": str(run_dir),
        "text_to_image": t2i,
        "image_to_text": i2t,
    }
    out_path = Path(args.report) if args.report else Path("work_dirs/flickr30k_hmonnx_multigpu_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    fmt = lambda d: "  ".join(f"{k}={v:.4f}" for k, v in d.items())
    print("===== Flickr30K Retrieval (HMONNX multi-GPU) =====", flush=True)
    print(f"text -> image  {fmt(t2i)}", flush=True)
    print(f"image -> text  {fmt(i2t)}", flush=True)
    print(f"Wrote report to {out_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--hmonnx-config", type=str, required=True)
    parser.add_argument("--hf-model", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="8B", choices=["2B", "4B", "8B"])
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--report", type=str, default="work_dirs/flickr30k_hmonnx_8B_multigpu_full_eval.json")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--pythonpath", type=str, default="/data01/home/she.gao/xh2modelzoo_new")
    parser.add_argument("--parent-log-every", type=float, default=30.0)
    parser.add_argument("--text-log-every", type=int, default=20)
    parser.add_argument("--image-log-every", type=int, default=10)
    parser.add_argument("--text-start", type=int, default=0)
    parser.add_argument("--text-end", type=int, default=0)
    parser.add_argument("--image-start", type=int, default=0)
    parser.add_argument("--image-end", type=int, default=0)
    parser.add_argument("--shard-out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        worker_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
