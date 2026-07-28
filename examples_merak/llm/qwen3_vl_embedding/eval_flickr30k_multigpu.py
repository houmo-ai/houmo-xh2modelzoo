import argparse
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import torch

from flickr30k_utils import (
    flatten_rows,
    load_flickr30k,
    retrieval_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Merak Qwen3-VL-Embedding HMONNX on "
            "Flickr30K with data parallel workers."
        )
    )
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated physical CUDA indices, for example 0,1",
    )
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--run-dir",
        default="work_dirs/qwen3_vl_embedding_multigpu",
    )
    parser.add_argument(
        "--report",
        default=(
            "work_dirs/"
            "qwen3_vl_embedding_flickr30k_multigpu.json"
        ),
    )
    return parser.parse_args()


def _export_meta_file(export_dir: str) -> Path:
    root = Path(export_dir).resolve()
    root_meta_file = root / "export_meta_info.json"
    if not root_meta_file.is_file():
        raise FileNotFoundError(
            f"export_meta_info.json not found under {root}"
        )
    return root_meta_file


def _shard_bounds(
    length: int,
    worker_index: int,
    worker_count: int,
) -> tuple[int, int]:
    start = length * worker_index // worker_count
    end = length * (worker_index + 1) // worker_count
    return start, end


def _embed_items(
    model: Any,
    items: list[dict[str, Any]],
    batch_size: int,
) -> torch.Tensor:
    embeddings = []
    for offset in range(0, len(items), batch_size):
        embeddings.append(
            model.embed_items(
                items[offset : offset + batch_size]
            ).cpu()
        )
    if not embeddings:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(embeddings, dim=0)


def _worker(
    worker_index: int,
    gpu_index: int,
    worker_count: int,
    export_dir: str,
    captions: list[str],
    images: list[str],
    text_batch_size: int,
    image_batch_size: int,
    run_dir: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from xhmodel_merak.xh_llm import AutoLLMHONNXModel
    from xhquant.api import xhquant_init

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    xhquant_init(None, False)
    model = AutoLLMHONNXModel.from_pretrained(
        str(_export_meta_file(export_dir)),
        device_map=[device],
    )
    model.to(device)

    text_start, text_end = _shard_bounds(
        len(captions),
        worker_index,
        worker_count,
    )
    image_start, image_end = _shard_bounds(
        len(images),
        worker_index,
        worker_count,
    )
    text_embeddings = _embed_items(
        model,
        [
            {"text": caption}
            for caption in captions[text_start:text_end]
        ],
        text_batch_size,
    )
    image_embeddings = _embed_items(
        model,
        [
            {"image": image}
            for image in images[image_start:image_end]
        ],
        image_batch_size,
    )
    shard_file = Path(run_dir) / f"worker_{worker_index:02d}.pt"
    torch.save(
        {
            "worker_index": worker_index,
            "text_start": text_start,
            "text_end": text_end,
            "image_start": image_start,
            "image_end": image_end,
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
        },
        shard_file,
    )


def _load_shards(
    run_dir: Path,
    worker_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    shards = [
        torch.load(
            run_dir / f"worker_{index:02d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for index in range(worker_count)
    ]
    text_embeddings = torch.cat(
        [
            shard["text_embeddings"]
            for shard in shards
            if shard["text_embeddings"].shape[0] > 0
        ],
        dim=0,
    )
    image_embeddings = torch.cat(
        [
            shard["image_embeddings"]
            for shard in shards
            if shard["image_embeddings"].shape[0] > 0
        ],
        dim=0,
    )
    return text_embeddings, image_embeddings


def main():
    args = parse_args()
    gpu_indices = [
        int(value)
        for value in args.gpus.split(",")
        if value.strip()
    ]
    if not gpu_indices:
        raise ValueError("--gpus must contain at least one CUDA index")

    rows = load_flickr30k(args.dataset_dir)
    if args.max_images:
        rows = rows[: args.max_images]
    images, captions, caption_owners, image_to_captions = (
        flatten_rows(rows)
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    context = multiprocessing.get_context("spawn")
    processes = []
    for worker_index, gpu_index in enumerate(gpu_indices):
        process = context.Process(
            target=_worker,
            args=(
                worker_index,
                gpu_index,
                len(gpu_indices),
                args.export_dir,
                captions,
                images,
                args.text_batch_size,
                args.image_batch_size,
                str(run_dir),
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"Worker {process.pid} exited with {process.exitcode}"
            )

    text_embeddings, image_embeddings = _load_shards(
        run_dir,
        len(gpu_indices),
    )
    report = {
        "backend": "hmonnx_multigpu",
        "dataset_dir": args.dataset_dir,
        "gpus": gpu_indices,
        "num_images": len(images),
        "num_captions": len(captions),
        "text_to_image": retrieval_metrics(
            text_embeddings,
            image_embeddings,
            [[owner] for owner in caption_owners],
        ),
        "image_to_text": retrieval_metrics(
            image_embeddings,
            text_embeddings,
            image_to_captions,
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
