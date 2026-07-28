import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def load_flickr30k(
    dataset_dir: str | Path,
) -> list[tuple[str, list[str]]]:
    dataset_dir = Path(dataset_dir)
    image_dirs = [
        dataset_dir / "flickr30k-images",
        dataset_dir / "flickr30k_images",
        dataset_dir / "images",
    ]
    image_dir = next(
        (path for path in image_dirs if path.is_dir()),
        None,
    )
    if image_dir is None:
        raise FileNotFoundError(
            f"No Flickr30K image directory found under {dataset_dir}"
        )

    annotations = dataset_dir / "flickr_annotations_30k.csv"
    if annotations.is_file():
        rows = []
        with annotations.open("r", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("split", "").strip() != "test":
                    continue
                filename = row.get("filename", "").strip()
                raw = row.get("raw", "").strip()
                image_path = image_dir / filename
                if not filename or not raw or not image_path.is_file():
                    continue
                try:
                    captions = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                captions = [
                    caption
                    for caption in captions
                    if isinstance(caption, str) and caption
                ]
                if captions:
                    rows.append((str(image_path), captions))
        if rows:
            return rows

    results_csv = dataset_dir / "results.csv"
    if results_csv.is_file():
        by_image: dict[str, list[str]] = {}
        with results_csv.open("r", encoding="utf-8") as file:
            sample = file.readline()
            file.seek(0)
            delimiter = "|" if "|" in sample else ","
            reader = csv.reader(file, delimiter=delimiter)
            header = [field.strip() for field in next(reader)]
            name_idx = (
                header.index("image_name")
                if "image_name" in header
                else 0
            )
            caption_idx = (
                header.index("comment")
                if "comment" in header
                else len(header) - 1
            )
            for row in reader:
                if len(row) <= max(name_idx, caption_idx):
                    continue
                image_path = image_dir / row[name_idx].strip()
                caption = row[caption_idx].strip()
                if image_path.is_file() and caption:
                    by_image.setdefault(
                        str(image_path),
                        [],
                    ).append(caption)
        if by_image:
            return list(by_image.items())

    captions_json = dataset_dir / "captions.json"
    if captions_json.is_file():
        data = json.loads(captions_json.read_text(encoding="utf-8"))
        rows = []
        for filename, captions in data.items():
            image_path = image_dir / filename
            captions = [
                caption
                for caption in captions
                if isinstance(caption, str) and caption
            ]
            if image_path.is_file() and captions:
                rows.append((str(image_path), captions))
        if rows:
            return rows

    raise RuntimeError(
        f"No supported Flickr30K annotations found under {dataset_dir}"
    )


def flatten_rows(
    rows: list[tuple[str, list[str]]],
) -> tuple[list[str], list[str], list[int], list[list[int]]]:
    images = []
    captions = []
    caption_owners = []
    image_to_captions = []
    for image_idx, (image, image_captions) in enumerate(rows):
        images.append(image)
        owned_caption_ids = []
        for caption in image_captions:
            caption_id = len(captions)
            captions.append(caption)
            caption_owners.append(image_idx)
            owned_caption_ids.append(caption_id)
        image_to_captions.append(owned_caption_ids)
    return images, captions, caption_owners, image_to_captions


def resize_with_padding(
    image_path: str | Path,
    image_size: int,
) -> Image.Image:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.thumbnail(
            (image_size, image_size),
            Image.Resampling.BICUBIC,
        )
    canvas = Image.new(
        "RGB",
        (image_size, image_size),
        color=(114, 114, 114),
    )
    offset = (
        (image_size - image.width) // 2,
        (image_size - image.height) // 2,
    )
    canvas.paste(image, offset)
    return canvas


def retrieval_metrics(
    query: torch.Tensor,
    documents: torch.Tensor,
    ground_truth: list[list[int]],
    recall_ks: tuple[int, ...] = (1, 5, 10),
    rank_limit: int = 10,
) -> dict[str, float]:
    scores = query @ documents.T
    max_k = min(
        max(max(recall_ks), rank_limit),
        scores.shape[1],
    )
    rankings = scores.topk(max_k, dim=-1).indices.tolist()
    discounts = torch.log2(
        torch.arange(2, max_k + 2, dtype=torch.float64)
    )

    recalls = {k: 0.0 for k in recall_ks}
    mrr = 0.0
    ndcg = 0.0
    for index, ranking in enumerate(rankings):
        relevant = set(ground_truth[index])
        for k in recall_ks:
            if any(doc_id in relevant for doc_id in ranking[:k]):
                recalls[k] += 1.0

        for rank, doc_id in enumerate(
            ranking[:rank_limit],
            start=1,
        ):
            if doc_id in relevant:
                mrr += 1.0 / rank
                break

        dcg = sum(
            1.0 / float(discounts[rank])
            for rank, doc_id in enumerate(ranking[:rank_limit])
            if doc_id in relevant
        )
        ideal_hits = min(len(relevant), rank_limit, max_k)
        idcg = sum(
            1.0 / float(discounts[rank])
            for rank in range(ideal_hits)
        )
        ndcg += dcg / idcg if idcg else 0.0

    count = max(len(rankings), 1)
    metrics: dict[str, Any] = {
        f"recall@{k}": recalls[k] / count
        for k in recall_ks
    }
    metrics[f"mrr@{rank_limit}"] = mrr / count
    metrics[f"ndcg@{rank_limit}"] = ndcg / count
    return metrics
