import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from flickr30k_utils import (
    flatten_rows,
    load_flickr30k,
    resize_with_padding,
    retrieval_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-Embedding on Flickr30K."
    )
    parser.add_argument(
        "--backend",
        choices=["native", "hmonnx"],
        required=True,
    )
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument(
        "--native-image-size",
        type=int,
        default=None,
        help=(
            "Letterbox native images to a fixed square size for "
            "resolution/padding ablation"
        ),
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--report",
        default="work_dirs/qwen3_vl_embedding_flickr30k.json",
    )
    return parser.parse_args()


def _load_hmonnx_model(export_dir: str, device: str):
    from xhmodel_merak.xh_llm import AutoLLMHONNXModel

    root = Path(export_dir)
    root_meta_file = root / "export_meta_info.json"
    root_meta = json.loads(
        root_meta_file.read_text(encoding="utf-8")
    )
    model = AutoLLMHONNXModel.from_pretrained(
        str(root_meta_file),
        device_map=[device],
    )
    model.to(torch.device(device))
    return model


def _embed_hmonnx(
    model: Any,
    items: list[dict[str, Any]],
    batch_size: int,
    label: str,
) -> torch.Tensor:
    embeddings = []
    start = time.time()
    for offset in range(0, len(items), batch_size):
        batch = items[offset : offset + batch_size]
        embeddings.append(model.embed_items(batch).cpu())
        completed = offset + len(batch)
        if completed == len(items) or completed % 100 == 0:
            rate = completed / max(time.time() - start, 1e-6)
            print(
                f"[{label}] {completed}/{len(items)} "
                f"({rate:.2f} samples/s)",
                flush=True,
            )
    return torch.cat(embeddings, dim=0)


def _embed_native(
    model: Any,
    items: list[dict[str, Any]],
    batch_size: int,
) -> torch.Tensor:
    from native_utils import embed_native_items

    return embed_native_items(
        model,
        items,
        batch_size=batch_size,
    )


def main():
    args = parse_args()
    rows = load_flickr30k(args.dataset_dir)
    if args.max_images:
        rows = rows[: args.max_images]
    images, captions, caption_owners, image_to_captions = (
        flatten_rows(rows)
    )

    if args.backend == "native":
        if not args.model_dir:
            raise ValueError(
                "--model-dir is required for the native backend"
            )
        from native_utils import load_native_embedder

        model = load_native_embedder(args.model_dir)
        text_embeddings = _embed_native(
            model,
            [{"text": caption} for caption in captions],
            args.text_batch_size,
        )
        native_images: list[Any] = images
        if args.native_image_size is not None:
            native_images = [
                resize_with_padding(
                    image,
                    args.native_image_size,
                )
                for image in images
            ]
        image_embeddings = _embed_native(
            model,
            [{"image": image} for image in native_images],
            args.image_batch_size,
        )
    else:
        if not args.export_dir:
            raise ValueError(
                "--export-dir is required for the hmonnx backend"
            )
        model = _load_hmonnx_model(
            args.export_dir,
            args.device,
        )
        text_embeddings = _embed_hmonnx(
            model,
            [{"text": caption} for caption in captions],
            args.text_batch_size,
            "text",
        )
        image_embeddings = _embed_hmonnx(
            model,
            [{"image": image} for image in images],
            args.image_batch_size,
            "image",
        )

    text_ground_truth = [
        [owner] for owner in caption_owners
    ]
    text_to_image = retrieval_metrics(
        text_embeddings,
        image_embeddings,
        text_ground_truth,
    )
    image_to_text = retrieval_metrics(
        image_embeddings,
        text_embeddings,
        image_to_captions,
    )
    report = {
        "backend": args.backend,
        "dataset_dir": args.dataset_dir,
        "num_images": len(images),
        "num_captions": len(captions),
        "native_image_size": args.native_image_size,
        "text_to_image": text_to_image,
        "image_to_text": image_to_text,
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
