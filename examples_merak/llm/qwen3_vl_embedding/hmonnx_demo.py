import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL-Embedding with Merak HMONNX."
    )
    parser.add_argument(
        "--export-dir",
        required=True,
        help="Directory containing export_meta_info.json",
    )
    parser.add_argument(
        "--text",
        nargs="+",
        default=[
            "A dog playing in the park",
            "A cat sitting on a chair",
        ],
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional local image path",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _load_export_meta(export_dir: Path) -> Path:
    root_meta_file = export_dir / "export_meta_info.json"
    if not root_meta_file.is_file():
        raise FileNotFoundError(
            f"export_meta_info.json not found under {export_dir}"
        )
    return root_meta_file


def main():
    args = parse_args()
    from xhmodel_merak.xh_llm import AutoLLMHONNXModel
    from xhquant.api import xhquant_init

    xhquant_init(None, args.debug)
    export_dir = Path(args.export_dir)
    export_meta = _load_export_meta(export_dir)
    model = AutoLLMHONNXModel.from_pretrained(
        str(export_meta),
        device_map=[args.device],
    )
    model.to(torch.device(args.device))
    if args.fast:
        model.to_fast()

    items = [{"text": text} for text in args.text]
    if args.image is not None:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )
        items.append({"image": str(image_path)})

    embeddings = model.embed_items(items)
    print(f"embedding_shape: {tuple(embeddings.shape)}")
    print(f"embedding_norms: {embeddings.norm(dim=-1).tolist()}")
    print("similarity:")
    print(embeddings @ embeddings.T)


if __name__ == "__main__":
    main()
