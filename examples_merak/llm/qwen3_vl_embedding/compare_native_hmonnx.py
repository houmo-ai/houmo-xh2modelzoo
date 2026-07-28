import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from native_utils import embed_native_items, load_native_embedder


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare native and Merak HMONNX Qwen3-VL embeddings."
        )
    )
    parser.add_argument("--model-dir", required=True)
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
        help="Optional image included in the parity comparison",
    )
    parser.add_argument("--device", default="cuda:0")
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

    export_meta_file = _load_export_meta(Path(args.export_dir))
    model_meta = json.loads(
        export_meta_file.read_text(encoding="utf-8")
    )
    items = [{"text": text} for text in args.text]
    if args.image is not None:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )
        visual_config = model_meta["visual_config"]
        image_size = (
            int(visual_config["image_size_w"]),
            int(visual_config["image_size_h"]),
        )
        image = Image.open(image_path).convert("RGB")
        items.append({"image": image.resize(image_size)})

    native_model = load_native_embedder(args.model_dir)
    native_embeddings = embed_native_items(
        native_model,
        items,
    )
    del native_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xhquant_init(None, args.debug)
    hmonnx_model = AutoLLMHONNXModel.from_pretrained(
        str(export_meta_file),
        device_map=[args.device],
    )
    hmonnx_model.to(torch.device(args.device))
    hmonnx_embeddings = hmonnx_model.embed_items(items).cpu()

    per_item_cosine = F.cosine_similarity(
        native_embeddings,
        hmonnx_embeddings,
        dim=-1,
    )
    print(f"embedding_shape: {tuple(hmonnx_embeddings.shape)}")
    print(
        "native_norms: "
        f"{native_embeddings.norm(dim=-1).tolist()}"
    )
    print(
        "hmonnx_norms: "
        f"{hmonnx_embeddings.norm(dim=-1).tolist()}"
    )
    print(f"per_item_cosine: {per_item_cosine.tolist()}")
    print("native_similarity:")
    print(native_embeddings @ native_embeddings.T)
    print("hmonnx_similarity:")
    print(hmonnx_embeddings @ hmonnx_embeddings.T)


if __name__ == "__main__":
    main()
