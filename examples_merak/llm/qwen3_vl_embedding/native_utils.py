import importlib
import sys
from pathlib import Path
from typing import Any

import torch


def load_native_embedder(
    model_dir: str,
    dtype: torch.dtype = torch.bfloat16,
):
    model_path = Path(model_dir).resolve()
    script_path = (
        model_path / "scripts" / "qwen3_vl_embedding.py"
    )
    if not script_path.is_file():
        raise FileNotFoundError(
            "The model-bundled Qwen3-VL-Embedding implementation "
            f"was not found: {script_path}"
        )

    model_root = str(model_path)
    sys.path.insert(0, model_root)
    try:
        module = importlib.import_module(
            "scripts.qwen3_vl_embedding"
        )
    finally:
        sys.path.remove(model_root)

    embedder = module.Qwen3VLEmbedder(
        model_name_or_path=str(model_path),
        torch_dtype=dtype,
    )
    embedder.model.eval()
    return embedder


def embed_native_items(
    embedder: Any,
    items: list[dict[str, Any]],
    batch_size: int = 8,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    embeddings = []
    with torch.no_grad():
        for offset in range(0, len(items), batch_size):
            batch = items[offset : offset + batch_size]
            embeddings.append(
                embedder.process(batch, normalize=True).float().cpu()
            )
    return torch.cat(embeddings, dim=0)
