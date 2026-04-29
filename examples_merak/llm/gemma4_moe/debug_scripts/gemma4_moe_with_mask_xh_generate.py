from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


def bootstrap_runtime() -> None:
    workspace_root = Path(__file__).resolve().parents[5]
    vendor_transformers = workspace_root / ".vendor" / "python" / "transformers"
    if vendor_transformers.exists():
        link_root = Path(tempfile.gettempdir()) / "xh2a_vendor_transformers_only"
        link_root.mkdir(parents=True, exist_ok=True)
        link_path = link_root / "transformers"
        if link_path.exists() or link_path.is_symlink():
            if not link_path.is_symlink() or link_path.resolve() != vendor_transformers:
                if link_path.is_dir() and not link_path.is_symlink():
                    shutil.rmtree(link_path)
                else:
                    link_path.unlink()
        if not link_path.exists():
            link_path.symlink_to(vendor_transformers, target_is_directory=True)
        sys.path.insert(0, str(link_root))


bootstrap_runtime()

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
from xhquant.api import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-type", choices=["wrap", "fronted", "quanted_fast"], default="wrap")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.only_first_block = True
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    model = AutoLLMModel.from_pretrained(model_cfg)
    model.to_wrap()
    if args.eval_type in {"fronted", "quanted_fast"}:
        model._models.pop("visual", None)
        try:
            model.to_fronted()
        except Exception as exc:
            print("Task12 text frontend diagnostic", type(exc).__name__, exc)
    assert model.wrap_model is not None
    assert model.get_input_embeddings() is not None
    assert model.kvcache_config.num_layers >= 1
    print(
        "Task06 wrap OK",
        type(model.wrap_model).__name__,
        model.kvcache_config.num_layers,
        model.kvcache_config.kv_cache_shape,
    )


if __name__ == "__main__":
    main()