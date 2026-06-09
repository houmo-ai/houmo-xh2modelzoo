import argparse
import copy
import json
import os.path as osp
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from qwen_vl_utils.vision_process import SPATIAL_MERGE_SIZE

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name, support_llm_model_types
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen2_vl import XHQwen2VLModel, XHQwen2VLModelConfig


MINERU_VISUAL_BUCKETS_MANIFEST = "mineru_visual_buckets.json"
DEFAULT_HF_MODEL_DIR = "/data02/datasets/MinerU2.5-Pro-2604-1.2B"


def _build_cfg_from_model(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = "mineru2_5_pro_1_2b"
    cfg_name = (
        f"{args.chip_arch}_{model_name}_{args.quant_type}_{args.prefill_chunk_length}_"
        f"{args.context_length // 1024}k_cli"
    )
    cfg = dict(
        chip_arch=args.chip_arch,
        model=dict(
            model_type=args.model_type,
            hf_model=hf_model_path,
            model_name=model_name,
            context_max_length=args.context_length,
            prefill_chunk_length=args.prefill_chunk_length,
            use_cache=True,
            num_logits_to_keep=1,
            quant_scheme=dict(
                quant_type=args.quant_type,
                nodes=dict(lm_head=dict(quant_type="w8a8h1_sefp")),
                ops={},
            ),
            only_first_block=False,
            quant_weight=args.quant_weight,
            visual_config=dict(
                max_size_w=args.image_size,
                max_size_h=args.image_size,
                patch_size=args.patch_size,
                temporal_patch_size=args.temporal_patch_size,
                quant_scheme=dict(
                    quant_type="w8a8h1_sefp",
                    ops={},
                ),
            ),
        ),
    )
    return cfg_name.lower(), Config(format_model_name(cfg))


def _parse_bucket(bucket) -> tuple[int, int]:
    if isinstance(bucket, dict):
        return int(bucket["max_size_h"]), int(bucket["max_size_w"])
    return int(bucket[0]), int(bucket[1])


def _validate_buckets(buckets: list[tuple[int, int]], patch_size: int) -> None:
    factor = patch_size * SPATIAL_MERGE_SIZE
    for bucket_h, bucket_w in buckets:
        if bucket_h <= 0 or bucket_w <= 0:
            raise ValueError(f"Invalid static visual bucket {(bucket_h, bucket_w)}")
        if bucket_h % factor != 0 or bucket_w % factor != 0:
            raise ValueError(f"Static visual bucket {(bucket_h, bucket_w)} must be divisible by {factor}")


def _load_visual_bucket_config(args, cfg) -> tuple[Any, list[tuple[int, int]], tuple[int, int]]:
    visual_cfg = Config.fromfile(args.visual_buckets_config)
    if args.model:
        visual_cfg.model.hf_model = args.model
    fallback_bucket = (int(cfg.model.visual_config.max_size_h), int(cfg.model.visual_config.max_size_w))
    buckets = [_parse_bucket(bucket) for bucket in visual_cfg.visual_buckets]
    if fallback_bucket not in buckets:
        buckets.append(fallback_bucket)
    _validate_buckets(buckets, int(cfg.model.visual_config.patch_size))
    return visual_cfg.model, sorted(set(buckets), key=lambda item: (item[0] * item[1], item[0], item[1])), fallback_bucket


def _resolve_exported_dir(work_dir: Path, meta_info) -> Path:
    prefill_hmonnx = getattr(meta_info, "prefill_hmonnx", None)
    if prefill_hmonnx:
        matches = [
            path
            for path in work_dir.iterdir()
            if path.is_dir() and (path / str(prefill_hmonnx)).exists()
        ]
        if len(matches) == 1:
            return matches[0]

    meta_files = sorted(work_dir.glob("*/golden_meta_info.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if meta_files:
        return meta_files[0].parent
    raise RuntimeError(f"Cannot locate exported HMONNX directory under {work_dir}")


def _relative_to_export_dir(path: str | Path, exported_dir: Path) -> str:
    path = Path(path)
    exported_dir_abs = exported_dir.resolve()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([exported_dir / path, Path.cwd() / path])
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return candidate.resolve().relative_to(exported_dir_abs).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def _find_existing_bucket_hmonnx(bucket_dir: Path) -> Path | None:
    for pattern in ("*with_act.onnx", "*.onnx"):
        matches = sorted(bucket_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def _export_visual_bucket(
    visual_cfg_model: Any,
    bucket: tuple[int, int],
    exported_dir: Path,
    skip_existing: bool,
    logger,
) -> str:
    max_size_h, max_size_w = bucket
    bucket_dir = exported_dir / "visual_buckets" / f"{max_size_h}x{max_size_w}"
    if skip_existing:
        existing_hmonnx = _find_existing_bucket_hmonnx(bucket_dir)
        if existing_hmonnx is not None:
            logger.info(f"Skip existing static visual bucket {bucket}: {existing_hmonnx}")
            return _relative_to_export_dir(existing_hmonnx, exported_dir)

    cfg_model = copy.deepcopy(visual_cfg_model)
    cfg_model.max_size_h = max_size_h
    cfg_model.max_size_w = max_size_w
    cfg_model.model_name = f"{cfg_model.model_name}_{max_size_h}x{max_size_w}"
    model_cfg = AutoLLMConfig.from_pretrained(cfg_model)
    model_cfg.work_dir = str(bucket_dir)
    visual_model = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(visual_model).__name__ == "XHQwen2VLVisualModel", (
        f"Expected model type XHQwen2VLVisualModel, but got {type(visual_model).__name__}"
    )
    logger.info(f"Exporting static visual bucket {bucket} to {bucket_dir}")
    visual_meta = visual_model.export_hmonnx(str(bucket_dir))
    return _relative_to_export_dir(visual_meta.hmonnx, exported_dir)


def _write_visual_bucket_manifest(
    exported_dir: Path,
    meta_info,
    visual_cfg_model: Any,
    buckets: list[tuple[int, int]],
    fallback_bucket: tuple[int, int],
    skip_existing: bool,
    logger,
) -> None:
    default_visual_meta = getattr(meta_info, "visual_config", None)
    default_bucket = (
        int(getattr(default_visual_meta, "image_size_h", fallback_bucket[0])),
        int(getattr(default_visual_meta, "image_size_w", fallback_bucket[1])),
    )
    default_hmonnx = getattr(default_visual_meta, "hmonnx", None)
    if default_hmonnx is None:
        raise RuntimeError("Default visual HMONNX path is missing from exported metadata.")

    manifest_buckets = []
    for bucket in buckets:
        if bucket == default_bucket:
            hmonnx = _relative_to_export_dir(default_hmonnx, exported_dir)
        else:
            hmonnx = _export_visual_bucket(visual_cfg_model, bucket, exported_dir, skip_existing, logger)
        manifest_buckets.append(
            {
                "max_size_h": bucket[0],
                "max_size_w": bucket[1],
                "hmonnx": hmonnx,
            }
        )

    manifest = {
        "buckets": manifest_buckets,
        "fallback_bucket": {
            "max_size_h": fallback_bucket[0],
            "max_size_w": fallback_bucket[1],
        },
        "patch_size": int(getattr(default_visual_meta, "patch_size", visual_cfg_model.patch_size)),
        "spatial_merge_size": int(getattr(default_visual_meta, "spatial_merge_size", SPATIAL_MERGE_SIZE)),
        "temporal_patch_size": int(
            getattr(default_visual_meta, "temporal_patch_size", visual_cfg_model.temporal_patch_size)
        ),
    }
    manifest_path = exported_dir / MINERU_VISUAL_BUCKETS_MANIFEST
    json.dump(manifest, open(manifest_path, "w"), indent=4)
    logger.info(f"MinerU static visual bucket manifest saved to {manifest_path}")


def main(args):
    config_file = args.config
    model_dir = args.model
    if config_file and model_dir == DEFAULT_HF_MODEL_DIR:
        args.model = ""
        model_dir = ""
    if config_file and model_dir:
        raise ValueError("Cannot specify both --config and --model at the same time.")
    if config_file:
        cfg_name = Path(config_file).stem
        cfg = Config.fromfile(args.config)
    elif model_dir:
        cfg_name, cfg = _build_cfg_from_model(args)
    else:
        raise ValueError("Either --config or --model must be specified.")

    if args.debug:
        cfg_name += "_debug"

    work_dir = Path("./work_dirs") / cfg_name
    if work_dir.exists():
        if args.force:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            from loguru import logger

            logger.info(f"Exported model already exists at {work_dir}, use --force to overwrite.")
            answer = input("Continue and overwrite? [y/N]: ").strip().lower()
            if answer != "y":
                return -1
    work_dir.mkdir(parents=True, exist_ok=True)

    xhquant_init(str(work_dir / "export_hmonnx.log"), args.debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()
    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    cfg.dump(work_dir / f"{cfg_name}.py")

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")
    model_cfg: XHQwen2VLModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    assert type(model_cfg).__name__ == "XHQwen2VLModelConfig", (
        f"Expected model config type XHQwen2VLModelConfig, but got {type(model_cfg).__name__}"
    )
    xh_model: XHQwen2VLModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHQwen2VLModel", (
        f"Expected model type XHQwen2VLModel, but got {type(xh_model).__name__}"
    )
    if args.visual_buckets_config:
        visual_cfg_model, buckets, fallback_bucket = _load_visual_bucket_config(args, cfg)
        logger.info(f"Static visual buckets: {buckets}; fallback: {fallback_bucket}")
    else:
        visual_cfg_model = None
        buckets = []
        fallback_bucket = None
        logger.info("Static visual bucket export disabled.")
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert2hmonnx", logger):
        meta_info = xh_model.export_hmonnx(str(work_dir))
        exported_dir = _resolve_exported_dir(work_dir, meta_info)
        if visual_cfg_model is not None:
            _write_visual_bucket_manifest(
                exported_dir=exported_dir,
                meta_info=meta_info,
                visual_cfg_model=visual_cfg_model,
                buckets=buckets,
                fallback_bucket=fallback_bucket,
                skip_existing=args.skip_existing_visual_buckets,
                logger=logger,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_llm_1_2b_xh2a_4k.py")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--force", action="store_true", help="Whether to force export even if the model exists.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="Qwen2VLForConditionalGeneration",
        choices=support_llm_model_types,
    )
    parser.add_argument("--chip-arch", type=str, default="XH2a", choices=["XH2a", "YueHui"])
    parser.add_argument("--model", type=str, default=DEFAULT_HF_MODEL_DIR)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--prefill-chunk-length", type=int, default=256)
    parser.add_argument("--quant-type", default="w8a8h1_sefp")
    parser.add_argument("--quant-weight", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=1260)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--temporal-patch-size", type=int, default=2)
    parser.add_argument(
        "--visual-buckets-config",
        type=str,
        default="configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_visual_buckets_1_2b_xh2a.py",
    )
    parser.add_argument(
        "--skip-existing-visual-buckets",
        action="store_true",
        help="Skip visual bucket export when an existing bucket HMONNX is found.",
    )
    args = parser.parse_args()
    main(args)
