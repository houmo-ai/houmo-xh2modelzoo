import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen3_5 import XHQwen3_5Model, XHQwen3_5ModelConfig


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _looks_like_export_work_dir(work_dir: Path) -> bool:
    marker_names = {
        "export_hmonnx.log",
        "golden_meta_info.json",
    }
    if any((work_dir / marker_name).is_file() for marker_name in marker_names):
        return True
    return any(path.is_file() for path in work_dir.glob("hmquant_*/golden_meta_info.json"))


def _remove_existing_work_dir(work_dir: Path, *, user_supplied: bool) -> None:
    import shutil

    resolved_work_dir = work_dir.resolve()
    cwd = Path.cwd().resolve()
    safe_default_root = (cwd / "work_dirs").resolve()
    unsafe_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        cwd,
        safe_default_root,
    }
    if work_dir.is_symlink() or resolved_work_dir in unsafe_paths:
        raise ValueError(f"Refusing to delete unsafe work_dir: {resolved_work_dir}")

    if not user_supplied:
        if not _is_relative_to(resolved_work_dir, safe_default_root):
            raise ValueError(f"Default work_dir resolved outside work_dirs: {resolved_work_dir}")
    elif not _looks_like_export_work_dir(work_dir):
        raise ValueError(
            f"Refusing to delete user-supplied work_dir without export markers: {resolved_work_dir}"
        )

    shutil.rmtree(resolved_work_dir)


def main(args):
    config_file = args.config
    if not config_file:
        raise ValueError("--config is required. Model/export shape must be selected by config file.")

    cfg_name = Path(config_file).stem
    if args.debug:
        cfg_name += "_debug"

    user_supplied_work_dir = bool(args.work_dir)
    work_dir = Path(args.work_dir) if user_supplied_work_dir else Path("./work_dirs") / cfg_name
    if work_dir.exists():
        if args.force:
            _remove_existing_work_dir(work_dir, user_supplied=user_supplied_work_dir)
        else:
            from loguru import logger

            logger.info(f"Exported model already exists at {work_dir}, use --force to overwrite.")
            answer = input("Continue and overwrite? [y/N]: ").strip().lower()
            if answer != "y":
                return -1
            # visual部分的onnx不会重新导出，需要注意

    work_dir.mkdir(parents=True, exist_ok=True)

    xhquant_init(str(work_dir / "export_hmonnx.log"), args.debug)
    seed = args.seed
    set_random_seed(seed)
    logger = get_xhquant_logger()

    cfg = Config.fromfile(config_file)
    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    cfg.dump(work_dir / f"{cfg_name}.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {torch.float16}")
    model_cfg: XHQwen3_5ModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    logger.info(f"Resolved model config type: {type(model_cfg).__name__}")
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model: XHQwen3_5Model = AutoLLMModel.from_pretrained(config=model_cfg)
    logger.info(f"Resolved model type: {type(xh_model).__name__}")
    xh_model.work_dir = str(work_dir)
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert2hmonnx", logger):
        xh_model.export_hmonnx(str(work_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Merak Qwen3.5/Qwen3.5-MoE HMONNX from a checked-in config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="model config file; all model/export choices must live in this config",
    )
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--force", action="store_true", help="Whether to force export even if the model exists")
    parser.add_argument("--seed", type=int, default=1024, help="random seed")
    parser.add_argument(
        "--work-dir",
        "--work_dir",
        dest="work_dir",
        type=str,
        default="",
        help="optional output work directory; defaults to work_dirs/<config_stem>",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
