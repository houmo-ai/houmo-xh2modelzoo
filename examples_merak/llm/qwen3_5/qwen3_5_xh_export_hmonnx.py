import argparse
import os.path as osp
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import AutoConfig

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name, support_llm_model_types
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen3_5 import XHQwen3_5Model, XHQwen3_5ModelConfig


def _infer_model_type_from_hf_config(hf_model_path: str) -> str:
    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    architectures = getattr(hf_config, "architectures", None)
    if not architectures:
        raise ValueError(f"architectures is missing in HF config: {hf_model_path}")
    return architectures[0]


def _is_quantized_hf_model_dir(path: str | None) -> bool:
    if not path:
        return False
    model_dir = Path(path)
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        return False
    hf_config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    return getattr(hf_config, "quantization_config", None) is not None


def _default_quant_type(model_type: str) -> str:
    if "Moe" in model_type:
        return "w8a8h0_ssfp"
    return "w8a8h1_sefp"


def _build_cfg_from_model(args):
    source_model_path = osp.normpath(osp.abspath(args.model))
    hf_model_path = source_model_path
    quant_weight = args.quant_weight
    if _is_quantized_hf_model_dir(quant_weight):
        hf_model_path = osp.normpath(osp.abspath(quant_weight))
        quant_weight = None
    elif quant_weight:
        quant_weight = osp.normpath(osp.abspath(quant_weight))

    model_name = Path(source_model_path).name
    model_type = args.model_type or _infer_model_type_from_hf_config(hf_model_path)
    target_device = args.chip_arch
    quant_type = args.quant_type or _default_quant_type(model_type)
    prefill_chunk_length = args.prefill_chunk_length
    context_length = args.context_length
    max_pe_length = args.max_pe_length or max(context_length, 32768)

    cfg_name = f"{target_device}_{model_name}_{quant_type}_{prefill_chunk_length}_{context_length // 1024}k_cli"
    cfg = dict(
        chip_arch=target_device,
        dtype=args.dtype,
        release=dict(
            xh_version=args.release_xh_version,
            modelscope_name=args.release_modelscope_name,
            wmix_amix=args.release_wmix_amix,
            date=args.release_date,
            package_release=args.package_release,
        ),
        export_options=dict(
            spec_draft_head_weight_bits=args.spec_draft_head_weight_bits,
        ),
        model=dict(
            model_type=model_type,
            hf_model=hf_model_path,
            model_name=model_name,
            batch_size=args.batch_size,
            context_max_length=context_length,
            prefill_chunk_length=prefill_chunk_length,
            max_pe_length=max_pe_length,
            use_cache=True,
            num_logits_to_keep=args.num_logits_to_keep,
            linear_attention_mode=args.linear_attention_mode,
            linear_chunk_size=args.linear_chunk_size,
            split_conv_cache=args.split_conv_cache,
            normalize_force_fp32=args.normalize_force_fp32,
            use_manual_depthwise_conv1d=args.use_manual_depthwise_conv1d,
            spec_decode_mode=args.spec_decode_mode,
            num_draft_tokens=args.num_draft_tokens,
            quant_scheme=dict(
                quant_type=quant_type,
            ),
            quant_weight=quant_weight,
            only_first_block=False,
        ),
    )
    cfg["model"]["visual_config"] = dict(
        max_size_w=args.max_size_w,
        max_size_h=args.max_size_h,
        quant_scheme=dict(
            quant_type=quant_type,
            ops={},
        ),
    )
    if args.spec_decode_mode == "dflash":
        if not args.dflash_model_dir:
            raise ValueError("--dflash-model-dir is required when --spec-decode-mode=dflash")
        cfg["model"]["output_hidden_state_indices"] = []
        cfg["model"]["dflash_config"] = dict(
            hf_model=osp.normpath(osp.abspath(args.dflash_model_dir)),
            target_model_dir=hf_model_path,
            mode="context",
            batch_size=args.batch_size,
            input_sequence_length=prefill_chunk_length,
            max_sequence_length=context_length,
            max_pe_length=max_pe_length,
        )
    elif args.spec_decode_mode == "mtp":
        cfg["model"]["output_post_norm_hidden"] = True
        cfg["model"]["mtp_config"] = dict(
            batch_size=args.batch_size,
            input_sequence_length=1,
            context_max_length=context_length,
            max_pe_length=max_pe_length,
            use_cache=True,
        )
    cfg = format_model_name(cfg)
    return cfg_name.lower(), Config(cfg)


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
    model_dir = args.model
    if config_file and model_dir:
        raise ValueError("Cannot specify both --config and --model at the same time.")
    if config_file:
        cfg_name = Path(config_file).stem
        cfg = Config.fromfile(config_file)
    elif model_dir:
        cfg_name, cfg = _build_cfg_from_model(args)
    else:
        raise ValueError("Either --config or --model must be specified.")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="model config file for development and debugging",
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
        help="optional output work directory; defaults to work_dirs/<generated_cfg_name>",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=support_llm_model_types,
        help="optional override; defaults to HF AutoConfig.architectures[0] when --model is used",
    )
    parser.add_argument("--chip-arch", type=str, default="XH2a", choices=["XH2a", "YueHui"])
    parser.add_argument("--model", type=str, default="")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["auto", "fp16", "float16", "half", "fp32", "float32", "bf16", "bfloat16"],
        help="compute dtype recorded in config for parity with legacy examples/llm entrypoints",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--context-length", type=int, default=2048, help="max context sequence length")
    parser.add_argument("--prefill-chunk-length", type=int, default=256, help="prefill chunk length")
    parser.add_argument("--max-pe-length", type=int, default=None, help="max RoPE table length")
    parser.add_argument("--num-logits-to-keep", type=int, default=1, help="number of final logits to keep")
    parser.add_argument(
        "--linear-attention-mode",
        type=str,
        default="auto",
        choices=["auto", "chunk", "recurrent"],
        help="linear attention computation mode",
    )
    parser.add_argument("--linear-chunk-size", type=int, default=64, help="linear attention chunk size")
    parser.add_argument("--quant-type", default=None, help="quant type")
    parser.add_argument("--quant-weight", type=str, default=None, help="optional quant weight path")
    parser.add_argument("--max-size-w", type=int, default=448, help="vision branch max width")
    parser.add_argument("--max-size-h", type=int, default=448, help="vision branch max height")
    parser.add_argument(
        "--spec-decode-mode",
        "--spec_decode_mode",
        dest="spec_decode_mode",
        type=str,
        default=None,
        choices=["none", "mtp", "dflash"],
        help="optional speculative decoding draft export mode",
    )
    parser.add_argument(
        "--dflash-model-dir",
        "--dflash_model_dir",
        dest="dflash_model_dir",
        type=str,
        default=None,
        help="DFlash draft model directory; required for --spec-decode-mode=dflash",
    )
    parser.add_argument("--num-draft-tokens", type=int, default=4, help="number of speculative draft tokens")
    parser.add_argument(
        "--spec-draft-head-weight-bits",
        "--spec_draft_head_weight_bits",
        dest="spec_draft_head_weight_bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="recorded draft lm_head weight bits for compatibility with legacy examples/llm args",
    )
    parser.add_argument(
        "--split-conv-cache",
        "--split_conv_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="split q/k/v conv cache tensors; enabled by default",
    )
    parser.add_argument(
        "--normalize-force-fp32",
        "--normalize_force_fp32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="force Normalize quant op to fp32; disabled by default",
    )
    parser.add_argument(
        "--use-manual-depthwise-conv1d",
        "--use_manual_depthwise_conv1d",
        action="store_true",
        default=False,
        help="use legacy manual depthwise conv1d tail instead of self.conv1d",
    )
    parser.add_argument("--release-xh-version", "--release_xh_version", dest="release_xh_version", default=None)
    parser.add_argument(
        "--release-modelscope-name", "--release_modelscope_name", dest="release_modelscope_name", default=None
    )
    parser.add_argument("--release-wmix-amix", "--release_wmix_amix", dest="release_wmix_amix", default=None)
    parser.add_argument("--release-date", "--release_date", dest="release_date", default=None)
    parser.add_argument(
        "--package-release",
        "--package_release",
        dest="package_release",
        action="store_true",
        help="record release packaging intent for compatibility with legacy examples/llm args",
    )
    args = parser.parse_args()
    if args.spec_decode_mode == "none":
        args.spec_decode_mode = None
    main(args)
