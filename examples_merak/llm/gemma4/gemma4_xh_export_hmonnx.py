import argparse
import os.path as osp
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name, support_llm_model_types
from xhquant.api import Config, HMONNXGoldenInference, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.core import CacheTensor
from xhquant.utils import MemoryTracker, TimeProfiler

if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.gemma4 import XHGemma4Model, XHGemma4ModelConfig


def _generate_golden_for_hmonnx(hmonnx_file: str, golden_dir: str, device: str = "cuda"):
    logger = get_xhquant_logger()
    if not Path(hmonnx_file).exists():
        logger.warning(f"HMONNX file not found, skipping golden: {hmonnx_file}")
        return
    logger.info(f"Generating golden for: {hmonnx_file}")
    session = HMONNXGoldenInference(hmonnx_file)
    session.to(torch.device(device))
    session.save_golden = True
    session.golden_dir = golden_dir
    session.initialize()
    input_names = session.get_input_names()
    net_inputs = []
    for input_name in input_names:
        input_tensor_info = session.get_input(input_name)
        if input_tensor_info.dtype in [torch.float32, torch.float16, torch.float64]:
            inp = torch.randn(input_tensor_info.shape, dtype=input_tensor_info.dtype, device=device)
        elif input_tensor_info.dtype in [torch.int32, torch.int64, torch.int16]:
            inp = torch.randint(0, 10, input_tensor_info.shape, dtype=input_tensor_info.dtype, device=device)
        elif input_tensor_info.dtype == torch.bool:
            inp = torch.randint(0, 2, input_tensor_info.shape, dtype=input_tensor_info.dtype, device=device)
        else:
            raise NotImplementedError(f"dtype {input_tensor_info.dtype} not supported for golden generation")
        if "past_key_cache" in input_name or "past_value_cache" in input_name:
            inp = CacheTensor(inp)
        net_inputs.append(inp)
    session(*net_inputs)
    logger.info(f"Golden saved to: {golden_dir}")


def _generate_all_golden(exported_dir: str, device: str = "cuda"):
    logger = get_xhquant_logger()
    exported_path = Path(exported_dir).resolve()
    hmonnx_files = list(exported_path.rglob("*_with_act.onnx"))
    if not hmonnx_files:
        logger.warning(f"No hmonnx files found in {exported_dir}")
        return
    for hmonnx_file in hmonnx_files:
        golden_dir = str(hmonnx_file.parent / f"{hmonnx_file.stem}_golden")
        _generate_golden_for_hmonnx(str(hmonnx_file), golden_dir, device)


def _build_cfg_from_model(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = args.chip_arch
    quant_type = args.quant_type
    prefill_chunk_length = args.prefill_chunk_length
    context_length = args.context_length

    cfg_name = f"{target_device}_{model_name}_{quant_type}_{prefill_chunk_length}_{context_length // 1024}k_cli"
    cfg = dict(
        chip_arch=target_device,
        model=dict(
            model_type=args.model_type,
            hf_model=hf_model_path,
            model_name=model_name,
            context_max_length=context_length,
            prefill_chunk_length=prefill_chunk_length,
            max_pe_length=max(context_length, 32768),
            use_cache=True,
            num_logits_to_keep=1,
            quant_scheme=dict(
                quant_type=quant_type,
            ),
            quant_weight=args.quant_weight,
            only_first_block=False,
            visual_config=dict(
                image_seq_length=280,
                patch_size=16,
                pooling_kernel_size=3,
                quant_scheme=dict(quant_type=quant_type),
            ),
        ),
    )
    cfg = format_model_name(cfg)
    return cfg_name.lower(), Config(cfg)


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

    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "export_hmonnx.log"), args.debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()
    cfg.seed = seed
    cfg.dump(work_dir / f"{cfg_name}.py")

    model_cfg: XHGemma4ModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    xh_model: XHGemma4Model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.work_dir = str(work_dir)
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert2hmonnx", logger):
        xh_model.export_hmonnx(str(work_dir))

    if args.golden:
        logger.info("Generating golden data for all exported modules...")
        exported_dirs = sorted(work_dir.glob("hmquant_*"))
        exported_dir = str(exported_dirs[-1]) if exported_dirs else str(work_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _generate_all_golden(exported_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/gemma4/31b/gemma4_31b_it_xh2a_2k.py",
        help="model config file for development and debugging",
    )
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--model-type", type=str, default="Gemma4ForConditionalGeneration", choices=support_llm_model_types)
    parser.add_argument("--chip-arch", type=str, default="XH2a", choices=["XH2a", "YueHui"])
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--context-length", type=int, default=2048, help="max context sequence length")
    parser.add_argument("--prefill-chunk-length", type=int, default=256, help="prefill chunk length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type")
    parser.add_argument("--quant-weight", type=str, default=None, help="optional quant weight path")
    parser.add_argument("--golden", action="store_true", help="generate golden data for each exported module")
    args = parser.parse_args()
    main(args)
