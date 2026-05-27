import argparse
from pathlib import Path

import torch

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name
from xhquant.api import Config, HMONNXGoldenInference, get_xhquant_logger, xhquant_init
from xhquant.core import CacheTensor


def _build_cfg_from_model(args):
    visual_config = {
        "export_mode": args.vision_export_mode,
    }
    if args.image_width is not None:
        visual_config["max_size_w"] = args.image_width
    if args.image_height is not None:
        visual_config["max_size_h"] = args.image_height

    cfg = dict(
        chip_arch=args.chip_arch,
        model=dict(
            model_type=args.model_type,
            hf_model=args.model,
            model_name=Path(args.model).name,
            context_max_length=args.context_length,
            prefill_chunk_length=args.prefill_chunk_length,
            use_cache=True,
            num_logits_to_keep=1,
            quant_scheme=dict(
                quant_type=args.quant_type,
            ),
            visual_config=visual_config,
            audio_config=dict(
                sampling_rate=args.audio_sampling_rate,
            ),
        ),
    )
    cfg = format_model_name(cfg)
    return Config(cfg)


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


def main(args):
    cfg = Config.fromfile(args.config) if args.config else _build_cfg_from_model(args)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "export_hmonnx.log"), args.debug)

    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.work_dir = str(work_dir)
    xh_model.export_hmonnx(str(work_dir))

    if args.golden:
        logger = get_xhquant_logger()
        logger.info("Generating golden data for all exported modules...")
        exported_dirs = sorted(work_dir.glob("hmquant_*"))
        exported_dir = str(exported_dirs[-1]) if exported_dirs else str(work_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _generate_all_golden(exported_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--model", type=str, default="/data01/datasets/gemma-4-E2B-it")
    parser.add_argument("--work-dir", type=str, default="work_dirs/gemma4_e2b_it")
    parser.add_argument("--model-type", type=str, default="Gemma4ForConditionalGeneration")
    parser.add_argument("--chip-arch", type=str, default="XH2a")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--prefill-chunk-length", type=int, default=256)
    parser.add_argument("--quant-type", type=str, default="w8a8h1_sefp")
    parser.add_argument("--vision-export-mode", type=str, default="full", choices=["full", "compact"])
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--audio-sampling-rate", type=int, default=16000)
    parser.add_argument("--golden", action="store_true", help="generate golden data for each exported module")
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())
