import argparse
from pathlib import Path

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name
from xhquant.api import Config, xhquant_init


def _build_cfg_from_model(args):
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
            visual_config=dict(
                max_size_w=args.image_width,
                max_size_h=args.image_height,
            ),
            audio_config=dict(
                sampling_rate=args.audio_sampling_rate,
            ),
        ),
    )
    cfg = format_model_name(cfg)
    return Config(cfg)


def main(args):
    cfg = Config.fromfile(args.config) if args.config else _build_cfg_from_model(args)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "export_hmonnx.log"), args.debug)

    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.work_dir = str(work_dir)
    xh_model.export_hmonnx(str(work_dir))


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
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--image-height", type=int, default=448)
    parser.add_argument("--audio-sampling-rate", type=int, default=16000)
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())
