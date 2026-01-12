import argparse
import os.path as osp
from pathlib import Path

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_tts_dvae_convert_config import MinicpmoTTSDVAEConvertConfig

from xhquant.api import DeviceType, QuantScheme, get_root_logger, xhquant_init  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)

    config = MinicpmoTTSDVAEConvertConfig(
        quant_scheme=quant_scheme,
        video=args.video,
        audio=args.audio,
        debug=args.debug,
        valid=args.valid,
        image_slice_max_size=[40, 40],
    )

    prefix = f"{model_name}-{target_device}-tts-dvae-{quant_type}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, "MiniCPMOTTSDVAEEncoder", config, str(work_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/datasets/MiniCPM-o-2_6", type=str, help="HuggingFace model path")
    parser.add_argument("--video", type=str, default="examples/llm/minicpmo/assets/Skiing.mp4")
    parser.add_argument("--audio", type=str, default="examples/llm/minicpmo/assets/demo.wav")
    parser.add_argument("--debug", type=bool, default=True, help="debug mode")
    parser.add_argument("--valid", type=bool, default=True, help="check hmonnx mode")
    parser.add_argument("--quant-type", default="w8a8h0_sefp", help="quant type, default is w8a8")
    args = parser.parse_args()
    main(args)

