import argparse
import os.path as osp
from pathlib import Path


from xh_model_zoo.utils import MemoryTracker
from xh_model_zoo.utils import TimeProfiler
from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen2_vl import Qwen2VLConvertConfig
from xh_model_zoo.xh_llm.models.qwen2_vl import VisualConfig

from xhquant.api import DeviceType, get_root_logger, xhquant_init, QuantScheme  # isort:skip


def main(args):
    args.batch_size = 1
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a

    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    # quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"
    config = Qwen2VLConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        visual_config=VisualConfig(
            image_max_size=1204,
            patch_size=14,
        ),
    )

    prefix = f"{model_name}-{target_device}-batch_{args.batch_size}-{args.context_length//1024}k-{quant_type}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)

    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, None, config, work_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default="data/models/Qwen2-VL-2B-Instruct-AWQ")
    parser.add_argument("--context-length", type=int, default=2048, help="max sequence length")
    parser.add_argument("--input-sequence-length", type=int, default=256, help="input sequence length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")

    args = parser.parse_args()
    main(args)
