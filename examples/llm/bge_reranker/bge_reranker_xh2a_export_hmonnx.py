import argparse
import os.path as osp
from pathlib import Path

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.bge_reranker import BGERerankerConvertConfig

from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    config = BGERerankerConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        quant_scheme=quant_scheme,
    )

    prefix = f"{model_name}-{target_device}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, "BertModel_Reranker", config, str(work_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data02/datasets/bge-reranker-v2_m3")
    parser.add_argument("--batch-size", type=int, default=10, help="batch size")
    parser.add_argument("--context-length", type=int, default=512, help="max sequence length")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8_sefp")
    args = parser.parse_args()
    main(args)
