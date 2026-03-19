import numpy as np
import argparse
import os.path as osp
from pathlib import Path

from xh_model_zoo.xh_llm.models.groot.gr00t.policy.gr00t_policy import Gr00tPolicy
from xh_model_zoo.xh_llm.models.groot.gr00t.data.embodiment_tags import EmbodimentTag
import torch
from xh_model_zoo.xh_llm.models.groot.groot_vision_convet import Groot_ConverterXH2a
from xh_model_zoo.xh_llm.models.groot.groot_convert_config import Groot_ConvertConfig

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen3_legacy import Qwen3LegacyConvertConfig
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip
from xh_model_zoo.xh_llm.models.qwen3_legacy import Qwen3LegacyConverterXH2a 

def main(args):
    np.random.seed(42)
    print('Loading GR00T N1.6...')
    with torch.no_grad():
        policy = Gr00tPolicy(
            model_path=args.model,
            embodiment_tag=EmbodimentTag.OXE_DROID,
            device=args.device,
        )
        # policy.model.backbone.model.vision_model

        config = Groot_ConvertConfig()
        config.quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)

        Groot_ConverterXH2a(config)._convert(
            policy.model.backbone.model.vision_model,
            args.output_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default='/data02/datasets/GR00T-N1.6-DROID')
    parser.add_argument("--output_path", type=str, default="work_dirs/groot_droid")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--context-length", type=int, default=2048, help="max sequence length"
    )
    parser.add_argument(
        "--input-sequence-length", type=int, default=256, help="input sequence length"
    )
    parser.add_argument(
        "--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8"
    )
    parser.add_argument(
        "--mix_search", type=str, default=None, help="mix search settings"
    )
    parser.add_argument(
        "--num_logits_to_keep", type=int, default=1, help="not for test ppl"
    )
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    args = parser.parse_args()
    main(args)