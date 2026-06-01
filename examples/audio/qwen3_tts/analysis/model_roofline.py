import argparse
from collections import defaultdict
from typing import cast

import torch
from loguru import logger

# from qwen_tts import Qwen3TTSModel
# from xh_model_zoo.xh_llm.models.qwen3_tts._common import *  # noqa: F401, F403
from xh_model_zoo.xh_llm.models.qwen3_tts.qwen3_tts import XHQwen3TTSModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Qwen3 TTS model")
    parser.add_argument(
        "--model",
        type=str,
        default="./data/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign/",
        help="Model name or path",
    )
    args = parser.parse_args()
    model_dir = args.model
    model = XHQwen3TTSModel.from_pretrained(
        model_dir,
        device_map="cuda:0",
        dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    model = cast(XHQwen3TTSModel, model)
    talker = model.model.talker.model
    total_params = sum(parameter.numel() for parameter in talker.parameters())
    trainable_params = sum(parameter.numel() for parameter in talker.parameters() if parameter.requires_grad)

    param_stats_by_type: dict[str, dict[str, object]] = defaultdict(
        lambda: {"module_count": 0, "param_count": 0, "sample_modules": []}
    )

    for module_name, module in talker.named_modules():
        direct_params = list(module.parameters(recurse=False))
        if not direct_params:
            continue

        module_type = type(module).__name__
        param_count = sum(parameter.numel() for parameter in direct_params)
        type_stats = param_stats_by_type[module_type]
        type_stats["module_count"] = int(type_stats["module_count"]) + 1
        type_stats["param_count"] = int(type_stats["param_count"]) + param_count
        sample_modules = cast(list[str], type_stats["sample_modules"])
        if len(sample_modules) < 3:
            sample_modules.append(module_name or "<root>")

    logger.info(f"talker_total_params={total_params}")
    logger.info(f"talker_total_params_b={total_params / 1e9:.6f}B")
    logger.info(f"talker_trainable_params={trainable_params}")
    logger.info("Parameter stats by module type:")

    for module_type, type_stats in sorted(
        param_stats_by_type.items(),
        key=lambda item: int(item[1]["param_count"]),
        reverse=True,
    ):
        param_count = int(type_stats["param_count"])
        module_count = int(type_stats["module_count"])
        sample_modules = cast(list[str], type_stats["sample_modules"])
        logger.info(
            f"  {module_type}: modules={module_count}, params={param_count}, "
            f"params_b={param_count / 1e9:.6f}B, ratio={param_count / total_params:.2%}, "
            f"samples={sample_modules}"
        )
