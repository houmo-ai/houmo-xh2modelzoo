import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image
from transformers import TextStreamer

from xhmodel_merak.xh_llm import (
    AutoLLMConfig,
    AutoLLMModel,
    LLMInferenceContextManager,
    LLMModelState,
)
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen3_5 import XHQwen3_5Model, XHQwen3_5ModelConfig
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import XHQwen3_5MoeModel, XHQwen3_5MoeModelConfig


def main(args):
    prompt = args.prompt
    if Path(prompt).is_file():
        prompt = Path(prompt).read_text()
    image = None
    if args.image_path is not None and len(args.image_path) > 0:
        image = Image.open(args.image_path).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_name = Path(args.config).stem
    if args.debug:
        cfg_name += "_debug"

    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / f"generate_{args.eval_type}.log"), args.debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()

    cfg = Config.fromfile(args.config)
    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    cfg.dump(work_dir / Path(args.config).name)

    model_cfg: XHQwen3_5ModelConfig | XHQwen3_5MoeModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    assert type(model_cfg).__name__ in ["XHQwen3_5ModelConfig", "XHQwen3_5MoeModelConfig"], (
        "Expected model config type XHQwen3_5ModelConfig or "
        f"XHQwen3_5MoeModelConfig, but got {type(model_cfg).__name__}"
    )
    model_cfg.enable_auto_offload = args.auto_offload  # 是否启用自动显存卸载

    enable_prefill_chunk = args.enable_prefill_chunk
    if enable_prefill_chunk:
        model_cfg.enable_prefill_chunk = True  # 开启prefill chunk后，设置chunk长度为512
        model_cfg.use_cache = True

    xh_model: XHQwen3_5Model = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ in ["XHQwen3_5Model", "XHQwen3_5MoeModel"], (
        f"Expected model type XHQwen3_5Model or XHQwen3_5MoeModel, but got {type(xh_model).__name__}"
    )
    xh_model.set_state(LLMModelState.from_string(args.eval_type))

    message = {
        "role": "user",
        "content": [],
    }
    if image is not None:
        message["content"].append(
            {
                "type": "image",
                "image": image,
            },
        )
    message["content"].append(
        {"type": "text", "text": prompt},
    )
    messages = [
        message,
    ]

    processor = xh_model.get_tf_processor()
    tokenizer = processor.tokenizer
    model_inputs = processor.apply_chat_template(
        messages,
        enable_thinking=args.think,
    ).to(device=device)
    streamer = TextStreamer(tokenizer=tokenizer)

    contexts = [
        TimeProfiler("generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(xh_model),
        torch.no_grad(),
    ]
    with ContextManagers(contexts):
        xh_model.to(device=device, dtype=torch.float16)
        xh_model.eval()
        xh_model.generate(
            **model_inputs,
            max_new_tokens=256,
            streamer=streamer,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Qwen3.5 model config path. Legacy xh2a Python config defaults were removed; pass an explicit compatible config or use AutoLLMWorkflow.",
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        default="wrap",
        choices=LLMModelState.get_all_values(),
    )
    parser.add_argument("--image-path", type=str)
    parser.add_argument("--prompt", type=str, default="你是谁？")
    parser.add_argument("--think", action="store_true", help="enable think mode")
    parser.add_argument("--debug", action="store_true", help="run in debug mode")
    parser.add_argument(
        "--auto-offload",
        action="store_true",
        help="Whether to enable auto offload, only for debug and development",
    )
    parser.add_argument(
        "--enable-prefill-chunk",
        action="store_true",
        help="Whether to enable prefill chunk, only for debug and development",
    )
    args = parser.parse_args()
    main(args)
