import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import TextStreamer

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.xh_llm import (
    AutoLLMConfig,
    AutoLLMModel,
    LLMInferenceContextManager,
    LLMModelState,
)
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen2_vl import XHQwen2VLModel, XHQwen2VLModelConfig, XHQwen2VLProcessor


def _build_messages(prompt, image_path, max_size_h, max_size_w):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "resized_height": max_size_h,
                    "resized_width": max_size_w,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main(args):
    cfg_name = Path(args.config).stem
    eval_type = args.eval_type
    if args.debug:
        cfg_name += "_debug"

    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / f"generate_{eval_type}.log"), args.debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()

    cfg = Config.fromfile(args.config)
    if args.model:
        cfg.model.hf_model = args.model
        cfg.model.visual_config.hf_model = args.model
    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    cfg.dump(work_dir / Path(args.config).name)

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")
    model_cfg: XHQwen2VLModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    model_cfg.work_dir = str(work_dir)
    model_cfg.visual_config.work_dir = str(work_dir / "visual")

    assert type(model_cfg).__name__ == "XHQwen2VLModelConfig", (
        f"Expected model config type XHQwen2VLModelConfig, but got {type(model_cfg).__name__}"
    )
    xh_model: XHQwen2VLModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHQwen2VLModel", (
        f"Expected model type XHQwen2VLModel, but got {type(xh_model).__name__}"
    )
    eval_state = LLMModelState.from_string(eval_type)
    xh_model.set_state(eval_state)

    max_size_h = model_cfg.visual_config.max_size_h
    max_size_w = model_cfg.visual_config.max_size_w
    messages = _build_messages(args.prompt, args.image_path, max_size_h, max_size_w)
    processor: XHQwen2VLProcessor = xh_model.get_tf_processor()
    model_inputs = processor.apply_chat_template(messages)
    model_inputs = model_inputs.to(device)

    streamer = TextStreamer(processor.tokenizer)
    xh_model.to(device=device, dtype=dtype)
    xh_model.eval()

    contexts = [
        TimeProfiler("llm_generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(xh_model),
        torch.no_grad(),
    ]
    with ContextManagers(contexts):
        generated_ids = xh_model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    logger.info(f"{'-' * 20} {eval_state} Output {'-' * 20}")
    logger.info(f"{output_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/qwen2_vl/2b/qwen2_vl_llm_2b_xh2a_4k.py",
    )
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--eval-type", type=str, default="wrap", choices=LLMModelState.get_all_values())
    parser.add_argument("--image-path", type=str, default="./data/images/qwen2_vl_demo.jpeg")
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    args = parser.parse_args()
    main(args)
