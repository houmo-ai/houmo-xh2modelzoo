import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import TextStreamer

from xhmodel_merak.xh_llm import (
    AutoLLMConfig,
    AutoLLMModel,
    LLMInferenceContextManager,
    LLMModelState,
)
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen3_vl import XHQwen3VLVisualConfig, XHQwen3VLVisualModel
    from xhmodel_merak.xh_llm.models.qwen3_vl.qwen3_vl_processor import XHQwen3VLProcessor

from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init


def main(args):
    cfg_name = Path(args.config).stem
    eval_type = args.eval_type
    debug = args.debug
    if debug:
        cfg_name += "_debug"

    args.work_dir = str(Path("./work_dirs") / cfg_name)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(work_dir / f"generate_{eval_type}.log")

    xhquant_init(log_file, debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()
    # 使用config文件,代替命令行参数,方便调试不同的配置
    cfg = Config.fromfile(args.config)
    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = work_dir / Path(args.config).name
    cfg.dump(config_file)

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")
    model_cfg: XHQwen3VLVisualConfig = AutoLLMConfig.from_pretrained(cfg.model)
    model_cfg.work_dir = str(work_dir)  # visual部分需要一个work_dir，存放中间的onnx

    assert type(model_cfg).__name__ == "XHQwen3VLVisualConfig", (
        f"Expected model config type XHQwen3VLVisualConfig, but got {type(model_cfg).__name__}"
    )
    xh_model: XHQwen3VLVisualModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHQwen3VLVisualModel", (
        f"Expected model type XHQwen3VLVisualModel, but got {type(xh_model).__name__}"
    )
    eval_type = LLMModelState.from_string(eval_type)
    xh_model.set_state(eval_type)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": args.image_path,  # 图片路径，模型会自动加载图片
                    "resized_height": model_cfg.max_size_h,
                    "resized_width": model_cfg.max_size_w,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # processor: Qwen3VLProcessor = xh_model.get_vl_processor()
    # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info(messages, image_patch_size=model_cfg.patch_size)
    # processor.image_processor.max_pixels = max(
    #     model_cfg.max_size_w * model_cfg.max_size_h + 1, processor.image_processor.max_pixels
    # )

    # model_inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )

    processor: XHQwen3VLProcessor = xh_model.get_tf_processor()
    # processor.config.patch_size = patch_size
    # processor.config.max_size_h = max_size_h
    # processor.config.max_size_w = max_size_w
    model_inputs = processor.apply_chat_template(messages)

    model_inputs = model_inputs.to(device)
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True)
    # logger.info(f"{'-' * 20} stream output {'-' * 20}")
    xh_model.to(device=device, dtype=dtype)
    xh_model.eval()
    contexts = [
        TimeProfiler("visual_generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(xh_model),
        torch.no_grad(),
    ]
    with ContextManagers(contexts):
        generated_ids = xh_model.generate(
            **model_inputs,
            max_new_tokens=128,
            streamer=streamer,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    logger.info(f"{'-' * 20} Output {'-' * 20}")
    logger.info(f"{output_text}")


if __name__ == "__main__":
    eval_types = LLMModelState.get_all_values()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/qwen3_vl/2b/qwen3_vl_visual_2b_xh2a_2k.py",
    )
    parser.add_argument("--model", type=str)
    parser.add_argument("--eval-type", type=str, default="quanted_fast", choices=eval_types)
    parser.add_argument("--image-path", type=str, default="./data/images/demo_qwen3_vl.jpeg")
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")

    args = parser.parse_args()
    main(args)
