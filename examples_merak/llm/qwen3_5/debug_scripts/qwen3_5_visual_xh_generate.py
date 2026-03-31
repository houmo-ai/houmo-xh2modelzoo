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
    from xhmodel_merak.xh_llm.models.qwen3_5 import XHQwen3_5_VisualConfig, XHQwen3_5VisionModel
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_processor import XHQwen3_5Processor

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
    model_cfg: XHQwen3_5_VisualConfig = AutoLLMConfig.from_pretrained(cfg.model)
    model_cfg.work_dir = str(work_dir)  # visual部分需要一个work_dir，存放中间的onnx

    assert type(model_cfg).__name__ == "XHQwen3_5_VisualConfig", (
        f"Expected model config type XHQwen3_5_VisualConfig, but got {type(model_cfg).__name__}"
    )
    xh_model: XHQwen3_5VisionModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHQwen3_5VisionModel", (
        f"Expected model type XHQwen3_5VisionModel, but got {type(xh_model).__name__}"
    )
    eval_type = LLMModelState.from_string(eval_type)
    xh_model.set_state(eval_type)

    messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image_url",
        #                 "image_url": {
        #                     "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
        #                 },
        #             },
        #             {
        #                 "type": "text",
        #                 "text": "The centres of the four illustrated circles are in the corners of the square. The two big circles touch each other and also the two little circles. With which factor do you have to multiply the radii of the little circles to obtain the radius of the big circles?\nChoices:\n(A) $\\frac{2}{9}$\n(B) $\\sqrt{5}$\n(C) $0.8 \\cdot \\pi$\n(D) 2.5\n(E) $1+\\sqrt{2}$",
        #             },
        #         ],
        #     },
        # {
        #     "role": "user",
        #     "content": [
        #         {
        #             "type": "image_url",
        #             "image_url": {
        #                 "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"
        #             },
        #         },
        #         {"type": "text", "text": "Where is this?"},
        #     ],
        # },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "data/images/demo_qwen3_vl.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        },
    ]

    processor: XHQwen3_5Processor = xh_model.get_tf_processor()

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
    output_text = output_text[0] if isinstance(output_text, list) else output_text
    logger.info(f"{'-' * 20} Output {'-' * 20}")
    logger.info(f"{output_text}")


if __name__ == "__main__":
    eval_types = LLMModelState.get_all_values()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_instruct_visual_xh2a_2k.py",
    )
    parser.add_argument("--model", type=str)
    parser.add_argument("--eval-type", type=str, default="wrap", choices=eval_types)
    parser.add_argument("--image-path", type=str, default="./data/images/demo_qwen3_vl.jpeg")
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")

    args = parser.parse_args()
    main(args)
