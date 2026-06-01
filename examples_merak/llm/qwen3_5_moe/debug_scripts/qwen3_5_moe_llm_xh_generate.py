import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image
from transformers import TextStreamer

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, LLMInferenceContextManager, LLMModelState
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import XHQwen3_5MoeModel, XHQwen3_5MoeModelConfig


def main(args):
    prompt = args.prompt
    if Path(prompt).is_file():
        prompt = Path(prompt).read_text()
    if not Path(args.image_path).exists():
        raise FileNotFoundError(f"Image file does not exist: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Failed to load image: {args.image_path}")

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

    model_cfg: XHQwen3_5MoeModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    assert type(model_cfg).__name__ == "XHQwen3_5MoeModelConfig", (
        f"Expected model config type XHQwen3_5MoeModelConfig, but got {type(model_cfg).__name__}"
    )
    model_cfg.enable_auto_offload = args.auto_offload  # 是否启用自动显存卸载
    enable_prefill_chunk = args.enable_prefill_chunk
    if enable_prefill_chunk:
        model_cfg.enable_prefill_chunk = True  # 开启prefill chunk后，设置chunk长度为512
        model_cfg.use_cache = True

    xh_model: XHQwen3_5MoeModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHQwen3_5MoeModel", (
        f"Expected model type XHQwen3_5MoeModel, but got {type(xh_model).__name__}"
    )
    xh_model.set_state(LLMModelState.from_string(args.eval_type))

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
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    processor = xh_model.get_tf_processor()
    tokenizer = processor.tokenizer
    model_inputs = processor.apply_chat_template(messages, enable_thinking=args.think).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    streamer = TextStreamer(tokenizer=tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        default="configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_instruct_hf_autoround_xh2a_2k.py",
    )
    parser.add_argument("--eval-type", type=str, default="wrap", choices=LLMModelState.get_all_values())
    parser.add_argument("--image-path", type=str, default="./data/images/RealWorld-04.png")
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--think", action="store_true", help="enable think mode")
    parser.add_argument("--debug", action="store_true", help="run in debug mode")
    parser.add_argument(
        "--auto-offload", action="store_true", help="Whether to enable auto offload, only for debug and development"
    )
    parser.add_argument(
        "--enable-prefill-chunk",
        action="store_true",
        help="Whether to enable prefill chunk, only for debug and development",
    )
    args = parser.parse_args()
    main(args)
