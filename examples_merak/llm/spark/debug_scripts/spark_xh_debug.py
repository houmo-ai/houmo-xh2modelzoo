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
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.spark_xh import XHSparkModeModel, XHSparkModeModelConfig


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

    cfg = Config.fromfile(args.config)
    cfg.model.num_logits_to_keep = 0
    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = work_dir / Path(args.config).name
    cfg.dump(config_file)

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")
    model_cfg: XHSparkModeModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    enable_prefill_chunk = args.enable_prefill_chunk
    if enable_prefill_chunk:
        model_cfg.enable_prefill_chunk = True
        model_cfg.use_cache = True
    model_cfg.enable_auto_offload = args.auto_offload

    assert type(model_cfg).__name__ == "XHSparkModeModelConfig", (
        f"Expected model config type XHSparkModeModelConfig, but got {type(model_cfg).__name__}"
    )
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model: XHSparkModeModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHSparkModeModel", (
        f"Expected model type XHSparkModeModel, but got {type(xh_model).__name__}"
    )
    hf_model = xh_model.get_native_model()

    eval_type = LLMModelState.from_string(eval_type)
    xh_model.set_state(eval_type)

    enable_think = args.think
    prompt = args.prompt
    if Path(prompt).is_file():
        with open(prompt, "r") as f:
            prompt = f.read()
    messages = [
        {"role": "user", "content": prompt},
    ]
    xh_model.to(device)
    processor = xh_model.get_tf_processor()
    tokenizer = processor.tokenizer
    model_inputs = processor.apply_chat_template(messages, enable_think=enable_think)
    model_inputs = model_inputs.to(device)
    seq_length = model_inputs["input_ids"].shape[1]
    logger.info(f"Input sequence length: {seq_length}")
    hf_model.to(device)
    inputs_embeds = hf_model.get_input_embeddings()(model_inputs["input_ids"])
    data_batch = {
        "input_ids": model_inputs["input_ids"],
        "past_seq_length": torch.tensor([0], dtype=torch.int32),
    }
    xh_model.to(device)
    xh_model.set_input_sequence_length(seq_length)
    xh_preprocessor = xh_model.get_data_preprocessor()
    xh_inputs = xh_preprocessor(data_batch)
    logger.info(f"{xh_inputs[0].shape}")

    hf_model.config._attn_implementation = "eager"
    output_gt = hf_model(inputs_embeds=inputs_embeds, attention_mask=model_inputs["attention_mask"], use_cache=True)
    logits_gt = output_gt.logits

    # inputs_embeds_gt = torch.load("inputs_embeds.pt", weights_only=True)
    # logits_gt_a = torch.load("logits_gt.pt", weights_only=True)
    # diff_1 = (inputs_embeds_gt - inputs_embeds).abs().max().item()
    # diff_2 = (logits_gt_a - logits_gt).abs().max().item()
    # logger.info(f"Max absolute difference in inputs_embeds: {diff_1}")
    # logger.info(f"Max absolute difference in logits: {diff_2}")
    contexts = [
        TimeProfiler("hmonnx_generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(xh_model),
    ]
    xh_model.to(dtype)
    with ContextManagers(contexts):
        logits = xh_model(*xh_inputs)

    diff = (logits_gt - logits).abs().max().item()
    logger.info(f"Max absolute difference in logits: {diff}")


if __name__ == "__main__":
    eval_types = LLMModelState.get_all_values()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/spark/30b/spark_xh_30b_xh2a_2k_2layers.py",
    )
    parser.add_argument("--model", type=str)
    parser.add_argument("--eval-type", type=str, default="fronted", choices=eval_types)
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    parser.add_argument("--think", action="store_true", help="enable think mode")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
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
