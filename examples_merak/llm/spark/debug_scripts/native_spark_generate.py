import argparse
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, TextStreamer

from xhmodel_merak.xh_llm.models.spark_xh.spark.modeling_ipt import IPTForCausalLM


CLS = 131632
SEP = 131633
MASK = 131634


def hf_auto_offload(hf_model, device_map="auto", max_memory=None):
    import inspect

    from transformers.modeling_utils import _get_device_map

    max_memory = None
    hf_quantizer = None
    dtype = torch.float16
    keep_in_fp32_regex = None
    _get_device_map_signature = inspect.signature(_get_device_map)
    if len(_get_device_map_signature.parameters) <= 4:
        device_map = _get_device_map(hf_model, device_map, max_memory, hf_quantizer)
    else:
        device_map = _get_device_map(hf_model, device_map, max_memory, hf_quantizer, dtype, keep_in_fp32_regex)

    offload_folder = None
    offload_index = None
    offload_buffers = False
    device_map_kwargs = {
        "device_map": device_map,
        "offload_dir": offload_folder,
        "offload_index": offload_index,
        "offload_buffers": offload_buffers,
    }

    device_map_kwargs["skip_keys"] = hf_model._skip_keys_device_placement
    from accelerate import dispatch_model

    dispatch_model(hf_model, **device_map_kwargs)
    return hf_model


def main(args):
    model_dir = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model: IPTForCausalLM = AutoModelForCausalLM.from_pretrained(  # type: ignore
        model_dir,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.config._attn_implementation = "eager"
    assert isinstance(model, IPTForCausalLM), f"Expected model type IPTForCausalLM, but got {type(model).__name__}"
    model.unfuse_experts(use_padding_expert=False)

    model = hf_auto_offload(model, device_map="auto")

    prompt = args.prompt
    if Path(prompt).is_file():
        with open(prompt, "r") as f:
            prompt = f.read()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.think,
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = model_inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    max_new_tokens = args.max_new_tokens
    prefill_len = model_inputs["input_ids"].shape[1]
    input_ids = model_inputs["input_ids"]
    # input_ids[input_ids >= 133120] = 131632
    model_inputs["input_ids"] = input_ids
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    model_inputs.pop("token_type_ids")
    generated_ids = model.generate(
        **model_inputs,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
    )
    output_ids = generated_ids[0][len(model_inputs["input_ids"][0]) :].tolist()

    # parsing thinking content
    try:
        think_end_token_id = tokenizer.encode("</think>")[-1]
        index = len(output_ids) - output_ids[::-1].index(think_end_token_id)
    except (ValueError, IndexError):
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    logger.info(f"prefill_len: {prefill_len}")
    if len(thinking_content) > 0:
        logger.info(f"thinking content: {thinking_content}")
    logger.info(f"content: {content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Spark MoE (IPT) model")
    parser.add_argument("--model-dir", type=str, default="./data/models/ipt_30b")
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    parser.add_argument("--think", action="store_true", help="enable think mode")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="max new tokens to generate")
    args = parser.parse_args()
    main(args)
