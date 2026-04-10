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
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.config._attn_implementation = "eager"
    assert isinstance(model, IPTForCausalLM), f"Expected model type IPTForCausalLM, but got {type(model).__name__}"
    # model.unfuse_experts(use_padding_expert=False)

    # model = hf_auto_offload(model, device_map="auto")

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

    # input_ids = model_inputs["input_ids"]
    # input_ids[input_ids >= 133120] = 131632
    # model_inputs["input_ids"] = input_ids

    model_inputs.pop("token_type_ids")

    inputs_embeds = model.get_input_embeddings()(model_inputs["input_ids"]).to(model_inputs["input_ids"].device)
    output_gt = model(inputs_embeds=inputs_embeds, attention_mask=model_inputs["attention_mask"], use_cache=True)

    torch.save(inputs_embeds, "inputs_embeds.pt")
    torch.save(output_gt.logits, "logits_gt.pt")
    model.unfuse_experts(use_padding_expert=False)
    output = model(inputs_embeds=inputs_embeds, attention_mask=model_inputs["attention_mask"], use_cache=True)

    diff = (output_gt.logits - output.logits).abs().max().item()
    logger.info(f"Max absolute difference in logits: {diff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Spark MoE (IPT) model")
    parser.add_argument("--model-dir", type=str, default="./data/models/ipt_30b")
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    parser.add_argument("--think", action="store_true", help="enable think mode")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="max new tokens to generate")
    args = parser.parse_args()
    main(args)
