import argparse
from pathlib import Path

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from xhmodel_merak.xh_llm.models.spark_xh.spark.modeling_ipt import IPTForCausalLM


CLS = 131632
SEP = 131633
MASK = 131634


def main(args):
    model_dir = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model: IPTForCausalLM = AutoModelForCausalLM.from_pretrained(  # type: ignore
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    assert isinstance(model, IPTForCausalLM), f"Expected model type IPTForCausalLM, but got {type(model).__name__}"
    model.unfuse_experts()

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
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    max_new_tokens = args.max_new_tokens
    prefill_len = model_inputs["input_ids"].shape[1]
    input_ids = model_inputs["input_ids"]
    input_ids[input_ids >= 133120] = 131632
    model_inputs["input_ids"] = input_ids
    generated_ids = model.generate(
        **model_inputs,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
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
