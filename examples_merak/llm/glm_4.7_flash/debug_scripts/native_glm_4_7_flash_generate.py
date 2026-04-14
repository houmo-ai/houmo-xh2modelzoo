import argparse
from pathlib import Path

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteForCausalLM


def main(args):
    model_dir = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model: Glm4MoeLiteForCausalLM = AutoModelForCausalLM.from_pretrained(  # type: ignore
        model_dir,
        dtype="auto",
        device_map="auto",
    )
    assert isinstance(model, Glm4MoeLiteForCausalLM), (
        f"Expected model type Glm4MoeLiteForCausalLM, but got {type(model).__name__}"
    )
    # prepare the model input
    prompt = args.prompt
    if Path(prompt).is_file():
        with open(prompt, "r") as f:
            prompt = f.read()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    max_new_tokens = 1024
    prefill_len = model_inputs["input_ids"].shape[1]
    generated_ids = model.generate(
        **model_inputs,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
    )
    output_ids = generated_ids[0][len(model_inputs["input_ids"][0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    logger.info(f"prefill_len: {prefill_len}")
    logger.info(f"content:{content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test GLM-4.7-Flash model")
    parser.add_argument("--model-dir", type=str, default="/data02/datasets/chuyuan.wei/GLM-4.7-Flash")
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    args = parser.parse_args()
    main(args)
