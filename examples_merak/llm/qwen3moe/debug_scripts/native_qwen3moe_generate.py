import argparse
from pathlib import Path

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import Qwen3MoeForCausalLM


def main(args):
    model_dir = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model: Qwen3MoeForCausalLM = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")  # type: ignore
    assert isinstance(model, Qwen3MoeForCausalLM), (
        f"Expected model type Qwen3MoeForCausalLM, but got {type(model).__name__}"
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
        enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
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

    # parsing thinking content
    try:
        # rindex finding 151668 (<tool_call>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    logger.info(f"prefill_len: {prefill_len}")
    logger.info(f"thinking content:{thinking_content}")
    logger.info(f"content:{content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Qwen3 MoE model")
    parser.add_argument("--model-dir", type=str, default="./data/models/Qwen3-30B-A3B")
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    args = parser.parse_args()
    main(args)
