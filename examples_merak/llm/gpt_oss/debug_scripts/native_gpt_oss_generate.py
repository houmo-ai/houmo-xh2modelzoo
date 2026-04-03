import argparse
from pathlib import Path

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model: GptOssForCausalLM = AutoModelForCausalLM.from_pretrained(  # type: ignore
        args.model_dir,
        torch_dtype="auto",
        device_map="auto",
    )
    assert isinstance(model, GptOssForCausalLM), (
        f"Expected model type GptOssForCausalLM, but got {type(model).__name__}"
    )

    prompt = args.prompt
    if Path(prompt).is_file():
        with open(prompt, "r") as f:
            prompt = f.read()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    output_ids = generated_ids[0][len(model_inputs["input_ids"][0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    logger.info(f"content: {content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test GPT-OSS model")
    parser.add_argument("--model-dir", type=str, default="./data/models/gpt-oss-20b-bfloat16")
    parser.add_argument("--prompt", type=str, default="Briefly introduce yourself.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    main(parser.parse_args())

