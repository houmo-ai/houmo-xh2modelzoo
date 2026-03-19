from typing import cast

import torch
import torch.nn.functional as F
from loguru import logger
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen3 import Qwen3Model


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def main(args):
    model_dir = args.model_dir
    ckpt_path = args.rot_ckpt

    task = "Given a web search query, retrieve relevant passages that answer the query"
    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    input_texts = queries + documents

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    model = AutoModel.from_pretrained(
        model_dir, attn_implementation="flash_attention_2", torch_dtype=torch.float16
    )
    model = cast(Qwen3Model, model)
    model.eval()

    state_dict = load_safetensors_file(ckpt_path)
    if "post_norm_linear.weight" in state_dict:
        hidden_size = model.config.hidden_size
        model.post_norm_linear = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    if hasattr(model, "post_norm_linear"):
        model.post_norm_linear = model.post_norm_linear.to(
            device=device, dtype=model.dtype
        )

    max_length = 8192
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict.to(model.device)
    with torch.no_grad():
        outputs = model(**batch_dict)
    last_hidden = outputs.last_hidden_state
    if hasattr(model, "post_norm_linear"):
        last_hidden = model.post_norm_linear(last_hidden)
    embeddings = last_token_pool(last_hidden, batch_dict["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = embeddings[:2] @ embeddings[2:].T

    logger.info(f"Model: {model_dir}")
    logger.info(f"Rot ckpt: {ckpt_path}")
    logger.info(f"{scores.tolist()}")


if __name__ == "__main__":
    import debugpy

    debugpy.listen(("0.0.0.0", 1160))
    print("✅ debugpy listening on 0.0.0.0:5678, waiting for VSCode attach...")
    debugpy.wait_for_client()
    print("✅ VSCode attached, continue running.")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-4B",
    )
    parser.add_argument(
        "--rot-ckpt",
        type=str,
        default="work_dirs/Qwen3-Embedding-4B_quarot_gptq_transformers-4.57.3/quarot-state-dict.safetensors",
    )
    args = parser.parse_args()
    main(args)
