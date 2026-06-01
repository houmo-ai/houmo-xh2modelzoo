from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from xhquant.api import HMONNXInference

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from kimi_vl_common import (  # noqa: E402
    build_language_model,
    build_processor_inputs,
    create_kv_caches,
    finalize_modelscope_weights,
    flatten_hmonnx_inputs,
    load_config_json,
    merge_image_embeds,
)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data02/datasets/kimivl")
    parser.add_argument("--work-dir", type=str, default="work_dirs/kimi_vl_a3b_xh2a")
    parser.add_argument("--image", type=str, default="example.png")
    parser.add_argument("--prompt", type=str, default="请详细描述这张图片中的内容。")
    parser.add_argument("--image-size-h", type=int, default=448)
    parser.add_argument("--image-size-w", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model).resolve()
    work_dir = Path(args.work_dir).resolve()
    finalize_modelscope_weights(model_dir)
    meta = json.loads((work_dir / "meta.json").read_text())
    cfg_json = load_config_json(model_dir)

    native_model, token_embedding, _ = build_language_model(model_dir)
    processor, prompt_text, raw_inputs = build_processor_inputs(
        model_dir=model_dir,
        image_path=args.image,
        prompt=args.prompt,
        image_size_h=args.image_size_h,
        image_size_w=args.image_size_w,
    )

    visual = HMONNXInference(str(work_dir / meta["visual_hmonnx"]))
    visual.exec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_embeds = visual(raw_inputs["pixel_values"].half().cpu())

    inputs_embeds = merge_image_embeds(
        token_embedding=token_embedding.cpu(),
        input_ids=raw_inputs["input_ids"].cpu(),
        image_embeds=image_embeds.cpu(),
        media_placeholder_token_id=cfg_json["media_placeholder_token_id"],
    ).cpu()
    seq_len = raw_inputs["input_ids"].shape[1]

    input_sequence_length = int(meta["input_sequence_length"])
    pad_token_id = int(cfg_json["text_config"]["pad_token_id"])
    if seq_len < input_sequence_length:
        pad_ids = torch.full((1, input_sequence_length - seq_len), pad_token_id, dtype=torch.long)
        pad_embeds = token_embedding(pad_ids)
        inputs_embeds = torch.cat([inputs_embeds, pad_embeds], dim=1)
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
    if seq_len < input_sequence_length:
        position_ids = torch.cat([position_ids, torch.ones((1, input_sequence_length - seq_len), dtype=torch.int32)], dim=1)

    key_head_dim = int(cfg_json["text_config"]["qk_nope_head_dim"] + cfg_json["text_config"]["qk_rope_head_dim"])
    value_head_dim = int(cfg_json["text_config"]["v_head_dim"])
    num_hidden_layers = int(cfg_json["text_config"]["num_hidden_layers"])
    num_key_value_heads = int(cfg_json["text_config"]["num_key_value_heads"])
    context_length = int(meta["context_length"])

    key_caches, value_caches = create_kv_caches(
        batch_size=1,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        context_length=context_length,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        dtype=torch.float16,
        cache_tensor=True,
    )

    prefill = HMONNXInference(str(work_dir / meta["prefill_hmonnx"]))
    prefill.exec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    decode = HMONNXInference(str(work_dir / meta["decode_hmonnx"]))
    decode.exec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generated_ids = raw_inputs["input_ids"].clone()
    current_embeds = inputs_embeds.half()
    prefill_logits = prefill(
        *flatten_hmonnx_inputs(
            (
                current_embeds,
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([seq_len], dtype=torch.int32),
                position_ids,
                key_caches,
                value_caches,
            )
        )
    )
    next_token = torch.argmax(prefill_logits[:, -1, :], dim=-1, keepdim=True).cpu().to(torch.long)
    generated_ids = torch.cat([generated_ids, next_token], dim=1)

    for step in range(args.max_new_tokens - 1):
        next_embeds = token_embedding(next_token).half()
        decode_logits = decode(
            *flatten_hmonnx_inputs(
                (
                    next_embeds,
                    torch.tensor([seq_len + step], dtype=torch.int32),
                    torch.tensor([1], dtype=torch.int32),
                    torch.tensor([[seq_len + step]], dtype=torch.int32),
                    key_caches,
                    value_caches,
                )
            )
        )
        next_token = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True).cpu().to(torch.long)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

    text = processor.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    print(f"prompt_text={prompt_text}")
    print(f"generated_text={text}")


if __name__ == "__main__":
    main()
