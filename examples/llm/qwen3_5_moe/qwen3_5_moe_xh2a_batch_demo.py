import argparse
from typing import List

import torch
import xhquant.utils.suppress_printing
import xhquant.xhonnxruntime.config as xhonnxruntime_config

from _runtime import (
    load_runtime_from_meta,
    parse_auto_offload_max_memory,
    parse_cuda_graph_modules,
    parse_dtype,
    postprocess_chat_output,
)


def _pad_to_len(input_ids: torch.Tensor, target_len: int, pad_token_id: int) -> torch.Tensor:
    if input_ids.shape[1] > target_len:
        return input_ids[:, :target_len]
    if input_ids.shape[1] == target_len:
        return input_ids
    pad = torch.full(
        (input_ids.shape[0], target_len - input_ids.shape[1]),
        pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    return torch.cat([input_ids, pad], dim=1)


def _position_ids(lengths: torch.Tensor, target_len: int, device: torch.device) -> torch.Tensor:
    rows = []
    for valid_len in lengths.tolist():
        valid_len = int(valid_len)
        if valid_len <= 0:
            rows.append(torch.zeros(target_len, dtype=torch.int32, device=device))
            continue
        row = torch.arange(valid_len, dtype=torch.int32, device=device)
        if valid_len < target_len:
            row = torch.cat([row, row[-1:].expand(target_len - valid_len)])
        rows.append(row[:target_len])
    return torch.stack(rows, dim=0)


def _linear_mask(lengths: torch.Tensor, target_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((lengths.numel(), target_len), dtype=dtype, device=device)
    for batch_idx, valid_len in enumerate(lengths.tolist()):
        mask[batch_idx, : max(int(valid_len), 0)] = 1
    return mask


def _chat_input_ids(tokenizer, prompts: List[str], system_prompt: str, enable_thinking: bool, device: torch.device):
    texts = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
            )
        )
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    return encoded.input_ids.to(device)


def _select_next_token_logits(logits: torch.Tensor, valid_lengths: torch.Tensor) -> torch.Tensor:
    """Select next-token logits from either full-sequence or last-token exports."""

    if logits.shape[1] == 1:
        return logits[:, -1]
    return logits[
        torch.arange(logits.shape[0], device=logits.device),
        valid_lengths.to(device=logits.device, dtype=torch.long) - 1,
    ]


@torch.no_grad()
def run_batch_greedy(runtime, tokenizer, prompts: List[str], args):
    batch_size = int(getattr(runtime, "batch_size", 1))
    if len(prompts) != batch_size:
        raise ValueError(f"Expected exactly {batch_size} prompts for this exported model, got {len(prompts)}")

    device = runtime.execution_device
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    input_ids = _chat_input_ids(tokenizer, prompts, args.system_prompt, args.enable_thinking, device)
    prompt_lengths = (input_ids != pad_token_id).sum(dim=1).to(torch.int32)
    prefill_len = runtime.get_input_sequence_length()
    if int(prompt_lengths.max().item()) > prefill_len:
        raise ValueError(
            f"Longest prompt length {int(prompt_lengths.max().item())} exceeds exported prefill length {prefill_len}"
        )

    runtime.set_phase_prefill(True)
    padded_ids = _pad_to_len(input_ids, prefill_len, pad_token_id)
    inputs_embeds = runtime.token_embedding(padded_ids).to(device=device, dtype=runtime.dtype)
    pos = _position_ids(prompt_lengths, prefill_len, device)
    past_seq = torch.zeros(batch_size, dtype=torch.int32, device=device)
    current_len = prompt_lengths.to(device=device, dtype=torch.int32)
    lin_mask = _linear_mask(prompt_lengths, prefill_len, inputs_embeds.dtype, device)

    logits = runtime._forward(
        inputs_embeds,
        pos,
        pos,
        pos,
        past_seq,
        current_len,
        lin_mask,
        runtime.past_key_caches,
        runtime.past_value_caches,
        runtime.past_conv_caches,
        runtime.past_recurrent_states,
    )

    next_tokens = _select_next_token_logits(logits, current_len).argmax(dim=-1, keepdim=True)
    generated = [list() for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    past_seq = current_len.clone()
    eos_ids = tokenizer.eos_token_id
    eos_set = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])

    runtime.set_phase_prefill(False)
    for _ in range(args.max_new_tokens):
        for batch_idx, token_id in enumerate(next_tokens.squeeze(1).tolist()):
            if not finished[batch_idx]:
                generated[batch_idx].append(int(token_id))
                if int(token_id) in eos_set:
                    finished[batch_idx] = True
        if bool(finished.all().item()):
            break

        decode_embeds = runtime.token_embedding(next_tokens).to(device=device, dtype=runtime.dtype)
        decode_pos = past_seq.view(batch_size, 1).to(torch.int32)
        decode_len = torch.ones(batch_size, dtype=torch.int32, device=device)
        decode_mask = torch.ones(batch_size, 1, dtype=decode_embeds.dtype, device=device)
        logits = runtime._forward(
            decode_embeds,
            decode_pos,
            decode_pos,
            decode_pos,
            past_seq,
            decode_len,
            decode_mask,
            runtime.past_key_caches,
            runtime.past_value_caches,
            runtime.past_conv_caches,
            runtime.past_recurrent_states,
        )
        past_seq = past_seq + 1
        next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    outputs = []
    for token_ids in generated:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        outputs.append(postprocess_chat_output(text, enable_thinking=args.enable_thinking))
    return outputs


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-MoE continue-batch HMONNX demo with one prompt per exported batch item",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt for one batch item. Repeat exactly batch_size times.",
    )
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exec-device", type=str, default="cuda")
    parser.set_defaults(auto_offload=False)
    parser.add_argument("--enable-auto-offload", dest="auto_offload", action="store_true")
    parser.add_argument("--disable-auto-offload", dest="auto_offload", action="store_false")
    parser.add_argument("--auto-offload-max-memory", type=str, default=None)
    parser.add_argument("--prefill-auto-offload-max-memory", type=str, default=None)
    parser.add_argument("--decode-auto-offload-max-memory", type=str, default=None)
    parser.add_argument("--resource-tight-mode", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--show-runtime-progress", action="store_true")
    parser.add_argument("--enable-cuda-graph", action="store_true")
    parser.add_argument("--cuda-graph-modules", type=str, default="")
    parser.add_argument("--cuda-graph-warmup-runs", type=int, default=3)
    parser.add_argument("--cuda-graph-graph-warmup-runs", type=int, default=6)
    return parser


def main():
    args = parse_arguments().parse_args()
    xhquant.utils.suppress_printing.disable_printing = True
    xhonnxruntime_config.disable_progress = not args.show_runtime_progress
    xhonnxruntime_config.verbose_progress = bool(args.show_runtime_progress)

    runtime, tokenizer, _ = load_runtime_from_meta(
        meta_path=args.config,
        dtype=parse_dtype(args.dtype),
        device=args.device,
        exec_device=args.exec_device,
        auto_offload=args.auto_offload,
        auto_offload_max_memory=parse_auto_offload_max_memory(args.auto_offload_max_memory),
        prefill_auto_offload_max_memory=parse_auto_offload_max_memory(args.prefill_auto_offload_max_memory),
        decode_auto_offload_max_memory=parse_auto_offload_max_memory(args.decode_auto_offload_max_memory),
        resource_tight_mode=args.resource_tight_mode,
        enable_cuda_graph=args.enable_cuda_graph,
        cuda_graph_modules=parse_cuda_graph_modules(args.cuda_graph_modules),
        cuda_graph_warmup_runs=args.cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=args.cuda_graph_graph_warmup_runs,
    )

    prompts = args.prompt
    if not prompts:
        prompts = [
            "请用一句话介绍混合专家模型。",
            "请用一句话解释线性注意力。",
            "请给出一个 Python 列表推导式示例。",
            "请用英文回答：what is batch inference?",
        ][: int(getattr(runtime, "batch_size", 1))]

    outputs = run_batch_greedy(runtime, tokenizer, prompts, args)
    for idx, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"[batch {idx}] prompt: {prompt}")
        print(f"[batch {idx}] output: {output}")


if __name__ == "__main__":
    main()
