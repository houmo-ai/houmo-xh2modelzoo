#!/usr/bin/env python3
"""Gemma4 Series MTP floating-point reference runner.

This script is intentionally independent from legacy gemma4/gemma4e/gemma4_moe
implementations.  It provides two things needed before HMONNX lowering:

1. config inspection for all local Gemma4 Series target/assistant pairs;
2. a small greedy speculative-decoding loop using Transformers Gemma4 assistant.

The loop recomputes the target on the accepted prefix each round.  That is not
optimized, but it is a clear correctness oracle for layer/KV mapping and draft
acceptance before implementing the HMONNX runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any



@dataclass(frozen=True)
class Preset:
    name: str
    target_dir: str
    assistant_dir: str


PRESETS: dict[str, Preset] = {
    "e2b": Preset("e2b", "weights/gemma-4-E2B-it", "weights/gemma-4-E2B-it-assistant"),
    "e4b": Preset("e4b", "weights/gemma-4-E4B-it", "weights/gemma-4-E4B-it-assistant"),
    "26b-a4b": Preset(
        "26b-a4b",
        "weights/gemma-4-26B-A4B-it",
        "weights/gemma-4-26B-A4B-it-assistant",
    ),
    "31b": Preset("31b", "weights/gemma-4-31B-it", "weights/gemma-4-31B-it-assistant"),
}


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _text_config(model_dir: str | Path) -> dict[str, Any]:
    cfg = _load_json(Path(model_dir) / "config.json")
    return cfg.get("text_config", cfg)


def _pattern(layer_types: list[str]) -> str:
    return "".join("S" if x == "sliding_attention" else "F" if x == "full_attention" else "?" for x in layer_types)


def build_kv_mapping(target_dir: str | Path, assistant_dir: str | Path) -> dict[str, Any]:
    target_text = _text_config(target_dir)
    assistant_cfg = _load_json(Path(assistant_dir) / "config.json")
    assistant_text = assistant_cfg.get("text_config", assistant_cfg)

    target_layer_types = list(target_text.get("layer_types", []))
    assistant_layer_types = list(assistant_text.get("layer_types", []))
    target_num_shared = int(target_text.get("num_kv_shared_layers", 0) or 0)
    num_non_shared = len(target_layer_types) - target_num_shared
    if num_non_shared <= 0:
        raise ValueError(f"Invalid target num_non_shared={num_non_shared} for {target_dir}")

    type_to_target_indices: dict[str, list[int]] = {}
    for idx, layer_type in enumerate(target_layer_types[:num_non_shared]):
        type_to_target_indices.setdefault(layer_type, []).append(idx)

    draft_to_target: list[dict[str, Any]] = []
    for draft_idx, layer_type in enumerate(assistant_layer_types):
        candidates = type_to_target_indices.get(layer_type, [])
        if not candidates:
            raise ValueError(f"No target non-shared layer for draft layer {draft_idx} type={layer_type}")
        draft_to_target.append(
            {
                "draft_layer": draft_idx,
                "layer_type": layer_type,
                "target_layer": candidates[-1],
            }
        )

    return {
        "target_layers": len(target_layer_types),
        "target_pattern": _pattern(target_layer_types),
        "target_num_kv_shared_layers": target_num_shared,
        "target_num_non_shared_layers": num_non_shared,
        "assistant_layers": len(assistant_layer_types),
        "assistant_pattern": _pattern(assistant_layer_types),
        "assistant_num_kv_shared_layers": int(assistant_text.get("num_kv_shared_layers", 0) or 0),
        "type_to_target_layer": {k: v[-1] for k, v in type_to_target_indices.items()},
        "draft_to_target": draft_to_target,
        "target_sliding_window": target_text.get("sliding_window"),
        "target_hidden_size": target_text.get("hidden_size"),
        "assistant_hidden_size": assistant_text.get("hidden_size"),
        "assistant_ordered_embedding": bool(assistant_cfg.get("use_ordered_embeddings", False)),
        "assistant_num_centroids": assistant_cfg.get("num_centroids"),
        "assistant_centroid_intermediate_top_k": assistant_cfg.get("centroid_intermediate_top_k"),
        "attention_k_eq_v": bool(assistant_text.get("attention_k_eq_v", False)),
        "num_key_value_heads": assistant_text.get("num_key_value_heads"),
        "num_global_key_value_heads": assistant_text.get("num_global_key_value_heads"),
        "head_dim": assistant_text.get("head_dim"),
        "global_head_dim": assistant_text.get("global_head_dim"),
    }


def inspect_presets(args: argparse.Namespace) -> None:
    names = list(PRESETS) if args.all else [args.preset]
    payload: dict[str, Any] = {}
    for name in names:
        preset = PRESETS[name]
        payload[name] = {
            "target_dir": preset.target_dir,
            "assistant_dir": preset.assistant_dir,
            **build_kv_mapping(preset.target_dir, preset.assistant_dir),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _import_transformers(transformers_src: str | None):
    if transformers_src:
        src = str(Path(transformers_src).resolve())
        if src not in sys.path:
            sys.path.insert(0, src)
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import Gemma4AssistantForCausalLM
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        raise RuntimeError(
            "Failed to import Transformers Gemma4 assistant. Use a Transformers checkout containing "
            "transformers.models.gemma4_assistant, e.g. --transformers-src /tmp/transformers-gemma4-main/src"
        ) from exc
    return AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, Gemma4AssistantForCausalLM


def _get_language_model(target_model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(target_model, "language_model"):
        return target_model.language_model
    if hasattr(target_model, "model") and hasattr(target_model.model, "language_model"):
        return target_model.model.language_model
    if hasattr(target_model, "get_language_model"):
        return target_model.get_language_model()
    raise AttributeError(f"Cannot locate Gemma4 language model inside {type(target_model)!r}")


def _target_forward(target_model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    return target_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
        return_shared_kv_states=True,
    )


def _token_embedding(target_model, token_ids: torch.Tensor) -> torch.Tensor:
    language_model = _get_language_model(target_model)
    embed = language_model.get_input_embeddings() if hasattr(language_model, "get_input_embeddings") else language_model.embed_tokens
    return embed(token_ids)


def _assistant_logits_argmax(assistant_model, last_token_embed, backbone_hidden, position_id, attention_mask, shared_kv_states):
    import torch

    assistant_input = torch.cat([last_token_embed, backbone_hidden], dim=-1)
    out = assistant_model(
        inputs_embeds=assistant_input,
        position_ids=position_id,
        attention_mask=attention_mask,
        shared_kv_states=shared_kv_states,
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    return next_token, out.last_hidden_state[:, -1:, :], logits


def _load_prompt_inputs(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    return tokenizer([text], return_tensors="pt")


def generate(args: argparse.Namespace) -> None:
    AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, Gemma4AssistantForCausalLM = _import_transformers(
        args.transformers_src
    )
    preset = PRESETS[args.preset]
    import torch

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(preset.target_dir, trust_remote_code=True)
    # AutoProcessor is available for multimodal checkpoints; tokenizer is enough for text-only prompts.
    try:
        processor = AutoProcessor.from_pretrained(preset.target_dir, trust_remote_code=True)
    except Exception:
        processor = None
    del processor

    target_model = AutoModelForImageTextToText.from_pretrained(
        preset.target_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation="eager",
    ).to(device)
    assistant_model = Gemma4AssistantForCausalLM.from_pretrained(
        preset.assistant_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation="eager",
    ).to(device)
    target_model.eval()
    assistant_model.eval()

    encoded = _load_prompt_inputs(tokenizer, args.prompt)
    input_ids = encoded["input_ids"].to(device)
    eos_token_id = tokenizer.eos_token_id
    accepted_total = 0
    drafted_total = 0
    iterations = 0
    full_accept_iterations = 0
    zero_accept_iterations = 0

    with torch.inference_mode():
        while input_ids.shape[1] < encoded["input_ids"].shape[1] + args.max_new_tokens:
            iterations += 1
            attn_mask = torch.ones_like(input_ids, device=device)
            target_out = _target_forward(target_model, input_ids, attn_mask)
            if getattr(target_out, "shared_kv_states", None) is None:
                raise RuntimeError("Target model did not return shared_kv_states; check Transformers Gemma4 version.")
            if not getattr(target_out, "hidden_states", None):
                raise RuntimeError("Target model did not return hidden_states.")

            prefix_len = input_ids.shape[1]
            target_hidden = target_out.hidden_states[-1][:, -1:, :]
            cur_token = input_ids[:, -1:]
            draft_tokens: list[torch.Tensor] = []
            draft_hidden = target_hidden
            shared_kv_states = target_out.shared_kv_states

            for step in range(args.num_assistant_tokens):
                token_embed = _token_embedding(target_model, cur_token)
                pos = torch.tensor([[prefix_len + step - 1]], dtype=torch.long, device=device)
                next_token, draft_hidden, _ = _assistant_logits_argmax(
                    assistant_model,
                    token_embed,
                    draft_hidden,
                    pos,
                    attn_mask,
                    shared_kv_states,
                )
                draft_tokens.append(next_token)
                cur_token = next_token
                drafted_total += 1
                if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                    break

            if not draft_tokens:
                break

            draft_tensor = torch.cat(draft_tokens, dim=1)
            verify_ids = torch.cat([input_ids, draft_tensor], dim=1)
            verify_mask = torch.ones_like(verify_ids, device=device)
            verify_out = _target_forward(target_model, verify_ids, verify_mask)

            accepted = 0
            for i in range(draft_tensor.shape[1]):
                target_token = torch.argmax(verify_out.logits[:, prefix_len - 1 + i, :], dim=-1, keepdim=True)
                if int(target_token.item()) == int(draft_tensor[:, i : i + 1].item()):
                    accepted += 1
                else:
                    break

            if accepted == draft_tensor.shape[1]:
                full_accept_iterations += 1
                bonus = torch.argmax(verify_out.logits[:, prefix_len + accepted - 1, :], dim=-1, keepdim=True)
                append_tokens = torch.cat([draft_tensor, bonus], dim=1)
            else:
                if accepted == 0:
                    zero_accept_iterations += 1
                reject_target = torch.argmax(verify_out.logits[:, prefix_len - 1 + accepted, :], dim=-1, keepdim=True)
                append_tokens = torch.cat([draft_tensor[:, :accepted], reject_target], dim=1)

            # Do not exceed requested max_new_tokens.
            remaining = encoded["input_ids"].shape[1] + args.max_new_tokens - input_ids.shape[1]
            append_tokens = append_tokens[:, :remaining]
            input_ids = torch.cat([input_ids, append_tokens], dim=1)
            accepted_total += accepted
            if eos_token_id is not None and (append_tokens == eos_token_id).any():
                break

    stats = {
        "preset": args.preset,
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "generated_tokens": int(input_ids.shape[1] - encoded["input_ids"].shape[1]),
        "iterations": iterations,
        "drafted_tokens": drafted_total,
        "accepted_draft_tokens": accepted_total,
        "acceptance_rate": (accepted_total / drafted_total) if drafted_total else 0.0,
        "full_accept_iterations": full_accept_iterations,
        "zero_accept_iterations": zero_accept_iterations,
    }
    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    print("\n[Gemma4 MTP float stats]")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect target/assistant configs and KV mapping")
    p_inspect.add_argument("--preset", choices=sorted(PRESETS), default="e2b")
    p_inspect.add_argument("--all", action="store_true")
    p_inspect.set_defaults(func=inspect_presets)

    p_gen = sub.add_parser("generate", help="Run greedy float speculative decoding")
    p_gen.add_argument("--preset", choices=sorted(PRESETS), default="e2b")
    p_gen.add_argument("--prompt", default="用一句话解释 Gemma4 MTP。")
    p_gen.add_argument("--max-new-tokens", type=int, default=32)
    p_gen.add_argument("--num-assistant-tokens", type=int, default=4)
    p_gen.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p_gen.add_argument("--device", default="cuda", help="Inference device; use cpu only for tiny/debug runs")
    p_gen.add_argument(
        "--transformers-src",
        default=os.environ.get("GEMMA4_TRANSFORMERS_SRC", "/tmp/transformers-gemma4-main/src"),
        help="Transformers source checkout containing gemma4_assistant; default: GEMMA4_TRANSFORMERS_SRC or /tmp/transformers-gemma4-main/src",
    )
    p_gen.set_defaults(func=generate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
