import argparse
import importlib.util
import json
import os
import shutil
import sys
import types
from glob import glob
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_model, safe_open, save_file
from transformers import PreTrainedTokenizerFast


DEFAULT_MODEL_PATH = "/data01/datasets/DeepSeek-V4-Flash"
DEFAULT_OUTPUT_PATH = "work_dirs/DeepSeek-V4-Flash-first6-mp1"
DEFAULT_NUM_LAYERS = 6


def load_module(module_name: str, module_path: Path):
    module_dir = module_path.parent.resolve()
    package_name = f"_copilot_dynamic_{module_dir.parent.name}_{module_dir.name}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(module_dir)]
        sys.modules[package_name] = package

    full_module_name = f"{package_name}.{module_name}"
    spec = importlib.util.spec_from_file_location(full_module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_module_name] = module
    parent_module_dir = str(module_dir)
    inserted = False
    if parent_module_dir not in sys.path:
        sys.path.insert(0, parent_module_dir)
        inserted = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.pop(0)
    return module


def resolve_paths(model_path: str) -> tuple[Path, Path, Path, Path, Path, Path]:
    model_root = Path(model_path).resolve()
    inference_dir = model_root / "inference"
    encoding_dir = model_root / "encoding"
    convert_script = inference_dir / "convert.py"
    generate_script = inference_dir / "generate.py"
    inference_config = inference_dir / "config.json"
    model_script = inference_dir / "model.py"
    for path, label in [
        (inference_dir, "inference directory"),
        (encoding_dir, "encoding directory"),
        (convert_script, "convert script"),
        (generate_script, "generate script"),
        (inference_config, "inference config"),
        (model_script, "model script"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    return model_root, inference_dir, encoding_dir, convert_script, generate_script, model_script


def load_encoder(model_path: str):
    _, _, encoding_dir, _, _, _ = resolve_paths(model_path)
    module = load_module("encoding_dsv4", encoding_dir / "encoding_dsv4.py")
    return module.encode_messages


def load_convert_helpers(model_path: str):
    _, _, _, convert_script, _, _ = resolve_paths(model_path)
    module = load_module("deepseek_v4_convert", convert_script)
    return module.mapping, module.cast_e2m1fn_to_e4m3fn


def rename_param(name: str, mapping: dict[str, tuple[str, int | None]]) -> tuple[str, int | None]:
    if name.startswith("model."):
        name = name[len("model."):]
    if name.startswith("mtp."):
        return "", None
    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    if any(token in name for token in ["hc", "attn_sink", "tie2eid", "ape"]):
        key = name.split(".")[-1]
    else:
        key = name.split(".")[-2]
    new_key, dim = mapping.get(key, (key, None))
    return name.replace(key, new_key), dim


def keep_param(name: str, num_layers: int) -> bool:
    if not name:
        return False
    parts = name.split(".")
    if parts[0] == "layers":
        return int(parts[1]) < num_layers
    if parts[0] == "mtp":
        return False
    return True


def build_runtime_config(model_path: str, num_layers: int, max_seq_len: int, max_batch_size: int) -> dict[str, Any]:
    _, _, _, _, _, _ = resolve_paths(model_path)
    with open(Path(model_path) / "inference" / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    config["n_layers"] = num_layers
    config["n_hash_layers"] = min(config.get("n_hash_layers", 0), num_layers)
    config["n_mtp_layers"] = 0
    config["max_seq_len"] = max_seq_len
    config["max_batch_size"] = max_batch_size
    return config


def prepare_truncated_checkpoint(
    model_path: str,
    output_path: str,
    num_layers: int,
    expert_dtype: str | None,
    max_seq_len: int,
    max_batch_size: int,
) -> None:
    model_root, _, _, _, _, _ = resolve_paths(model_path)
    mapping, cast_e2m1fn_to_e4m3fn = load_convert_helpers(model_path)
    state_dict: dict[str, torch.Tensor] = {}
    torch.set_num_threads(8)
    for file_path in sorted(glob(os.path.join(model_root, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for original_name in handle.keys():
                renamed_name, _ = rename_param(original_name, mapping)
                if not keep_param(renamed_name, num_layers):
                    continue
                param = handle.get_tensor(original_name)
                state_dict[renamed_name] = param

    names = list(state_dict.keys())
    for name in names:
        if name.endswith("wo_a.weight"):
            weight = state_dict[name]
            scale = state_dict.pop(name.replace("weight", "scale"))
            weight = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float() * scale[:, None, :, None].float()
            state_dict[name] = weight.flatten(2, 3).flatten(0, 1).bfloat16()
        elif "experts" in name and state_dict[name].dtype == torch.int8:
            if expert_dtype == "fp8":
                scale_name = name.replace("weight", "scale")
                weight = state_dict.pop(name)
                scale = state_dict.pop(scale_name)
                state_dict[name], state_dict[scale_name] = cast_e2m1fn_to_e4m3fn(weight, scale)
            else:
                state_dict[name] = state_dict[name].view(torch.float4_e2m1fn_x2)

    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, output_root / "model0-mp1.safetensors")
    for file_name in ["tokenizer.json", "tokenizer_config.json"]:
        source = model_root / file_name
        if source.exists():
            shutil.copyfile(source, output_root / file_name)
    runtime_config = build_runtime_config(model_path, num_layers, max_seq_len, max_batch_size)
    with open(output_root / "config.json", "w", encoding="utf-8") as handle:
        json.dump(runtime_config, handle, indent=2)
    print(f"Saved truncated checkpoint to {output_root}")


def summarize_tensor(name: str, tensor: torch.Tensor) -> None:
    data = tensor.detach().float()
    print(
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"mean={data.mean().item():.6f} std={data.std(unbiased=False).item():.6f} "
        f"min={data.min().item():.6f} max={data.max().item():.6f}"
    )


def sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


def hc_split_sinkhorn_torch(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb_logits = mixes[..., 2 * hc_mult :].view(*mixes.shape[:-1], hc_mult, hc_mult)
    comb_logits = comb_logits * hc_scale[2] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult)

    comb = torch.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def act_quant_torch(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    scales = torch.ones(*x.shape[:-1], max(x.shape[-1] // block_size, 1), dtype=scale_dtype, device=x.device)
    if inplace:
        return x
    return x, scales


def fp4_act_quant_torch(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
):
    scales = torch.ones(*x.shape[:-1], max(x.shape[-1] // block_size, 1), dtype=torch.float32, device=x.device)
    if inplace:
        return x
    return x, scales


def dequantize_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
    scale = weight.scale.float()
    out_features, in_features = weight.shape
    expanded_scale = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    expanded_scale = expanded_scale[:out_features, :in_features]
    return weight.float() * expanded_scale


def dequantize_fp4_weight(weight: torch.Tensor) -> torch.Tensor:
    fp4_table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=weight.device,
    )
    packed = weight.view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack([fp4_table[low.long()], fp4_table[high.long()]], dim=-1).flatten(1, 2)
    scale = weight.scale.float().repeat_interleave(32, dim=1)
    scale = scale[:, : unpacked.size(1)]
    return unpacked * scale


def linear_torch(weight: torch.Tensor, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    if weight.dtype == torch.float4_e2m1fn_x2:
        dense_weight = dequantize_fp4_weight(weight)
    elif weight.dtype == torch.float8_e4m3fn:
        dense_weight = dequantize_fp8_weight(weight)
    else:
        dense_weight = weight.float()
    output = F.linear(x.float(), dense_weight, None if bias is None else bias.float())
    return output.to(x.dtype)


def sparse_attn_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    q_float = q.float()
    kv_float = kv.float()
    idxs = topk_idxs.long()
    valid = idxs >= 0
    batch_idx = torch.arange(kv.size(0), device=kv.device).view(-1, 1, 1)
    gathered_kv = kv_float[batch_idx, idxs.clamp_min(0)]

    scores = torch.einsum("bshd,bstd->bsht", q_float, gathered_kv) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    sink_scores = attn_sink.float().view(1, 1, -1, 1).expand(scores.size(0), scores.size(1), -1, -1)
    weights = torch.softmax(torch.cat([scores, sink_scores], dim=-1), dim=-1)[..., :-1]
    output = torch.einsum("bsht,bstd->bshd", weights, gathered_kv)
    return output.to(q.dtype)


def load_runtime(model_path: str, ckpt_path: str, device: str):
    if device != "cuda":
        raise RuntimeError("This tracing script is GPU-only because DeepSeek-V4 official kernels depend on CUDA and TileLang.")
    _, _, _, _, _, model_script = resolve_paths(model_path)
    model_module = load_module("deepseek_v4_model", model_script)
    encode_messages = load_encoder(model_path)

    runtime_config_path = Path(ckpt_path) / "config.json"
    if not runtime_config_path.exists():
        raise FileNotFoundError(
            f"Missing runtime config: {runtime_config_path}. Run the prepare subcommand first."
        )
    with open(runtime_config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    torch.cuda.set_device(0)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    model_module.hc_split_sinkhorn = hc_split_sinkhorn_torch
    model_module.act_quant = act_quant_torch
    model_module.fp4_act_quant = fp4_act_quant_torch
    model_module.sparse_attn = sparse_attn_torch
    model_module.linear = lambda x, weight, bias=None: linear_torch(weight, x, bias)
    args = model_module.ModelArgs(**config)
    with torch.device(device):
        model = model_module.Transformer(args)
    load_model(model, str(Path(ckpt_path) / "model0-mp1.safetensors"), strict=False)
    torch.set_default_device(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(ckpt_path))
    return model.eval(), tokenizer, encode_messages


@torch.inference_mode()
def traced_forward(model, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
    hidden = model.embed(input_ids)
    summarize_tensor("embed", hidden)
    hidden = hidden.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    summarize_tensor("hc_expand", hidden)
    for layer_idx, layer in enumerate(model.layers):
        print(f"\n=== layer {layer_idx} ===")
        residual = hidden
        attn_pre, post, comb = layer.hc_pre(hidden, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
        summarize_tensor(f"layer{layer_idx}.attn_pre", attn_pre)
        summarize_tensor(f"layer{layer_idx}.attn_post_weight", post)
        summarize_tensor(f"layer{layer_idx}.attn_comb", comb)
        attn_norm = layer.attn_norm(attn_pre)
        summarize_tensor(f"layer{layer_idx}.attn_norm", attn_norm)
        attn_out = layer.attn(attn_norm, start_pos)
        summarize_tensor(f"layer{layer_idx}.attn_out", attn_out)
        hidden = layer.hc_post(attn_out, residual, post, comb)
        summarize_tensor(f"layer{layer_idx}.after_attn", hidden)

        residual = hidden
        ffn_pre, post, comb = layer.hc_pre(hidden, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base)
        summarize_tensor(f"layer{layer_idx}.ffn_pre", ffn_pre)
        summarize_tensor(f"layer{layer_idx}.ffn_post_weight", post)
        summarize_tensor(f"layer{layer_idx}.ffn_comb", comb)
        ffn_norm = layer.ffn_norm(ffn_pre)
        summarize_tensor(f"layer{layer_idx}.ffn_norm", ffn_norm)
        ffn_out = layer.ffn(ffn_norm, input_ids)
        summarize_tensor(f"layer{layer_idx}.ffn_out", ffn_out)
        hidden = layer.hc_post(ffn_out, residual, post, comb)
        summarize_tensor(f"layer{layer_idx}.after_ffn", hidden)
    logits = model.head(hidden, model.hc_head_fn, model.hc_head_scale, model.hc_head_base, model.norm)
    summarize_tensor("logits", logits)
    return logits


@torch.inference_mode()
def plain_forward(model, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
    hidden = model.embed(input_ids)
    hidden = hidden.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    for layer in model.layers:
        hidden = layer(hidden, start_pos, input_ids)
    return model.head(hidden, model.hc_head_fn, model.hc_head_scale, model.hc_head_base, model.norm)


@torch.inference_mode()
def trace_generation(
    model_path: str,
    ckpt_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
    forward_mode: str,
) -> None:
    model, tokenizer, encode_messages = load_runtime(model_path, ckpt_path, device)
    encoded_prompt = encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat")
    prompt_tokens = tokenizer.encode(encoded_prompt)
    tokens = torch.full((1, len(prompt_tokens) + max_new_tokens), -1, dtype=torch.long, device=device)
    tokens[0, :len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    prompt_mask = tokens != -1
    prev_pos = 0
    finished = False

    print("Encoded prompt:")
    print(encoded_prompt)
    print(f"Prompt token count: {len(prompt_tokens)}")

    for cur_pos in range(len(prompt_tokens), tokens.size(1)):
        step_tokens = tokens[:, prev_pos:cur_pos]
        if forward_mode == "trace":
            print(f"\n######## trace step prev_pos={prev_pos} cur_pos={cur_pos} ########")
            logits = traced_forward(model, step_tokens, prev_pos)
        else:
            logits = plain_forward(model, step_tokens, prev_pos)
        topk = torch.topk(logits[0], k=5)
        decoded_topk = [(tokenizer.decode([idx]), score.item()) for score, idx in zip(topk.values, topk.indices)]
        print("Top-5 next tokens:", decoded_topk)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        prev_pos = cur_pos
        if not prompt_mask[:, cur_pos].item() and next_token.item() == tokenizer.eos_token_id:
            finished = True
            break

    generated = tokens[0, len(prompt_tokens):prev_pos + (0 if finished else 1)].tolist()
    if tokenizer.eos_token_id in generated:
        generated = generated[:generated.index(tokenizer.eos_token_id)]
    print("\nFinal completion:")
    print(tokenizer.decode(generated))


def apply_runtime_env(cuda_visible_devices: str) -> None:
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def parse_args() -> argparse.Namespace:
    prompt = '''秋日的山林落满金黄树叶，小松鼠忙着捡拾松果，为寒冬储备粮食。它在老槐树下，发现一只崴了腿的小刺猬，蜷缩成小小的刺球，孤零零地躲在落叶堆里，眼里满是慌张。
小刺猬没法奔走觅食，眼看就要挨过萧瑟秋日。善良的小松鼠停下忙碌，把辛苦攒下的松果分出大半，又每天穿梭林间，衔来清甜野果、汲来山泉。它找来柔软干草，在树洞旁为刺猬铺了温暖小窝，日日相伴照料。
秋风吹尽，寒霜渐起，小刺猬的腿终于痊愈。离别那日清晨，小松鼠出门觅食，赫然发现洞口堆满了圆润野枣与饱满干果。
世间温柔从来双向奔赴，一份小小的善意，终会化作冬日里最暖心的馈赠。帮我续写这个故事，结尾要温暖治愈。'''
    parser = argparse.ArgumentParser(
        description="Prepare and trace a single-GPU DeepSeek-V4 checkpoint that keeps only the first 6 transformer layers."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    subparsers = parser.add_subparsers(dest="command", required=False)

    prepare_parser = subparsers.add_parser("prepare", help="Create a single-GPU checkpoint that keeps only the first N layers.")
    prepare_parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    prepare_parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    prepare_parser.add_argument("--expert-dtype", choices=["fp4", "fp8"], default=None)
    prepare_parser.add_argument("--max-seq-len", type=int, default=4096)
    prepare_parser.add_argument("--max-batch-size", type=int, default=1)

    trace_parser = subparsers.add_parser("trace", help="Run traced single-GPU generation with the truncated checkpoint.")
    trace_parser.add_argument("--ckpt-path", default=DEFAULT_OUTPUT_PATH)
    trace_parser.add_argument("--prompt", default=prompt)
    trace_parser.add_argument("--max-new-tokens", type=int, default=4)
    trace_parser.add_argument("--temperature", type=float, default=0.0)
    trace_parser.add_argument("--device", choices=["cuda"], default="cuda")
    trace_parser.add_argument("--cuda-visible-devices", default="0")
    trace_parser.add_argument("--forward-mode", choices=["plain", "trace"], default="plain")

    if len(sys.argv) == 1:
        return parser.parse_args(["trace"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_truncated_checkpoint(
            args.model_path,
            args.output_path,
            args.num_layers,
            args.expert_dtype,
            args.max_seq_len,
            args.max_batch_size,
        )
        return
    if args.command == "trace":
        apply_runtime_env(args.cuda_visible_devices)
        trace_generation(
            args.model_path,
            args.ckpt_path,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.device,
            args.forward_mode,
        )
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()