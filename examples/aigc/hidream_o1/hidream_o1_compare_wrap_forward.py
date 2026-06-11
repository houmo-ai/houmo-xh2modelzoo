import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor
from xhquant.utils.registry import DynamicModule

from xh_model_zoo.xh_llm.models.hidream_o1 import (
    HiDreamO1DenoiseExportWrapper,
    build_rotary_inputs,
    build_t2i_sample_inputs,
    ensure_hidream_o1_imports,
    register_hidream_o1_wrap_modules,
)

ensure_hidream_o1_imports()
from xh_model_zoo.xh_llm.models.hidream_o1.models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration  # pyright: ignore[reportMissingImports]  # noqa: E402


def force_eager_attention(model: torch.nn.Module) -> None:
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"


def summarize_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    lhs_f = lhs.float()
    rhs_f = rhs.float()
    diff = (lhs_f - rhs_f).abs()
    cosine = torch.nn.functional.cosine_similarity(lhs_f.flatten(), rhs_f.flatten(), dim=0)
    print(
        f"{name}: shape={tuple(lhs.shape)} max={float(diff.max()):.8f} "
        f"mean={float(diff.mean()):.8f} cosine={float(cosine):.8f}"
    )


def build_origin_attention_mask(
    token_types: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    pad_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    token_types = token_types.to(device)
    if token_types.dim() == 1:
        token_types = token_types.unsqueeze(0)
    batch_size, total_seq_len = token_types.shape
    min_val = torch.finfo(dtype).min
    attn_masks = []
    for batch_idx in range(batch_size):
        causal = torch.full((total_seq_len, total_seq_len), min_val, device=device, dtype=dtype)
        causal = torch.triu(causal, diagonal=1)
        gen_positions = token_types[batch_idx].bool()
        causal[gen_positions, :] = 0
        if pad_positions is not None and pad_positions.numel() > 0:
            causal[:, pad_positions] = min_val
            causal[pad_positions, :] = min_val
            causal[pad_positions, pad_positions] = 0
        attn_masks.append(causal)
    return torch.stack(attn_masks, dim=0).unsqueeze(1)


def print_mask_regions(attention_mask: torch.Tensor, txt_seq_len: int) -> None:
    mask = attention_mask[0, 0]
    regions = {
        "text_to_text": mask[:txt_seq_len, :txt_seq_len],
        "text_to_image": mask[:txt_seq_len, txt_seq_len:],
        "image_to_text": mask[txt_seq_len:, :txt_seq_len],
        "image_to_image": mask[txt_seq_len:, txt_seq_len:],
    }
    min_val = torch.finfo(attention_mask.dtype).min
    for name, value in regions.items():
        allowed = (value == 0).sum().item()
        blocked = (value == min_val).sum().item()
        print(f"mask_{name}: shape={tuple(value.shape)} allowed_zero={allowed} blocked_min={blocked}")


def run_decoder_layer_parts(layer, hidden_states, position_embeddings, attention_mask):
    residual = hidden_states
    input_norm = layer.input_layernorm(hidden_states)
    attn_output, _ = layer.self_attn(
        hidden_states=input_norm,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
    )
    after_attn = residual + attn_output
    post_norm = layer.post_attention_layernorm(after_attn)
    mlp_output = layer.mlp(post_norm)
    after_mlp = after_attn + mlp_output
    return {
        "input_norm": input_norm,
        "attn_output": attn_output,
        "after_attn": after_attn,
        "post_norm": post_norm,
        "mlp_output": mlp_output,
        "after_mlp": after_mlp,
    }


def run_text_decoder(language_model, hidden_states, position_embeddings, attention_mask, compare_layers=None):
    mid_results = [] if compare_layers else None
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        if compare_layers is not None and layer_idx in compare_layers:
            mid_results.append(hidden_states)
    hidden_states = language_model.norm(hidden_states)
    return hidden_states, mid_results


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data02/datasets/HiDream-O1-Image")
    parser.add_argument("--prompt", type=str, default="A beautiful castle beside a lake, cinematic, highly detailed")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--text-seq-len", type=int, default=512)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--compare-layers", action="store_true", help="同时对比每层 decoder 输出，定位第一处差异")
    parser.add_argument(
        "--compare-unpadded-origin",
        action="store_true",
        help="origin 使用未 pad 原始 prompt 长度，wrapped 使用 --text-seq-len pad 后长度",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_dir = Path(args.model).resolve()

    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        device_map="cuda" if device.type == "cuda" else None,
    ).eval()
    force_eager_attention(model)

    origin_sample = None
    origin_vinputs = None
    if args.compare_unpadded_origin:
        origin_sample, origin_vinputs = build_t2i_sample_inputs(
            model=model,
            processor=processor,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            seed=args.seed,
            dtype=dtype,
            device=device,
            timestep=args.timestep,
            text_seq_len=0,
        )

    sample, vinputs = build_t2i_sample_inputs(
        model=model,
        processor=processor,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        dtype=dtype,
        device=device,
        timestep=args.timestep,
        text_seq_len=args.text_seq_len,
    )
    txt_seq_len = int(sample["input_ids"].shape[-1])

    sample_check, vinputs_check = build_t2i_sample_inputs(
        model=model,
        processor=processor,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        dtype=dtype,
        device=device,
        timestep=args.timestep,
        text_seq_len=args.text_seq_len,
    )
    summarize_diff("vinputs_repeat_check", vinputs, vinputs_check)
    print(f"sample_input_ids_repeat_equal: {torch.equal(sample['input_ids'], sample_check['input_ids'])}")
    origin_attention_mask = build_origin_attention_mask(sample["token_types"], dtype, device, sample["pad_positions"])
    summarize_diff("attention_mask_vs_origin", sample["attention_mask"], origin_attention_mask)
    print_mask_regions(sample["attention_mask"], txt_seq_len)

    compare_layers = list(range(len(model.model.language_model.layers))) if args.compare_layers else None
    if args.compare_unpadded_origin:
        unpadded_txt_seq_len = int(origin_sample["input_ids"].shape[-1])
        origin_outputs = model(
            input_ids=origin_sample["input_ids"],
            position_ids=origin_sample["position_ids"],
            vinputs=origin_vinputs,
            timestep=origin_sample["timestep"],
            token_types=origin_sample["token_types"],
            use_flash_attn=False,
            return_mid_results_layers=compare_layers,
        )
        origin_x_pred = origin_outputs.x_pred[:, unpadded_txt_seq_len:, :].float()
    else:
        origin_x_pred = None

    # RoPE only depends on position_ids/model rotary constants. Precompute it before
    # wrapping so the comparison uses the original model's exact rotary path.
    text_embeds = model.model.get_input_embeddings()(sample["input_ids"])
    t_emb = model.model.t_embedder1(sample["timestep"].to(text_embeds.device))
    tms_mask = (sample["input_ids"] == model.model.tms_token_id).unsqueeze(-1).expand_as(text_embeds)
    origin_text_embeds = torch.where(tms_mask, t_emb.unsqueeze(1).expand_as(text_embeds), text_embeds)
    wrapped_text_embeds = torch.cat([text_embeds[:, :-1, :], t_emb.unsqueeze(1)], dim=1)
    summarize_diff("text_embeds_after_timestep", origin_text_embeds, wrapped_text_embeds)
    vinputs_embedded = model.model.x_embedder(vinputs.to(text_embeds.device)).to(text_embeds.dtype)
    rotary_dummy_embeds = torch.cat([wrapped_text_embeds, vinputs_embedded], dim=1)
    origin_decoder_inputs = torch.cat([origin_text_embeds, vinputs_embedded], dim=1)
    wrapped_decoder_inputs = torch.cat([wrapped_text_embeds, vinputs_embedded], dim=1)
    summarize_diff("decoder_inputs_before_layers", origin_decoder_inputs, wrapped_decoder_inputs)
    rotary_position_ids = sample["position_ids"]
    if rotary_position_ids.ndim == 2:
        rotary_position_ids = rotary_position_ids[None, ...].expand(3, rotary_position_ids.shape[0], -1)
    elif rotary_position_ids.ndim == 3 and rotary_position_ids.shape[0] == 4:
        rotary_position_ids = rotary_position_ids[1:]
    rotary_cos, rotary_sin = build_rotary_inputs(
        model.model.language_model.rotary_emb,
        rotary_dummy_embeds,
        rotary_position_ids,
        dtype=dtype,
    )
    origin_layer0 = model.model.language_model.layers[0]
    origin_layer0_parts = run_decoder_layer_parts(
        origin_layer0,
        origin_decoder_inputs,
        position_embeddings=(rotary_cos, rotary_sin),
        attention_mask=sample["attention_mask"],
    )
    origin_layer0_out = origin_layer0(
        origin_decoder_inputs,
        position_embeddings=(rotary_cos, rotary_sin),
        attention_mask=sample["attention_mask"],
    )
    origin_hidden_states, origin_mid_results = run_text_decoder(
        model.model.language_model,
        origin_decoder_inputs,
        position_embeddings=(rotary_cos, rotary_sin),
        attention_mask=sample["attention_mask"],
        compare_layers=compare_layers,
    )
    if origin_x_pred is None:
        origin_x_pred = model.model.final_layer2(origin_hidden_states)[:, txt_seq_len:, :].float()

    register_hidream_o1_wrap_modules(model)
    print(
        "wrapped_types: "
        f"model={isinstance(model, DynamicModule)} "
        f"inner={isinstance(model.model, DynamicModule)} "
        f"lm={isinstance(model.model.language_model, DynamicModule)} "
        f"layer0_attn={isinstance(model.model.language_model.layers[0].self_attn, DynamicModule)}"
    )
    wrapped_layer0 = model.model.language_model.layers[0]
    wrapped_layer0_parts = run_decoder_layer_parts(
        wrapped_layer0,
        wrapped_decoder_inputs,
        position_embeddings=(rotary_cos, rotary_sin),
        attention_mask=sample["attention_mask"],
    )
    wrapped_layer0_out = wrapped_layer0(
        wrapped_decoder_inputs,
        position_embeddings=(rotary_cos, rotary_sin),
        attention_mask=sample["attention_mask"],
    )
    for part_name in origin_layer0_parts:
        summarize_diff(f"layer0_{part_name}", origin_layer0_parts[part_name], wrapped_layer0_parts[part_name])
    summarize_diff("layer0_direct_out", origin_layer0_out, wrapped_layer0_out)

    wrapper = HiDreamO1DenoiseExportWrapper(model, txt_seq_len=txt_seq_len).to(device).eval()
    wrapped_x_pred = wrapper(
        text_embeds,
        sample["attention_mask"],
        rotary_cos,
        rotary_sin,
        vinputs,
        sample["timestep_index"],
    ).float()

    diff = (origin_x_pred - wrapped_x_pred).abs()
    cosine = torch.nn.functional.cosine_similarity(origin_x_pred.flatten(), wrapped_x_pred.flatten(), dim=0)
    print(f"origin shape: {tuple(origin_x_pred.shape)}, wrapped shape: {tuple(wrapped_x_pred.shape)}")
    print(f"origin mean/std: {float(origin_x_pred.mean()):.8f}/{float(origin_x_pred.std()):.8f}")
    print(f"wrapped mean/std: {float(wrapped_x_pred.mean()):.8f}/{float(wrapped_x_pred.std()):.8f}")
    print(f"max_abs_diff: {float(diff.max()):.8f}")
    print(f"mean_abs_diff: {float(diff.mean()):.8f}")
    print(f"cosine: {float(cosine):.8f}")
    is_close = torch.allclose(origin_x_pred, wrapped_x_pred, atol=args.atol, rtol=args.rtol)
    print(f"allclose(atol={args.atol}, rtol={args.rtol}): {is_close}")

    if args.compare_layers:
        wrapped_outputs = model.model(
            inputs_embeds=text_embeds,
            attention_mask=sample["attention_mask"],
            position_embeddings=(rotary_cos, rotary_sin),
            vinputs=vinputs,
            timestep=sample["timestep_index"],
            token_types=sample["token_types"].to(dtype=torch.int32),
            use_flash_attn=False,
            return_mid_results_layers=compare_layers,
        )
        print("layer_diff:")
        layer_pairs = zip(origin_mid_results, wrapped_outputs.mid_results, strict=True)
        for idx, (origin_mid, wrapped_mid) in enumerate(layer_pairs):
            layer_diff = (origin_mid.float() - wrapped_mid.float()).abs()
            layer_cos = torch.nn.functional.cosine_similarity(
                origin_mid.float().flatten(),
                wrapped_mid.float().flatten(),
                dim=0,
            )
            print(
                f"  layer {idx:02d}: max={float(layer_diff.max()):.8f} "
                f"mean={float(layer_diff.mean()):.8f} cosine={float(layer_cos):.8f}"
            )


if __name__ == "__main__":
    main()
