from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from xhquant.api import (
    CacheTensor,
    DeviceType,
    HMONNXGoldenInference,
    HMONNXInference,
    QuantScheme,
    convert_fx_model_to_quanted_model,
    convert_onnx_to_hmonnx,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)
from xhquant.api import ConfigDict
from xh_model_zoo.utils.time_profiler import TimeProfiler

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from kimi_vl_common import (  # noqa: E402
    build_language_model,
    build_processor_inputs,
    build_visual_model,
    build_wrapped_language_model,
    create_kv_caches,
    finalize_modelscope_weights,
    flatten_hmonnx_inputs,
    load_config_json,
    merge_image_embeds,
    pad_prefill_inputs,
)


def compare_tensors(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> dict:
    lhs = lhs.detach().float().cpu()
    rhs = rhs.detach().float().cpu()
    max_abs_diff = float((lhs - rhs).abs().max().item())
    mean_abs_diff = float((lhs - rhs).abs().mean().item())
    return {"name": name, "max_abs_diff": max_abs_diff, "mean_abs_diff": mean_abs_diff}


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data02/datasets/kimivl")
    parser.add_argument("--image", type=str, default="example.png")
    parser.add_argument("--prompt", type=str, default="请详细描述这张图片中的内容。")
    parser.add_argument("--work-dir", type=str, default="work_dirs/kimi_vl_a3b_xh2a")
    parser.add_argument("--image-size-h", type=int, default=448)
    parser.add_argument("--image-size-w", type=int, default=448)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--input-sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--quant-type", type=str, default="w8a8h1_sefp")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "hmonnx").mkdir(exist_ok=True, parents=True)
    (work_dir / "golden").mkdir(exist_ok=True, parents=True)

    finalize_modelscope_weights(model_dir)

    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    logger.info("Preparing Kimi-VL export")

    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a,
        quant_type=args.quant_type,
        ops=dict(
            MatMul=dict(
                act_scheme=dict(bits=8, fp_mode="sefp"),
                act_schema_2=dict(bits=16, fp_mode="sefp"),
            )
        ),
    )
    quant_config = ConfigDict(create_quant_config(quant_scheme))

    cfg_json = load_config_json(model_dir)
    native_model, token_embedding, _ = build_language_model(model_dir)
    processor, prompt_text, raw_inputs = build_processor_inputs(
        model_dir=model_dir,
        image_path=args.image,
        prompt=args.prompt,
        image_size_h=args.image_size_h,
        image_size_w=args.image_size_w,
    )
    visual_model = build_visual_model(model_dir, raw_inputs["image_grid_hws"], dtype=torch.float16).eval().cpu()

    with torch.no_grad(), TimeProfiler("visual_reference", logger):
        image_embeds_ref = visual_model(raw_inputs["pixel_values"].half().cpu()).cpu()

    inputs_embeds = merge_image_embeds(
        token_embedding=token_embedding.cpu(),
        input_ids=raw_inputs["input_ids"].cpu(),
        image_embeds=image_embeds_ref.cpu(),
        media_placeholder_token_id=cfg_json["media_placeholder_token_id"],
    ).cpu()
    seq_len = raw_inputs["input_ids"].shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)

    seq_len = int(raw_inputs["input_ids"].shape[1])
    effective_input_sequence_length = max(args.input_sequence_length, ((seq_len + 63) // 64) * 64)

    wrapped_model, wrap_cfg = build_wrapped_language_model(
        native_model=native_model.cpu(),
        batch_size=args.batch_size,
        context_length=args.context_length,
        input_sequence_length=effective_input_sequence_length,
    )
    wrapped_model = wrapped_model.eval().half().cpu()

    padded_inputs_embeds, padded_position_ids, current_input_length = pad_prefill_inputs(
        prepared=type("Prepared", (), {"inputs_embeds": inputs_embeds, "position_ids": position_ids, "current_input_length": seq_len})(),
        token_embedding=token_embedding.cpu(),
        input_sequence_length=effective_input_sequence_length,
        pad_token_id=cfg_json["text_config"]["pad_token_id"],
    )

    key_head_dim = int(cfg_json["text_config"]["qk_nope_head_dim"] + cfg_json["text_config"]["qk_rope_head_dim"])
    value_head_dim = int(cfg_json["text_config"]["v_head_dim"])
    num_hidden_layers = int(cfg_json["text_config"]["num_hidden_layers"])
    num_key_value_heads = int(cfg_json["text_config"]["num_key_value_heads"])

    pt_prefill_k, pt_prefill_v = create_kv_caches(
        batch_size=args.batch_size,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        context_length=args.context_length,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        dtype=torch.float16,
        cache_tensor=True,
    )
    past_seq_length = torch.tensor([0], dtype=torch.int32)
    prefill_inputs = (
        padded_inputs_embeds.half(),
        past_seq_length,
        current_input_length,
        padded_position_ids,
        pt_prefill_k,
        pt_prefill_v,
    )

    with torch.no_grad(), TimeProfiler("prefill_reference", logger):
        ref_prefill_logits = wrapped_model(*prefill_inputs).cpu()

    ref_next_token = torch.argmax(ref_prefill_logits[:, -1, :], dim=-1, keepdim=True).cpu().to(torch.long)
    ref_next_token_text = processor.batch_decode(ref_next_token, skip_special_tokens=False)[0]
    logger.info("Prefill next token: %s", ref_next_token_text)

    update_cfg = ConfigDict(dict(wrap_cfg))
    update_cfg.input_sequence_length = 1

    def _update_cfg_fn(module):
        if hasattr(module, "_update_cfg"):
            module._update_cfg(update_cfg)

    decode_position_ids = torch.tensor([[seq_len]], dtype=torch.int32)
    decode_inputs_embeds = token_embedding(ref_next_token)
    decode_past_seq_length = torch.tensor([seq_len], dtype=torch.int32)
    decode_current_input_length = torch.tensor([1], dtype=torch.int32)
    decode_inputs = (
        decode_inputs_embeds.half(),
        decode_past_seq_length,
        decode_current_input_length,
        decode_position_ids,
        pt_prefill_k,
        pt_prefill_v,
    )
    with torch.no_grad(), TimeProfiler("decode_reference", logger):
        ref_decode_logits = wrapped_model(*decode_inputs).cpu()

    prefix = f"{model_dir.name}-xh2a-{args.quant_type}"
    visual_hmonnx_file = work_dir / "hmonnx" / f"{prefix}-vision.onnx"
    prefill_hmonnx_file = work_dir / "hmonnx" / f"{prefix}-prefill.onnx"
    decode_hmonnx_file = work_dir / "hmonnx" / f"{prefix}-decode.onnx"

    if not visual_hmonnx_file.exists():
        with TimeProfiler("export_visual_onnx", logger):
            tmp_visual_onnx = work_dir / "hmonnx" / f"{prefix}-vision-tmp.onnx"
            torch.onnx.export(
                visual_model.float().cpu(),
                (raw_inputs["pixel_values"].float().cpu(),),
                str(tmp_visual_onnx),
                input_names=["pixel_values"],
                output_names=["image_embeds"],
                opset_version=18,
                do_constant_folding=True,
            )
            convert_onnx_to_hmonnx(
                str(tmp_visual_onnx),
                (raw_inputs["pixel_values"].float().cpu(),),
                DeviceType.XH2a,
                str(visual_hmonnx_file),
            )
            tmp_visual_onnx.unlink(missing_ok=True)

    onnx_input_names = ["inputs_embeds", "past_seq_length", "current_input_length", "position_ids"]
    onnx_input_names += [f"past_key_cache_{i}" for i in range(num_hidden_layers)]
    onnx_input_names += [f"past_value_cache_{i}" for i in range(num_hidden_layers)]

    if not prefill_hmonnx_file.exists():
        with TimeProfiler("export_prefill_hmonnx", logger):
            quant_graph_model = convert_fx_model_to_quanted_model(
                wrapped_model, prefill_inputs, DeviceType.XH2a, quant_config
            )
            convert_quanted_model_to_hmonnx(
                quant_graph_model,
                prefill_inputs,
                str(prefill_hmonnx_file),
                onnx_input_names,
                ["logits"],
            )
    else:
        quant_graph_model = None

    if not decode_hmonnx_file.exists():
        with TimeProfiler("export_decode_hmonnx", logger):
            if quant_graph_model is None:
                quant_graph_model = convert_fx_model_to_quanted_model(
                    wrapped_model, prefill_inputs, DeviceType.XH2a, quant_config
                )
            quant_graph_model.apply(_update_cfg_fn)
            convert_quanted_model_to_hmonnx(
                quant_graph_model,
                decode_inputs,
                str(decode_hmonnx_file),
                onnx_input_names,
                ["logits"],
            )

    visual_golden_dir = work_dir / "golden" / f"{prefix}-vision"
    prefill_golden_dir = work_dir / "golden" / f"{prefix}-prefill"
    decode_golden_dir = work_dir / "golden" / f"{prefix}-decode"

    if not visual_golden_dir.exists():
        visual_golden = HMONNXGoldenInference(str(visual_hmonnx_file))
        visual_golden.exec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        visual_golden.save_golden = True
        visual_golden.golden_dir = str(visual_golden_dir)
        with torch.no_grad():
            visual_golden(raw_inputs["pixel_values"].half().cpu())

    if not prefill_golden_dir.exists():
        hm_prefill_k, hm_prefill_v = create_kv_caches(
            batch_size=args.batch_size,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            context_length=args.context_length,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            dtype=torch.float16,
            cache_tensor=True,
        )
        prefill_golden = HMONNXGoldenInference(str(prefill_hmonnx_file))
        prefill_golden.exec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        prefill_golden.save_golden = True
        prefill_golden.golden_dir = str(prefill_golden_dir)
        prefill_args = flatten_hmonnx_inputs(
            (
                padded_inputs_embeds.half(),
                past_seq_length,
                current_input_length,
                padded_position_ids,
                hm_prefill_k,
                hm_prefill_v,
            )
        )
        with torch.no_grad():
            prefill_golden(*prefill_args)

    if not decode_golden_dir.exists():
        hm_decode_k, hm_decode_v = create_kv_caches(
            batch_size=args.batch_size,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            context_length=args.context_length,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            dtype=torch.float16,
            cache_tensor=True,
        )
        prefill_runtime_for_decode = HMONNXInference(str(prefill_hmonnx_file))
        prefill_runtime_for_decode.exec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        prefill_runtime_for_decode(*flatten_hmonnx_inputs(
            (
                padded_inputs_embeds.half(),
                past_seq_length,
                current_input_length,
                padded_position_ids,
                hm_decode_k,
                hm_decode_v,
            )
        ))
        decode_golden = HMONNXGoldenInference(str(decode_hmonnx_file))
        decode_golden.exec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        decode_golden.save_golden = True
        decode_golden.golden_dir = str(decode_golden_dir)
        decode_args = flatten_hmonnx_inputs(
            (
                decode_inputs_embeds.half(),
                decode_past_seq_length,
                decode_current_input_length,
                decode_position_ids,
                hm_decode_k,
                hm_decode_v,
            )
        )
        with torch.no_grad():
            decode_golden(*decode_args)

    hm_visual = HMONNXInference(str(visual_hmonnx_file))
    hm_visual.exec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hm_visual_embeds = hm_visual(raw_inputs["pixel_values"].half().cpu())

    hm_prefill_k, hm_prefill_v = create_kv_caches(
        batch_size=args.batch_size,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        context_length=args.context_length,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        dtype=torch.float16,
        cache_tensor=True,
    )
    hm_prefill = HMONNXInference(str(prefill_hmonnx_file))
    hm_prefill.exec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hm_prefill_logits = hm_prefill(
        *flatten_hmonnx_inputs(
            (
                padded_inputs_embeds.half(),
                past_seq_length,
                current_input_length,
                padded_position_ids,
                hm_prefill_k,
                hm_prefill_v,
            )
        )
    )

    hm_decode = HMONNXInference(str(decode_hmonnx_file))
    hm_decode.exec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hm_decode_logits = hm_decode(
        *flatten_hmonnx_inputs(
            (
                decode_inputs_embeds.half(),
                decode_past_seq_length,
                decode_current_input_length,
                decode_position_ids,
                hm_prefill_k,
                hm_prefill_v,
            )
        )
    )

    validation = {
        "visual": compare_tensors("visual", image_embeds_ref, hm_visual_embeds),
        "prefill": compare_tensors("prefill", ref_prefill_logits, hm_prefill_logits),
        "decode": compare_tensors("decode", ref_decode_logits, hm_decode_logits),
    }

    meta = {
        "model_dir": str(model_dir),
        "work_dir": str(work_dir),
        "prompt": args.prompt,
        "prompt_text": prompt_text,
        "image_path": str(Path(args.image).resolve()),
        "image_size_h": args.image_size_h,
        "image_size_w": args.image_size_w,
        "context_length": args.context_length,
        "input_sequence_length": effective_input_sequence_length,
        "visual_hmonnx": str(visual_hmonnx_file.relative_to(work_dir)),
        "prefill_hmonnx": str(prefill_hmonnx_file.relative_to(work_dir)),
        "decode_hmonnx": str(decode_hmonnx_file.relative_to(work_dir)),
        "golden": {
            "visual": str(visual_golden_dir.relative_to(work_dir)),
            "prefill": str(prefill_golden_dir.relative_to(work_dir)),
            "decode": str(decode_golden_dir.relative_to(work_dir)),
        },
        "validation": validation,
        "next_token_id": int(ref_next_token.item()),
        "next_token_text": ref_next_token_text,
    }
    (work_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info("Validation summary: %s", json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
