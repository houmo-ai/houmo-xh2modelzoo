# Copyright 2026 HOUMO AI

"""Forward sanity suite for exported Qwen3-Omni HMONNX artifacts."""

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hmonnx_pipeline import discover_artifacts, run_text_hmonnx_chain_forward, save_json

from xhquant.api import CacheTensor, get_root_logger, xhquant_init
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference


def _build_prefill_fused_inputs(meta, captured_entry, inputs_embeds: torch.Tensor):
    seq_len = int(inputs_embeds.shape[1])
    target_seq_len = int(meta.get("talker_input_sequence_length", seq_len))
    batch = int(inputs_embeds.shape[0])
    hidden_state_size = int(
        meta.get(
            "talker_hidden_state_size",
            meta.get("talker_thinker_hidden_size", meta.get("talker_projection_in_features", inputs_embeds.shape[-1])),
        )
    )
    if captured_entry is not None:
        hidden_state = captured_entry.get("hidden_state")
        role_mask = captured_entry.get("role_mask")
        bypass_embeds = captured_entry.get("bypass_embeds")
        bypass_mask = captured_entry.get("bypass_mask")
        if all(isinstance(item, torch.Tensor) for item in (hidden_state, role_mask, bypass_embeds, bypass_mask)):
            if int(hidden_state.shape[1]) >= seq_len and int(bypass_embeds.shape[1]) >= seq_len:
                hidden_state = hidden_state[:, :seq_len, :].to(torch.float16).cpu()
                role_mask = role_mask[:, :seq_len, :].to(torch.float16).cpu()
                bypass_embeds = bypass_embeds[:, :seq_len, :].to(torch.float16).cpu()
                bypass_mask = bypass_mask[:, :seq_len, :].to(torch.float16).cpu()
                if seq_len < target_seq_len:
                    hidden_state = torch.cat(
                        [hidden_state, torch.zeros(batch, target_seq_len - seq_len, hidden_state_size, dtype=torch.float16)],
                        dim=1,
                    )
                    role_mask = torch.cat(
                        [role_mask, torch.zeros(batch, target_seq_len - seq_len, 1, dtype=torch.float16)],
                        dim=1,
                    )
                    bypass_embeds = torch.cat(
                        [bypass_embeds, torch.zeros(batch, target_seq_len - seq_len, bypass_embeds.shape[-1], dtype=torch.float16)],
                        dim=1,
                    )
                    bypass_mask = torch.cat(
                        [bypass_mask, torch.zeros(batch, target_seq_len - seq_len, 1, dtype=torch.float16)],
                        dim=1,
                    )
                return (
                    hidden_state,
                    role_mask,
                    bypass_embeds,
                    bypass_mask,
                )
    return (
        torch.zeros(batch, target_seq_len, hidden_state_size, dtype=torch.float16),
        torch.zeros(batch, target_seq_len, 1, dtype=torch.float16),
        torch.cat(
            [
                inputs_embeds,
                torch.zeros(batch, target_seq_len - seq_len, inputs_embeds.shape[-1], dtype=torch.float16),
            ],
            dim=1,
        )
        if seq_len < target_seq_len
        else inputs_embeds,
        torch.cat(
            [
                torch.ones(batch, seq_len, 1, dtype=torch.float16),
                torch.zeros(batch, target_seq_len - seq_len, 1, dtype=torch.float16),
            ],
            dim=1,
        )
        if seq_len < target_seq_len
        else torch.ones(batch, seq_len, 1, dtype=torch.float16),
    )


def _run_talker_forward(meta, report):
    capture_path = Path(meta["_root_dir"]) / "talker_model_inputs.pth"
    if not capture_path.exists():
        report["talker"] = {"status": "skipped", "reason": f"missing {capture_path.name}"}
        return

    captured = torch.load(capture_path, map_location="cpu", weights_only=False)
    captured_entry = captured[0]
    inputs_embeds = captured_entry["inputs_embeds"].to(torch.float16).cpu()
    kv_info = meta["talker_kv_cache"]
    kv_shape = kv_info["shape"]
    num_layers = kv_info["num_decoder_layers"]
    hidden_state_size = int(
        meta.get(
            "talker_hidden_state_size",
            meta.get("talker_thinker_hidden_size", meta.get("talker_projection_in_features", 0)),
        )
    )
    past_key_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)]
    past_value_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)]
    prefill = HMONNXInference(str(Path(meta["_root_dir"]) / meta["talker_prefill_onnx"]))
    decode = HMONNXInference(str(Path(meta["_root_dir"]) / meta["talker_decode_onnx"]))

    batch = int(inputs_embeds.shape[0])
    prefill_seq = int(meta.get("talker_input_sequence_length", inputs_embeds.shape[1]))
    prefill_hidden_state, prefill_role_mask, prefill_bypass_embeds, prefill_bypass_mask = _build_prefill_fused_inputs(
        meta, captured_entry, inputs_embeds
    )
    prefill_out = prefill.forward(
        prefill_hidden_state,
        prefill_role_mask,
        prefill_bypass_embeds,
        prefill_bypass_mask,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([prefill_seq], dtype=torch.int32),
        *past_key_caches,
        *past_value_caches,
    )
    decode_out = decode.forward(
        torch.zeros(batch, 1, hidden_state_size, dtype=torch.float16),
        torch.zeros(batch, 1, 1, dtype=torch.float16),
        inputs_embeds[:, :1, :],
        torch.ones(batch, 1, 1, dtype=torch.float16),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        *past_key_caches,
        *past_value_caches,
    )
    prefill_outputs = list(prefill_out) if isinstance(prefill_out, (list, tuple)) else [prefill_out]
    decode_outputs = list(decode_out) if isinstance(decode_out, (list, tuple)) else [decode_out]
    talker_report = {
        "status": "ok",
        "prefill_output_shapes": [list(tensor.shape) for tensor in prefill_outputs],
        "decode_output_shapes": [list(tensor.shape) for tensor in decode_outputs],
        "prefill_hidden_state_shape": list(prefill_hidden_state.shape),
    }
    trailing_text_hidden = captured_entry.get("trailing_text_hidden")
    if isinstance(trailing_text_hidden, torch.Tensor):
        talker_report["style_guidance_hidden_shape"] = list(trailing_text_hidden.shape)
    report["talker"] = talker_report


def _run_talker_prediction_forward(meta, report):
    capture_path = Path(meta["_root_dir"]) / "talker_prediction_inputs.pth"
    if not capture_path.exists():
        report["talker_prediction"] = {"status": "skipped", "reason": f"missing {capture_path.name}"}
        return

    captured = torch.load(capture_path, map_location="cpu", weights_only=False)
    inputs_embeds = captured[0]["inputs_embeds"].to(torch.float16).cpu()
    kv_info = meta["talker_prediction_kv_cache"]
    kv_shape = kv_info["shape"]
    num_layers = kv_info["num_decoder_layers"]
    num_lm_heads = int(meta.get("lm_head_count", 15))
    batch = int(inputs_embeds.shape[0])
    prefill_seq = int(inputs_embeds.shape[1])
    past_key_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)]
    past_value_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)]
    prefill = HMONNXInference(str(Path(meta["_root_dir"]) / meta["talker_prediction_prefill_onnx"]))
    decode = HMONNXInference(str(Path(meta["_root_dir"]) / meta["talker_prediction_decode_onnx"]))

    prefill_step = max(0, min(prefill_seq - 2, num_lm_heads - 1))
    head_mask_prefill = torch.zeros(batch, prefill_seq, num_lm_heads, 1, dtype=torch.float16)
    head_mask_prefill[:, :, prefill_step, 0] = 1.0
    head_mask_decode = torch.zeros(batch, 1, num_lm_heads, 1, dtype=torch.float16)
    head_mask_decode[0, 0, 0, 0] = 1.0

    prefill_out = prefill.forward(
        inputs_embeds,
        head_mask_prefill,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([prefill_seq], dtype=torch.int32),
        *past_key_caches,
        *past_value_caches,
    )
    decode_out = decode.forward(
        inputs_embeds[:, :1, :],
        head_mask_decode,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        *past_key_caches,
        *past_value_caches,
    )
    prefill_outputs = list(prefill_out) if isinstance(prefill_out, (list, tuple)) else [prefill_out]
    decode_outputs = list(decode_out) if isinstance(decode_out, (list, tuple)) else [decode_out]
    report["talker_prediction"] = {
        "status": "ok",
        "prefill_output_shapes": [list(tensor.shape) for tensor in prefill_outputs],
        "decode_output_shapes": [list(tensor.shape) for tensor in decode_outputs],
        "residual_hidden_contract": "second output should match predictor input embeddings for talker residual sum",
    }


def _run_code2wav_forward(meta, report, work_dir: Path):
    session = HMONNXInference(str(Path(meta["_root_dir"]) / meta["code2wav_hmonnx"]))
    static_code_len = int(meta["static_code_len"])
    codes = torch.randint(0, 100, (1, 16, static_code_len), dtype=torch.int32)
    output = session.forward(codes)
    tensor = output[0] if isinstance(output, (list, tuple)) else output
    wav_path = work_dir / "hmonnx_forward_code2wav.wav"
    waveform = tensor.reshape(-1).detach().cpu().to(torch.float32).numpy()
    sf.write(str(wav_path), waveform, samplerate=24000)
    report["code2wav"] = {
        "status": "ok",
        "output_shape": list(tensor.shape),
        "audio_file": str(wav_path.relative_to(work_dir)),
    }


def main(args):
    work_dir = Path(args.work_dir)
    work_dir.mkdir(exist_ok=True, parents=True)
    xhquant_init(work_dir / "hmonnx_forward.log", debug=args.debug)
    logger = get_root_logger()

    quick_mode = args.quick or args.case == "vision"
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (1 if quick_mode else 256)

    artifacts = discover_artifacts(work_dir)
    if "text" not in artifacts:
        raise RuntimeError(f"No qwen3omni text meta.json found under {work_dir}")

    report = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "artifacts": sorted(list(artifacts.keys())),
        "quick_mode": quick_mode,
        "max_new_tokens": max_new_tokens,
    }

    logger.info(
        f"running hmonnx forward suite with case={args.case}, quick_mode={quick_mode}, max_new_tokens={max_new_tokens}"
    )

    report["text_chain"] = run_text_hmonnx_chain_forward(
        args.model,
        artifacts["text"],
        logger,
        case=args.case,
        audio_meta=artifacts.get("audio"),
        vision_meta=artifacts.get("vision"),
        max_new_tokens=max_new_tokens,
        device_map=args.device_map,
    )

    if quick_mode:
        report["extra_modules"] = {
            "status": "skipped",
            "reason": "quick_mode enabled",
        }
    elif "talker" in artifacts:
        _run_talker_forward(artifacts["talker"], report)
        if "talker_prediction" in artifacts:
            _run_talker_prediction_forward(artifacts["talker_prediction"], report)
        if "code2wav" in artifacts:
            _run_code2wav_forward(artifacts["code2wav"], report, work_dir)

    report_path = work_dir / "hmonnx_forward_report.json"
    save_json(report_path, report)
    logger.info(f"HMONNX forward suite report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run forward sanity checks for exported Qwen3-Omni HMONNX artifacts")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--case", type=str, default="multimodal", choices=["text", "vision", "audio", "multimodal"])
    parser.add_argument("--device-map", type=str, default="auto", choices=["auto", "cpu", "cuda:0"])
    parser.add_argument(
        "--quick", action="store_true", help="run only a fast text-chain sanity check and skip extra module forwards"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="override generated token count for text-chain validation"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
