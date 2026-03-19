# Copyright 2026 HOUMO AI

"""Forward sanity suite for exported Qwen3-Omni HMONNX artifacts."""

import argparse
import time
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hmonnx_pipeline import discover_artifacts, run_text_hmonnx_chain_forward, save_json
from xhquant.api import CacheTensor, get_root_logger, xhquant_init
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference


def _run_talker_forward(meta, report):
    capture_path = Path(meta["_root_dir"]) / "talker_model_inputs.pth"
    if not capture_path.exists():
        report["talker"] = {"status": "skipped", "reason": f"missing {capture_path.name}"}
        return

    captured = torch.load(capture_path, map_location="cpu", weights_only=False)
    inputs_embeds = captured[0]["inputs_embeds"].to(torch.float16).cpu()
    kv_info = meta["talker_kv_cache"]
    kv_shape = kv_info["shape"]
    num_layers = kv_info["num_decoder_layers"]
    past_key_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)]
    past_value_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)]
    prefill = HMONNXInference(str(Path(meta["_root_dir"]) / meta["talker_prefill_onnx"]))
    decode = HMONNXInference(str(Path(meta["_root_dir"]) / meta["talker_decode_onnx"]))
    prefill_out = prefill.forward(
        inputs_embeds,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([inputs_embeds.shape[1]], dtype=torch.int32),
        *past_key_caches,
        *past_value_caches,
    )
    decode_out = decode.forward(
        inputs_embeds[:, :1, :],
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        *past_key_caches,
        *past_value_caches,
    )
    prefill_tensor = prefill_out[0] if isinstance(prefill_out, (list, tuple)) else prefill_out
    decode_tensor = decode_out[0] if isinstance(decode_out, (list, tuple)) else decode_out
    report["talker"] = {
        "status": "ok",
        "prefill_shape": list(prefill_tensor.shape),
        "decode_shape": list(decode_tensor.shape),
    }


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
    past_key_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)]
    past_value_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)]
    prefill = HMONNXInference(str(Path(meta["_root_dir"]) / meta["talker_prediction_prefill_onnx"]))
    decode = HMONNXInference(str(Path(meta["_root_dir"]) / meta["talker_prediction_decode_onnx"]))
    prefill_out = prefill.forward(
        inputs_embeds,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([inputs_embeds.shape[1]], dtype=torch.int32),
        *past_key_caches,
        *past_value_caches,
    )
    decode_out = decode.forward(
        inputs_embeds[:, :1, :],
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        *past_key_caches,
        *past_value_caches,
    )
    prefill_tensor = prefill_out[0] if isinstance(prefill_out, (list, tuple)) else prefill_out
    decode_tensor = decode_out[0] if isinstance(decode_out, (list, tuple)) else decode_out
    report["talker_prediction"] = {
        "status": "ok",
        "prefill_shape": list(prefill_tensor.shape),
        "decode_shape": list(decode_tensor.shape),
    }


def _run_projection_forward(meta, report):
    hidden_projection = HMONNXInference(str(Path(meta["_root_dir"]) / meta["hidden_projection_hmonnx"]))
    text_projection = HMONNXInference(str(Path(meta["_root_dir"]) / meta["text_projection_hmonnx"]))
    hidden_size = int(meta["hidden_size"])
    sample = torch.randn(1, 1, hidden_size, dtype=torch.float16)
    hidden_out = hidden_projection.forward(sample)
    text_out = text_projection.forward(sample)
    hidden_tensor = hidden_out[0] if isinstance(hidden_out, (list, tuple)) else hidden_out
    text_tensor = text_out[0] if isinstance(text_out, (list, tuple)) else text_out
    report["projection"] = {
        "status": "ok",
        "hidden_projection_shape": list(hidden_tensor.shape),
        "text_projection_shape": list(text_tensor.shape),
    }


def _run_code2wav_forward(meta, report):
    session = HMONNXInference(str(Path(meta["_root_dir"]) / meta["code2wav_hmonnx"]))
    static_code_len = int(meta["static_code_len"])
    codes = torch.randint(0, 100, (1, 16, static_code_len), dtype=torch.int32)
    output = session.forward(codes)
    tensor = output[0] if isinstance(output, (list, tuple)) else output
    report["code2wav"] = {
        "status": "ok",
        "output_shape": list(tensor.shape),
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
        if "projection" in artifacts:
            _run_projection_forward(artifacts["projection"], report)
        if "code2wav" in artifacts:
            _run_code2wav_forward(artifacts["code2wav"], report)

    report_path = work_dir / "hmonnx_forward_report.json"
    save_json(report_path, report)
    logger.info(f"HMONNX forward suite report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run forward sanity checks for exported Qwen3-Omni HMONNX artifacts")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--case", type=str, default="multimodal", choices=["text", "vision", "audio", "multimodal"])
    parser.add_argument("--device-map", type=str, default="auto", choices=["auto", "cpu", "cuda:0"])
    parser.add_argument("--quick", action="store_true", help="run only a fast text-chain sanity check and skip extra module forwards")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="override generated token count for text-chain validation")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)