#!/usr/bin/env python3
"""Run greedy text generation with an exported DeepSeek-V4 Flash HMONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_DEEPSEEK_V4_BOS = "<｜begin▁of▁sentence｜>"
_DEEPSEEK_V4_EOS = "<｜end▁of▁sentence｜>"
_DEEPSEEK_V4_USER = "<｜User｜>"
_DEEPSEEK_V4_ASSISTANT = "<｜Assistant｜>"


def resolve_meta_path(config: str | Path) -> Path:
    """Resolve either one metadata file or one unambiguous export directory."""

    path = Path(config).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"HMONNX config does not exist: {path}")
    direct = path / "golden_meta_info.json"
    if direct.is_file():
        return direct
    matches = sorted(path.glob("hmquant_*/golden_meta_info.json"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one golden_meta_info.json below {path}, got {len(matches)}")
    return matches[0]


def parse_devices(value: str) -> list[str | int]:
    """Parse ``cpu`` or a comma-separated CUDA device list."""

    tokens = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--device must be 'cpu' or comma-separated GPU ids")
    if tokens == ["cpu"]:
        return ["cpu"]
    if "cpu" in tokens:
        raise ValueError("--device cannot mix CPU and GPUs")

    devices: list[int] = []
    for token in tokens:
        token = token.removeprefix("cuda:")
        if not token.isdigit():
            raise ValueError(f"unsupported device token: {token!r}")
        device = int(token)
        if device not in devices:
            devices.append(device)
    return devices


def validate_packed_shared_weights(summary: dict[str, int]) -> None:
    """Verify that runtime W4 tensors were packed once and shared by both graphs."""

    planned = int(summary.get("runtime_w4_planned_references", 0))
    packed = int(summary.get("runtime_w4_packed_references", 0))
    unique = int(summary.get("runtime_w4_packed_initializers", 0))
    shared = int(summary.get("runtime_w4_shared_references", 0))
    if planned <= 0:
        raise RuntimeError("HMONNX package contains no runtime W4 initializers")
    if packed != planned or unique != shared or planned != 2 * shared:
        raise RuntimeError(
            f"invalid packed/shared W4 contract: planned={planned}, packed={packed}, unique={unique}, shared={shared}"
        )


def build_user_content(prompt: str, context_file: str | None) -> str:
    """Place an optional real context before the final user question."""

    if context_file is None:
        return prompt
    path = Path(context_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"context file does not exist: {path}")
    context = path.read_text(encoding="utf-8")
    if not context.strip():
        raise ValueError(f"context file is empty: {path}")
    return f"{context}\n\n请根据以上内容回答：{prompt}"


def render_prompt(tokenizer, user_content: str, *, raw_prompt: bool) -> str:
    if raw_prompt:
        return user_content
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if (
        getattr(tokenizer, "bos_token", None) == _DEEPSEEK_V4_BOS
        and getattr(tokenizer, "eos_token", None) == _DEEPSEEK_V4_EOS
    ):
        # DeepSeek-V4 releases include a reference encoder but intentionally do
        # not embed a Jinja chat template in tokenizer_config.json.  This is its
        # documented single-user-turn encoding in non-thinking (chat) mode.
        return f"{_DEEPSEEK_V4_BOS}{_DEEPSEEK_V4_USER}{user_content}{_DEEPSEEK_V4_ASSISTANT}</think>"
    raise ValueError(
        "tokenizer has no chat_template and is not a recognized DeepSeek-V4 tokenizer; "
        "pass --raw-prompt only when the input already contains the complete model prompt"
    )


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import TextStreamer

    from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
    from xhmodel_merak.xh_llm.models.deepseek_v4 import XHDeepSeekV4HMONNXModel
    from xhmodel_merak.xh_llm.utils import configure_hmonnx_validation_runtime
    from xhquant.api import xhquant_init

    meta_path = resolve_meta_path(args.config)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    devices = parse_devices(args.device)
    auto_offload = len(devices) > 1
    if auto_offload and args.cuda_graph:
        raise ValueError("CUDA Graph is not supported with multi-device auto-offload")

    configure_hmonnx_validation_runtime(use_v2=True, pack_w4=args.pack_w4)
    xhquant_init(args.log_file, args.debug)
    model = AutoLLMHONNXModel.from_pretrained(
        str(meta_path),
        device_map=devices,
        enable_auto_offload=auto_offload,
        enable_cuda_graph=args.cuda_graph,
        enable_golden=args.golden,
    )
    if not isinstance(model, XHDeepSeekV4HMONNXModel):
        raise TypeError(f"expected XHDeepSeekV4HMONNXModel, got {type(model).__name__}")
    if args.pack_w4:
        validate_packed_shared_weights(model.shared_weight_summary)

    # Match the workflow golden contract used by Qwen3.5: enabling golden on
    # the fully constructed model resets both prefill/decode sessions to
    # step 0, so files are emitted below ``<stage>/step_0`` rather than being
    # scattered directly beside the HMONNX graph.
    if args.golden:
        model.enable_golden = True

    cache_summary = model._kvcache_mixin.residency_summary()
    if cache_summary["mismatches"]:
        raise RuntimeError(f"cache tensors do not match graph placement: {cache_summary['mismatches'][:8]}")
    if args.load_only:
        print(
            json.dumps(
                {
                    "meta": str(meta_path),
                    "devices": [str(device) for device in model._valid_devices],
                    "auto_offload": auto_offload,
                    "weights": model.shared_weight_summary,
                    "caches": cache_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    tokenizer = model.get_tokenizer()
    user_content = build_user_content(args.prompt, args.context_file)
    text = render_prompt(tokenizer, user_content, raw_prompt=args.raw_prompt)
    inputs = tokenizer([text], return_tensors="pt")
    prompt_tokens = int(inputs.input_ids.shape[1])
    context_limit = int(meta["model_config"]["context_max_length"])
    if prompt_tokens + args.max_new_tokens > context_limit:
        raise ValueError(
            "prompt and continuation exceed the exported context: "
            f"{prompt_tokens} + {args.max_new_tokens} > {context_limit}"
        )

    runtime_device = torch.device(model.device)
    inputs = inputs.to(runtime_device)
    model.to(runtime_device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = model.pad_token_id
    streamer = TextStreamer(tokenizer, skip_prompt=True) if args.stream else None
    with LLMInferenceContextManager(model, devices=model._valid_devices):
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=int(pad_token_id),
            streamer=streamer,
        )
    continuation = generated[:, prompt_tokens:]
    print(tokenizer.batch_decode(continuation, skip_special_tokens=True)[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="golden_meta_info.json or export directory")
    parser.add_argument("--prompt", default="请简要解释为什么天空看起来是蓝色的。")
    parser.add_argument("--context-file", help="optional UTF-8 long context placed before the question")
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="0,1,2,3")
    parser.add_argument("--pack-w4", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--golden", "--dump-golden", dest="golden", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    run(args)


if __name__ == "__main__":
    main()
