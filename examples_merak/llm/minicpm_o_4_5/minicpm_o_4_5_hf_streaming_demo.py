from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Final, TypedDict

import librosa
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, DynamicCache

from examples_merak.llm.minicpm_o_4_5.media_utils import (
    flatten_audio_chunks,
    video_chunks,
    write_wav,
)
from examples_merak.llm.minicpm_o_4_5.minicpm_o_4_5_hf_streaming_support import (
    environment_versions,
    make_failure_packet,
    utc_timestamp,
    write_failure_packet,
    write_json,
)
from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import (
    ensure_tts_sampling_config,
    patch_audio_attention_return_compat,
    patch_dynamic_cache_legacy_methods,
    patch_dynamic_cache_seen_tokens,
    patch_empty_audio_cache,
    patch_remote_cache_helpers,
    patch_speech_generation_capture,
    patch_tts_cache_position_compat,
)


SEED: Final = 42
SAMPLE_RATE: Final = 16000
OUTPUT_AUDIO_RATE: Final = 24000
CASE_NAMES: Final = (
    "session_audio_text",
    "session_audio_reply",
    "duplex_audio_text",
    "duplex_audio_reply",
    "duplex_omni_reply",
)
PARAMETERS: Final = {
    "seed": SEED,
    "sampling": False,
    "do_sample": False,
    "chunk_ms": 1000,
    "first_chunk_ms": 1035,
    "cnn_redundancy_ms": 20,
    "sample_rate": SAMPLE_RATE,
    "batch_size": 1,
}


class CasePayload(TypedDict):
    name: str
    api_family: str
    prompt: str
    generate_audio: bool
    omni: bool
    asset_paths: list[str]


class RunState:
    def __init__(self, output_dir: Path, events_path: Path, log_path: Path) -> None:
        self.output_dir = output_dir
        self.events_path = events_path
        self.log_path = log_path


def _assets(model_dir: Path) -> dict[str, str]:
    root = model_dir / "assets"
    return {
        "skiing_video": str(root / "Skiing.mp4"),
        "system_ref_audio": str(root / "system_ref_audio.wav"),
        "duplex_omni_1": str(root / "omni_duplex1.mp4"),
        "duplex_omni_2": str(root / "omni_duplex2.mp4"),
    }


def build_manifest(model_dir: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_dir": str(model_dir),
        "parameters": dict(PARAMETERS),
        "assets": _assets(model_dir),
        "cases": [case_manifest(name, model_dir) for name in CASE_NAMES],
        "parity_claim": False,
    }


def case_manifest(name: str, model_dir: Path) -> CasePayload:
    assets = _assets(model_dir)
    cases: dict[str, CasePayload] = {
        "session_audio_text": {
            "name": "session_audio_text",
            "api_family": "session",
            "prompt": "Listen to the streamed skiing video audio and briefly transcribe or summarize what you hear.",
            "generate_audio": False,
            "omni": False,
            "asset_paths": [assets["skiing_video"]],
        },
        "session_audio_reply": {
            "name": "session_audio_reply",
            "api_family": "session",
            "prompt": "Listen to the streamed skiing video audio and answer aloud with a brief summary.",
            "generate_audio": True,
            "omni": False,
            "asset_paths": [assets["skiing_video"], assets["system_ref_audio"]],
        },
        "duplex_audio_text": {
            "name": "duplex_audio_text",
            "api_family": "duplex",
            "prompt": "Streaming audio conversation. Respond briefly in text when appropriate.",
            "generate_audio": False,
            "omni": False,
            "asset_paths": [assets["duplex_omni_1"]],
        },
        "duplex_audio_reply": {
            "name": "duplex_audio_reply",
            "api_family": "duplex",
            "prompt": "Streaming audio conversation. Respond briefly with speech when appropriate.",
            "generate_audio": True,
            "omni": False,
            "asset_paths": [assets["duplex_omni_1"], assets["system_ref_audio"]],
        },
        "duplex_omni_reply": {
            "name": "duplex_omni_reply",
            "api_family": "duplex",
            "prompt": (
                "Streaming audiovisual conversation. Use both the frames and audio and respond briefly with speech."
            ),
            "generate_audio": True,
            "omni": True,
            "asset_paths": [assets["duplex_omni_2"], assets["system_ref_audio"]],
        },
    }
    try:
        return cases[name]
    except KeyError as error:
        raise ValueError(f"unknown case: {name}") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real MiniCPM-o-4.5 Hugging Face streaming baseline")
    parser.add_argument("--case", required=True, choices=CASE_NAMES)
    parser.add_argument("--model-dir", type=Path, required=True, help="Hugging Face MiniCPM-o-4.5 model directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args(argv)


def _event(state: RunState, stage: str, **fields: object) -> None:
    payload = {"timestamp": utc_timestamp(), "stage": stage, **fields}
    with state.events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def _load_model(model_dir: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(model_dir),
        config=ensure_tts_sampling_config(config),
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    ).eval()
    patch_dynamic_cache_seen_tokens(DynamicCache)
    patch_dynamic_cache_legacy_methods(DynamicCache)
    patch_remote_cache_helpers(model)
    patch_empty_audio_cache(model)
    patch_audio_attention_return_compat(model)
    patch_tts_cache_position_compat(model)
    patch_speech_generation_capture(model)
    return model.to(device), tokenizer


def _session_run(model, tokenizer, case: CasePayload, chunks, max_new_tokens: int, state: RunState):
    session_id = f"hf-baseline-{case['name']}"
    ref_audio = None
    if case["generate_audio"]:
        model.init_tts(model_dir=str(Path(case["asset_paths"][1]).parent / "token2wav"))
        ref_audio, _ = librosa.load(case["asset_paths"][1], sr=SAMPLE_RATE, mono=True)
        model.init_token2wav_cache(ref_audio)
        _event(state, "init_token2wav_cache")
        model.reset_session(False)
        _event(state, "reset_session", reset_token2wav_cache=False)
    else:
        model.reset_session()
        _event(state, "reset_session", reset_token2wav_cache=True)
    for index, (audio, _) in enumerate(chunks):
        content = [audio]
        if index == 0:
            content.insert(0, case["prompt"])
        model.streaming_prefill(
            session_id,
            [{"role": "user", "content": content}],
            tokenizer=tokenizer,
            is_last_chunk=index == len(chunks) - 1,
            omni_mode=True,
            use_tts_template=case["generate_audio"],
        )
        _event(state, "streaming_prefill", chunk_index=index, final=index == len(chunks) - 1)
    outputs = list(
        model.streaming_generate(
            session_id,
            generate_audio=case["generate_audio"],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
            use_tts_template=case["generate_audio"],
        )
    )
    _event(state, "streaming_generate", output_count=len(outputs))
    if case["generate_audio"]:
        waves = [item[0].detach().cpu().numpy() for item in outputs if item[0] is not None]
        texts = [str(item[1]) for item in outputs if item[1]]
        return texts, list(getattr(model, "_streaming_generated_token_ids", [])), waves
    texts = [str(item[0]) for item in outputs]
    return texts, [], []


def _duplex_run(model, case: CasePayload, chunks, state: RunState):
    duplex = model.as_duplex(
        device=str(model.device),
        generate_audio=case["generate_audio"],
        chunk_ms=PARAMETERS["chunk_ms"],
        first_chunk_ms=PARAMETERS["first_chunk_ms"],
        cnn_redundancy_ms=PARAMETERS["cnn_redundancy_ms"],
        sample_rate=PARAMETERS["sample_rate"],
        temperature=0.0,
        top_k=1,
        top_p=1.0,
    )
    _event(state, "as_duplex")
    ref_path = case["asset_paths"][1] if case["generate_audio"] else None
    ref_audio = None if ref_path is None else librosa.load(ref_path, sr=SAMPLE_RATE, mono=True)[0]
    duplex.prepare(prefix_system_prompt=case["prompt"], ref_audio=ref_audio, prompt_wav_path=ref_path)
    _event(state, "prepare")
    texts: list[str] = []
    token_ids: list[int] = []
    waves: list[np.ndarray] = []
    for index, (audio, frames) in enumerate(chunks):
        prefill = duplex.streaming_prefill(audio_waveform=audio, frame_list=frames if case["omni"] else None)
        _event(state, "streaming_prefill", chunk_index=index, success=bool(prefill.get("success")))
        result = duplex.streaming_generate(decode_mode="greedy", top_k=1, top_p=1.0, temperature=0.0)
        _event(state, "streaming_generate", chunk_index=index, end_of_turn=bool(result["end_of_turn"]))
        texts.append(str(result["text"]))
        token_ids.extend(duplex.total_ids[len(token_ids) :])
        waveform = result.get("audio_waveform")
        if waveform is not None:
            waves.append(np.asarray(waveform, dtype=np.float32))
    return texts, token_ids, waves


def run_case(args: argparse.Namespace) -> dict[str, object]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    case = case_manifest(args.case, args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(args.output_dir, args.output_dir / "api_events.jsonl", args.output_dir / "runner.log")
    logging.basicConfig(filename=state.log_path, level=logging.INFO, force=True)
    write_json(args.output_dir / "manifest.json", build_manifest(args.model_dir))
    write_json(args.output_dir / "environment.json", environment_versions())
    for path in case["asset_paths"]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    chunks = video_chunks(Path(case["asset_paths"][0]), SAMPLE_RATE)
    _event(state, "media_loaded", chunk_count=len(chunks))
    model, tokenizer = _load_model(args.model_dir, args.device)
    _event(state, "model_loaded", device=args.device)
    if case["api_family"] == "session":
        texts, token_ids, waves = _session_run(model, tokenizer, case, chunks, args.max_new_tokens, state)
    else:
        texts, token_ids, waves = _duplex_run(model, case, chunks, state)
    text = "".join(texts)
    (args.output_dir / "text.txt").write_text(text + "\n", encoding="utf-8")
    write_json(args.output_dir / "tokens.json", {"token_ids": token_ids})
    if waves:
        write_wav(args.output_dir / "audio.wav", flatten_audio_chunks(waves), OUTPUT_AUDIO_RATE)
    result = {
        "status": "passed",
        "case": args.case,
        "api_family": case["api_family"],
        "text_path": str(args.output_dir / "text.txt"),
        "tokens_path": str(args.output_dir / "tokens.json"),
        "audio_path": str(args.output_dir / "audio.wav") if waves else None,
        "api_events_path": str(state.events_path),
        "log_path": str(state.log_path),
        "parity_claim": False,
    }
    write_json(args.output_dir / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_paths = tuple(args.output_dir / name for name in ("result.json", "api_events.jsonl", "runner.log"))
    stage = "startup"
    last_successful = "argument_parse"
    try:
        stage = "case_execution"
        result = run_case(args)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as error:
        packet = make_failure_packet(
            case_name=args.case,
            command=(sys.executable, *sys.argv),
            asset_paths=case_manifest(args.case, args.model_dir)["asset_paths"],
            api_stage=stage,
            error=error,
            last_successful_stage=last_successful,
            output_paths=output_paths,
            environment=environment_versions(),
            timestamp=utc_timestamp(),
        )
        write_failure_packet(args.output_dir, packet)
        logging.exception("HF streaming baseline failed")
        print(json.dumps(packet, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_manifest", "case_manifest", "make_failure_packet", "write_failure_packet"]
