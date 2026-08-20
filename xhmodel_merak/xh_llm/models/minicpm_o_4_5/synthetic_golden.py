from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from xhquant.api import HMONNXGoldenInference
from xhquant.core import CacheTensor

from .streaming_fixtures import CERTIFIED_STREAMING_CASES, streaming_case_manifest, validate_streaming_graph_coverage


def dump_synthetic_golden(work_dir: Path, meta: Mapping[str, Any], device: str) -> dict[str, Any]:
    validate_streaming_graph_coverage(meta)
    result: dict[str, Any] = {"mode": "synthetic", "components": {}}
    for name, component in meta["components"].items():
        graph_groups = ("graphs", "projection_graphs")
        for graph_group in graph_groups:
            for role, relative_path in (component.get(graph_group) or {}).items():
                graph_path = work_dir / relative_path
                golden_dir = graph_path.parent / "golden" / graph_path.stem
                if golden_dir.exists():
                    shutil.rmtree(golden_dir)
                golden_dir.mkdir(parents=True, exist_ok=True)
                inputs = synthetic_golden_inputs(work_dir, name, role, component)
                session = HMONNXGoldenInference(str(graph_path))
                session.to(device)
                session.save_golden = True
                session.golden_dir = str(golden_dir)
                session.step = 0
                session(*(value.to(device) for value in inputs))
                result["components"][f"{name}_{role}"] = {
                    "graph": str(graph_path.relative_to(work_dir)),
                    "golden_dir": str(golden_dir.relative_to(work_dir)),
                }
                del session
                torch.cuda.empty_cache()
    result["streaming_manifest"] = streaming_case_manifest(meta, CERTIFIED_STREAMING_CASES)
    return result


def synthetic_golden_inputs(
    work_dir: Path,
    name: str,
    role: str,
    component: Mapping[str, Any],
) -> tuple[torch.Tensor, ...]:
    if name == "vision":
        return (
            torch.zeros((1, 3, 14, 22400), dtype=torch.float16),
            torch.arange(1600, dtype=torch.int32).unsqueeze(0),
            torch.zeros((1, 1, 1600, 1600), dtype=torch.float16),
            torch.tensor([[40, 40]], dtype=torch.int32),
        )
    if name == "audio":
        if role in {"stream_prefill", "stream_decode", "session_prefill", "session_decode"}:
            contract = component["graph_contracts"][role]
            cache_shape = tuple(int(value) for value in component["kv_cache_shape"])
            frame_count = int(component[f"{role}_frames"])
            values: list[torch.Tensor] = []
            for input_name in contract["input_names"]:
                if input_name == "input_features":
                    values.append(torch.zeros((1, 80, frame_count), dtype=torch.float16))
                elif input_name in {"valid_mel_length", "past_seq_length", "current_input_length"}:
                    values.append(torch.zeros((1,), dtype=torch.int32))
                elif input_name == "attention_mask":
                    prefix_overlap = int(contract.get("prefix_overlap", 0))
                    suffix_overlap = int(contract.get("suffix_overlap", 0))
                    query_capacity = (
                        (int(component[f"{role}_frames"]) + 1) // 2
                        - (prefix_overlap + 1) // 2
                        - (suffix_overlap + 1) // 2
                    )
                    values.append(
                        torch.zeros(
                            (1, 1, query_capacity, int(component["cache_capacity"])),
                            dtype=torch.float16,
                        )
                    )
                elif input_name.startswith("past_"):
                    values.append(CacheTensor(torch.zeros(cache_shape, dtype=torch.float16)))
                else:
                    raise RuntimeError(f"Unsupported streaming Audio input: {input_name}")
            return tuple(values)
        return torch.zeros((4, 80, 3000), dtype=torch.float16), torch.zeros((4, 1, 1500, 1500), dtype=torch.float16)
    if name == "llm":
        length = int(component["prefill_input_sequence_length"]) if role == "prefill" else 1
        past = 0 if role == "prefill" else int(component["prefill_input_sequence_length"])
        hidden_size = int(component["kv_cache_shape"][-1]) * 32
        values: list[torch.Tensor] = [
            torch.zeros((1, length, hidden_size), dtype=torch.float16),
            torch.tensor([past], dtype=torch.int32),
            torch.tensor([length], dtype=torch.int32),
        ]
        cache_shape = tuple(int(value) for value in component["kv_cache_shape"])
        cache_count = 2 * int(component["num_hidden_layers"])
        values += [CacheTensor(torch.zeros(cache_shape, dtype=torch.float16)) for _ in range(cache_count)]
        return tuple(values)
    if name == "tts":
        if role in {"projector_semantic", "head_code"}:
            sequence_capacity = int(
                component.get("projection_seq_capacity", component["prefill_input_sequence_length"])
            )
            hidden_size = 4096 if role == "projector_semantic" else 768
            return (torch.zeros((1, sequence_capacity, hidden_size), dtype=torch.float16),)
        length = int(component["prefill_input_sequence_length"]) if role == "prefill" else 1
        past = 0 if role == "prefill" else int(component["prefill_input_sequence_length"])
        values = [
            torch.zeros((1, length, 768), dtype=torch.float16),
            torch.tensor([past], dtype=torch.int32),
            torch.tensor([length], dtype=torch.int32),
        ]
        cache_shape = tuple(int(value) for value in component["kv_cache_shape"])
        cache_count = 2 * int(component["num_hidden_layers"])
        values += [CacheTensor(torch.zeros(cache_shape, dtype=torch.float16)) for _ in range(cache_count)]
        values.append(torch.zeros((1, 1, length, int(component["kv_cache_shape"][-2])), dtype=torch.float16))
        return tuple(values)
    if name == "token2wav_flow_frontend":
        if role in {"stream_flow_frontend", "stream_flow_frontend_final"}:
            contract = component["stream_contract"]
            token_capacity = int(contract["chunk_token_capacity"])
            frontend_shapes = {
                "conformer_cnn_cache": tuple(
                    int(value) for value in contract["base_cache_shapes"]["conformer_cnn_cache"]
                ),
                "conformer_att_cache": tuple(
                    int(value) for value in contract["frontend_cache_shapes"]["conformer_att_cache"]
                ),
            }
            values: list[torch.Tensor] = []
            for input_name in contract["frontend_input_names"]:
                if input_name == "tokens":
                    values.append(torch.zeros((1, token_capacity), dtype=torch.int32))
                elif input_name == "token_valid_length":
                    values.append(torch.tensor([token_capacity], dtype=torch.int32))
                elif input_name == "embedding":
                    values.append(torch.zeros((1, 192), dtype=torch.float16))
                elif input_name.endswith("_valid_length"):
                    values.append(torch.zeros((1,), dtype=torch.int32))
                elif input_name.startswith("past_"):
                    cache_name = input_name.removeprefix("past_")
                    values.append(CacheTensor(torch.zeros(frontend_shapes[cache_name], dtype=torch.float16)))
                else:
                    raise RuntimeError(f"Unsupported streaming Flow frontend input: {input_name}")
            return tuple(values)
        token_capacity = int(component["token_capacity"])
        mel_capacity = int(component["mel_capacity"])
        return (
            torch.zeros((1, token_capacity), dtype=torch.int32),
            torch.tensor([token_capacity], dtype=torch.int32),
            torch.zeros((1, mel_capacity, 80), dtype=torch.float16),
            torch.tensor([mel_capacity // 2], dtype=torch.int32),
            torch.zeros((1, 192), dtype=torch.float16),
        )
    if name == "token2wav_flow_decoder":
        if role == "stream_flow_estimator_step":
            contract = component["stream_contract"]
            shapes = contract["estimator_step_cache_shapes"]
            frame_capacity = int(shapes["frame_capacity"])
            return (
                torch.zeros((2, 80, frame_capacity), dtype=torch.float16),
                torch.zeros((2, 80, frame_capacity), dtype=torch.float16),
                torch.zeros((2,), dtype=torch.float16),
                torch.zeros((2, 80), dtype=torch.float16),
                torch.zeros((2, 80, frame_capacity), dtype=torch.float16),
                CacheTensor(torch.zeros(tuple(int(value) for value in shapes["input_cnn"]), dtype=torch.float16)),
                CacheTensor(torch.zeros(tuple(int(value) for value in shapes["input_att"]), dtype=torch.float16)),
                torch.tensor([int(contract["base_cache_valid_length"])], dtype=torch.int32),
                torch.tensor([frame_capacity], dtype=torch.int32),
            )
        capacity = int(component["mel_capacity"])
        return (
            torch.zeros((2, 80, capacity), dtype=torch.float16),
            torch.ones((2, 1, capacity), dtype=torch.float16),
            torch.zeros((2, 80, capacity), dtype=torch.float16),
            torch.zeros((2,), dtype=torch.float16),
            torch.zeros((2, 80), dtype=torch.float16),
            torch.zeros((2, 80, capacity), dtype=torch.float16),
        )
    if name == "token2wav_hift":
        if role in {"stream_hift", "stream_hift_final"}:
            contract = component["stream_contract"]
            return (
                torch.zeros((1, 80, int(contract["frame_capacity"])), dtype=torch.float16),
                torch.tensor([int(contract["frame_capacity"])], dtype=torch.int32),
                torch.zeros((1, 80, int(contract["mel_cache_length"])), dtype=torch.float16),
                torch.tensor([0], dtype=torch.int32),
                torch.zeros((1, 1, int(contract["source_cache_length"])), dtype=torch.float16),
                torch.tensor([0], dtype=torch.int32),
                torch.load(work_dir / str(contract["phase_noise_file"]), map_location="cpu", weights_only=True).to(
                    torch.float16
                ),
                torch.load(work_dir / str(contract["source_noise_file"]), map_location="cpu", weights_only=True).to(
                    torch.float16
                ),
            )
        capacity = int(component["frame_capacity"])
        return (
            torch.zeros((1, 80, capacity), dtype=torch.float16),
            torch.load(work_dir / str(component["phase_noise_file"]), map_location="cpu", weights_only=True).to(
                torch.float16
            ),
            torch.load(work_dir / str(component["source_noise_file"]), map_location="cpu", weights_only=True).to(
                torch.float16
            ),
        )
    if name == "speaker":
        # Runtime feeds these graphs float16 (runtime_token2wav calls
        # feat.to(torch.float16) / mel.to(torch.float16)); match qwen3_tts
        # speaker_encoder golden dtype and the float16 convention used by
        # every other minicpm golden component.
        if role == "campplus":
            seq_capacity = int(component.get("sequence_length", 1000))
            return (torch.zeros((1, seq_capacity, 80), dtype=torch.float16),)
        if role == "speech_tokenizer":
            n_mels = int(component.get("n_mels", 128))
            feats_length = int(component.get("feats_length", 3000))
            return (
                torch.zeros((1, n_mels, feats_length), dtype=torch.float16),
                torch.tensor([feats_length], dtype=torch.int32),
            )
    raise RuntimeError(f"Unsupported MiniCPM-o-4.5 Golden component: {name}_{role}")


__all__ = ["dump_synthetic_golden", "synthetic_golden_inputs"]
