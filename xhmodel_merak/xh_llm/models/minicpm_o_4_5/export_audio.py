from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .export_common import build_component_model, quantize_and_export


class StreamingAudioConfigError(ValueError):
    pass


def _streaming_io_names(num_hidden_layers: int) -> tuple[list[str], list[str]]:
    input_names = [
        "input_features",
        "valid_mel_length",
        "past_seq_length",
        "current_input_length",
        "attention_mask",
    ]
    input_names.extend(f"past_k_cache_{index}" for index in range(num_hidden_layers))
    input_names.extend(f"past_v_cache_{index}" for index in range(num_hidden_layers))
    output_names = ["audio_hidden_states"]
    output_names.extend(f"present_k_cache_{index}" for index in range(num_hidden_layers))
    output_names.extend(f"present_v_cache_{index}" for index in range(num_hidden_layers))
    return input_names, output_names


def capture_audio_inputs(component_cfg: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    batch = int(component_cfg["static_batch_size"])
    frames = int(component_cfg["max_audio_frames"])
    features = torch.zeros((batch, 80, frames), dtype=torch.float16)
    mask = torch.zeros((batch, 1, frames // 2, frames // 2), dtype=torch.float16)
    return features, mask


def capture_streaming_audio_inputs(
    processor: Any,
    streaming_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_rate = int(streaming_cfg["sample_rate"])
    if sample_rate != 16000:
        raise StreamingAudioConfigError(f"Streaming Audio requires sample_rate=16000, got sample_rate={sample_rate}")
    processor.set_streaming_mode(
        mode="exact",
        chunk_ms=int(streaming_cfg["chunk_ms"]),
        first_chunk_ms=int(streaming_cfg["first_chunk_ms"]),
        cnn_redundancy_ms=int(streaming_cfg["cnn_redundancy_ms"]),
        enable_sliding_window=False,
    )
    processor.reset_streaming()
    chunks = []
    overlaps = (
        int(streaming_cfg["prefix_overlap_first"]),
        int(streaming_cfg["prefix_overlap_later"]),
    )
    suffix_overlap = int(streaming_cfg["suffix_overlap"])
    for past_seq_length, prefix_overlap in zip((0, 1), overlaps, strict=True):
        sample_count = int(processor.get_streaming_chunk_size())
        batch = processor.process_audio_streaming(
            np.zeros(sample_count, dtype=np.float32),
            return_batch_feature=True,
        )
        frames = int(batch.audio_feature_lens[0].reshape(-1)[0].item())
        current_input_length = (frames + 1) // 2 - (prefix_overlap + 1) // 2 - (suffix_overlap + 1) // 2
        valid_cache_length = past_seq_length + current_input_length
        attention_mask = torch.full(
            (1, 1, current_input_length, int(streaming_cfg["cache_capacity"])),
            float("-inf"),
            dtype=torch.float16,
        )
        attention_mask[..., :valid_cache_length] = 0
        chunks.append(
            {
                "input_features": batch.audio_features.to(torch.float16),
                "valid_mel_length": frames,
                "past_seq_length": past_seq_length,
                "current_input_length": current_input_length,
                "attention_mask": attention_mask,
            }
        )
    return chunks[0], chunks[1]


def capture_session_audio_inputs(
    processor: Any,
    streaming_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    del processor
    frames = int(streaming_cfg["session_frames"])
    chunks = []
    for past_seq_length in (0, (frames + 1) // 2):
        current_input_length = (frames + 1) // 2
        attention_mask = torch.full(
            (1, 1, current_input_length, int(streaming_cfg["cache_capacity"])),
            float("-inf"),
            dtype=torch.float16,
        )
        attention_mask[..., : past_seq_length + current_input_length] = 0
        chunks.append(
            {
                "input_features": torch.zeros((1, 80, frames), dtype=torch.float16),
                "valid_mel_length": frames,
                "past_seq_length": past_seq_length,
                "current_input_length": current_input_length,
                "attention_mask": attention_mask,
            }
        )
    return chunks[0], chunks[1]


class _AudioFullEncoder(torch.nn.Module):
    """Official audio encoder chain: apm -> projection -> avg pooling.

    Used for the ONNX direct-export of the non-streaming main graph. The
    torch-fx export chain applies W8 weight quantization that diverges
    numerically for this encoder path (embeddings cosine ~0.2 vs 1.0), while
    the ONNX direct route (torch.onnx.export + convert_onnx_to_hmonnx) is
    accurate (cosine ~0.9998, verified). This mirrors the standalone whisper
    model export path.
    """

    def __init__(self, apm: torch.nn.Module, proj: torch.nn.Module, pooler: torch.nn.Module) -> None:
        super().__init__()
        self.apm = apm
        self.proj = proj
        self.pooler = pooler

    def forward(self, input_features: torch.Tensor, audio_attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.apm(input_features, attention_mask=audio_attention_mask)
        hidden = out.last_hidden_state if out.last_hidden_state is not None else out.hidden_states[-1]
        embeds = self.proj(hidden)
        embeds = embeds.transpose(1, 2)
        embeds = self.pooler(embeds)
        embeds = embeds.transpose(1, 2)
        return embeds


def _chunked_attention_mask(host: Any, num_frames: int, chunk_length: float = 1.0) -> torch.Tensor:
    """Build the official chunked attention mask for `num_frames` conv frames."""
    max_seq = int((num_frames - 1) // 2 + 1)
    chunk_size = int(chunk_length * 50)
    chunk_mask = host.subsequent_chunk_mask(size=max_seq, chunk_size=chunk_size, num_left_chunks=-1, device="cpu")
    seq_range = torch.arange(0, max_seq, device="cpu").unsqueeze(0).expand(1, max_seq)
    lens = torch.tensor([max_seq], device="cpu").unsqueeze(1).expand(1, max_seq)
    padded = (seq_range >= lens).view(1, 1, 1, max_seq).expand(1, 1, max_seq, max_seq).clone()
    padded = torch.logical_or(padded, torch.logical_not(chunk_mask))
    mask = torch.zeros(1, 1, max_seq, max_seq, dtype=torch.float16)
    mask[padded] = float("-inf")
    return mask


def _export_audio_main_onnx_direct(
    *,
    host: Any,
    component_cfg: Mapping[str, Any],
    target_device: str,
    output_dir: Path,
    device: str,
    model_name: str = "minicpm_o_4_5",
) -> str:
    """Export the non-streaming audio main graph via the ONNX direct route.

    Mirrors the standalone whisper model export: torch.onnx.export the official
    encoder chain, simplify, then convert_onnx_to_hmonnx with the requested
    quant type. The graph uses a fixed frame capacity (max_audio_frames) with
    the official chunked attention mask, so the runtime pads to the same
    capacity and masks the tail.
    """
    import onnx
    import onnxsim

    from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config
    from xhquant.patch.core import RewriterContext

    from .hf_compatible import patch_audio_attention_return_compat

    patch_audio_attention_return_compat(host)

    onnx_dir = output_dir / "onnx"
    hmonnx_dir = output_dir / "hmonnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_dir.mkdir(parents=True, exist_ok=True)
    capacity = int(component_cfg["max_audio_frames"])
    quant_type = str(component_cfg["quant_type"])
    prefix = f"{model_name}_audio_offline_{target_device}_{quant_type}"

    encoder = _AudioFullEncoder(host.apm, host.audio_projection_layer, host.audio_avg_pooler).to(device)

    # Fixed-capacity dummy inputs: zero mel frames padded to capacity + official
    # chunked attention mask (mirrors the runtime contract). The graph is
    # exported with the configured static batch so the runtime can pad
    # multi-segment audio (e.g. Daily-Omni video segments) to the same batch.
    batch = int(component_cfg.get("static_batch_size", 1))
    features = torch.zeros((batch, 80, capacity), dtype=torch.float16, device=device)
    mask = _chunked_attention_mask(host, capacity).expand(batch, -1, -1, -1).contiguous().to(device)

    raw_onnx = onnx_dir / f"{prefix}_raw.onnx"
    with RewriterContext(None, backend="onnxruntime"):
        with torch.no_grad():
            torch.onnx.export(
                encoder,
                (features, mask),
                str(raw_onnx),
                input_names=["input_features", "audio_attention_mask"],
                output_names=["audio_embeddings"],
                opset_version=17,
            )
    onnx_model = onnx.load(str(raw_onnx))
    skipped_optimizers = [
        "fuse_pad_into_conv",
        "fuse_consecutive_slices",
        "eliminate_common_subexpression",
        "fuse_qkv",
    ]
    onnx_model_sim, checked = onnxsim.simplify(onnx_model, skipped_optimizers=skipped_optimizers)
    if checked:
        onnx_model = onnx_model_sim
    onnx_file = onnx_dir / f"{prefix}.onnx"
    onnx.save(
        onnx_model,
        str(onnx_file),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{onnx_file.stem}_external_data",
    )
    raw_onnx.unlink(missing_ok=True)
    onnx_path = hmonnx_dir / f"{prefix}.onnx"
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    convert_onnx_to_hmonnx(
        str(onnx_file),
        [features.cpu(), mask.cpu()],
        DeviceType.XH2a,
        str(onnx_path),
        quant_config=create_quant_config(quant_scheme),
        input_names=["input_features", "audio_attention_mask"],
        output_names=["audio_embeddings"],
    )
    return str(onnx_path)


def export_minicpm_o_4_5_audio(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, Any],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
) -> dict[str, Any]:
    model_name = model_name or "minicpm_o_4_5"
    model = build_component_model(model_dir, component_cfg)
    host = model.get_hf_model(device_map="cpu")
    graph = Path(
        _export_audio_main_onnx_direct(
            host=host,
            component_cfg=component_cfg,
            target_device=target_device,
            output_dir=work_dir / "Audio" / "Offline",
            device=device,
            model_name=model_name,
        )
    )
    result = {"quant_type": component_cfg["quant_type"], "graphs": {"main": str(graph.relative_to(work_dir))}}
    streaming_cfg = component_cfg.get("streaming") or {}
    if not streaming_cfg.get("enabled", False):
        return result

    probe_model = build_component_model(model_dir, component_cfg)
    probe_host = probe_model.get_hf_model(device_map="cpu")
    prefill_data, decode_data = capture_streaming_audio_inputs(probe_host.processor, streaming_cfg)
    num_hidden_layers = int(probe_host.apm.config.encoder_layers)
    audio_encoder_layer = int(probe_host.audio_encoder_layer)
    if audio_encoder_layer != -1:
        raise StreamingAudioConfigError(
            f"Streaming Audio selected-state graph requires audio_encoder_layer=-1, got {audio_encoder_layer}"
        )
    input_names, output_names = _streaming_io_names(num_hidden_layers)
    streaming_component_cfg = dict(component_cfg)
    streaming_component_cfg["input_names"] = input_names
    streaming_component_cfg["output_names"] = output_names

    prefill_model = build_component_model(model_dir, streaming_component_cfg)
    prefill_host = prefill_model.get_hf_model(device_map="cpu")
    prefill_model.init_wrap_model(
        prefill_host,
        streaming=True,
        prefix_extra_frames=int(streaming_cfg["prefix_overlap_first"]),
        suffix_extra_frames=int(streaming_cfg["suffix_overlap"]),
        input_frame_capacity=int(prefill_data["input_features"].shape[-1]),
    )
    prefill_graph = Path(
        quantize_and_export(
            prefill_model,
            prefill_data,
            work_dir / "Audio" / "Stream",
            f"{model_name}_audio_stream_prefill_{target_device}_{component_cfg['quant_type']}",
            device,
        )
    )
    decode_data["past_seq_length"] = int(prefill_data["current_input_length"])
    decode_model = build_component_model(model_dir, streaming_component_cfg)
    decode_host = decode_model.get_hf_model(device_map="cpu")
    decode_model.init_wrap_model(
        decode_host,
        streaming=True,
        prefix_extra_frames=int(streaming_cfg["prefix_overlap_later"]),
        suffix_extra_frames=int(streaming_cfg["suffix_overlap"]),
        input_frame_capacity=int(decode_data["input_features"].shape[-1]),
    )
    decode_graph = Path(
        quantize_and_export(
            decode_model,
            decode_data,
            work_dir / "Audio" / "Stream",
            f"{model_name}_audio_stream_decode_{target_device}_{component_cfg['quant_type']}",
            device,
        )
    )
    session_prefill_data, session_decode_data = capture_session_audio_inputs(probe_host.processor, streaming_cfg)
    session_prefill_model = build_component_model(model_dir, streaming_component_cfg)
    session_prefill_host = session_prefill_model.get_hf_model(device_map="cpu")
    session_prefill_model.init_wrap_model(
        session_prefill_host,
        streaming=True,
        prefix_extra_frames=0,
        suffix_extra_frames=0,
        input_frame_capacity=int(session_prefill_data["input_features"].shape[-1]),
    )
    session_prefill_graph = Path(
        quantize_and_export(
            session_prefill_model,
            session_prefill_data,
            work_dir / "Audio" / "Session",
            f"{model_name}_audio_session_prefill_{target_device}_{component_cfg['quant_type']}",
            device,
        )
    )
    session_decode_data["past_seq_length"] = int(session_prefill_data["current_input_length"])
    session_decode_model = build_component_model(model_dir, streaming_component_cfg)
    session_decode_host = session_decode_model.get_hf_model(device_map="cpu")
    session_decode_model.init_wrap_model(
        session_decode_host,
        streaming=True,
        prefix_extra_frames=0,
        suffix_extra_frames=0,
        input_frame_capacity=int(session_decode_data["input_features"].shape[-1]),
    )
    session_decode_graph = Path(
        quantize_and_export(
            session_decode_model,
            session_decode_data,
            work_dir / "Audio" / "Session",
            f"{model_name}_audio_session_decode_{target_device}_{component_cfg['quant_type']}",
            device,
        )
    )
    cache_shape = list(prefill_model.past_key_caches[0].shape)
    current_length_rule = "ceil(valid_mel_length / 2) - ceil(prefix_overlap / 2) - ceil(suffix_overlap / 2)"

    def graph_contract(prefix_overlap: int, suffix_overlap: int) -> dict[str, Any]:
        contract = {
            "input_names": input_names,
            "output_names": output_names,
            "prefix_overlap": prefix_overlap,
            "suffix_overlap": suffix_overlap,
            "current_length_rule": (
                "ceil(valid_mel_length / 2)" if prefix_overlap == 0 and suffix_overlap == 0 else current_length_rule
            ),
            "cache_capacity": int(streaming_cfg["cache_capacity"]),
            "kv_cache_shape": cache_shape,
            "selected_hidden_state_index": audio_encoder_layer,
            # The streaming graphs export audio_projection_layer in-graph (1024 ->
            # 4096), mirroring the non-streaming main graph; the Host still applies
            # AvgPool1d frame pooling and the runtime owns the pooled-length slicing.
            "projection_in_graph": True,
            # The streaming graphs export projection AND AvgPool1d(pool_step=5)
            # pooling in-graph, so the valid output region is the pooled frame
            # count ((current_input_length - pool_step) // pool_step + 1).
            "pooling_in_graph": True,
        }
        if prefix_overlap == 0 and suffix_overlap == 0:
            contract["output_valid_region"] = (
                "audio_hidden_states[:, :(current_input_length - pool_step) // pool_step + 1, :]"
            )
        return contract

    result.update(
        {
            "graphs": {
                "main": str(graph.relative_to(work_dir)),
                "stream_prefill": str(prefill_graph.relative_to(work_dir)),
                "stream_decode": str(decode_graph.relative_to(work_dir)),
                "session_prefill": str(session_prefill_graph.relative_to(work_dir)),
                "session_decode": str(session_decode_graph.relative_to(work_dir)),
            },
            "stream_prefill_frames": int(prefill_data["valid_mel_length"]),
            "stream_decode_frames": int(decode_data["valid_mel_length"]),
            "session_prefill_frames": int(session_prefill_data["valid_mel_length"]),
            "session_decode_frames": int(session_decode_data["valid_mel_length"]),
            "num_hidden_layers": len(prefill_model.past_key_caches),
            "kv_cache_shape": cache_shape,
            "cache_capacity": int(streaming_cfg["cache_capacity"]),
            "audio_encoder_layer": audio_encoder_layer,
            "selected_hidden_state_index": audio_encoder_layer,
            "prefix_overlap_first": int(streaming_cfg["prefix_overlap_first"]),
            "prefix_overlap_later": int(streaming_cfg["prefix_overlap_later"]),
            "suffix_overlap": int(streaming_cfg["suffix_overlap"]),
            # Older official remote-code revisions do not expose this field;
            # the streaming wrapper and graph contract use the same default.
            "pool_step": int(getattr(getattr(probe_host, "config", None), "audio_pool_step", 5)),
            "projection_in_graph": True,
            "graph_contracts": {
                "stream_prefill": graph_contract(
                    int(streaming_cfg["prefix_overlap_first"]),
                    int(streaming_cfg["suffix_overlap"]),
                ),
                "stream_decode": graph_contract(
                    int(streaming_cfg["prefix_overlap_later"]),
                    int(streaming_cfg["suffix_overlap"]),
                ),
                "session_prefill": graph_contract(0, 0),
                "session_decode": graph_contract(0, 0),
            },
        }
    )
    return result


__all__ = [
    "StreamingAudioConfigError",
    "capture_audio_inputs",
    "capture_streaming_audio_inputs",
    "export_minicpm_o_4_5_audio",
]
