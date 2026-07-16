from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from xhmodel_merak.xh_llm.kv_cache_mixin import KVCacheMixin
from xhmodel_merak.xh_llm.utils import unfold_args
from xhquant.core import HybridCacheTensor
from xhquant.xhonnxruntime.parsers import PageAttentionContext

from ...hmonnx.hmonnx_model import HMONNXModel
from ...hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
from ...types import KVCacheConfig, VLLMModelMeta
from .data_preprocess import Gemma4DataPreprocess, Gemma4MoeDataPreprocess, Gemma4PerLayerInputEmbedding
from .gemma4_series_processor import XHGemma4Processor


def _as_positive_int(value, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _gemma4_prefill_graph_length_from_meta(meta_info) -> int:
    """Return the single Gemma4 prefill graph width from runtime metadata."""

    model_config = getattr(meta_info, "model_config", None)
    prefill_len = _as_positive_int(getattr(model_config, "prefill_chunk_length", None))

    raw_graphs = getattr(meta_info, "prefill_graphs", None)
    graph_len = None
    if isinstance(raw_graphs, dict) and raw_graphs:
        value = raw_graphs.get("prefill")
        if isinstance(value, dict):
            graph_len = _as_positive_int(value.get("input_sequence_length", value.get("prefill_chunk_length")))
        elif value is not None:
            length = getattr(value, "input_sequence_length", None)
            if length is None:
                length = getattr(value, "prefill_chunk_length", None)
            graph_len = _as_positive_int(length)
        if any(str(name) != "prefill" for name in raw_graphs):
            raise ValueError(
                "Gemma4 Series metadata must declare only the single 'prefill' graph; "
                f"got prefill_graphs={list(raw_graphs)}."
            )

    resolved = graph_len or prefill_len
    if resolved is None:
        raise ValueError(
            "Gemma4 Series metadata must declare model_config.prefill_chunk_length "
            "or prefill_graphs['prefill'].input_sequence_length."
        )
    if prefill_len is not None and graph_len is not None and graph_len != prefill_len:
        raise ValueError(
            "Gemma4 prefill graph length does not match model_config.prefill_chunk_length: "
            f"prefill_graphs.prefill={graph_len}, prefill_chunk_length={prefill_len}."
        )
    return int(resolved)


def _gemma4_prefill_graph_lengths_from_meta(meta_info) -> dict[str, int]:
    """Compatibility wrapper returning the single declared prefill graph."""

    return {"prefill": _gemma4_prefill_graph_length_from_meta(meta_info)}


def _select_gemma4_prefill_graph_name(requested_input_sequence_length: int, graph_lengths: dict[str, int]) -> str:
    requested = int(requested_input_sequence_length)
    prefill_len = int(graph_lengths.get("prefill", -1))
    if requested == prefill_len:
        return "prefill"
    raise ValueError(
        "No Gemma4 prefill graph matches requested input_sequence_length "
        f"{requested}; available={{'prefill': {prefill_len}}}."
    )


def _gemma4_meta_base_dir(meta_info) -> Path | None:
    meta_path = getattr(meta_info, "_meta_path_", None)
    if meta_path:
        return Path(meta_path).parent
    for attr_name in ("prefill_hmonnx", "decode_hmonnx"):
        path_value = getattr(meta_info, attr_name, None)
        if not path_value:
            continue
        path = Path(str(path_value))
        if path.is_absolute():
            # Runtime meta stores graph paths as "<graph-dir>/<file.onnx>".
            # The export root is therefore the parent of that graph directory.
            return path.parent.parent
    return None


def _gemma4_resolve_meta_path(meta_info, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    base_dir = _gemma4_meta_base_dir(meta_info)
    return str(base_dir / path) if base_dir is not None else str(path)


def _gemma4_prefill_graph_hmonnx_from_meta(meta_info, graph_name: str) -> str | None:
    if graph_name != "prefill":
        raise ValueError(f"Gemma4 Series runtime only supports the single 'prefill' graph; got {graph_name!r}.")
    raw_graphs = getattr(meta_info, "prefill_graphs", None) or {}
    value = raw_graphs.get("prefill") if isinstance(raw_graphs, dict) else None
    if isinstance(value, dict) and (value.get("hmonnx") or value.get("path")):
        return _gemma4_resolve_meta_path(meta_info, value.get("hmonnx") or value.get("path"))
    hmonnx = getattr(value, "hmonnx", None) if value is not None else None
    if hmonnx:
        return _gemma4_resolve_meta_path(meta_info, hmonnx)
    return _gemma4_resolve_meta_path(meta_info, getattr(meta_info, "prefill_hmonnx", None))


def _is_mtp_mode(value) -> bool:
    return str(value).lower() == "mtp"


class Gemma4VisualHMONNXModel(HMONNXModel):
    def forward(self, *args):
        return super().forward(*args)


class Gemma4AudioHMONNXModel(HMONNXModel):
    def __init__(
        self,
        hmonnx: str,
        input_feature_length: int = 0,
        *,
        attention_chunk_size: int = 12,
        attention_context_left: int = 13,
        attention_context_right: int = 0,
    ):
        super().__init__(hmonnx)
        self.input_feature_length = int(input_feature_length or 0)
        self.attention_chunk_size = int(attention_chunk_size)
        self.attention_context_left = int(attention_context_left)
        self.attention_context_right = int(attention_context_right)

    @staticmethod
    def normalize_audio_inputs(
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        target_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_len = int(target_len or 0)
        if target_len <= 0:
            return input_features, input_features_mask
        seq_len = input_features.shape[1]
        if seq_len > target_len:
            return input_features[:, :target_len, :], input_features_mask[:, :target_len]
        if seq_len < target_len:
            pad_len = target_len - seq_len
            input_features = torch.nn.functional.pad(input_features, (0, 0, 0, pad_len))
            input_features_mask = torch.nn.functional.pad(input_features_mask, (0, pad_len), value=0)
        return input_features, input_features_mask

    def forward(self, *args):
        args = list(args)
        if self.input_feature_length > 0 and len(args) >= 2:
            input_features, input_features_mask = self.normalize_audio_inputs(
                args[0],
                args[1],
                self.input_feature_length,
            )
            args[0], args[1] = input_features, input_features_mask.to(torch.float16)
        if len(args) == 2:
            args.append(
                XHGemma4Processor.build_audio_attention_mask(
                    args[1],
                    chunk_size=self.attention_chunk_size,
                    context_left=self.attention_context_left,
                    context_right=self.attention_context_right,
                    dtype=torch.float16,
                )
            )
        out = super().forward(*args)
        if isinstance(out, (tuple, list)) and len(out) == 1:
            return out[0]
        return out


class Gemma4KVCacheMixinHMONNX(KVCacheMixin):
    """KV-cache mixin that supports heterogeneous layer shapes (Gemma4 has
    sliding_attention + full_attention with different num_kv_heads / head_dim)."""

    def __init__(self, kv_cache_config: KVCacheConfig, layer_kv_shapes: list[list[int]]):
        super().__init__(kv_cache_config)
        self.layer_kv_shapes = layer_kv_shapes

    def prepare_kv_cache(self, dtype=torch.float16):
        if not self.use_cache:
            return
        self.past_key_caches.clear()
        self.past_value_caches.clear()
        full_cache_len = int(getattr(self.kvcache_config, "max_sequence_length", 0) or 0)
        if full_cache_len <= 0 and self.layer_kv_shapes:
            full_cache_len = max(int(shape[2]) for shape in self.layer_kv_shapes if len(shape) > 2)
        for shape in self.layer_kv_shapes:
            cache_type = (
                HybridCacheTensor if len(shape) > 2 and int(shape[2]) < full_cache_len else self.CACHCE_TENSOR_TYPE
            )
            self.past_key_caches.append(cache_type(torch.zeros(shape, dtype=dtype)))
            self.past_value_caches.append(cache_type(torch.zeros(shape, dtype=dtype)))


class XHGemma4SeriesHMONNXModel(VisonLLMHMONNXModel):
    _PAGE_ATTENTION_EAGER_REASON = "Gemma4 PageAttention eager execution explicitly requested"

    def __init__(self, meta_info: VLLMModelMeta, **kwargs):
        enable_cuda_graph = bool(kwargs.get("enable_cuda_graph", False))
        super().__init__(meta_info, **kwargs)
        self.visual_meta = meta_info.visual_config
        self.visual = (
            Gemma4VisualHMONNXModel(
                self.visual_meta.hmonnx,
                enable_cuda_graph=enable_cuda_graph,
            )
            if self.visual_meta is not None
            else None
        )
        self.video_visual_meta = getattr(meta_info, "video_visual_config", None)
        self.video_visual = (
            Gemma4VisualHMONNXModel(
                self.video_visual_meta.hmonnx,
                enable_cuda_graph=enable_cuda_graph,
            )
            if self.video_visual_meta is not None and getattr(self.video_visual_meta, "hmonnx", None)
            else None
        )
        self.audio_meta = getattr(meta_info, "audio_config", None)
        self.audio = (
            Gemma4AudioHMONNXModel(
                self.audio_meta.hmonnx,
                input_feature_length=getattr(self.audio_meta, "input_feature_length", 0) or 0,
                attention_chunk_size=getattr(self.audio_meta, "attention_chunk_size", 12) or 12,
                attention_context_left=getattr(self.audio_meta, "attention_context_left", 13) or 13,
                attention_context_right=getattr(self.audio_meta, "attention_context_right", 0) or 0,
            )
            if self.audio_meta is not None and getattr(self.audio_meta, "hmonnx", None)
            else None
        )
        per_layer_artifact = getattr(meta_info, "per_layer_input_embedding", None)
        self.per_layer_input_embedding = (
            Gemma4PerLayerInputEmbedding.from_artifact(per_layer_artifact) if per_layer_artifact is not None else None
        )

        layer_kv_shapes = getattr(meta_info, "layer_kv_shapes", None) or getattr(
            meta_info,
            "kv_cache_shapes_per_layer",
            [],
        )
        self._kvcache_mixin = Gemma4KVCacheMixinHMONNX(self.kvcache_config, layer_kv_shapes)
        self.prefill_graph_lengths = _gemma4_prefill_graph_lengths_from_meta(meta_info)
        self.prefill_models = {"prefill": self.prefill_model}
        self._active_prefill_graph_name = "prefill"
        self._active_prefill_model = self.prefill_model
        self._active_prefill_input_sequence_length = int(self.prefill_graph_lengths["prefill"])
        self.layer_types = getattr(meta_info, "layer_types", [])
        self.layer_cache_types = getattr(meta_info, "layer_cache_types", [])
        self.layer_cache_indices = getattr(meta_info, "layer_cache_indices", [])
        self.sliding_window = getattr(meta_info, "sliding_window", 1024)
        self._validate_mtp_sliding_kv_cache_input_mode()

    def _attention_contract_version(self) -> int:
        model_config = getattr(self.meta_info, "model_config", None)
        return int(
            getattr(
                self.meta_info,
                "attention_contract_version",
                getattr(model_config, "attention_contract_version", 1),
            )
        )

    def _active_page_attention_model(self) -> HMONNXModel:
        if self.is_prefill():
            return self._active_prefill_model
        return self.decode_model

    def set_page_attention_context(
        self,
        paged_kv_caches=None,
        block_ids=None,
        slot_mapping=None,
        block_size: int | None = None,
        *,
        contexts_by_cache_index: dict[int, PageAttentionContext] | None = None,
    ) -> None:
        """Bind Gemma4 contract-v2 contexts by exported layer cache index.

        The positional arguments remain available only as a Gemma-local
        compatibility delegation.  Qwen continues to own and call the base
        contract-v1 implementation unchanged.
        """

        if contexts_by_cache_index is None:
            return super().set_page_attention_context(
                paged_kv_caches=paged_kv_caches,
                block_ids=block_ids,
                slot_mapping=slot_mapping,
                block_size=block_size,
            )
        if self._attention_contract_version() < 2:
            raise RuntimeError("Gemma4 contexts_by_cache_index requires attention contract-v2")

        page_attention_modules = self._get_page_attention_modules(self._active_page_attention_model())
        cache_indices = self._page_attention_cache_index_by_layer(
            len(page_attention_modules), set(contexts_by_cache_index)
        )
        for page_attention_module, cache_index in zip(page_attention_modules, cache_indices, strict=True):
            # Preserve the exact context object.  In particular, layers that
            # repeat a local cache index share its typed cache handle and staged
            # metadata, while full/local cache indices remain independent.
            page_attention_module.set_context(contexts_by_cache_index[cache_index])

    def _page_attention_cache_index_by_layer(
        self,
        module_count: int,
        available_cache_indices: set[int],
    ) -> list[int]:
        """Return the exported physical cache index for every attention layer.

        Contract-v2 makes ``layer_cache_indices`` the authoritative per-layer
        mapping.  Repeated values express shared caches directly; runtime must
        not infer sharing from attention type or from a shorter owner list.
        """

        authoritative = [int(index) for index in self.layer_cache_indices]
        layer_types = [str(value) for value in self.layer_types]
        if len(authoritative) != module_count or len(layer_types) != module_count:
            raise ValueError(
                "Gemma4 contract-v2 requires one physical cache index per "
                "PageAttention layer: "
                f"modules={module_count}, layer_cache_indices={len(authoritative)}, "
                f"layer_types={len(layer_types)}"
            )
        if any(index < 0 for index in authoritative):
            raise ValueError(
                "Gemma4 contract-v2 layer_cache_indices must contain only "
                f"non-negative physical cache indices, got {authoritative}"
            )
        expected_cache_indices = set(authoritative)
        if available_cache_indices != expected_cache_indices:
            raise ValueError(
                "Gemma4 contexts_by_cache_index does not match physical caches "
                "declared by authoritative per-layer layer_cache_indices: "
                f"expected={sorted(expected_cache_indices)}, "
                f"got={sorted(available_cache_indices)}"
            )
        return authoritative

    def _active_page_attention_interpreter(self) -> Any | None:
        active_model = self._active_page_attention_model()
        session = getattr(active_model, "hmonnx_session", None)
        return getattr(session, "interpreter", None)

    def set_page_attention_execution_mode(self, mode: str) -> None:
        """Select eager visual execution or HMONNX V2 CUDA Graph execution."""

        if mode not in {"eager", "cuda_graph"}:
            raise ValueError(f"Gemma4 PageAttention execution mode must be 'eager' or 'cuda_graph', got {mode!r}")
        interpreter = self._active_page_attention_interpreter()
        active_model = self._active_page_attention_model()
        if getattr(active_model, "enable_cuda_graph", False) and interpreter is None:
            raise RuntimeError("Gemma4 dynamic PageAttention execution mode requires the HMONNXInferenceV2 interpreter")
        if getattr(active_model, "enable_cuda_graph", False) and not hasattr(interpreter, "_capture_disabled_reason"):
            raise RuntimeError(
                "Gemma4 dynamic PageAttention execution mode requires a "
                "single-stage HMONNXInferenceV2 CUDA Graph interpreter"
            )

        stage = "prefill" if self.is_prefill() else "decode"
        clear_counts = getattr(self, "_page_attention_graph_clear_counts", None)
        if clear_counts is None:
            clear_counts = {"prefill": 0, "decode": 0}
            self._page_attention_graph_clear_counts = clear_counts

        if interpreter is not None:
            if mode == "eager":
                prior_disabled_reason = getattr(interpreter, "_capture_disabled_reason", None)
                clear = getattr(interpreter, "clear", None)
                if callable(clear):
                    # Drop captured/replay state without erasing a genuine V2
                    # capture failure. Gemma owns only its exact eager marker.
                    clear(clear_disabled_reason=False)
                    clear_counts[stage] += 1
                if prior_disabled_reason in {
                    None,
                    self._PAGE_ATTENTION_EAGER_REASON,
                }:
                    interpreter._capture_disabled_reason = self._PAGE_ATTENTION_EAGER_REASON
            elif getattr(interpreter, "_capture_disabled_reason", None) == self._PAGE_ATTENTION_EAGER_REASON:
                # Re-enable capture without discarding a stable graph owned by
                # another range-free step.  Genuine capture failures remain.
                interpreter._capture_disabled_reason = None
        self._page_attention_execution_modes = getattr(self, "_page_attention_execution_modes", {})
        self._page_attention_execution_modes[stage] = mode

    @staticmethod
    def _read_interpreter_state(interpreter: Any, name: str, default: Any) -> Any:
        value = getattr(interpreter, name, default)
        return value() if callable(value) else value

    def get_page_attention_execution_state(self) -> dict[str, Any]:
        """Return the minimal active-stage capture/replay/clear evidence."""

        stage = "prefill" if self.is_prefill() else "decode"
        active_model = self._active_page_attention_model()
        interpreter = self._active_page_attention_interpreter()
        modes = getattr(self, "_page_attention_execution_modes", {})
        clear_counts = getattr(self, "_page_attention_graph_clear_counts", {})
        return {
            "stage": stage,
            "requested_mode": modes.get(stage),
            "cuda_graph_enabled": bool(getattr(active_model, "enable_cuda_graph", False)),
            "has_captured_graph": bool(
                self._read_interpreter_state(interpreter, "has_captured_graph", False)
                if interpreter is not None
                else False
            ),
            "replay_active": bool(getattr(interpreter, "_replay_logged", False) if interpreter is not None else False),
            "clear_count": int(clear_counts.get(stage, 0)),
            "capture_disabled_reason": (
                self._read_interpreter_state(interpreter, "capture_disabled_reason", None)
                if interpreter is not None
                else None
            ),
        }

    def _is_mtp_export(self) -> bool:
        return _is_mtp_mode(getattr(self.meta_info, "spec_decode_mode", None))

    def _sliding_kv_cache_input_mode(self) -> str:
        return str(
            getattr(
                self.meta_info,
                "sliding_kv_cache_input_mode",
                getattr(self.meta_info.model_config, "sliding_kv_cache_input_mode", "slice_window"),
            )
        ).lower()

    def _uses_target_verify_decode_accepted_count(self) -> bool:
        return self._is_mtp_export() and self._sliding_kv_cache_input_mode() == "slice_window" and self.is_decode()

    def _validate_mtp_sliding_kv_cache_input_mode(self) -> None:
        if self._is_mtp_export() and self._sliding_kv_cache_input_mode() != "slice_window":
            raise ValueError(
                "Gemma4 Series MTP requires sliding_kv_cache_input_mode='slice_window'; "
                f"got {self._sliding_kv_cache_input_mode()!r}."
            )

    def _mtp_verify_length(self) -> int:
        spec_decode = getattr(self.meta_info, "spec_decode", None) or {}
        if hasattr(spec_decode, "get"):
            verify_length = spec_decode.get("verify_length")
            block_size = spec_decode.get("block_size")
        else:
            verify_length = getattr(spec_decode, "verify_length", None)
            block_size = getattr(spec_decode, "block_size", None)
        if verify_length:
            return int(verify_length)
        verify_length = getattr(self.meta_info, "spec_decode_verify_length", None)
        if verify_length:
            return int(verify_length)
        block_size = block_size or getattr(self.meta_info, "spec_decode_block_size", None)
        block_size = block_size or getattr(self.meta_info.model_config, "num_draft_tokens", 4)
        return int(block_size) + 1

    def get_input_sequence_length(self) -> int:
        if self.is_prefill():
            return int(self._active_prefill_input_sequence_length)
        if self.is_decode() and self._is_mtp_export():
            return self._mtp_verify_length()
        return super().get_input_sequence_length()

    def _set_device(self, device):
        super()._set_device(device)
        if self.visual is not None:
            self.visual.to(device)
        if self.video_visual is not None:
            self.video_visual.to(device)
        if self.audio is not None:
            self.audio.to(device)
        if self.per_layer_input_embedding is not None:
            self.per_layer_input_embedding.to(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        if self.visual is not None:
            self.visual._set_dtype(dtype)
        if self.video_visual is not None:
            self.video_visual._set_dtype(dtype)
        if self.audio is not None:
            self.audio._set_dtype(dtype)
        if self.per_layer_input_embedding is not None:
            self.per_layer_input_embedding.to(dtype=dtype)
        return self

    def _select_prefill_graph_for_length(self, input_sequence_length: int) -> str:
        graph_name = _select_gemma4_prefill_graph_name(input_sequence_length, self.prefill_graph_lengths)
        self._active_prefill_graph_name = graph_name
        self._active_prefill_model = self.prefill_model
        self._active_prefill_input_sequence_length = int(input_sequence_length)
        return graph_name

    def set_input_sequence_length(self, seq_length: int):
        if self.is_prefill():
            self._select_prefill_graph_for_length(int(seq_length))
        return super().set_input_sequence_length(seq_length)

    def set_prefill(self):
        self._llm_prefill = True
        input_sequence_length = int(getattr(self.meta_info.model_config, "prefill_chunk_length", 320))
        self._select_prefill_graph_for_length(input_sequence_length)
        self._active_prefill_model.to(device=self.device)
        self.set_input_sequence_length(input_sequence_length)

    def forward(self, *args, **kwargs):
        args = unfold_args(args)
        args = [arg.to(torch.int32) if arg.dtype == torch.int64 else arg for arg in args]
        if self._llm_prefill:
            self._active_prefill_model.to(device=self.device)
            outs = self._active_prefill_model(*args)
        else:
            outs = super().forward(*args, **kwargs)

        if isinstance(outs, (tuple, list)):
            logits = outs[0]
        else:
            logits = outs
        if getattr(self.meta_info.model_config, "enable_mtp_outputs", False):
            return outs
        return logits

    def get_tf_processor(self):
        processor = XHGemma4Processor.from_pretrained(
            self.hf_model_dir,
            video_max_patches=getattr(self.video_visual_meta, "max_patches", None),
            video_image_seq_length=getattr(self.video_visual_meta, "num_image_tokens", None),
            video_pooling_kernel_size=getattr(self.video_visual_meta, "pooling_kernel_size", None),
        )
        if self.audio_meta is not None:
            processor.config.audio_feature_length = getattr(self.audio_meta, "input_feature_length", None)
            processor.config.audio_attention_chunk_size = getattr(self.audio_meta, "attention_chunk_size", 12) or 12
            processor.config.audio_attention_context_left = getattr(self.audio_meta, "attention_context_left", 13) or 13
            processor.config.audio_attention_context_right = getattr(self.audio_meta, "attention_context_right", 0) or 0
        return processor

    def get_data_preprocessor(self):
        # Keep the runtime input contract phase-aligned with the exported
        # prefill/decode graphs. Full-attention layers use the graph-internal
        # causal MaskedSoftmax path; runtime feeds only the sliding mask.
        self._data_processor = None
        return super().get_data_preprocessor()

    def _get_data_preprocessor(self) -> Gemma4DataPreprocess:
        bidirectional_vision_attention = getattr(
            self.meta_info.model_config,
            "bidirectional_vision_attention",
            False,
        )
        return Gemma4DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.get_input_sequence_length(),
            context_length=self.meta_info.model_config.context_max_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            pad_token_id=self.pad_token_id,
            image_token_id=getattr(self.meta_info.model_config, "image_token_id", -1) or -1,
            audio_token_id=getattr(self.meta_info.model_config, "audio_token_id", -1) or -1,
            video_token_id=getattr(self.meta_info.model_config, "video_token_id", -1) or -1,
            per_layer_input_embedding=self.per_layer_input_embedding,
            sliding_window=self.sliding_window,
            bidirectional_vision_attention=bidirectional_vision_attention,
            attention_contract_version=self._attention_contract_version(),
            max_mm_ranges_per_chunk=int(getattr(self.meta_info, "max_mm_ranges_per_chunk", 1)),
            emit_full_attention_mask=False,
            emit_accepted_count_input=self._uses_target_verify_decode_accepted_count(),
        )

    def _set_enable_golden(self, enable: bool) -> None:
        super()._set_enable_golden(enable)
        if self.visual is not None:
            self.visual.enable_golden = enable
        if self.video_visual is not None:
            self.video_visual.enable_golden = enable
        if self.audio is not None:
            self.audio.enable_golden = enable


class XHGemma4MoeHMONNXModel(XHGemma4SeriesHMONNXModel):
    """Public Gemma4 runtime for the 26B-A4B MoE topology.

    The exported metadata deliberately keeps the public model type as
    ``Gemma4ForConditionalGeneration`` to avoid public-registration collisions.
    Runtime still needs the MoE text input contract: ``inputs_embeds`` followed
    by local/global masks and caches.  Reuse the unified visual/audio wrapper
    from ``XHGemma4HMONNXModel`` so image and video both follow the padded VIT
    API, and only swap the text-side data preprocessor.
    """

    def _get_data_preprocessor(self) -> Gemma4MoeDataPreprocess:
        model_config = self.meta_info.model_config
        sliding_window_cfg = getattr(self.meta_info, "sliding_window_cfg", None)
        if sliding_window_cfg is None:
            sliding_window_cfg = {
                "sliding_window": getattr(model_config, "sliding_window", 1024),
                "local_attention_window_size": getattr(model_config, "local_attention_window_size", 1024),
                "global_attention_window_size": getattr(
                    model_config,
                    "global_attention_window_size",
                    getattr(model_config, "context_max_length", 2048),
                ),
                "has_local_attention": getattr(model_config, "has_local_attention", True),
                "has_global_attention": getattr(model_config, "has_global_attention", True),
            }
        return Gemma4MoeDataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.get_input_sequence_length(),
            context_length=model_config.context_max_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            pad_token_id=self.pad_token_id,
            image_token_id=getattr(model_config, "image_token_id", -1) or -1,
            audio_token_id=-1,
            video_token_id=getattr(model_config, "video_token_id", -1) or -1,
            sliding_window_cfg=sliding_window_cfg,
            bidirectional_vision_attention=getattr(model_config, "bidirectional_vision_attention", False),
            emit_full_attention_mask=False,
            emit_accepted_count_input=self._uses_target_verify_decode_accepted_count(),
        )


XHGemma4HMONNXModel = XHGemma4SeriesHMONNXModel
XHGemma4_HMONNXModel = XHGemma4SeriesHMONNXModel


__all__ = [
    "Gemma4AudioHMONNXModel",
    "Gemma4KVCacheMixinHMONNX",
    "Gemma4VisualHMONNXModel",
    "XHGemma4HMONNXModel",
    "XHGemma4MoeHMONNXModel",
    "XHGemma4SeriesHMONNXModel",
    "XHGemma4_HMONNXModel",
]
