from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from torch import Tensor, nn

from ...hmonnx.hmonnx_model import HMONNXBaseModel, HMONNXModel
from .hf_compatible import (
    MiniCPMOHFCompatible,
    bind_duplex_hmonnx_reset,
    patch_speech_generation_capture,
    reuse_initialized_tts,
)
from .runtime_audio import MiniCPMO45AudioHMONNXRuntime
from .runtime_llm import MiniCPMO45LLMHMONNXRuntime
from .runtime_token2wav import MiniCPMO45Token2WavHMONNXRuntime, install_hmonnx_token2wav
from .runtime_tts import MiniCPMO45TTSHMONNXRuntime
from .runtime_vision import MiniCPMO45VisionHMONNXRuntime


STREAMING_CASE_SESSION_AUDIO_TEXT = "session_audio_text"
STREAMING_CASE_SESSION_AUDIO_REPLY = "session_audio_reply"
STREAMING_CASE_DUPLEX_AUDIO_TEXT = "duplex_audio_text"
STREAMING_CASE_DUPLEX_AUDIO_REPLY = "duplex_audio_reply"
STREAMING_CASE_DUPLEX_OMNI_REPLY = "duplex_omni_reply"
STREAMING_CASES = (
    STREAMING_CASE_SESSION_AUDIO_TEXT,
    STREAMING_CASE_SESSION_AUDIO_REPLY,
    STREAMING_CASE_DUPLEX_AUDIO_TEXT,
    STREAMING_CASE_DUPLEX_AUDIO_REPLY,
    STREAMING_CASE_DUPLEX_OMNI_REPLY,
)


def move_native_tts_modules(host_model: nn.Module, device: str) -> None:
    tts = getattr(host_model, "tts", None)
    if tts is None:
        return
    for name in ("emb_text", "projector_semantic", "emb_code", "head_code"):
        module = getattr(tts, name, None)
        if module is not None:
            module.to(device)


def move_native_audio_modules(host_model: nn.Module, device: str) -> None:
    for name in ("audio_projection_layer", "audio_avg_pooler", "resampler"):
        module = getattr(host_model, name, None)
        if module is not None:
            module.to(device)


def move_native_llm_modules(host_model: nn.Module, device: str) -> None:
    llm = getattr(host_model, "llm", None)
    for name in ("embed_tokens", "lm_head"):
        module = getattr(llm, name, None)
        if module is not None:
            module.to(device)


def _release_hmonnx_component(component: object) -> None:
    """Release shared graph wrappers once, then drop model-specific state.

    ``HMONNXBaseModel`` intentionally has no public unregister/release API.
    Keeping this one top-level owner avoids giving Audio, Vision, text and
    Token2Wav subtly different release semantics while leaving all ordinary
    lifecycle operations to the repository classes.
    """
    release_state = getattr(component, "release_state", None)
    if callable(release_state):
        release_state()

    if isinstance(component, HMONNXModel):
        component.hmonnx_session = None
        component._onnx_graph = None
        return
    if not isinstance(component, HMONNXBaseModel):
        return
    for name, model in tuple(component._models.items()):
        model.hmonnx_session = None
        model._onnx_graph = None
        setattr(component, name, None)
    component._models.clear()


class MiniCPMO45HMONNXRuntime:
    """Compose MiniCPM's official multimodal APIs with shared HMONNX graphs.

    This remains a component coordinator rather than a ``BaseLLMHMONNXModel``:
    the official session/duplex APIs jointly manage audio, vision, TTS and
    Token2Wav state.  The component graph sessions themselves use the common
    ``HMONNXModel`` lifecycle.
    """

    def __init__(
        self,
        work_dir: str | Path,
        host_model: nn.Module | None = None,
        *,
        device_map: str | list[str] | None = None,
        enable_cuda_graph: bool = False,
        enable_auto_offload: bool = False,
        enable_golden: bool = False,
    ) -> None:
        root = Path(work_dir)
        meta = json.loads((root / "export_meta_info.json").read_text(encoding="utf-8"))
        components = meta["components"]
        graph_options = {
            "device_map": device_map,
            "enable_cuda_graph": enable_cuda_graph,
            "enable_auto_offload": enable_auto_offload,
            "enable_golden": enable_golden,
        }
        self.vision = MiniCPMO45VisionHMONNXRuntime(root / components["vision"]["graphs"]["main"], **graph_options)
        self.audio = MiniCPMO45AudioHMONNXRuntime(root, components["audio"], **graph_options)
        self.llm = MiniCPMO45LLMHMONNXRuntime(root, components["llm"], **graph_options)
        self.tts = MiniCPMO45TTSHMONNXRuntime(root, components["tts"], **graph_options)
        self.token2wav = MiniCPMO45Token2WavHMONNXRuntime(root, components, **graph_options)
        self.host_model = None
        if host_model is not None:
            self.attach_host(host_model)

    def attach_host(self, host_model: nn.Module) -> nn.Module:
        native_tts_model = getattr(getattr(host_model, "tts", None), "model", None)
        native_tts_config = getattr(native_tts_model, "config", None)
        for name in (
            "reset_session",
            "init_token2wav_cache",
            "streaming_prefill",
            "streaming_generate",
            "as_duplex",
        ):
            method = getattr(host_model, name, None)
            if method is not None:
                setattr(self, f"_official_{name}", method)
        self._streaming_api_mode: str | None = None
        self._session_id: str | None = None
        embeddings = host_model.vpm.embeddings
        self.vision.patch_size = embeddings.patch_size
        self.vision.num_patches_per_side = embeddings.num_patches_per_side
        self.vision.wrap_cfg = SimpleNamespace(image_slice_max_size=[40, 40])
        self.host_model = MiniCPMOHFCompatible.to_hf_compatible(
            host_model,
            vision_model=self.vision,
            audio_model=self.audio,
            llm_model=self.llm,
            tts_llama_model=self.tts if hasattr(host_model, "tts") else None,
        )
        if native_tts_config is not None:
            self.tts.config = native_tts_config
        if hasattr(host_model, "tts") and hasattr(host_model, "_generate_speech_non_streaming"):
            audio_tokenizer = getattr(host_model.tts, "audio_tokenizer", None)
            if audio_tokenizer is None:
                raise RuntimeError("MiniCPM-o-4.5 TTS audio tokenizer is not initialized")
            install_hmonnx_token2wav(audio_tokenizer, self.token2wav)
            patch_speech_generation_capture(self.host_model)
        return self.host_model

    def set_exec_device(self, device: str) -> None:
        """Place the complete official MiniCPM pipeline on one execution device."""
        for component in (self.vision, self.audio, self.llm, self.tts, self.token2wav):
            component.to(device)
        if self.host_model is not None:
            move_native_audio_modules(self.host_model, device)
            move_native_llm_modules(self.host_model, device)
            move_native_tts_modules(self.host_model, device)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        if self.host_model is None:
            raise RuntimeError("MiniCPM-o-4.5 host model is not attached")
        return self.host_model.chat(*args, **kwargs)

    def reset_session(self, reset_token2wav_cache: bool = True) -> Any:
        if self.host_model is None:
            raise RuntimeError("MiniCPM-o-4.5 host model is not attached")
        reset = getattr(self, "_official_reset_session", self.host_model.reset_session)
        result = reset(reset_token2wav_cache)
        token2wav = getattr(self, "token2wav", None)
        if reset_token2wav_cache and token2wav is not None:
            token2wav.reset_state()
        self._reset_cache_components()
        self._streaming_api_mode = None
        self._session_id = None
        return result

    def init_token2wav_cache(self, prompt_speech_16k: Any) -> Any:
        if self.host_model is None:
            raise RuntimeError("MiniCPM-o-4.5 host model is not attached")
        init_cache = getattr(self, "_official_init_token2wav_cache", self.host_model.init_token2wav_cache)
        result = init_cache(prompt_speech_16k)
        token2wav = getattr(self, "token2wav", None)
        bridge = getattr(token2wav, "bridge_official_init", None)
        if bridge is not None:
            official_cache = result
            if official_cache is None:
                official_cache = getattr(self.host_model, "token2wav_cache", None)
            bridge(official_cache)
        return result

    def streaming_prefill(self, session_id: str, msgs: list[dict[str, Any]], **kwargs: Any) -> Any:
        if self.host_model is None:
            raise RuntimeError("MiniCPM-o-4.5 host model is not attached")
        self._claim_streaming_mode("session")
        session_id_before = getattr(self, "_session_id", None)
        if session_id_before is not None and session_id_before != session_id:
            self._reset_streaming_components(reset_token2wav_cache=True)
        self._session_id = session_id
        prefill = getattr(self, "_official_streaming_prefill", self.host_model.streaming_prefill)
        return prefill(session_id, msgs, **kwargs)

    def streaming_generate(self, session_id: str, **kwargs: Any) -> Any:
        if self.host_model is None:
            raise RuntimeError("MiniCPM-o-4.5 host model is not attached")
        self._claim_streaming_mode("session")
        session_id_before = getattr(self, "_session_id", None)
        if session_id_before is not None and session_id_before != session_id:
            reset = getattr(self, "_official_reset_session", self.host_model.reset_session)
            reset(False)
            self._reset_streaming_components(reset_token2wav_cache=True)
        self._session_id = session_id
        generate = getattr(self, "_official_streaming_generate", self.host_model.streaming_generate)
        return generate(session_id, **kwargs)

    def as_duplex(self, device: str | None = None, **kwargs: Any) -> Any:
        if self.host_model is None:
            raise RuntimeError("MiniCPM-o-4.5 host model is not attached")
        self._claim_streaming_mode("duplex")
        as_duplex = getattr(self, "_official_as_duplex", self.host_model.as_duplex)
        tokenizer = getattr(getattr(self.host_model, "tts", None), "audio_tokenizer", None)
        if tokenizer is None or not hasattr(self.host_model, "init_tts"):
            return as_duplex(device=device, **kwargs)
        duplex = reuse_initialized_tts(
            self.host_model,
            tokenizer,
            lambda: as_duplex(device=device, **kwargs),
        )
        return bind_duplex_hmonnx_reset(
            duplex,
            self._reset_streaming_components,
            lambda: getattr(self, "_streaming_api_mode", None) == "duplex",
        )

    def _claim_streaming_mode(self, mode: str) -> None:
        active = getattr(self, "_streaming_api_mode", None)
        if active is not None and active != mode:
            raise RuntimeError(f"{active.title()} API is active; reset the runtime before switching to {mode}")
        self._streaming_api_mode = mode

    def _reset_streaming_components(self, reset_token2wav_cache: bool = False) -> None:
        self._reset_cache_components()
        token2wav = getattr(self, "token2wav", None)
        if reset_token2wav_cache and token2wav is not None:
            token2wav.reset_state()

    def _reset_cache_components(self) -> None:
        for component in (self.audio, self.llm, self.tts):
            component.reset_state()

    def reset_state(self) -> None:
        self._reset_streaming_components(reset_token2wav_cache=True)
        if self.host_model is not None:
            self.host_model._xh_last_speech_tokens = None
            self.host_model._xh_last_waveform = None
        self._streaming_api_mode = None
        self._session_id = None

    def to_fast(self) -> "MiniCPMO45HMONNXRuntime":
        """Enable the shared HMONNX fast path for every graph component."""
        for component in (self.vision, self.audio, self.llm, self.tts, self.token2wav):
            component.to_fast()
        return self

    @property
    def token2wav_backends(self) -> dict[str, str]:
        return self.token2wav.report.backends

    @property
    def token2wav_execution_counts(self) -> dict[str, int]:
        return self.token2wav.report.execution_counts

    @property
    def token2wav_streaming_backends(self) -> dict[str, str]:
        return self.token2wav.report.streaming_backends

    @property
    def token2wav_streaming_execution_counts(self) -> dict[str, int]:
        return self.token2wav.report.streaming_execution_counts

    @property
    def last_speech_token_ids(self) -> Tensor | None:
        if self.host_model is None:
            return None
        return getattr(self.host_model, "_xh_last_speech_tokens", None)

    @property
    def last_waveform(self) -> Tensor | None:
        if self.host_model is None:
            return None
        return getattr(self.host_model, "_xh_last_waveform", None)

    def release(self) -> None:
        for component in (self.vision, self.audio, self.llm, self.tts, self.token2wav):
            _release_hmonnx_component(component)
        self._streaming_api_mode = None
        self._session_id = None


__all__ = [
    "MiniCPMO45HMONNXRuntime",
    "STREAMING_CASE_SESSION_AUDIO_TEXT",
    "STREAMING_CASE_SESSION_AUDIO_REPLY",
    "STREAMING_CASE_DUPLEX_AUDIO_TEXT",
    "STREAMING_CASE_DUPLEX_AUDIO_REPLY",
    "STREAMING_CASE_DUPLEX_OMNI_REPLY",
    "STREAMING_CASES",
    "move_native_audio_modules",
    "move_native_llm_modules",
    "move_native_tts_modules",
]
