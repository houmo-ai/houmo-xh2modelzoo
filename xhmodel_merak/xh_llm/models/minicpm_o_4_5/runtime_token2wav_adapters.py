from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Protocol

import torch
from torch import Tensor


class Token2WavRuntimeProtocol(Protocol):
    up_rate: int

    def flow_inference(self, *args: Tensor | int) -> Tensor: ...

    def hift_inference(self, speech_feat: Tensor) -> Tensor: ...

    def init_stream_cache(self, base_cache: Mapping[str, Tensor], prompt_mel_length: int) -> Any: ...

    def stream_flow(
        self, tokens: Tensor, embedding: Tensor, last_chunk: bool
    ) -> tuple[Tensor, Mapping[str, Tensor]]: ...

    def stream_hift(
        self, mel: Tensor, cache_source: Tensor, last_chunk: bool | None = None
    ) -> tuple[Tensor, Tensor]: ...


class _HMONNXFlowAdapter:
    def __init__(self, native_flow: object, runtime: Token2WavRuntimeProtocol) -> None:
        self.native_flow = native_flow
        self.runtime = runtime
        self.up_rate = runtime.up_rate
        self.pre_lookahead_len = int(getattr(runtime, "pre_lookahead_len", 0))

    def __getattr__(self, name: str) -> object:
        return getattr(self.native_flow, name)

    def inference(self, *args: Tensor | int) -> Tensor:
        reference = next((value for value in args if isinstance(value, Tensor)), None)
        device_type = reference.device.type if reference is not None else "cuda"
        with torch.autocast(device_type, enabled=False):
            return self.runtime.flow_inference(*args)

    def setup_cache(self, token: Tensor, mel: Tensor, spk: Tensor, n_timesteps: int = 10) -> Mapping[str, Tensor]:
        result = self.native_flow.setup_cache(token, mel, spk, n_timesteps=n_timesteps)
        if not isinstance(result, Mapping):
            raise RuntimeError("official Flow setup_cache must return a cache mapping")
        if mel.dim() != 3 or mel.shape[0] != 1 or (mel.shape[1] != 80 and mel.shape[2] != 80):
            raise RuntimeError("official Flow prompt mel must have layout [1, frames, 80] or [1, 80, frames]")
        prompt_mel_length = int(mel.shape[2] if mel.shape[1] == 80 else mel.shape[1])
        self.runtime.init_stream_cache(result, prompt_mel_length)
        report = getattr(self.runtime, "report", None)
        if report is not None:
            report.record_host_initialization()
        return result

    def inference_chunk(self, *args: Tensor, **kwargs: object) -> tuple[Tensor, Mapping[str, Tensor]]:
        token = kwargs.get("token", args[0] if args else None)
        embedding = kwargs.get("spk", args[1] if len(args) > 1 else None)
        last_chunk = kwargs.get("last_chunk", False)
        if not isinstance(token, Tensor) or not isinstance(embedding, Tensor) or not isinstance(last_chunk, bool):
            raise RuntimeError("official Flow streaming arguments are incomplete")
        with torch.autocast(token.device.type, enabled=False):
            return self.runtime.stream_flow(token, embedding, last_chunk)


class _HMONNXHiFTAdapter:
    def __init__(self, runtime: Token2WavRuntimeProtocol) -> None:
        self.runtime = runtime

    def __call__(self, speech_feat: Tensor, cache_source: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        if cache_source is None:
            with torch.autocast(speech_feat.device.type, enabled=False):
                return self.runtime.hift_inference(speech_feat).float(), None
        stream_hift = getattr(self.runtime, "stream_hift", None)
        if stream_hift is None:
            raise RuntimeError("Token2Wav stream_hift role is unavailable after HMONNX attachment")
        with torch.autocast(speech_feat.device.type, enabled=False):
            return stream_hift(speech_feat, cache_source)


class _HMONNXCampplusAdapter:
    def __init__(self, runtime: Token2WavRuntimeProtocol, *, sequence_length: int) -> None:
        self._runtime = runtime
        self._sequence_length = sequence_length
        self._input_name = "input"

    def get_inputs(self) -> list[Any]:
        return [type("_In", (), {"name": self._input_name})()]

    def run(self, _output_names: Any, feed: dict[str, Any]) -> list[Any]:
        session = getattr(self._runtime, "campplus_session", None)
        if session is None:
            raise RuntimeError("HMONNX campplus session has been released")
        feat = torch.as_tensor(feed[self._input_name])
        if feat.shape[1] > self._sequence_length:
            feat = feat[:, : self._sequence_length]
        elif feat.shape[1] < self._sequence_length:
            feat = torch.nn.functional.pad(feat, (0, 0, 0, self._sequence_length - feat.shape[1]))
        return [torch.as_tensor(session(feat.to(torch.float16))).float().cpu().numpy()]


class _HMONNXSpeechTokenizerAdapter:
    def __init__(self, runtime: Token2WavRuntimeProtocol, *, feats_length: int, n_mels: int) -> None:
        self._runtime = runtime
        self._feats_length = feats_length
        self._n_mels = n_mels

    def quantize(self, mel: Tensor, mel_len: Tensor) -> tuple[Tensor, Tensor]:
        session = getattr(self._runtime, "speech_tokenizer_session", None)
        if session is None:
            raise RuntimeError("HMONNX speech tokenizer session has been released")
        mel = torch.as_tensor(mel)
        batch, n_mels, frames = mel.shape
        if batch != 1 or n_mels != self._n_mels:
            raise RuntimeError(f"HMONNX speech_tokenizer requires [1, {self._n_mels}, frames], got {tuple(mel.shape)}")
        valid = int(torch.as_tensor(mel_len).reshape(-1)[0].item())
        if frames > self._feats_length:
            mel = mel[:, :, : self._feats_length]
        elif frames < self._feats_length:
            mel = torch.nn.functional.pad(mel, (0, self._feats_length - frames, 0, 0, 0, 0))
        capacity = torch.tensor([self._feats_length], dtype=torch.int32, device=mel.device)
        tokens = torch.as_tensor(session(mel.to(torch.float16), capacity))
        token_len = _official_token_length(valid)
        return tokens[:, :token_len], torch.tensor([token_len], dtype=torch.int64, device=mel.device)


def _official_token_length(feats_length: int) -> int:
    x = (feats_length - 3) // 2
    x = int(x) + 1
    x = x - 3 + 2
    return int((x // 2) + 1)


def install_hmonnx_token2wav(audio_tokenizer: SimpleNamespace, runtime: Token2WavRuntimeProtocol) -> None:
    audio_tokenizer.flow = _HMONNXFlowAdapter(audio_tokenizer.flow, runtime)
    audio_tokenizer.hift = _HMONNXHiFTAdapter(runtime)
    audio_tokenizer.hift.inference_chunk = audio_tokenizer.hift
    if getattr(runtime, "campplus_session", None) is not None:
        audio_tokenizer.spk_model = _HMONNXCampplusAdapter(
            runtime, sequence_length=getattr(runtime, "campplus_sequence_length", 1000)
        )
    if getattr(runtime, "speech_tokenizer_session", None) is not None:
        audio_tokenizer.audio_tokenizer = _HMONNXSpeechTokenizerAdapter(
            runtime, feats_length=getattr(runtime, "speech_tokenizer_feats_length", 3000), n_mels=128
        )
    audio_tokenizer._xh_hmonnx_streaming_attached = True
    report = getattr(runtime, "report", None)
    if report is not None:
        report.streaming_attached = True


__all__ = [
    "Token2WavRuntimeProtocol",
    "_HMONNXCampplusAdapter",
    "_HMONNXFlowAdapter",
    "_HMONNXHiFTAdapter",
    "_HMONNXSpeechTokenizerAdapter",
    "install_hmonnx_token2wav",
]
