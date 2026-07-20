# Copyright 2025 HOUMO AI
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"SenseVoice config must be a YAML mapping: {path}")
    return data


def _load_cmvn(cmvn_file: Path) -> Any:
    import numpy as np

    lines = cmvn_file.read_text(encoding="utf-8").splitlines()
    means_list: list[str] = []
    vars_list: list[str] = []
    for index, line in enumerate(lines):
        line_item = line.split()
        if not line_item or index + 1 >= len(lines):
            continue
        if line_item[0] == "<AddShift>":
            line_item = lines[index + 1].split()
            if line_item and line_item[0] == "<LearnRateCoef>":
                means_list = line_item[3:-1]
        elif line_item[0] == "<Rescale>":
            line_item = lines[index + 1].split()
            if line_item and line_item[0] == "<LearnRateCoef>":
                vars_list = line_item[3:-1]

    means = np.asarray(means_list, dtype=np.float64)
    vars_ = np.asarray(vars_list, dtype=np.float64)
    if means.size == 0 or vars_.size == 0:
        raise ValueError(f"Failed to parse CMVN file: {cmvn_file}")
    return np.stack([means, vars_], axis=0)


def _apply_cmvn(feat: Any, cmvn: Any) -> Any:
    import numpy as np

    frame, dim = feat.shape
    means = np.tile(cmvn[0:1, :dim], (frame, 1))
    vars_ = np.tile(cmvn[1:2, :dim], (frame, 1))
    return (feat + means) * vars_


def _apply_lfr(inputs: Any, lfr_m: int, lfr_n: int) -> Any:
    import numpy as np

    if lfr_m == 1 and lfr_n == 1:
        return inputs.astype(np.float32)
    if inputs.shape[0] == 0:
        raise ValueError("Cannot apply LFR to an empty feature sequence")

    lfr_inputs = []
    input_length = inputs.shape[0]
    output_length = int(np.ceil(input_length / lfr_n))
    left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
    inputs = np.vstack((left_padding, inputs))
    padded_length = input_length + (lfr_m - 1) // 2
    for index in range(output_length):
        start = index * lfr_n
        if lfr_m <= padded_length - start:
            lfr_inputs.append(inputs[start : start + lfr_m].reshape(1, -1))
            continue
        num_padding = lfr_m - (padded_length - start)
        frame = inputs[start:].reshape(-1)
        for _ in range(num_padding):
            frame = np.hstack((frame, inputs[-1]))
        lfr_inputs.append(frame)
    return np.vstack(lfr_inputs).astype(np.float32)


@dataclass(frozen=True)
class FrontendConfig:
    fs: int = 16000
    window: str = "hamming"
    n_mels: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    lfr_m: int = 7
    lfr_n: int = 6
    dither: float = 1.0
    cmvn_file: str = ""


class SenseVoiceFrontend:
    """FunASR-compatible fbank + LFR + CMVN frontend."""

    def __init__(self, cfg: FrontendConfig):
        self.cfg = cfg
        self.cmvn = _load_cmvn(Path(cfg.cmvn_file)) if cfg.cmvn_file else None

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "SenseVoiceFrontend":
        model_path = Path(model_dir).expanduser().resolve()
        config_file = model_path / "config.yaml"
        cmvn_file = model_path / "am.mvn"
        if not config_file.is_file():
            raise FileNotFoundError(f"SenseVoice config.yaml not found: {config_file}")
        if not cmvn_file.is_file():
            raise FileNotFoundError(f"SenseVoice am.mvn not found: {cmvn_file}")

        config = _read_yaml(config_file)
        frontend_conf = dict(config.get("frontend_conf") or {})
        return cls(
            FrontendConfig(
                fs=int(frontend_conf.get("fs", 16000)),
                window=str(frontend_conf.get("window", "hamming")),
                n_mels=int(frontend_conf.get("n_mels", 80)),
                frame_length=int(frontend_conf.get("frame_length", 25)),
                frame_shift=int(frontend_conf.get("frame_shift", 10)),
                lfr_m=int(frontend_conf.get("lfr_m", 7)),
                lfr_n=int(frontend_conf.get("lfr_n", 6)),
                dither=float(frontend_conf.get("dither", 1.0)),
                cmvn_file=str(cmvn_file),
            )
        )

    def fbank(self, waveform: Any) -> tuple[Any, int]:
        import torch

        try:
            import torchaudio
        except Exception as exc:
            raise RuntimeError(f"torchaudio is required for Kaldi fbank extraction: {exc}") from exc

        wav = torch.as_tensor(waveform, dtype=torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim != 2:
            raise ValueError(f"Expected waveform with one or two dimensions, got shape {tuple(wav.shape)}")
        if wav.shape[0] != 1:
            wav = wav.mean(dim=0, keepdim=True)

        feat = torchaudio.compliance.kaldi.fbank(
            wav * (1 << 15),
            num_mel_bins=self.cfg.n_mels,
            sample_frequency=self.cfg.fs,
            frame_length=float(self.cfg.frame_length),
            frame_shift=float(self.cfg.frame_shift),
            dither=float(self.cfg.dither),
            window_type=self.cfg.window,
            snip_edges=True,
            energy_floor=0.0,
            use_energy=False,
        )
        feat = feat.cpu().numpy().astype("float32")
        return feat, int(feat.shape[0])

    def extract(self, waveform: Any) -> tuple[Any, int]:
        feat, _ = self.fbank(waveform)
        feat = _apply_lfr(feat, self.cfg.lfr_m, self.cfg.lfr_n)
        if self.cmvn is not None:
            feat = _apply_cmvn(feat, self.cmvn).astype("float32")
        return feat, int(feat.shape[0])


__all__ = ["FrontendConfig", "SenseVoiceFrontend"]
