"""Shared helpers for FunASR-Nano xhquant/HMONNX examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch


def resolve_model_dir(model_dir: str) -> Path:
    path = Path(model_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_funasr_nano_model(model_dir: str, device: str = "cpu"):
    """Load FunASR-Nano with FunASR AutoModel and return ``(model, kwargs)``."""
    from funasr import AutoModel

    model, kwargs = AutoModel.build_model(
        model=str(resolve_model_dir(model_dir)),
        device=device,
        trust_remote_code=True,
        disable_update=True,
    )
    model.eval()
    return model, kwargs


def load_checkpoint_state(model_dir: str) -> dict[str, torch.Tensor]:
    ckpt_path = resolve_model_dir(model_dir) / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    return checkpoint.get("state_dict", checkpoint)


def prefixed_state_dict(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def save_token_embedding_from_model(model: torch.nn.Module, out_file: Path) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    weight = model.llm.get_input_embeddings().weight.detach().cpu()
    torch.save({"weight": weight}, out_file)
    return out_file


def load_audio_for_frontend(audio: Optional[str], frontend, seconds: float = 3.0) -> torch.Tensor:
    if audio:
        from funasr.utils.load_utils import load_audio_text_image_video

        return load_audio_text_image_video(audio, fs=frontend.fs).float()
    sample_count = int(frontend.fs * seconds)
    return torch.zeros(sample_count, dtype=torch.float32)


def extract_fbank(audio: torch.Tensor, frontend) -> tuple[torch.Tensor, torch.Tensor]:
    from funasr.utils.load_utils import extract_fbank as funasr_extract_fbank

    return funasr_extract_fbank(audio, data_type="sound", frontend=frontend, is_final=True)


def low_frame_rate_len(fbank_len: torch.Tensor) -> torch.Tensor:
    """Match FunASR-Nano's use_low_frame_rate length correction."""
    olens = 1 + (fbank_len - 3 + 2 * 1) // 2
    olens = 1 + (olens - 3 + 2 * 1) // 2
    return ((olens - 1) // 2 + 1).to(torch.int32)


def output_path_in_work_dir(work_dir: Path, subdir: str, name: str) -> Path:
    path = work_dir / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
