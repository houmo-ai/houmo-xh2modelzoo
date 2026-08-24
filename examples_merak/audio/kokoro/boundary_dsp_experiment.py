from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch

from xhmodel_merak.xh_other_model.models.kokoro.assets import resolve_model_assets
from xhmodel_merak.xh_other_model.models.kokoro.host import (
    DEFAULT_INPUT_IDS,
    SAMPLE_RATE,
    WAVEFORM_SAMPLES_PER_FRAME,
    istft_waveform,
    load_voice_style,
)
from xhmodel_merak.xh_other_model.models.kokoro.runtime import (
    KokoroStaticRuntime,
    Runner,
)
from xhmodel_merak.xh_other_model.models.kokoro.static_dsp import (
    StaticISTFT20,
    StaticSTFT20,
)


class CaptureRunner:
    def __init__(self, inner: Runner) -> None:
        self.inner = inner
        self.feeds: list[dict[str, np.ndarray]] = []
        self.outputs: list[dict[str, np.ndarray]] = []

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = self.inner.run(feed)
        self.feeds.append({name: np.asarray(value).copy() for name, value in feed.items()})
        self.outputs.append({name: np.asarray(value).copy() for name, value in output.items()})
        return output


def tensor_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference64 = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate64 = np.asarray(candidate, dtype=np.float64).reshape(-1)
    difference = reference64 - candidate64
    denominator = float(np.linalg.norm(reference64) * np.linalg.norm(candidate64))
    return {
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "mean_abs": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
        "cosine": float(np.dot(reference64, candidate64) / denominator) if denominator else 1.0,
    }


def phase_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.angle(np.exp(1j * (np.asarray(candidate) - np.asarray(reference))))
    return {
        "max_abs_radians": float(np.max(np.abs(difference), initial=0.0)),
        "mean_abs_radians": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "rmse_radians": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
    }


def write_wav(path: Path, waveform: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(waveform, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure fixed-bucket STFT padding and static iSTFT boundaries")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="work_dirs/kokoro_merak/boundary_dsp_experiment",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    export_dir = Path(args.export_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    assets = resolve_model_assets(args.model_dir)
    tokens = np.asarray(DEFAULT_INPUT_IDS, dtype=np.int32)
    style = load_voice_style(assets.voice, phoneme_count=tokens.size - 2)
    runtime = KokoroStaticRuntime.from_export(export_dir, backend="ort")
    captures = {role: CaptureRunner(runner) for role, runner in runtime.runners.items()}
    runtime.runners = captures
    reference_waveform, synthesis = runtime.synthesize(
        tokens,
        style,
        seed=args.seed,
    )

    valid_frames = int(synthesis["frame_length"])
    valid_samples = valid_frames * WAVEFORM_SAMPLES_PER_FRAME
    frame_max_length = runtime.frame_max_length
    bucket_samples = frame_max_length * WAVEFORM_SAMPLES_PER_FRAME
    valid_spectral_frames = 120 * valid_frames + 1
    source = torch.from_numpy(captures["source_merge"].outputs[0]["harmonic_source"]).float()
    source_mask = (torch.arange(bucket_samples).reshape(1, -1, 1) < valid_samples).to(source.dtype)
    source = source * source_mask
    generator_feed = captures["generator"].feeds[0]
    reference_harmonic = np.asarray(generator_feed["harmonic"], dtype=np.float32)
    reference_spec_phase = torch.from_numpy(captures["generator"].outputs[0]["spec_phase"]).float()
    valid_frames_tensor = torch.tensor([valid_frames], dtype=torch.int32)
    waveform_mask = (torch.arange(bucket_samples).reshape(1, 1, -1) < valid_samples).float()
    static_istft = StaticISTFT20(frame_max_length).eval()
    baseline_static_istft = static_istft(reference_spec_phase, waveform_mask).reshape(-1)[:valid_samples].numpy()

    report: dict[str, Any] = {
        "frame_max_length": frame_max_length,
        "valid_frames": valid_frames,
        "bucket_samples": bucket_samples,
        "valid_samples": valid_samples,
        "static_istft_vs_torch_istft": tensor_metrics(
            reference_waveform,
            baseline_static_istft,
        ),
        "padding_variants": {},
    }
    write_wav(output_dir / "reference_reflect_torch_istft.wav", reference_waveform)
    write_wav(output_dir / "reference_reflect_static_istft.wav", baseline_static_istft)

    generator_runner = captures["generator"].inner
    for pad_mode in ("constant", "reflect", "replicate", "length_aware_reflect"):
        stft = StaticSTFT20(
            pad_mode=pad_mode,
            waveform_length=(bucket_samples if pad_mode == "length_aware_reflect" else None),
        ).eval()
        harmonic = stft(
            source,
            torch.tensor([valid_samples], dtype=torch.int32) if pad_mode == "length_aware_reflect" else None,
        )
        feed = {name: np.asarray(value).copy() for name, value in generator_feed.items()}
        feed["harmonic"] = harmonic.detach().numpy()
        generated = generator_runner.run(feed)
        spec_phase = torch.from_numpy(generated["spec_phase"]).float()
        torch_waveform = istft_waveform(spec_phase, valid_frames_tensor).reshape(-1).numpy()
        linear_waveform = static_istft(spec_phase, waveform_mask).reshape(-1)[:valid_samples].numpy()
        harmonic_array = harmonic.detach().numpy()
        report["padding_variants"][pad_mode] = {
            "harmonic_magnitude": tensor_metrics(
                reference_harmonic[:, :11, :valid_spectral_frames],
                harmonic_array[:, :11, :valid_spectral_frames],
            ),
            "harmonic_phase": phase_metrics(
                reference_harmonic[:, 11:, :valid_spectral_frames],
                harmonic_array[:, 11:, :valid_spectral_frames],
            ),
            "torch_istft_waveform_vs_reference": tensor_metrics(
                reference_waveform,
                torch_waveform,
            ),
            "static_istft_waveform_vs_reference": tensor_metrics(
                reference_waveform,
                linear_waveform,
            ),
            "static_vs_torch_istft": tensor_metrics(
                torch_waveform,
                linear_waveform,
            ),
        }
        write_wav(output_dir / f"{pad_mode}_torch_istft.wav", torch_waveform)
        write_wav(output_dir / f"{pad_mode}_static_istft.wav", linear_waveform)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "report.json"
    report_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(report_file)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
