from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import jieba
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from xhmodel_merak.xh_other_model.models.melotts.assets import (
    resolve_model_assets,
)
from xhmodel_merak.xh_other_model.models.melotts.modeling import (
    load_official_model,
)
from xhmodel_merak.xh_other_model.models.melotts.runtime import (
    HmonnxRunner,
    MeloTTSRuntime,
    OrtRunner,
    error_stats,
)


class Lexicon:
    def __init__(self, lexicon_path: Path, tokens_path: Path):
        symbols = {}
        for line in tokens_path.read_text(encoding="utf-8").splitlines():
            symbol, index = line.split()
            symbols[symbol] = int(index)
        self.entries = {}
        for line in lexicon_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            word = fields[0]
            values = fields[1:]
            if len(values) % 2:
                raise ValueError(f"invalid lexicon entry: {word}")
            half = len(values) // 2
            self.entries[word] = (
                [symbols[value] for value in values[:half]],
                [int(value) for value in values[half:]],
            )
        self.entries["呣"] = self.entries["母"]
        self.entries["嗯"] = self.entries["恩"]
        for symbol in ("!", "?", "…", ",", ".", "'", "-"):
            self.entries[symbol] = ([symbols[symbol]], [0])
        self.entries[" "] = ([symbols["_"]], [0])

    def _lookup(self, word: str) -> tuple[list[int], list[int]]:
        word = {"，": ",", "。": ".", "！": "!", "？": "?"}.get(word, word)
        if word in self.entries:
            return self.entries[word]
        phones: list[int] = []
        tones: list[int] = []
        if len(word) > 1:
            for character in word:
                part_phones, part_tones = self._lookup(character)
                phones.extend(part_phones)
                tones.extend(part_tones)
        return phones, tones

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        words = list(jieba.cut(text, HMM=True))
        phones: list[int] = []
        tones: list[int] = []
        for word in words:
            part_phones, part_tones = self._lookup(word)
            phones.extend(part_phones)
            tones.extend(part_tones)
        blank_phones = [0] * (2 * len(phones) + 1)
        blank_tones = [0] * (2 * len(tones) + 1)
        blank_phones[1::2] = phones
        blank_tones[1::2] = tones
        return (
            np.asarray(blank_phones, dtype=np.int64),
            np.asarray(blank_tones, dtype=np.int64),
            words,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate official/static/HMONNX MeloTTS on real text")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--text", default="你好。")
    parser.add_argument("--output")
    parser.add_argument("--wav-dir")
    return parser.parse_args()


def dynamic_pytorch(
    model: torch.nn.Module,
    values: np.ndarray,
    tones: np.ndarray,
) -> np.ndarray:
    x = torch.from_numpy(values.reshape(1, -1)).long()
    tone = torch.from_numpy(tones.reshape(1, -1)).long()
    length = torch.tensor([values.size]).long()
    language = torch.zeros_like(x)
    language[:, 1::2] = 3
    sid = torch.tensor([1]).long()
    with torch.no_grad():
        waveform = model.infer(
            x=x,
            x_lengths=length,
            sid=sid,
            tone=tone,
            language=language,
            bert=torch.zeros(1, 1024, values.size),
            ja_bert=torch.zeros(1, 768, values.size),
            noise_scale=0.0,
            noise_scale_w=0.8,
            length_scale=1.0,
        )[0]
    return waveform.detach().cpu().numpy().reshape(-1)


def release_onnx(path: Path, values: np.ndarray, tones: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(
        ["y"],
        {
            "x": values.reshape(1, -1),
            "x_lengths": np.asarray([values.size], dtype=np.int64),
            "tones": tones.reshape(1, -1),
            "sid": np.asarray([1], dtype=np.int64),
            "noise_scale": np.asarray([0.0], dtype=np.float32),
            "length_scale": np.asarray([1.0], dtype=np.float32),
            "noise_scale_w": np.asarray([0.8], dtype=np.float32),
        },
    )[0].reshape(-1)


def spectral_metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    ref = torch.from_numpy(reference.astype(np.float32).reshape(-1))
    got = torch.from_numpy(actual.astype(np.float32).reshape(-1))
    window = torch.hann_window(1024)
    ref_spec = torch.stft(ref, 1024, hop_length=256, window=window, return_complex=True).abs()
    got_spec = torch.stft(got, 1024, hop_length=256, window=window, return_complex=True).abs()
    ref_db = 20 * torch.log10(ref_spec.clamp_min(1e-7))
    got_db = 20 * torch.log10(got_spec.clamp_min(1e-7))
    delta = ref_db - got_db
    active = ref_db >= ref_db.max() - 60
    active_delta = delta[active]
    return {
        "log_spectral_all_bin_mae_db": float(delta.abs().mean()),
        "log_spectral_active_mae_db": float(active_delta.abs().mean()),
        "log_spectral_active_rmse_db": float(torch.sqrt((active_delta * active_delta).mean())),
        "active_bin_ratio": float(active.float().mean()),
    }


def compare(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    if reference.shape != actual.shape:
        raise ValueError(f"waveform shape mismatch: {reference.shape} vs {actual.shape}")
    result: dict[str, float | int] = error_stats(reference, actual)
    result.update(spectral_metrics(reference, actual))
    result["samples"] = int(reference.size)
    return result


def main() -> None:
    args = parse_args()
    assets = resolve_model_assets(args.model_dir)
    export_dir = Path(args.export_dir).expanduser().resolve()
    metadata = json.loads((export_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    lexicon = Lexicon(
        assets.release_package / "lexicon.txt",
        assets.release_package / "tokens.txt",
    )
    values, tones, words = lexicon.encode(args.text)
    lmax = int(metadata["text_max_length"])
    if values.size > lmax:
        raise ValueError(f"text uses {values.size} tokens, exceeds Lmax={lmax}")
    encoder_onnx = export_dir / metadata["components"]["encoder"]["onnx_file"]
    decoder_onnx = export_dir / metadata["components"]["decoder"]["onnx_file"]
    encoder_hm = export_dir / metadata["components"]["encoder"]["hmonnx_file"]
    decoder_hm = export_dir / metadata["components"]["decoder"]["hmonnx_file"]
    static_runtime = MeloTTSRuntime(
        OrtRunner(encoder_onnx),
        OrtRunner(decoder_onnx),
        text_max_length=lmax,
        acoustic_max_length=int(metadata["acoustic_max_length"]),
    )
    hmonnx_runtime = MeloTTSRuntime(
        HmonnxRunner(encoder_hm),
        HmonnxRunner(decoder_hm),
        text_max_length=lmax,
        acoustic_max_length=int(metadata["acoustic_max_length"]),
    )
    static_wave, static_info = static_runtime.synthesize(values, tones)
    hmonnx_wave, hmonnx_info = hmonnx_runtime.synthesize(values, tones)
    if static_info["durations"] != hmonnx_info["durations"]:
        raise AssertionError("W16 HMONNX changed ceil durations")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    official_model, _ = load_official_model(assets.source_root, assets.config, assets.checkpoint)
    pytorch_wave = dynamic_pytorch(official_model, values, tones)
    official_onnx_wave = release_onnx(assets.release_package / "model.onnx", values, tones)
    waves = {
        "pytorch": pytorch_wave,
        "release_onnx": official_onnx_wave,
        "static_onnx": static_wave,
        "hmonnx": hmonnx_wave,
    }
    wav_dir = Path(args.wav_dir).expanduser().resolve() if args.wav_dir else export_dir / "accuracy_audio"
    wav_dir.mkdir(parents=True, exist_ok=True)
    for name, waveform in waves.items():
        sf.write(
            wav_dir / f"melotts_{name}.wav",
            waveform.astype(np.float32),
            44100,
        )
    result = {
        "provenance": {
            "melo_git_commit": subprocess.check_output(
                ["git", "-C", str(assets.source_root), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "release_model_sha256": metadata["release_model_sha256"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "config_sha256": metadata["config_sha256"],
        },
        "input": {
            "text": args.text,
            "jieba_words": words,
            "token_length": int(values.size),
            "tokens": values.tolist(),
            "tones": tones.tolist(),
            "lmax": lmax,
            "acoustic_length": static_info["acoustic_length"],
            "tmax": metadata["acoustic_max_length"],
            "valid_samples": static_info["valid_samples"],
            "duration_exact": True,
        },
        "release_onnx_vs_pytorch": compare(pytorch_wave, official_onnx_wave),
        "static_onnx_vs_pytorch": compare(pytorch_wave, static_wave),
        "hmonnx_vs_pytorch": compare(pytorch_wave, hmonnx_wave),
        "hmonnx_vs_static_onnx": compare(static_wave, hmonnx_wave),
        "wav_dir": str(wav_dir),
    }
    output = Path(args.output).expanduser().resolve() if args.output else export_dir / "real_text_eval.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
