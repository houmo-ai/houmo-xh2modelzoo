from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from xhmodel_merak.xh_other_model.models.zipformer.runtime import (
    HmonnxRunner,
    OrtRunner,
    StreamingEncoder,
    error_stats,
    feature_windows,
    make_features,
)


class TransducerHead:
    def __init__(self, decoder: Path, joiner: Path):
        self.decoder = ort.InferenceSession(str(decoder), providers=["CPUExecutionProvider"])
        self.joiner = ort.InferenceSession(str(joiner), providers=["CPUExecutionProvider"])
        metadata = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(metadata["context_size"])
        self.tokens = [0] * self.context_size
        self.decoder_out = self._run_decoder()

    def _run_decoder(self) -> np.ndarray:
        value = np.asarray([self.tokens[-self.context_size :]], dtype=np.int64)
        return self.decoder.run(None, {"y": value})[0]

    def decode(self, encoder_out: np.ndarray) -> None:
        for frame in encoder_out[0]:
            logits = self.joiner.run(
                None,
                {
                    "encoder_out": frame.reshape(1, -1).astype(np.float32),
                    "decoder_out": self.decoder_out,
                },
            )[0]
            token = int(np.argmax(logits[0]))
            if token != 0:
                self.tokens.append(token)
                self.decoder_out = self._run_decoder()

    @property
    def result_tokens(self) -> list[int]:
        return self.tokens[self.context_size :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate source/static/HMONNX Zipformer on real audio")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--audio")
    parser.add_argument(
        "--expected-text",
        default="对我做了介绍那么我想说的是大家如果对我的研究感兴趣",
    )
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def load_tokens(path: Path) -> dict[int, str]:
    table: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        symbol, index = line.rsplit(maxsplit=1)
        table[int(index)] = symbol
    return table


def text_from_tokens(tokens: list[int], table: dict[int, str]) -> str:
    return "".join(table[index] for index in tokens).replace("▁", " ").strip()


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, observed in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != observed),
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> dict[str, float | int]:
    edits = edit_distance(reference, hypothesis)
    return {
        "edits": edits,
        "reference_characters": len(reference),
        "cer": float(edits / max(len(reference), 1)),
    }


def summarize(
    metrics: list[dict[str, float]],
) -> dict[str, float | list[dict[str, float]]]:
    return {
        "per_chunk": metrics,
        "worst_max_abs": max(item["max_abs"] for item in metrics),
        "worst_mse": max(item["mse"] for item in metrics),
        "minimum_cosine": min(item["cosine"] for item in metrics),
        "minimum_snr_db": min(item["snr_db"] for item in metrics),
    }


def official_text(package: Path, waveform: np.ndarray, sample_rate: int) -> str:
    import sherpa_onnx

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(package / "encoder-epoch-99-avg-1.onnx"),
        decoder=str(package / "decoder-epoch-99-avg-1.onnx"),
        joiner=str(package / "joiner-epoch-99-avg-1.onnx"),
        tokens=str(package / "tokens.txt"),
        num_threads=1,
        decoding_method="greedy_search",
        provider="cpu",
    )
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, waveform)
    stream.accept_waveform(sample_rate, np.zeros(int(sample_rate * 0.3), dtype=np.float32))
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    return str(recognizer.get_result(stream))


def main() -> None:
    args = parse_args()
    package = Path(args.model_dir).expanduser().resolve()
    export_dir = Path(args.export_dir).expanduser().resolve()
    metadata = json.loads((export_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    source_path = package / "encoder-epoch-99-avg-1.onnx"
    static_path = export_dir / metadata["onnx_file"]
    hmonnx_path = export_dir / metadata["hmonnx_file"]
    audio_path = Path(args.audio).expanduser().resolve() if args.audio else package / "test_wavs/0.wav"
    waveform, sample_rate = sf.read(audio_path, dtype="float32")
    if waveform.ndim != 1:
        waveform = waveform.mean(axis=1)
    windows = feature_windows(make_features(waveform, sample_rate))
    if args.max_chunks is not None:
        windows = windows[: args.max_chunks]

    encoders = {
        "source_onnx": StreamingEncoder(OrtRunner(source_path)),
        "static_onnx": StreamingEncoder(OrtRunner(static_path)),
        "hmonnx": StreamingEncoder(HmonnxRunner(static_path, hmonnx_path)),
    }
    decoder = package / "decoder-epoch-99-avg-1.onnx"
    joiner = package / "joiner-epoch-99-avg-1.onnx"
    heads = {name: TransducerHead(decoder, joiner) for name in encoders}
    static_metrics = []
    hmonnx_metrics = []
    for window in windows:
        outputs = {name: encoder.run(window) for name, encoder in encoders.items()}
        for name, head in heads.items():
            head.decode(outputs[name])
        static_metrics.append(error_stats(outputs["source_onnx"], outputs["static_onnx"]))
        hmonnx_metrics.append(error_stats(outputs["source_onnx"], outputs["hmonnx"]))

    token_table = load_tokens(package / "tokens.txt")
    transcripts = {name: text_from_tokens(head.result_tokens, token_table) for name, head in heads.items()}
    sherpa_text = official_text(package, waveform, sample_rate)
    result = {
        "model_dir": str(package),
        "audio": str(audio_path),
        "sample_rate": sample_rate,
        "audio_seconds": float(waveform.size / sample_rate),
        "processed_chunks": len(windows),
        "chunk_shape": [1, 39, 80],
        "chunk_shift_frames": 32,
        "official_sherpa_onnx_text": sherpa_text,
        "transcripts": transcripts,
        "expected_text": args.expected_text,
        "cer": {
            "official_sherpa_onnx": cer(args.expected_text, sherpa_text),
            **{name: cer(args.expected_text, text) for name, text in transcripts.items()},
        },
        "transcript_exact_vs_source": {
            name: text == transcripts["source_onnx"] for name, text in transcripts.items() if name != "source_onnx"
        },
        "source_vs_static_encoder": summarize(static_metrics),
        "source_vs_hmonnx_encoder": summarize(hmonnx_metrics),
    }
    output = Path(args.output).expanduser().resolve() if args.output else export_dir / "real_audio_eval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
