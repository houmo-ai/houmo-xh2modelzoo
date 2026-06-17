from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import types
from pathlib import Path
from typing import Optional

import onnx
import torch
import torch.nn as nn
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiEuclideanCodebook
from xhquant.api import Config, convert_onnx_to_hmonnx, set_random_seed


class SpeechTokenizerEncodeWrapper(nn.Module):
    """Static tensor wrapper for Qwen3-TTS 12Hz speech_tokenizer.encode."""

    def __init__(self, tokenizer_model: Qwen3TTSTokenizerV2Model):
        super().__init__()
        self.encoder = tokenizer_model.encoder.encoder
        self.encoder_transformer = tokenizer_model.encoder.encoder_transformer
        self.downsample = tokenizer_model.encoder.downsample
        self.quantizer = tokenizer_model.encoder.quantizer
        self.valid_num_quantizers = int(tokenizer_model.encoder_valid_num_quantizers)
        self.encode_downsample_rate = int(tokenizer_model.encode_downsample_rate)
        self.num_quantizers = int(tokenizer_model.encoder.config.num_quantizers)

    def forward(self, input_values: torch.Tensor, padding_mask: torch.Tensor):
        hidden = self.encoder(input_values.unsqueeze(1), padding_cache=None)
        hidden = hidden.transpose(1, 2)

        seq_len = hidden.shape[1]
        device = hidden.device
        cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
        position_ids = cache_position.unsqueeze(0)
        q_idx = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(1)
        kv_idx = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        causal_mask = torch.where(kv_idx <= q_idx, 0.0, -10000.0).unsqueeze(0).unsqueeze(0)

        for layer in self.encoder_transformer.layers:
            hidden = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
            )[0]

        hidden = hidden.transpose(1, 2)
        hidden = self.downsample(hidden, padding_cache=None)
        audio_codes = self.quantizer.encode(hidden, self.num_quantizers)
        audio_codes = audio_codes.transpose(0, 1)[:, : self.valid_num_quantizers].transpose(1, 2).to(torch.int32)
        valid_samples = padding_mask.to(torch.int64).sum(dim=1)
        valid_frames = ((valid_samples + self.encode_downsample_rate - 1) // self.encode_downsample_rate).to(torch.int32)
        return audio_codes, valid_frames


class SpeakerEncoderWrapper(nn.Module):
    """Export the ECAPA speaker_encoder. Input is precomputed mel [B, T, 128]."""

    def __init__(self, speaker_encoder: nn.Module):
        super().__init__()
        self.speaker_encoder = speaker_encoder

    def forward(self, mels: torch.Tensor):
        return self.speaker_encoder(mels)


def _patch_mimi_conv1d_static_padding(module: nn.Module) -> None:
    """Use Python int padding for fixed-shape ONNX export."""

    def static_forward(self, hidden_states, padding_cache=None):
        if not self.causal and padding_cache is not None:
            raise ValueError("`padding_cache` is not supported for non-causal convolutions.")
        if padding_cache is not None:
            layer_padding_cache = padding_cache.update(hidden_states, self.layer_idx)
            hidden_states = torch.cat([layer_padding_cache, hidden_states], dim=2)
            return self.conv(hidden_states)

        length = int(hidden_states.shape[-1])
        kernel_size = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0] + 1
        stride = self.conv.stride[0]
        padding_total = kernel_size - stride
        n_frames = math.ceil((length - kernel_size + padding_total) / stride + 1) - 1
        ideal_length = n_frames * stride + kernel_size - padding_total
        extra_padding = int(ideal_length - length)

        if self.causal:
            paddings = (padding_total, extra_padding)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            paddings = (padding_left, padding_right + extra_padding)

        hidden_states = MimiConv1d._pad1d(hidden_states, paddings, mode=self.pad_mode)
        return self.conv(hidden_states)

    for submodule in module.modules():
        if isinstance(submodule, MimiConv1d):
            submodule.forward = types.MethodType(static_forward, submodule)


def _patch_mimi_codebook_matmul_distance(module: nn.Module) -> None:
    """Replace cdist with an ONNX-friendly squared-distance matmul."""

    def quantize_without_cdist(self, hidden_states):
        hidden_states = hidden_states.float()
        embed = self.embed.float()
        hidden_norm = hidden_states.square().sum(dim=-1, keepdim=True)
        embed_norm = embed.square().sum(dim=-1).unsqueeze(0)
        dists = hidden_norm + embed_norm - 2.0 * hidden_states.matmul(embed.transpose(0, 1))
        return dists.argmin(dim=-1)

    for submodule in module.modules():
        if isinstance(submodule, MimiEuclideanCodebook):
            submodule.quantize = types.MethodType(quantize_without_cdist, submodule)


def _load_cfg(config_path: Optional[str], variant: Optional[str]):
    if config_path is None:
        return None
    cfg = Config.fromfile(config_path)
    if variant:
        from config.llm._components import apply_variant

        apply_variant(cfg, variant)
    return cfg


def _resolve_model_dir(args: argparse.Namespace) -> str:
    if args.hf_model_dir:
        return args.hf_model_dir
    cfg = _load_cfg(args.config, args.variant)
    if cfg is None or not getattr(cfg, "hf_model_dir", None):
        raise ValueError("provide --hf-model-dir or a --config that defines hf_model_dir")
    return cfg.hf_model_dir


def _resolve_target_device(args: argparse.Namespace) -> str:
    if args.target_device:
        return args.target_device
    cfg = _load_cfg(args.config, args.variant)
    if cfg is not None and getattr(cfg, "target_device", None):
        return cfg.target_device
    return "xh2a"


def _resolve_work_dir(args: argparse.Namespace) -> Path:
    if args.work_dir:
        return Path(args.work_dir)
    name = args.name or "qwen3_tts_12hz_0_6B_base_frontend_xh2a"
    return Path("./work_dirs") / name


def _resolve_tokenizer_dir(model_dir: str) -> str:
    speech_tokenizer_dir = Path(model_dir) / "speech_tokenizer"
    return str(speech_tokenizer_dir if speech_tokenizer_dir.exists() else Path(model_dir))


def _export_onnx(model, dummy_inputs, onnx_file: Path, input_names, output_names, opset: int, use_dynamo: bool = True):
    export_kwargs = dict(
        model=model,
        args=dummy_inputs,
        f=str(onnx_file),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
    )
    with torch.no_grad():
        if use_dynamo:
            try:
                torch.onnx.export(**export_kwargs, dynamo=True)
            except Exception as exc:
                print(f"dynamo ONNX export failed for {onnx_file.name}, retry legacy tracer: {exc}")
                torch.onnx.export(**export_kwargs, dynamo=False)
        else:
            torch.onnx.export(**export_kwargs, dynamo=False)
    onnx_model = onnx.load(str(onnx_file))
    onnx.save(onnx_model, str(onnx_file))


def _convert_to_hmonnx(onnx_file: Path, dummy_inputs, target_device: str, hmonnx_file: Path):
    convert_onnx_to_hmonnx(
        str(onnx_file),
        [x.cpu() for x in dummy_inputs],
        target_device,
        str(hmonnx_file),
    )


def main(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)
    model_dir = _resolve_model_dir(args)
    target_device = _resolve_target_device(args)
    work_dir = _resolve_work_dir(args)
    if work_dir.exists() and any(work_dir.iterdir()):
        if not args.force:
            raise FileExistsError(f"work_dir already exists and is not empty: {work_dir}; pass --force to overwrite")
        shutil.rmtree(work_dir)

    onnx_dir = work_dir / "onnx"
    hmonnx_dir = work_dir / "hmonnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = _resolve_tokenizer_dir(model_dir)
    tokenizer_model = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_dir).float().cpu().eval()
    tokenizer_model.config._attn_implementation = "eager"
    tokenizer_model.encoder.config._attn_implementation = "eager"
    tokenizer_model.encoder.encoder_transformer.config._attn_implementation = "eager"
    _patch_mimi_conv1d_static_padding(tokenizer_model.encoder)
    _patch_mimi_codebook_matmul_distance(tokenizer_model.encoder.quantizer)
    encode_model = SpeechTokenizerEncodeWrapper(tokenizer_model).float().cpu().eval()

    input_values = torch.zeros(args.batch_size, args.audio_samples, dtype=torch.float32)
    padding_mask = torch.ones(args.batch_size, args.audio_samples, dtype=torch.int32)
    encode_inputs = (input_values, padding_mask)
    encode_onnx = onnx_dir / "speech_tokenizer_encode.onnx"
    encode_hmonnx = hmonnx_dir / f"speech_tokenizer_encode_{target_device}.onnx"
    _export_onnx(
        encode_model,
        encode_inputs,
        encode_onnx,
        ["input_values", "padding_mask"],
        ["audio_codes", "valid_frames"],
        args.opset,
        use_dynamo=False,
    )
    if not args.onnx_only:
        _convert_to_hmonnx(encode_onnx, encode_inputs, target_device, encode_hmonnx)
        if args.golden:
            from _golden import run_hmonnx_golden

            encode_golden_dir = work_dir / "golden" / "speech_tokenizer_encode"
            run_hmonnx_golden(encode_hmonnx, encode_golden_dir, tuple(x.cpu() for x in encode_inputs), args.golden_device)

    tts_model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).model.float().cpu().eval()
    if tts_model.speaker_encoder is None:
        raise ValueError(f"model does not have speaker_encoder: {model_dir}")
    speaker_model = SpeakerEncoderWrapper(tts_model.speaker_encoder).float().cpu().eval()
    mels = torch.zeros(args.batch_size, args.mel_frames, args.mel_dim, dtype=torch.float32)
    speaker_inputs = (mels,)
    speaker_onnx = onnx_dir / "speaker_encoder.onnx"
    speaker_hmonnx = hmonnx_dir / f"speaker_encoder_{target_device}.onnx"
    _export_onnx(
        speaker_model,
        speaker_inputs,
        speaker_onnx,
        ["mels"],
        ["speaker_embedding"],
        args.opset,
    )
    if not args.onnx_only:
        _convert_to_hmonnx(speaker_onnx, speaker_inputs, target_device, speaker_hmonnx)
        if args.golden:
            from _golden import run_hmonnx_golden

            speaker_golden_dir = work_dir / "golden" / "speaker_encoder"
            run_hmonnx_golden(speaker_hmonnx, speaker_golden_dir, tuple(x.cpu() for x in speaker_inputs), args.golden_device)

    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hf_model": model_dir,
        "tokenizer_dir": tokenizer_dir,
        "target_device": target_device,
        "speech_tokenizer_encode_onnx": str(encode_onnx.relative_to(work_dir)),
        "speech_tokenizer_encode_hmonnx": str(encode_hmonnx.relative_to(work_dir)),
        "speaker_encoder_onnx": str(speaker_onnx.relative_to(work_dir)),
        "speaker_encoder_hmonnx": str(speaker_hmonnx.relative_to(work_dir)),
        "batch_size": int(args.batch_size),
        "audio_samples": int(args.audio_samples),
        "input_sample_rate": int(tokenizer_model.input_sample_rate),
        "encode_downsample_rate": int(tokenizer_model.encode_downsample_rate),
        "encoder_valid_num_quantizers": int(tokenizer_model.encoder_valid_num_quantizers),
        "mel_frames": int(args.mel_frames),
        "mel_dim": int(args.mel_dim),
        "speaker_encoder_sample_rate": int(tts_model.speaker_encoder_sample_rate),
        "interfaces": {
            "speech_tokenizer_encode": {
                "inputs": ["input_values: float32[B, audio_samples]", "padding_mask: int32[B, audio_samples]"],
                "outputs": ["audio_codes: int32[B, T, 16]", "valid_frames: int32[B]"],
                "note": "Crop audio_codes by valid_frames. valid_frames = ceil(valid_samples / encode_downsample_rate).",
            },
            "speaker_encoder": {
                "inputs": ["mels: float32[B, mel_frames, mel_dim]"],
                "outputs": ["speaker_embedding: float32[B, 1024]"],
                "note": "mels should match qwen_tts.core.models.modeling_qwen3_tts.mel_spectrogram(...).transpose(1, 2).",
            },
        },
    }
    if args.golden and not args.onnx_only:
        meta["speech_tokenizer_encode_golden_dir"] = str(encode_golden_dir.relative_to(work_dir))
        meta["speaker_encoder_golden_dir"] = str(speaker_golden_dir.relative_to(work_dir))
    if args.onnx_only:
        meta["onnx_only"] = True
    with open(work_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(f"speech_tokenizer.encode ONNX saved to: {encode_onnx}")
    if not args.onnx_only:
        print(f"speech_tokenizer.encode HMONNX saved to: {encode_hmonnx}")
    print(f"speaker_encoder ONNX saved to: {speaker_onnx}")
    if not args.onnx_only:
        print(f"speaker_encoder HMONNX saved to: {speaker_hmonnx}")
    print(f"meta saved to: {work_dir / 'meta.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="./config/llm/qwen3_tts_12hz_speech_tokenizer_xh2a.py")
    parser.add_argument("--variant", choices=["0_6B_base", "0_6B_customvoice", "1_7B_voicedesign"], default="0_6B_base")
    parser.add_argument("--hf-model-dir", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--target-device", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-samples", type=int, default=101760)
    parser.add_argument("--mel-frames", type=int, default=400)
    parser.add_argument("--mel-dim", type=int, default=128)
    parser.add_argument("--golden", action="store_true", help="export hmonnx golden")
    parser.add_argument("--golden-device", type=str, default="cuda", help="device for golden inference")
    main(parser.parse_args())
