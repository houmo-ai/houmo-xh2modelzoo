# Export an HMONNX-friendly stateful Qwen3-TTS speech tokenizer decoder.
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Optional

import onnx
import torch
import torch.nn as nn
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
from xhquant.api import Config, convert_onnx_to_hmonnx, set_random_seed
from xhquant.export.onnx.transforms import hmonnx_transforms


class DecoderPart1PreConv(nn.Module):
    def __init__(self, decoder_model, chunk_size: int):
        super().__init__()
        self.quantizer = decoder_model.quantizer
        self.pre_conv = decoder_model.pre_conv
        self.chunk_size = int(chunk_size)
        self.pre_conv_history_window = 2

    def forward(self, audio_codes: torch.Tensor, pre_conv_history: torch.Tensor):
        codes = audio_codes.to(torch.long).transpose(1, 2)
        quantized = self.quantizer.decode(codes)
        quant_full = torch.cat([pre_conv_history, quantized], dim=-1)
        hidden_all = self.pre_conv(quant_full)
        hidden = hidden_all[:, :, -self.chunk_size :].transpose(1, 2)
        next_pre_conv_hist = quant_full[:, :, -self.pre_conv_history_window :]
        return hidden, next_pre_conv_hist


class FixedKVStack:
    def __init__(self, keys, values, window_size: int):
        self.key_cache = keys
        self.value_cache = values
        self.window_size = int(window_size)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k_combined = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
        v_combined = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        self.key_cache[layer_idx] = k_combined[:, :, -self.window_size :, :]
        self.value_cache[layer_idx] = v_combined[:, :, -self.window_size :, :]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.window_size

    def __len__(self):
        return len(self.key_cache)


class DecoderPart2Transformer(nn.Module):
    def __init__(self, decoder_model, chunk_size: int):
        super().__init__()
        self.trans = decoder_model.pre_transformer
        self.num_layers = int(self.trans.config.num_hidden_layers)
        self.window_size = int(getattr(self.trans.config, "sliding_window", 72) or 72)
        self.chunk_size = int(chunk_size)

    def forward(self, hidden, kv_valid_len: torch.Tensor, *past_kv_flat):
        device = hidden.device
        keys_in = list(past_kv_flat[: self.num_layers])
        values_in = list(past_kv_flat[self.num_layers :])
        kv_stack = FixedKVStack(keys_in, values_in, self.window_size)

        past_len = kv_valid_len.to(torch.long).reshape(1)
        hidden = self.trans.input_proj(hidden)
        frame_idx = torch.arange(self.chunk_size, device=device, dtype=torch.long)
        position_ids = (past_len + frame_idx).unsqueeze(0)
        position_embeddings = self.trans.rotary_emb(hidden, position_ids)

        query_pos = (past_len + frame_idx).unsqueeze(1)
        key_idx = torch.arange(self.window_size, device=device, dtype=torch.long).unsqueeze(0)
        key_pos = past_len + self.chunk_size - self.window_size + key_idx
        mask_cond = (key_pos >= 0) & (key_pos <= query_pos) & (key_pos > query_pos - self.window_size)
        attention_mask = torch.where(mask_cond, 0.0, -10000.0).unsqueeze(0).unsqueeze(0)

        for layer in self.trans.layers:
            layer_out = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=kv_stack,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
            hidden = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

        hidden = self.trans.norm(hidden)
        new_hidden = self.trans.output_proj(hidden).transpose(1, 2)
        return (new_hidden,) + tuple(kv_stack.key_cache) + tuple(kv_stack.value_cache)


class DecoderPart3Upsample(nn.Module):
    def __init__(self, decoder_model, chunk_size: int):
        super().__init__()
        self.upsample = decoder_model.upsample
        self.decoder = decoder_model.decoder
        self.samples_per_frame = int(decoder_model.total_upsample)
        self.lookahead_frames = 4
        self.conv_history_window = 4
        self.chunk_size = int(chunk_size)

    def forward(
        self,
        new_hidden: torch.Tensor,
        latent_buffer: torch.Tensor,
        conv_history: torch.Tensor,
        is_last: torch.Tensor,
        kv_valid_len: torch.Tensor,
        valid_frames: torch.Tensor,
    ):
        device = new_hidden.device
        accumulated = torch.cat([latent_buffer, new_hidden], dim=-1)

        has_history = (kv_valid_len.to(torch.float32) > 0).to(torch.float32).view(1)
        latent_valid = has_history * float(self.lookahead_frames)
        valid_frames_f = valid_frames.to(torch.float32).view(1)
        total_valid = latent_valid + valid_frames_f
        lookahead = torch.full_like(total_valid, float(self.lookahead_frames))
        is_last_f = is_last.to(torch.float32).view(1)
        num_finalize_f = is_last_f * total_valid + (1.0 - is_last_f) * torch.clamp(
            total_valid - lookahead, min=0.0
        )
        num_finalize = num_finalize_f.to(torch.long)
        num_finalize_idx = num_finalize[0]

        curr = torch.cat([conv_history, accumulated], dim=-1)
        for blocks in self.upsample:
            for block in blocks:
                curr = block(curr)
        for block in self.decoder:
            curr = block(curr)
        wav = curr.squeeze(1).clamp(min=-1, max=1)

        start_samples_idx = self.conv_history_window * self.samples_per_frame
        valid_samples = (num_finalize * self.samples_per_frame).view(1)
        final_wav = wav[:, start_samples_idx:]

        next_latent_buf = accumulated[:, :, -self.lookahead_frames :]
        batch, channels = accumulated.size(0), accumulated.size(1)
        indices = torch.arange(self.conv_history_window, device=device, dtype=torch.long)
        latent_pad = (1.0 - has_history).to(torch.long).view(1) * self.lookahead_frames
        target_indices = latent_pad + (num_finalize_idx - self.conv_history_window) + indices
        gather_indices = torch.clamp(target_indices, min=0).unsqueeze(0).unsqueeze(0).expand(batch, channels, -1)
        next_conv_hist = torch.gather(accumulated, 2, gather_indices)
        return final_wav, valid_samples, next_latent_buf, next_conv_hist


class StatefulDecoderDynamoCombined(nn.Module):
    def __init__(self, decoder_model, chunk_size: int = 12):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.part1 = DecoderPart1PreConv(decoder_model, self.chunk_size)
        self.part2 = DecoderPart2Transformer(decoder_model, self.chunk_size)
        self.part3 = DecoderPart3Upsample(decoder_model, self.chunk_size)
        self.num_layers = self.part2.num_layers
        self.kv_cache_window = self.part2.window_size
        self.samples_per_frame = self.part3.samples_per_frame

    def forward(
        self,
        audio_codes: torch.Tensor,
        pre_conv_history: torch.Tensor,
        latent_buffer: torch.Tensor,
        conv_history: torch.Tensor,
        is_last: torch.Tensor,
        kv_valid_len: torch.Tensor,
        valid_frames: torch.Tensor,
        *past_kv_flat,
    ):
        hidden, next_pre_conv_hist = self.part1(audio_codes, pre_conv_history)
        trans_outputs = self.part2(hidden, kv_valid_len, *past_kv_flat)
        new_hidden = trans_outputs[0]
        next_kv_flat = trans_outputs[1:]
        final_wav, valid_samples, next_latent_buf, next_conv_hist = self.part3(
            new_hidden, latent_buffer, conv_history, is_last, kv_valid_len, valid_frames
        )
        return (
            final_wav,
            valid_samples,
            next_pre_conv_hist,
            next_latent_buf,
            next_conv_hist,
            *next_kv_flat,
        )


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
    name = args.name or "qwen3_tts_stateful_decoder_xh2a"
    return Path("./work_dirs") / name


def _resolve_tokenizer_dir(model_dir: str) -> str:
    speech_tokenizer_dir = Path(model_dir) / "speech_tokenizer"
    return str(speech_tokenizer_dir if speech_tokenizer_dir.exists() else Path(model_dir))


def _make_dummy_inputs(wrapper: StatefulDecoderDynamoCombined, num_heads: int, head_dim: int, args):
    batch = int(args.dummy_batch)
    frames = int(args.chunk_size)
    kv_valid_len = max(0, min(int(args.dummy_history), int(wrapper.kv_cache_window)))
    valid_frames = max(1, min(int(args.dummy_valid_frames), frames))

    audio_codes = torch.zeros(batch, frames, 16, dtype=torch.int32)
    pre_conv_history = torch.zeros(batch, 512, 2, dtype=torch.float32)
    latent_buffer = torch.zeros(batch, 1024, 4, dtype=torch.float32)
    conv_history = torch.zeros(batch, 1024, 4, dtype=torch.float32)
    is_last = torch.tensor([0.0], dtype=torch.float32)
    kv_valid_len_tensor = torch.tensor([kv_valid_len], dtype=torch.int32)
    valid_frames_tensor = torch.tensor([valid_frames], dtype=torch.int32)
    kv = [
        torch.zeros(batch, num_heads, wrapper.kv_cache_window, head_dim, dtype=torch.float32)
        for _ in range(wrapper.num_layers * 2)
    ]
    return (audio_codes, pre_conv_history, latent_buffer, conv_history, is_last, kv_valid_len_tensor, valid_frames_tensor, *kv)


def _make_dynamic_shapes(num_layers: int):
    batch = torch.export.Dim("batch", min=1, max=8)
    return (
        {0: batch},
        {0: batch},
        {0: batch},
        {0: batch},
        None,
        None,
        None,
        tuple([{0: batch}] * (num_layers * 2)),
    )


def _input_output_names(num_layers: int):
    input_names = [
        "audio_codes",
        "pre_conv_history",
        "latent_buffer",
        "conv_history",
        "is_last",
        "kv_valid_len",
        "valid_frames",
    ]
    output_names = [
        "final_wav",
        "valid_samples",
        "next_pre_conv_history",
        "next_latent_buffer",
        "next_conv_history",
    ]
    input_names.extend([f"past_key_{i}" for i in range(num_layers)])
    input_names.extend([f"past_value_{i}" for i in range(num_layers)])
    output_names.extend([f"next_key_{i}" for i in range(num_layers)])
    output_names.extend([f"next_value_{i}" for i in range(num_layers)])
    return input_names, output_names


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
    model = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_dir).float().cpu().eval()
    if hasattr(model.config, "decoder_config"):
        model.config.decoder_config._attn_implementation = "eager"
        model.config.decoder_config.head_dim = args.head_dim
    if hasattr(model.decoder.pre_transformer, "config"):
        model.decoder.pre_transformer.config._attn_implementation = "eager"
        model.decoder.pre_transformer.config.head_dim = args.head_dim

    wrapper = StatefulDecoderDynamoCombined(model.decoder, chunk_size=args.chunk_size).float().cpu().eval()
    cfg = model.decoder.config
    num_layers = int(wrapper.num_layers)
    num_heads = int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 16)))
    head_dim = int(getattr(cfg, "head_dim", args.head_dim))
    dummy_inputs = _make_dummy_inputs(wrapper, num_heads, head_dim, args)
    input_names, output_names = _input_output_names(num_layers)

    onnx_file = onnx_dir / "qwen3_tts_decoder_stateful_static.onnx"
    export_kwargs = dict(
        model=wrapper,
        args=dummy_inputs,
        f=str(onnx_file),
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
        dynamo=True,
    )
    if args.dynamic_batch:
        export_kwargs["dynamic_shapes"] = _make_dynamic_shapes(num_layers)
    with torch.no_grad():
        torch.onnx.export(**export_kwargs)

    onnx_model = onnx.load(str(onnx_file))
    hmonnx_transforms(onnx_model)
    onnx.save(onnx_model, str(onnx_file))

    hmonnx_file = hmonnx_dir / f"qwen3_tts_decoder_stateful_static_{target_device}.onnx"
    if not args.onnx_only:
        convert_onnx_to_hmonnx(
            str(onnx_file),
            [x.cpu() for x in dummy_inputs],
            target_device,
            str(hmonnx_file),
        )

    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hf_model": model_dir,
        "tokenizer_dir": tokenizer_dir,
        "target_device": target_device,
        "stateful_onnx": str(onnx_file.relative_to(work_dir)),
        "stateful_hmonnx": str(hmonnx_file.relative_to(work_dir)),
        "stateful_static_buffers": True,
        "stateful_num_layers": num_layers,
        "stateful_num_heads": num_heads,
        "stateful_head_dim": head_dim,
        "stateful_kv_cache_window": int(wrapper.kv_cache_window),
        "stateful_chunk_size": int(args.chunk_size),
        "stateful_samples_per_frame": int(wrapper.samples_per_frame),
        "stateful_initial_output_skip_frames": int(wrapper.part3.lookahead_frames),
        "stateful_dynamic_batch": bool(args.dynamic_batch),
        "input_names": input_names,
        "output_names": output_names,
    }

    if args.golden and not args.onnx_only:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _golden import run_hmonnx_golden

        golden_dir = work_dir / "golden" / "stateful_decoder"
        run_hmonnx_golden(str(hmonnx_file), golden_dir, tuple(x.cpu() for x in dummy_inputs), args.golden_device)
        meta["stateful_golden_dir"] = str(golden_dir.relative_to(work_dir))

    with open(work_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(f"stateful decoder ONNX saved to: {onnx_file}")
    if not args.onnx_only:
        print(f"stateful decoder HMONNX saved to: {hmonnx_file}")
    print(f"meta saved to: {work_dir / 'meta.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="./config/llm/qwen3_tts_12hz_speech_tokenizer_xh2a.py")
    parser.add_argument("--variant", choices=["0_6B_base", "0_6B_customvoice", "1_7B_voicedesign"], default=None)
    parser.add_argument("--hf-model-dir", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--target-device", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument("--dynamic-batch", action="store_true", help="experimental: keep batch axis dynamic")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--golden-device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--dummy-batch", type=int, default=1)
    parser.add_argument("--dummy-history", type=int, default=0)
    parser.add_argument("--dummy-valid-frames", type=int, default=12)
    main(parser.parse_args())
