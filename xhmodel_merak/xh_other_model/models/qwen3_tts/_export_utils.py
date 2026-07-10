import math
import types
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiEuclideanCodebook


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


def patch_mimi_conv1d_static_padding(module: nn.Module) -> None:
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


def patch_mimi_codebook_matmul_distance(module: nn.Module) -> None:
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


def export_onnx(model, dummy_inputs, onnx_file: Path, input_names, output_names, opset: int, use_dynamo: bool = True):
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
        gather_indices = target_indices.unsqueeze(0).unsqueeze(0).expand(batch, channels, -1)
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


def make_stateful_decoder_dummy_inputs(wrapper: StatefulDecoderDynamoCombined, num_heads: int, head_dim: int, args):
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
    return (
        audio_codes,
        pre_conv_history,
        latent_buffer,
        conv_history,
        is_last,
        kv_valid_len_tensor,
        valid_frames_tensor,
        *kv,
    )


def stateful_decoder_input_output_names(num_layers: int):
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
