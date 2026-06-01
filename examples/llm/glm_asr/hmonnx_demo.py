# Copyright 2025 HOUMO AI
#
# File: hmonnx_demo.py
# Description: Demo script for running end-to-end inference with exported GLM-ASR HMONNX models
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import argparse
import json

import torch
import torch.nn as nn
import numpy as np

from pathlib import Path

from xhquant.api import HMONNXInference
from transformers import AutoConfig, AutoProcessor

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class GLMASRInference:
    """End-to-end GLM-ASR HMONNX inference pipeline.

    Pipeline:
        1. processor.apply_transcription_request(audio) -> input_ids, input_features, input_features_mask
        2. Encoder HMONNX (audio_tower + projector): input_features -> audio_embeds (B, T/4, text_hidden)
        3. Feature fusion: replace audio_token positions in text embeddings with audio_embeds
        4. Prefill HMONNX: fused embeddings -> first token logits
        5. Decode HMONNX: autoregressive generation
    """

    def __init__(self, model_dir: str, hf_model: str, device="cuda", encoder_quant_type="w8a8_sefp"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.hf_model = hf_model

        # Load meta info
        meta_file = self.model_dir / "export_meta_info.json"
        with open(meta_file, "r") as f:
            self.meta_info = json.load(f)

        # Load processor (contains tokenizer + feature_extractor) from HF model
        print(f"Loading processor from: {self.hf_model}")
        self.processor = AutoProcessor.from_pretrained(self.hf_model, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        # Load config from HF model
        self.config = AutoConfig.from_pretrained(self.hf_model, trust_remote_code=True)
        self.text_config = self.config.text_config

        # Load encoder HMONNX (includes audio_tower + reshape + multi_modal_projector)
        encoder_work_dir = self.model_dir / "Encoder"
        model_name = Path(self.meta_info.get("hf_model", "")).name
        encoder_hmonnx = encoder_work_dir / "hmonnx" / f"{model_name}_Encoder_xh2a_{encoder_quant_type}.onnx"
        if not encoder_hmonnx.exists():
            encoder_candidates = list((encoder_work_dir / "hmonnx").glob("*_Encoder_*.onnx"))
            if encoder_candidates:
                encoder_hmonnx = encoder_candidates[0]
        print(f"Loading encoder: {encoder_hmonnx}")
        self.encoder_sess = HMONNXInference(str(encoder_hmonnx))
        self.encoder_sess.to(device)

        # Load prefill HMONNX
        prefill_path = self.model_dir / self.meta_info["prefill_onnx_file"]
        print(f"Loading prefill: {prefill_path}")
        self.prefill_sess = HMONNXInference(str(prefill_path))
        self.prefill_sess.to(device)

        # Load decode HMONNX
        decode_path = self.model_dir / self.meta_info["decode_onnx_file"]
        print(f"Loading decode: {decode_path}")
        self.decode_sess = HMONNXInference(str(decode_path))
        self.decode_sess.to(device)

        # Load token embedding
        token_embedding_path = self.model_dir / self.meta_info["token_embedding_file"]
        w = torch.load(token_embedding_path, map_location="cpu", weights_only=True)["weight"]
        self.embed_tokens = nn.Embedding(*w.shape).to(device, dtype=torch.float16).eval()
        self.embed_tokens.weight.data.copy_(w.to(device=device, dtype=torch.float16))

        # KV cache setup from meta info
        self.num_hidden_layers = self.meta_info["num_hidden_layers"]
        self.kv_cache_shape = self.meta_info["kv_cache_shape"]

        # Model dimensions
        self.audio_token_id = getattr(self.config, "audio_token_id", 59260)
        self.eos_token_ids = getattr(self.text_config, "eos_token_id", [59246, 59253, 59255])
        if not isinstance(self.eos_token_ids, list):
            self.eos_token_ids = [self.eos_token_ids]

        self.hidden_size = self.text_config.hidden_size
        self.num_kv_heads = self.text_config.num_key_value_heads
        self.head_dim = getattr(
            self.text_config,
            "head_dim",
            self.hidden_size // self.text_config.num_attention_heads,
        )
        self.cache_len = 2048

    # ------------------------------------------------------------------
    # Audio length computation (mirrors GlmAsrForConditionalGeneration.get_audio_features)
    # ------------------------------------------------------------------
    def _compute_audio_output_length(self, input_features_mask: torch.Tensor) -> torch.Tensor:
        """Compute the valid audio embedding length after conv down-sampling + merge."""
        audio_lengths = input_features_mask.sum(-1)
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
            audio_lengths = (audio_lengths + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        merge_factor = 4
        post_lengths = (audio_lengths - merge_factor) // merge_factor + 1
        return post_lengths

    # ------------------------------------------------------------------
    # Transcribe
    # ------------------------------------------------------------------
    def transcribe(self, audio_input, max_new_tokens: int = 2048) -> str:
        """Transcribe audio to text.

        Args:
            audio_input: file path (str) or raw audio numpy array (float32, 16 kHz).
            max_new_tokens: maximum tokens to generate.

        Returns:
            Transcription text.
        """
        DEVICE = self.device

        # ===================== 1. Preprocessing via processor =====================
        if isinstance(audio_input, str):
            import librosa

            audio_array, _ = librosa.load(audio_input, sr=self.processor.feature_extractor.sampling_rate, mono=True)
        else:
            audio_array = audio_input

        # Split into 30s chunks and transcribe each independently
        sr = self.processor.feature_extractor.sampling_rate
        chunk_size = int(sr * 30.0)  # 30s
        n_samples = len(audio_array)
        n_chunks = max(1, (n_samples + chunk_size - 1) // chunk_size)

        if n_chunks == 1:
            inputs = self.processor.apply_transcription_request(audio_array)
            inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
            return self._run_inference(inputs, max_new_tokens)
        else:
            print(f"Audio {n_samples / sr:.1f}s → splitting into {n_chunks} chunks of 30s each")
            results = []
            for i in range(n_chunks):
                chunk = audio_array[i * chunk_size : (i + 1) * chunk_size]
                print(f"\n--- Chunk {i + 1}/{n_chunks} ({len(chunk) / sr:.1f}s) ---")
                inputs = self.processor.apply_transcription_request(chunk)
                inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
                result = self._run_inference(inputs, max_new_tokens)
                results.append(result)
            return " ".join(filter(None, results))

    def _run_inference(self, inputs: dict, max_new_tokens: int) -> str:
        """Run encoder → prefill → decode on a single-chunk inputs dict (batch=1).

        Args:
            inputs: processor output with input_features (1, mel, 3000),
                    input_features_mask (1, 3000), input_ids (1, seq_len).
            max_new_tokens: maximum tokens to generate.

        Returns:
            Transcription text string.
        """
        DEVICE = self.device

        input_features = inputs["input_features"].float()
        input_features_mask = inputs["input_features_mask"]
        input_ids = inputs["input_ids"]

        # ===================== 2. Audio Encoder =====================
        # Pad input_features to fixed length 3000
        feat_len = input_features.shape[2]
        if feat_len < 3000:
            input_features = torch.nn.functional.pad(input_features, (0, 3000 - feat_len), value=0.0)

        print(f"Encoder input shape: {input_features.shape}")
        audio_embeds = self.encoder_sess.run({"input_features": input_features.half()})
        audio_embeds = audio_embeds.to(DEVICE)
        print(f"Encoder output shape (after projector): {audio_embeds.shape}")

        # Compute valid length after conv down-sampling + merge, then trim
        T_out = self._compute_audio_output_length(input_features_mask).item()
        audio_embeds = audio_embeds[:, :T_out, :]
        if audio_embeds.dim() == 2:
            audio_embeds = audio_embeds.unsqueeze(0)
        print(f"Audio embeddings after trim: {audio_embeds.shape} (T_out={T_out})")

        # ===================== 3. Feature Fusion =====================
        text_embeds = self.embed_tokens(input_ids)

        pad_indices = (input_ids == self.audio_token_id).nonzero(as_tuple=True)[1]
        if len(pad_indices) > 0:
            start_idx = pad_indices[0].item()
            end_idx = pad_indices[-1].item()
            final_inputs_embeds = torch.cat(
                [
                    text_embeds[:, :start_idx, :],
                    audio_embeds.to(text_embeds.dtype),
                    text_embeds[:, end_idx + 1 :, :],
                ],
                dim=1,
            )
        else:
            final_inputs_embeds = text_embeds
        print(f"Fused embeddings shape: {final_inputs_embeds.shape}")

        # ===================== 4. Prefill =====================
        num_layers = self.num_hidden_layers
        max_prefill = 411
        seq_len = final_inputs_embeds.shape[1]
        L = min(seq_len, max_prefill)

        prefill_embeds = torch.zeros((1, max_prefill, self.hidden_size), dtype=torch.float16, device=DEVICE)
        prefill_embeds[:, :L, :] = final_inputs_embeds[:, :L, :].half()

        kcache = [
            torch.zeros((1, self.num_kv_heads, self.cache_len, self.head_dim), dtype=torch.float16, device=DEVICE)
            for _ in range(num_layers)
        ]
        vcache = [
            torch.zeros((1, self.num_kv_heads, self.cache_len, self.head_dim), dtype=torch.float16, device=DEVICE)
            for _ in range(num_layers)
        ]

        valid_length = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        current_length = torch.tensor([L], dtype=torch.int32, device=DEVICE)

        prefill_inputs = {
            "input_embeds": prefill_embeds,
            "valid_length": valid_length,
            "current_length": current_length,
        }
        prefill_input_names = self.prefill_sess.get_input_names()
        for i in range(num_layers):
            k_key = f"model_layers_{i}_self_attn_kcache_input"
            v_key = f"model_layers_{i}_self_attn_vcache_input"
            if k_key in prefill_input_names:
                prefill_inputs[k_key] = kcache[i]
            if v_key in prefill_input_names:
                prefill_inputs[v_key] = vcache[i]

        outputs = self.prefill_sess.run(prefill_inputs)

        next_token_id = torch.argmax(outputs, dim=-1).item()
        generated_ids = [next_token_id]
        print(f"Prefill first token: {next_token_id}")

        # ===================== 5. Decode =====================
        valid_length = torch.tensor([L], dtype=torch.int32, device=DEVICE)
        current_length = torch.tensor([1], dtype=torch.int32, device=DEVICE)

        decode_input_names = self.decode_sess.get_input_names()

        for step in range(max_new_tokens):
            if generated_ids[-1] in self.eos_token_ids:
                break

            token_tensor = torch.tensor([[generated_ids[-1]]], device=DEVICE)
            next_embed = self.embed_tokens(token_tensor).half()

            decode_inputs = {
                "input_embeds": next_embed,
                "valid_length": valid_length,
                "current_length": current_length,
            }
            for i in range(num_layers):
                k_key = f"model_layers_{i}_self_attn_kcache_input"
                v_key = f"model_layers_{i}_self_attn_vcache_input"
                if k_key in decode_input_names:
                    decode_inputs[k_key] = kcache[i]
                if v_key in decode_input_names:
                    decode_inputs[v_key] = vcache[i]

            decode_outputs = self.decode_sess.run(decode_inputs)
            next_id = torch.argmax(decode_outputs, dim=-1).item()
            generated_ids.append(next_id)

            valid_length = valid_length + 1

        # ===================== 6. Decode text =====================
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return result


def main(args):
    print(f"Initializing GLM-ASR inference engine...")
    inference = GLMASRInference(
        model_dir=args.model_dir,
        hf_model=args.hf_model,
        device=args.device,
        encoder_quant_type=args.encoder_quant_type,
    )
    result = inference.transcribe(args.audio, max_new_tokens=args.max_new_tokens)

    # Try to extract <asr_text> content if present
    match = re.search(r"(?<=<asr_text>)[\s\S]*", result)
    if match:
        result = match.group().strip()

    print("\n" + "=" * 60)
    print(f"Transcription Result:")
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GLM-ASR HMONNX Inference Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="glm-asr-nano-2512_XH2a",
        help="Path to exported work_dir (e.g., work_dirs/glm-asr-nano-2512_XH2a)",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default="glm-asr-nano-2512",
        help="HF model path or Hub ID for loading processor/config"
        "(e.g., glm-asr-nano-2512 or zai-org/GLM-ASR-Nano-2512)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=f"{FILE_DIR}/audio.mp3",
        help="Path to audio file (.wav, .mp3, .flac, etc.)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda / cpu)",
    )
    parser.add_argument(
        "--encoder_quant_type",
        type=str,
        default="w8a8_sefp",
        help="Encoder quantization type",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate",
    )
    args = parser.parse_args()
    main(args)
