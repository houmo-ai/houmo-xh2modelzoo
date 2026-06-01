import re
import os
import math
import torch
import argparse
import librosa

import numpy as np
import torch.nn as nn

from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
from xhquant.api import HMONNXInference as InferenceEngine

from qwen_asr.core.transformers_backend import (
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """
    # 8 = [100, ... 100]
    input_lengths_leave = input_lengths % 100 # [0, 0, ..., 0]
    feat_lengths = (input_lengths_leave - 1) // 2 + 1 # 0
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    # output_lengths: tensor([13, 13, 13, 13, 13, 13, 13, 13], device='cuda:0')
    return output_lengths

def main(args):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    AUDIO_PATH = "./dsj_20251212.wav"
    # AUDIO_PATH = "/LibriSpeech/test-clean/2094/142345/2094-142345-0008.flac" # 长文本推理测试
    CFG_DIR = "./work_dirs/Qwen3-ASR-0.6B_XH2a/ConfigFiles"
    WORK_DIR = Path("./work_dirs/Qwen3-ASR-0.6B_XH2a/")
    ENCODE_ONNX = WORK_DIR / "Encoder/hmonnx/Qwen3-ASR-0.6B_Encoder_xh2a_w8a8_sefp.onnx"
    PREFILL_ONNX = WORK_DIR / "Prefill/Qwen3-ASR-0.6B_XH2a_prefill.onnx"
    DECODE_ONNX  = WORK_DIR / "Decoder/Qwen3-ASR-0.6B_XH2a_decode.onnx"
    encoder_sess = InferenceEngine(str(ENCODE_ONNX))
    encoder_sess.to(str(DEVICE))
    prefill_sess = InferenceEngine(str(PREFILL_ONNX))
    prefill_sess.to(str(DEVICE))
    decode_sess = InferenceEngine(str(DECODE_ONNX))
    decode_sess.to(str(DEVICE))

    processor = Qwen3ASRProcessor.from_pretrained(CFG_DIR, fix_mistral_regex=True)
    tokenizer = AutoTokenizer.from_pretrained(CFG_DIR, trust_remote_code=True, use_fast=True)
    config = AutoConfig.from_pretrained(CFG_DIR, trust_remote_code=True)

    EMB_PT_PATH = WORK_DIR / "token_embedding.pt"
    w = torch.load(EMB_PT_PATH, map_location="cpu")["weight"]
    embed_tokens = nn.Embedding(*w.shape).to(DEVICE, dtype=torch.float16).eval()
    embed_tokens.weight.data.copy_(w.to(device=DEVICE, dtype=torch.float16))

    text_config = config.thinker_config.text_config
    num_layers = text_config.num_hidden_layers
    num_kv_heads = text_config.num_key_value_heads
    hidden_size = text_config.hidden_size
    head_dim = text_config.head_dim
    cache_len = 2048
    max_new_tokens = 2048

    max_audio_length = int(args.max_audio_length)
    max_prefill = _get_feat_extract_output_lengths(max_audio_length) + 21

    proc_tokenizer = processor.tokenizer
    if "<|audio_pad|>" in proc_tokenizer.get_vocab():
        audio_pad_id = proc_tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    else:
        audio_pad_id = proc_tokenizer.encode("<|audio_pad|>", add_special_tokens=False)[0]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "audio", "audio": "placeholder"}]},
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def run_single_chunk(audio_array: np.ndarray) -> str:
        inputs = processor(text=prompt, audio=audio_array, return_tensors="pt", padding=True)
        inputs = inputs.to(DEVICE)
        inputs["input_features"] = inputs["input_features"].float()

        feat_len = int(inputs["input_features"].shape[2])
        origin_feature_lens = inputs["feature_attention_mask"].sum(dim=-1).to(torch.int32)

        if feat_len > max_audio_length:
            inputs["input_features"] = inputs["input_features"][:, :, :max_audio_length]
            inputs["feature_attention_mask"] = inputs["feature_attention_mask"][:, :max_audio_length]
            origin_feature_lens = torch.tensor([max_audio_length], dtype=torch.int32, device=DEVICE)
            feat_len = max_audio_length

        if feat_len < max_audio_length:
            pad_width = (0, max_audio_length - feat_len)
            inputs["input_features"] = torch.nn.functional.pad(inputs["input_features"], pad_width, mode="constant", value=0.0)
            inputs["feature_attention_mask"] = torch.nn.functional.pad(inputs["feature_attention_mask"], pad_width, mode="constant", value=0)

        outputs = encoder_sess.run(
            {
                "input_features": inputs["input_features"].to(torch.float16),
                "feature_lens": origin_feature_lens,
            }
        )
        audio_embeds = outputs.to(DEVICE)
        print(f"audio_embeds.shape: {audio_embeds.shape}")
        T_out = int(_get_feat_extract_output_lengths(origin_feature_lens).item())
        audio_embeds = audio_embeds[:, :T_out, :]
        if audio_embeds.dim() == 2:
            audio_embeds = audio_embeds.unsqueeze(0)

        text_input_ids = inputs["input_ids"]
        text_embeds = embed_tokens(text_input_ids)
        pad_indices = (text_input_ids == audio_pad_id).nonzero(as_tuple=True)[1]
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

        seq_len = final_inputs_embeds.shape[1]
        L = min(seq_len, max_prefill)
        prefill_embeds = torch.zeros((1, max_prefill, hidden_size), dtype=torch.float16, device=DEVICE)
        prefill_embeds[:, :L, :] = final_inputs_embeds[:, :L, :].to(torch.float16).to(DEVICE)
        valid_length = torch.tensor([0], dtype=torch.int32, device=DEVICE)
        current_length = torch.tensor([L], dtype=torch.int32, device=DEVICE)

        kcache = [torch.zeros((1, num_kv_heads, cache_len, head_dim), dtype=torch.float16, device=DEVICE) for _ in range(num_layers)]
        vcache = [torch.zeros((1, num_kv_heads, cache_len, head_dim), dtype=torch.float16, device=DEVICE) for _ in range(num_layers)]

        prefill_input_names = prefill_sess.get_input_names()
        prefill_inputs_dict = {
            "input_embeds": prefill_embeds,
            "valid_length": valid_length,
            "current_length": current_length,
        }
        for i in range(num_layers):
            k_key = f"model_layers_{i}_self_attn_kcache_input"
            v_key = f"model_layers_{i}_self_attn_vcache_input"
            if k_key in prefill_input_names:
                prefill_inputs_dict[k_key] = kcache[i]
            if v_key in prefill_input_names:
                prefill_inputs_dict[v_key] = vcache[i]

        outputs = prefill_sess.run(prefill_inputs_dict)
        next_token_id = torch.argmax(outputs, dim=-1).item()

        valid_length = torch.tensor([L], dtype=torch.int32, device=DEVICE)
        current_length = torch.tensor([1], dtype=torch.int32, device=DEVICE)
        generated_ids = [next_token_id]
        decode_input_names = decode_sess.get_input_names()

        for _ in range(max_new_tokens):
            token_tensor = torch.tensor([[generated_ids[-1]]], device=DEVICE)
            next_embed = embed_tokens(token_tensor).to(torch.float16)
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

            decode_outputs = decode_sess.run(decode_inputs)
            next_id = torch.argmax(decode_outputs, dim=-1).item()
            generated_ids.append(next_id)
            valid_length = valid_length + 1
            if next_id == processor.tokenizer.eos_token_id:
                break

        result = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        match = re.search(r"(?<=<asr_text>)[\s\S]*", result)
        if match:
            return match.group().strip()
        return result

    print(f">>> 处理音频: {AUDIO_PATH}")
    if os.path.exists(AUDIO_PATH):
        audio, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)
    else:
        audio = np.zeros(16000, dtype=np.float32)
        sr = 16000

    chunk_seconds = max_audio_length / 100.0
    chunk_size = int(sr * chunk_seconds)
    n_samples = len(audio)
    n_chunks = max(1, (n_samples + chunk_size - 1) // chunk_size)

    if n_chunks == 1:
        result_text = run_single_chunk(audio)
        print("识别结果:", result_text)
        return

    results = []
    for i in range(n_chunks):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        results.append(run_single_chunk(chunk))

    print("识别结果:", " ".join(filter(None, results)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_audio_length",
        type=int,
        default=1500,
        help="手动固定 Encoder 输入的时间维度 T"
    )
    args = parser.parse_args()
    main(args)

# python hmonnx_demo.py --max_audio_length 1500