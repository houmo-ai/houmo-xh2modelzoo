import re
import os
import math
import torch
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

# 辅助函数
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

# =========================== 1. 路径与配置 ===========================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

AUDIO_PATH = "./dsj_20251212.wav"

CFG_DIR = "./work_dirs/Qwen3-ASR-1.7B_XH2a/ConfigFiles"
WORK_DIR = Path("./work_dirs/Qwen3-ASR-1.7B_XH2a/")
ENCODE_ONNX = WORK_DIR / "Encoder/hmonnx/Qwen3-ASR-1.7B_Encoder_xh2a_w8a8_sefp.onnx"
PREFILL_ONNX = WORK_DIR / "Prefill/Qwen3-ASR-1.7B_XH2a_prefill.onnx"
DECODE_ONNX  = WORK_DIR / "Decoder/Qwen3-ASR-1.7B_XH2a_decode.onnx"
encoder_sess = InferenceEngine(str(ENCODE_ONNX))
encoder_sess.to(str(DEVICE))
encoder_sess.save_golden = True
encoder_sess.save_golden_dir = f"./{WORK_DIR}/golden/encode_golden"

# =========================== 2. 加载 PyTorch 模型 ===========================
processor = Qwen3ASRProcessor.from_pretrained(CFG_DIR, fix_mistral_regex=True)
tokenizer = AutoTokenizer.from_pretrained(CFG_DIR, trust_remote_code=True, use_fast=True)
config = AutoConfig.from_pretrained(CFG_DIR, trust_remote_code=True)

EMB_PT_PATH = WORK_DIR / "token_embedding.pt"

w = torch.load(EMB_PT_PATH, map_location="cpu")["weight"]   # [vocab, hidden]
embed_tokens = nn.Embedding(*w.shape).to(DEVICE, dtype=torch.float16).eval()
embed_tokens.weight.data.copy_(w.to(device=DEVICE, dtype=torch.float16))

# =========================== 3. 音频文本预处理与特征融合 ===========================
print(f">>> 处理音频: {AUDIO_PATH}")
if os.path.exists(AUDIO_PATH):
    audio, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)
else:
    audio = np.zeros(16000)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [{"type": "audio", "audio": "placeholder"}]},
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

inputs = processor(text=prompt, audio=audio, return_tensors="pt", padding=True)
inputs = inputs.to(DEVICE)
# 只把需要的张量转 dtype
inputs['input_features'] = inputs['input_features'].float()   # 给 ONNX encoder 用
# input_ids 保持 long，不要转

origin_feature_lens = inputs['feature_attention_mask'].sum(dim=-1).to(torch.int32) # 原始长度，未填充前的有效长度

pad_width = (0, 3000 - inputs['input_features'].shape[2])

# # 执行填充操作
inputs['input_features'] = torch.nn.functional.pad(
    inputs['input_features'], 
    pad_width, 
    mode='constant',
    value=0.0
)

inputs['feature_attention_mask'] = torch.nn.functional.pad(
    inputs['feature_attention_mask'], 
    pad_width, 
    mode='constant',
    value=0
)

print(f"encoder input shape: {inputs['input_features'].shape}, feature_attention_mask shape: {inputs['feature_attention_mask'].shape}")

# 运行推理
outputs = encoder_sess.run({
    "input_features": inputs['input_features'].to(torch.float16),
    "feature_lens": origin_feature_lens,
})
audio_embeds = outputs.to(DEVICE)

# 处理输出结果
print(f"audio_embeds shape: {audio_embeds.shape}")

T_out = _get_feat_extract_output_lengths(origin_feature_lens).item()
audio_embeds = audio_embeds[:, :T_out, :]

print(f"origin_feature_lens: {origin_feature_lens}, T_out: {T_out}, audio_embeds shape after trim: {audio_embeds.shape}")

if audio_embeds.dim() == 2: 
    audio_embeds = audio_embeds.unsqueeze(0)

text_input_ids = inputs['input_ids']
text_embeds = embed_tokens(text_input_ids)

tokenizer = processor.tokenizer
if "<|audio_pad|>" in tokenizer.get_vocab():
    audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
else:
    audio_pad_id = tokenizer.encode("<|audio_pad|>", add_special_tokens=False)[0]

pad_indices = (text_input_ids == audio_pad_id).nonzero(as_tuple=True)[1]

if len(pad_indices) > 0:
    start_idx = pad_indices[0].item()
    end_idx = pad_indices[-1].item()

    final_inputs_embeds = torch.cat([
        text_embeds[:, :start_idx, :], 
        audio_embeds, 
        text_embeds[:, end_idx+1:, :]
    ], dim=1)
else:
    final_inputs_embeds = text_embeds

print(f"✅ 融合后输入 Shape: {final_inputs_embeds.shape}")

# =========================== final_inputs_embeds =========================== 
text_config = config.thinker_config.text_config

num_layers = text_config.num_hidden_layers
num_kv_heads = text_config.num_key_value_heads
num_attention_heads = text_config.num_attention_heads
hidden_size = text_config.hidden_size
# head_dim = hidden_size // num_attention_heads 
head_dim = text_config.head_dim

cache_len = 2048 # 暂时没用，用的默认 2048

print(f">>> 初始化 KV Cache (L={num_layers}, H={num_kv_heads}, D={head_dim}, MaxLen={cache_len})...")

print(f">>> 加载 Prefill ONNX: {PREFILL_ONNX}")
prefill_sess = InferenceEngine(str(PREFILL_ONNX))
print("DEVICE:", DEVICE)
prefill_sess.to(str(DEVICE))
prefill_sess.save_golden = True
prefill_sess.save_golden_dir = f"./{WORK_DIR}/golden/prefill_golden"

max_prefill = 411
seq_len = final_inputs_embeds.shape[1]
L = min(seq_len, max_prefill)

prefill_embeds = torch.zeros((1, max_prefill, hidden_size), dtype=torch.float16, device=DEVICE)
prefill_embeds[:, :L, :] = final_inputs_embeds[:, :L, :].to(torch.float16).to(DEVICE)

valid_length = torch.tensor([0], dtype=torch.int32, device=DEVICE)   # [1] 当前有效输入长度（区别于 padding）
current_length = torch.tensor([L], dtype=torch.int32, device=DEVICE)   # scalar 当前生成位置

# kv cache 所需维度 num_layers * (batch_size, num_heads, cache_len/seq_len, head_dim)
kcache = [torch.zeros((1, num_kv_heads, cache_len, head_dim), dtype=torch.float16, device=DEVICE)
          for _ in range(num_layers)]
vcache = [torch.zeros((1, num_kv_heads, cache_len, head_dim), dtype=torch.float16, device=DEVICE)
          for _ in range(num_layers)]

prefill_input_names = prefill_sess.get_input_names()
prefill_inputs_dict = {
    "input_embeds": prefill_embeds,
    "valid_length": valid_length,
    "current_length": current_length,
}

# print(f"check inputs:{prefill_inputs_dict['current_length']}")

for i in range(num_layers):
    prefill_inputs_dict[f"model_layers_{i}_self_attn_kcache_input"] = kcache[i]
    prefill_inputs_dict[f"model_layers_{i}_self_attn_vcache_input"] = vcache[i]

# check if input dict fully
missing = [n for n in prefill_input_names if n not in prefill_inputs_dict]
extra = [k for k in prefill_inputs_dict.keys() if k not in prefill_input_names]
print("构造输入中缺失字段与多余字段检查: ", "missing:", missing, "extra:", extra)


# 5. 运行 prefill
try:
    outputs = prefill_sess.run(prefill_inputs_dict)
except Exception as e:
    print("Prefill ONNX 运行失败:", str(e))
    raise e

print("outputs shape:", outputs.shape)


next_token_id = torch.argmax(outputs, dim=-1).item()

print(f"Prefill 预测的第一个 Token ID: {next_token_id}")

# # ======================================================================= decode =======================================================================

max_new_tokens = 2048
# max_new_tokens = 100

print(f">>> 加载 Decode ONNX: {DECODE_ONNX}")
decode_sess = InferenceEngine(str(DECODE_ONNX))
decode_sess.to(str(DEVICE))
decode_sess.save_golden = True
decode_sess.save_golden_dir = f"./{WORK_DIR}/golden/decode_golden"

valid_length = torch.tensor([L], dtype=torch.int32, device=DEVICE)   # [1] 当前有效输入长度（区别于 padding）
current_length = torch.tensor([1], dtype=torch.int32, device=DEVICE)   # scalar 当前生成位置

generated_ids = [next_token_id]

for step in range(max_new_tokens):
    # 准备当前 token 的 embedding
    token_tensor = torch.tensor([[generated_ids[-1]]], device=DEVICE)
    next_embed = embed_tokens(token_tensor).to(torch.float16)  # [1, 1, hidden_size]

    decode_inputs = {
        "input_embeds": next_embed,
        "valid_length": valid_length,
        "current_length": current_length,
    }
    for i in range(num_layers):
        decode_inputs[f"model_layers_{i}_self_attn_kcache_input"] = kcache[i]
        decode_inputs[f"model_layers_{i}_self_attn_vcache_input"] = vcache[i]

    decode_outputs = decode_sess.run(decode_inputs)

    # # 取 hidden state，过 lm_head 得到下一个 token
    # hidden = decode_outputs[0]
    # if isinstance(hidden, np.ndarray):
    #     hidden = torch.from_numpy(hidden).to(DEVICE)

    # logits = lm_head_sess.run({"hidden": hidden[-1]})
    # logits = lm_head(hidden[-1])
    
    next_id = torch.argmax(decode_outputs, dim=-1).item()
    generated_ids.append(next_id)

    # 更新 valid_length
    valid_length = valid_length + 1

    # 遇到 eos 停止
    if next_id == processor.tokenizer.eos_token_id:
        break

result = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
match = re.search(r'(?<=<asr_text>)[\s\S]*', result)
if match:
    print(match.group())
print("识别结果:", result)