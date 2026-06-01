import torch
import numpy as np
import librosa
from pathlib import Path
import torch.nn as nn

from xhquant.api import HMONNXInference as InferenceEngine
from transformers import AutoConfig, AutoTokenizer

from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor
from qwen_asr.core.transformers_backend import (
    Qwen3ASRProcessor,
    Qwen3ASRForConditionalGeneration
)

# ========================= 基本配置 =========================

SAMPLE_RATE = 16000
MAX_AUDIO_LEN = 3000
MAX_PREFILL = 411

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CFG_DIR = "work_dirs/Qwen3-ForcedAligner-0.6B_XH2a/ConfigFiles"
WORK_DIR = Path("./work_dirs/Qwen3-ForcedAligner-0.6B_XH2a")
ENCODE_ONNX = WORK_DIR / "Encoder/hmonnx/Qwen3-ForcedAligner-0.6B_Encoder_xh2a_w8a8_sefp.onnx"
PREFILL_ONNX = WORK_DIR / "Prefill/Qwen3-ForcedAligner-0.6B_XH2a_prefill_fullseq.onnx"

audio_path = "./61-70968-0000.wav"

text = "HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD WHO HAD VANISHED BEHIND THE CURTAIN ON THE LEFT"
language = "Chinese"


# ========================= helper =========================

def _get_feat_extract_output_lengths(input_lengths):

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1

    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1
        + (input_lengths // 100) * 13
    )

    return output_lengths


# ========================= 加载模型 =========================

print("Loading PyTorch components...")
processor = Qwen3ASRProcessor.from_pretrained(CFG_DIR, fix_mistral_regex=True)
tokenizer = AutoTokenizer.from_pretrained(CFG_DIR, trust_remote_code=True, use_fast=True)
config = AutoConfig.from_pretrained(CFG_DIR, trust_remote_code=True)

EMB_PT_PATH = WORK_DIR / "token_embedding.pt"

w = torch.load(EMB_PT_PATH, map_location="cpu")["weight"]   # [vocab, hidden]
embed_tokens = nn.Embedding(*w.shape).to(DEVICE, dtype=torch.float16).eval()
embed_tokens.weight.data.copy_(w.to(device=DEVICE, dtype=torch.float16))

timestamp_token_id = config.timestamp_token_id
timestamp_segment_time = config.timestamp_segment_time

tokenizer = processor.tokenizer

if "<|audio_pad|>" in tokenizer.get_vocab():
    audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
else:
    audio_pad_id = tokenizer.encode("<|audio_pad|>", add_special_tokens=False)[0]


# ========================= ONNX =========================

encoder_sess = InferenceEngine(ENCODE_ONNX)
encoder_sess.to(str(DEVICE))
encoder_sess.save_golden = True
encoder_sess.save_golden_dir = f"./{WORK_DIR}/golden/encoder_golden"


prefill_sess = InferenceEngine(str(PREFILL_ONNX))
prefill_sess.to(str(DEVICE))
prefill_sess.save_golden = True
prefill_sess.save_golden_dir = f"./{WORK_DIR}/golden/prefill_golden"


# ========================= 音频读取 =========================

def load_audio(path):

    audio, sr = librosa.load(path, sr=None, mono=False)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)

    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    audio = audio.astype(np.float32)

    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak

    return audio


audio = load_audio(audio_path)


# ========================= 文本处理 =========================

aligner_processor = Qwen3ForceAlignProcessor()

word_list, aligner_input_text = aligner_processor.encode_timestamp(
    text,
    language
)

inputs = processor(
    text=[aligner_input_text],
    audio=[audio],
    return_tensors="pt",
    padding=True
)

inputs = inputs.to(DEVICE)

input_features = inputs["input_features"]
feature_attention_mask = inputs["feature_attention_mask"]
input_ids = inputs["input_ids"]


# ========================= encoder padding =========================

origin_feature_lens = feature_attention_mask.sum(dim=-1).to(torch.int32)

pad_width = (0, MAX_AUDIO_LEN - input_features.shape[2])

input_features = torch.nn.functional.pad(
    input_features,
    pad_width,
    mode="constant",
    value=0.0
)

# ========================= encoder =========================

encoder_outputs = encoder_sess.run({
    "input_features": input_features.to(torch.float16),
    "feature_lens": origin_feature_lens,
})

audio_embeds = encoder_outputs

if isinstance(audio_embeds, np.ndarray):
    audio_embeds = torch.from_numpy(audio_embeds)

audio_embeds = audio_embeds.to(DEVICE).to(torch.float16)

print("audio_embeds raw:", audio_embeds.shape)

T_out = _get_feat_extract_output_lengths(origin_feature_lens).item()

audio_embeds = audio_embeds[:, :T_out, :]

print("audio_embeds trimmed:", audio_embeds.shape)


# ========================= text embedding =========================

text_embeds = embed_tokens(input_ids)
print("text_embeds:", text_embeds.shape)

# ========================= merge =========================

pad_indices = (input_ids == audio_pad_id).nonzero(as_tuple=True)[1]

start_idx = pad_indices[0].item()
end_idx = pad_indices[-1].item()

assert audio_embeds.shape[1] == (end_idx - start_idx + 1)

inputs_embeds = torch.cat(
    [
        text_embeds[:, :start_idx],
        audio_embeds,
        text_embeds[:, end_idx + 1 :]
    ],
    dim=1
)

print("final_inputs_embeds:", inputs_embeds.shape)

# ========================= Prefill padding =========================

text_config = config.thinker_config.text_config

num_layers = text_config.num_hidden_layers
num_kv_heads = text_config.num_key_value_heads
head_dim = text_config.head_dim
hidden_size = text_config.hidden_size

cache_len = 2048

seq_len = inputs_embeds.shape[1]
L = min(seq_len, MAX_PREFILL)

prefill_embeds = torch.zeros(
    (1, MAX_PREFILL, hidden_size),
    dtype=torch.float16,
    device=DEVICE
)

prefill_embeds[:, :L, :] = inputs_embeds[:, :L, :].to(torch.float16)

valid_length = torch.tensor([0], dtype=torch.int32, device=DEVICE)
current_length = torch.tensor([L], dtype=torch.int32, device=DEVICE)


# ========================= KV cache =========================

kcache = [
    torch.zeros((1, num_kv_heads, cache_len, head_dim),
    dtype=torch.float16,
    device=DEVICE)
    for _ in range(num_layers)
]

vcache = [
    torch.zeros((1, num_kv_heads, cache_len, head_dim),
    dtype=torch.float16,
    device=DEVICE)
    for _ in range(num_layers)
]

# ========================= prefill inputs =========================

prefill_inputs = {
    "input_embeds": prefill_embeds,
    "valid_length": valid_length,
    "current_length": current_length,
}

for i in range(num_layers):
    prefill_inputs[f"model_layers_{i}_self_attn_kcache_input"] = kcache[i]
    prefill_inputs[f"model_layers_{i}_self_attn_vcache_input"] = vcache[i]


# ========================= ONNX forward =========================
prefill_outputs = prefill_sess.run(prefill_inputs)

def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(DEVICE)
    return x.to(DEVICE)

print(f"prefill returned {len(prefill_outputs)} outputs")
for i, out in enumerate(prefill_outputs):
    t = to_torch(out)
    print(f"  output[{i}]: {tuple(t.shape)}  dtype={t.dtype}")

logits = None
for out in prefill_outputs:
    t = to_torch(out)
    # [B,T,V]
    if t.ndim == 3:
        logits = t
        break
    # [T,V]
    if t.ndim == 2:
        logits = t.unsqueeze(0)
        break

if logits is None:
    raise RuntimeError("No suitable logits found from prefill outputs.")

print("picked logits:", logits.shape)  # expected [1, T, V]

output_ids = logits.argmax(dim=-1)     # [1, T]

if input_ids.shape[1] != output_ids.shape[1]:
    print(f"WARNING: input_ids len={input_ids.shape[1]} != output_ids len={output_ids.shape[1]}")
    T = min(input_ids.shape[1], output_ids.shape[1])
    input_ids_cut = input_ids[:, :T]
    output_ids_cut = output_ids[:, :T]
else:
    input_ids_cut = input_ids
    output_ids_cut = output_ids

masked_output_id = output_ids_cut[input_ids_cut == timestamp_token_id]
timestamp_ms = (masked_output_id * timestamp_segment_time).cpu().numpy()

timestamp_output = aligner_processor.parse_timestamp(word_list, timestamp_ms)
for it in timestamp_output:
    it["start_time"] = round(it["start_time"] / 1000.0, 3)
    it["end_time"] = round(it["end_time"] / 1000.0, 3)

print("\n输出:\n")
for w in timestamp_output:
    print(f"{w['text']:10s} {w['start_time']:6.3f}  {w['end_time']:6.3f}")