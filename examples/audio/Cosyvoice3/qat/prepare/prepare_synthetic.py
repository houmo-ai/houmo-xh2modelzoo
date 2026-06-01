"""生成合成数据用于 QAT smoke test。

不依赖任何音频数据集，直接生成随机 tensor 保存为 .pt 文件。
每个样本包含：speech_token, embedding, speech_feat, speech, pitch_feat, text_token, instruct_token。
"""
import torch
import os
from pathlib import Path

N_TRAIN = 32
N_VAL = 8
SEQ_LEN_TOKEN = 64   # speech token 长度
MEL_LEN = 128        # mel 帧数
SPK_DIM = 192
SPEECH_TOKEN_SIZE = 6561

def make_sample(idx):
    speech_token_len = SEQ_LEN_TOKEN
    speech_token = torch.randint(0, SPEECH_TOKEN_SIZE, (speech_token_len,))
    speech_feat = torch.randn(MEL_LEN, 80)
    speech = torch.randn(MEL_LEN * 480)  # 24kHz * mel_len * hop_size
    pitch_feat = torch.randn(MEL_LEN, 1)
    embedding = torch.randn(SPK_DIM)
    # 模拟 text/instruct token (Qwen2 tokenizer 格式)
    text_token = torch.randint(100, 5000, (20,))
    instruct_token = torch.tensor([100, 101, 102])  # 固定 instruct
    return {
        "speech_token": speech_token,
        "speech_token_len": torch.tensor(speech_token_len),
        "speech_feat": speech_feat,
        "speech_feat_len": torch.tensor(MEL_LEN),
        "speech": speech,
        "pitch_feat": pitch_feat,
        "embedding": embedding,
        "text_token": text_token,
        "text_token_len": torch.tensor(len(text_token)),
        "instruct_token": instruct_token,
        "instruct_token_len": torch.tensor(len(instruct_token)),
    }

def save_dataset(samples, path):
    torch.save(samples, path)
    print(f"  saved {len(samples)} samples to {path}")

if __name__ == "__main__":
    out_dir = Path("./synthetic_data")
    out_dir.mkdir(exist_ok=True)

    print("Generating synthetic data...")
    train_samples = [make_sample(i) for i in range(N_TRAIN)]
    val_samples = [make_sample(i) for i in range(N_VAL)]

    save_dataset(train_samples, out_dir / "train.pt")
    save_dataset(val_samples, out_dir / "val.pt")
    print("Done!")
