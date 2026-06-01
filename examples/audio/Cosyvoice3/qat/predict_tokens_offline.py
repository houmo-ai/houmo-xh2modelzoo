"""
LLM 预测 Token 离线预计算
===========================

对每条数据运行 LLM forward，提取 argmax 预测 speech token，保存新 parquet。
本脚本是 cosyvoice3_qat_flow_distill.py（Flow 蒸馏 QAT）的数据准备步骤。

支持两种模式：
    - FP 模式：加载原始 llm.pt，FP 推理（给 Teacher 用）
    - QAT 模式：加载 dequant 权重 + prepare_quanted_model_to_compile(deploy=True)，
             真实量化推理（给 Student 用，与部署对齐）

用法:
    # FP LLM 预测 (teacher)
    python predict_tokens_offline.py \
        --model_dir /data01/nfs_shared/ASR_TTS/CosyVoice3-0.5B-2512 \
        --input_list ./data_zero_shot_zh/train_abs.list \
        --output_dir ./data_zero_shot_zh_fp_pred_tokens

    # QAT LLM 预测 (student)
    python predict_tokens_offline.py \
        --model_dir /data01/nfs_shared/ASR_TTS/CosyVoice3-0.5B-2512 \
        --llm_weights ./output_cosyvoice3_qat_zh_pred100/llm/dequant_steps5000.pt \
        --quantize \
        --input_list ./data_zero_shot_zh_tokw8a16/train/train_abs_w8a16.list \
        --output_dir ./data_zero_shot_zh_qat_pred_tokens
"""

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qat_module_llm import load_llm_model
from qat_utils import print_stage, load_checkpoint, _sanitize_numpy_attrs

# ================================================================
#  简化版 Dataset — 只需 LLM 输入，不需要 mel/audio
# ================================================================

EOP_ID = 151646
EOP_STR = "<|endofprompt|>"


class LLMPredDataset(torch.utils.data.Dataset):
    """最小化数据集：只提供 LLM forward 所需的字段。"""

    def __init__(self, data_list_path, tokenizer, max_samples=-1):
        self.tokenizer = tokenizer
        self.samples = self._load_index(data_list_path, max_samples)

    @staticmethod
    def _load_index(data_list_path, max_samples):
        import pyarrow.parquet as pq

        files = [l.strip() for l in open(data_list_path) if l.strip()]
        samples = []
        for pf in files:
            for _, row in pq.read_table(pf).to_pandas().iterrows():
                samples.append({
                    "text": str(row.get("text", "")),
                    "spk_embedding": row.get("spk_embedding"),
                    "speech_token": row.get("speech_token"),
                    "audio_data": row.get("audio_data"),
                })
                if 0 < max_samples <= len(samples):
                    return samples
        return samples

    def _tokenize(self, text):
        if EOP_STR in text:
            inst, content = text.split(EOP_STR, 1)
            return (
                torch.tensor(self.tokenizer.encode(content, add_special_tokens=False), dtype=torch.long),
                torch.tensor(self.tokenizer.encode(inst, add_special_tokens=False) + [EOP_ID], dtype=torch.long),
            )
        return (
            torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long),
            torch.tensor([], dtype=torch.long),
        )

    @staticmethod
    def _to_tensor(val, dtype=torch.float32):
        if val is None:
            return torch.zeros(192, dtype=dtype)
        if isinstance(val, (bytes, memoryview)):
            return torch.from_numpy(np.frombuffer(bytes(val), dtype=np.float32 if dtype.is_floating_point else np.int32))
        if isinstance(val, np.ndarray):
            return torch.from_numpy(val.copy()).to(dtype)
        return torch.tensor(val, dtype=dtype)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text_token, instruct_token = self._tokenize(s["text"])
        emb = self._to_tensor(s["spk_embedding"])
        st = self._to_tensor(s["speech_token"], dtype=torch.long)

        return {
            "text_token": text_token,
            "text_token_len": torch.tensor(len(text_token), dtype=torch.int32),
            "instruct_token": instruct_token,
            "instruct_token_len": torch.tensor(len(instruct_token), dtype=torch.int32),
            "speech_token": st,
            "speech_token_len": torch.tensor(len(st), dtype=torch.int32),
            "embedding": emb,
            "speech_feat": torch.zeros(len(st) * 2, 80),    # placeholder
            "speech_feat_len": torch.tensor(len(st) * 2, dtype=torch.int32),
        }


# ================================================================
#  LLM 预测 hook — 从 criterion_ce 捕获 logits
# ================================================================

class _LogitCapture:
    """临时 hook：捕获 criterion_ce 的 logits 和 target。"""

    def __init__(self):
        self.logits = None
        self.target = None

    def __call__(self, module, inputs, output):
        self.logits = inputs[0]
        self.target = inputs[1]


def extract_pred_tokens(llm, batch, device):
    """运行 LLM forward，提取 speech token 位置的 argmax 预测。"""
    capture = _LogitCapture()
    hook = llm.criterion_ce.register_forward_hook(capture)

    try:
        llm.forward(batch, device)
    finally:
        hook.remove()

    if capture.logits is None:
        return None

    pred_all = capture.logits.argmax(dim=-1)
    max_token_id = llm.speech_token_size - 1
    pred_all = pred_all.clamp(max=max_token_id)

    speech_mask = capture.target != -1
    speech_token_len = batch["speech_token_len"].to(device)
    bs = speech_token_len.shape[0]

    pred_tokens = []
    for i in range(bs):
        pos = speech_mask[i].nonzero(as_tuple=True)[0]
        n = int(speech_token_len[i].item())
        if len(pos) >= n:
            pred_tokens.append(pred_all[i, pos[:n]].cpu())
        else:
            pred_tokens.append(batch["speech_token"][i, :n].cpu())

    return torch.nn.utils.rnn.pad_sequence(pred_tokens, batch_first=True, padding_value=0)


# ================================================================
#  主流程
# ================================================================

def main():
    p = argparse.ArgumentParser(description="LLM 预测 token 离线预计算")
    p.add_argument("--model_dir", default=os.environ.get("MODEL_DIR", ""))
    p.add_argument("--llm_weights", default=None, help="LLM 权重路径（默认 model_dir/llm.pt）")
    p.add_argument("--quantize", action="store_true", help="QAT deploy mode 量化推理")
    p.add_argument("--w_man_bit", type=int, default=8)
    p.add_argument("--input_list", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--gpu_id", type=int, default=0)
    args = p.parse_args()

    model_dir = args.model_dir
    hf_model_dir = os.path.join(model_dir, "CosyVoice-BlankEN")
    yaml_path = os.path.join(model_dir, "cosyvoice3.yaml")
    llm_path = args.llm_weights or os.path.join(model_dir, "llm.pt")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # ---- 加载 tokenizer ----
    print_stage("加载 Text Tokenizer")
    from transformers import AutoTokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)

    # ---- 加载数据 ----
    print_stage("加载数据")
    ds = LLMPredDataset(args.input_list, text_tokenizer, args.max_samples)
    print(f"  samples = {len(ds)}")

    # ---- 加载 LLM ----
    print_stage(f"加载 LLM ({'QAT deploy' if args.quantize else 'FP'} mode)")
    llm = load_llm_model(yaml_path, hf_model_dir)
    load_checkpoint(llm, llm_path, tag="LLM")
    _sanitize_numpy_attrs(llm)
    llm.to(device)

    if args.quantize:
        print("[QAT] apply prepare_quanted_model_to_compile (deploy mode) ...")
        from qat_utils import prepare_quanted_model_to_compile
        qconfig = {
            "precision_mode": "aligned",
            "enable_qat_compiler_ops": False,
            "w_schema": {
                "fp_mode": "sefp", "man_bit": args.w_man_bit,
                "hidden_bit": True, "nshare": 64, "rounding": "rne",
            },
            "act_schema": {
                "fp_mode": "sefp", "man_bit": 8,
                "hidden_bit": True, "nshare": 64, "rounding": "rne",
                "max_exp_boost": 0,
            },
        }
        llm, _ = prepare_quanted_model_to_compile(
            "llm_pred_offline", llm, "xh2a", qconfig)
        llm.to(device)

    llm.eval()

    # ---- 逐条预测 ----
    print_stage("预测 token")
    out_dir = Path(args.output_dir)
    results = []

    for idx in range(len(ds)):
        sample = ds[idx]
        # 构造 batch（batch_size=1）
        batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v
                 for k, v in sample.items()}

        with torch.no_grad():
            pred_tokens = extract_pred_tokens(llm, batch, device)

        if pred_tokens is None:
            print(f"  [{idx}] WARNING: failed to predict, using GT")
            pred_tokens = sample["speech_token"].unsqueeze(0)

        pred_np = pred_tokens[0].numpy().astype(np.int32)
        orig_np = sample["speech_token"].numpy().astype(np.int32)

        diff = (pred_np != orig_np).sum()

        # 保存时需要 raw bytes，不是 tensor
        emb = sample["embedding"]
        if torch.is_tensor(emb):
            emb = emb.numpy().tobytes()
        elif isinstance(emb, np.ndarray):
            emb = emb.tobytes()

        audio = ds.samples[idx].get("audio_data")
        if isinstance(audio, memoryview):
            audio = bytes(audio)

        results.append({
            "utt": str(ds.samples[idx].get("utt", f"utt_{idx}")),
            "text": str(ds.samples[idx].get("text", "")),
            "spk_embedding": emb,
            "speech_token": pred_np.tobytes(),
            "audio_data": audio,
        })

        if (idx + 1) % 10 == 0 or idx == len(ds) - 1:
            print(f"  [{idx + 1}/{len(ds)}] token_diff={diff}")

    # ---- 保存 ----
    print_stage("保存结果")
    import pyarrow as pa
    import pyarrow.parquet as pq

    # 构造 Arrow table
    schema = pa.schema([
        ("utt", pa.string()),
        ("text", pa.string()),
        ("spk_embedding", pa.binary()),
        ("speech_token", pa.binary()),
        ("audio_data", pa.binary()),
    ])
    table = pa.table({
        "utt": [r["utt"] for r in results],
        "text": [r["text"] for r in results],
        "spk_embedding": [r["spk_embedding"] for r in results],
        "speech_token": [r["speech_token"] for r in results],
        "audio_data": [r["audio_data"] for r in results],
    }, schema=schema)

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "pred_tokens.parquet"
    pq.write_table(table, str(parquet_path))
    print(f"  saved: {parquet_path}")

    list_path = out_dir / "train_abs_pred.list"
    list_path.write_text(str(parquet_path) + "\n")
    print(f"  list: {list_path}")

    print(f"\n[DONE] {len(results)} samples processed")


if __name__ == "__main__":
    main()
