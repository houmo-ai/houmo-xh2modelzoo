"""
CosyVoice3 QAT 评测脚本
========================

支持两种数据源和三种模型类型，统一评测 WER、Speaker Similarity、DNSMOS。

数据源 (--data_source):
    cv3_eval  — CV3-Eval Kaldi 格式目录 (text, prompt_text, prompt_wav.scp)
    parquet   — 预处理 parquet 文件 (text, audio_data 字段)

模型类型 (--model_type):
    fp        — 原始 FP 模型，不做量化
    ptq       — 运行时 PTQ 量化 (w4a8 / w8a8)
    qat       — 运行时 PTQ + 加载 QAT checkpoint
    prebuilt  — 已替换反量化权重的评测目录，不做额外量化

用法:
    # CV3-Eval 评测 (中文 zero-shot)
    conda activate xhquant
    python eval_qat.py --data_source cv3_eval --model_type qat \\
        --qat_ckpt_dir ./output_cosyvoice3_qat --qat_steps 5000

    # Parquet dev 评测 (Pred100 验证)
    conda activate cosyvoice-deploy
    python eval_qat.py --data_source parquet --model_type prebuilt \\
        --dev_parquet ../data/data_librispeech_full_tokw8a16/cv/dev_w8a16.list \\
        --skip_inference --max_samples 100
"""

import argparse
import io
import os
import string
import sys

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# ================================================================
#  CosyVoice 路径
# ================================================================

_COSYVOICE_ROOT = os.getenv(
    "COSYVOICE_ROOT",
    os.path.expanduser("~/workspace/repo/develop/CosyVoice"),
)
sys.path.insert(0, _COSYVOICE_ROOT)
_MATCHA_PATH = os.path.join(_COSYVOICE_ROOT, "third_party", "Matcha-TTS")
if os.path.isdir(_MATCHA_PATH):
    sys.path.insert(0, _MATCHA_PATH)


# ================================================================
#  CLI 参数
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="CosyVoice3 QAT 评测")
    # 数据源
    p.add_argument("--data_source", choices=["cv3_eval", "parquet"],
                   default="cv3_eval", help="评测数据来源")
    p.add_argument("--cv3_eval_dir", default="/data01/home/she.gao/CV3-Eval")
    p.add_argument("--eval_subset", default="zero_shot/zh")
    p.add_argument("--dev_parquet", help="parquet 数据 list 路径")
    # 模型
    p.add_argument("--model_type", choices=["fp", "ptq", "qat", "prebuilt"],
                   default="fp", help="模型类型")
    p.add_argument("--model_dir",
                   default="/data01/nfs_shared/ASR_TTS/CosyVoice3-0.5B-2512")
    # 量化参数 (仅 ptq/qat)
    p.add_argument("--w_man_bit", type=int, default=4,
                   help="权重尾数位数 (4=w4a8, 8=w8a8)")
    p.add_argument("--qat_ckpt_dir", default="./output_cosyvoice3_qat")
    p.add_argument("--qat_steps", type=int, default=200)
    # 评测控制
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--output_dir", default="./eval_output")
    p.add_argument("--skip_inference", action="store_true",
                   help="跳过推理，仅对已有 wav 做评测")
    p.add_argument("--overwrite", action="store_true",
                   help="覆盖已有 wav")
    p.add_argument("--lang", default="auto",
                   help="WER 语言: auto/zh/en (auto=根据 subset 自动判断)")
    return p.parse_args()


# ================================================================
#  数据加载 — CV3-Eval
# ================================================================

def load_cv3_eval(cv3_dir, subset, max_samples):
    """加载 CV3-Eval Kaldi 格式数据。

    Returns: [(utt_id, tts_text, prompt_text, prompt_wav_path)]
    """
    subset_dir = os.path.join(cv3_dir, "data", subset)

    texts = {}
    with open(os.path.join(subset_dir, "text")) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1]

    prompt_texts = {}
    pt_path = os.path.join(subset_dir, "prompt_text")
    if os.path.exists(pt_path):
        with open(pt_path) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    prompt_texts[parts[0]] = parts[1]

    prompt_wavs = {}
    with open(os.path.join(subset_dir, "prompt_wav.scp")) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wav_path = parts[1]
                if not os.path.isabs(wav_path):
                    wav_path = os.path.join(cv3_dir, wav_path)
                prompt_wavs[parts[0]] = wav_path

    samples = []
    for utt_id in texts:
        if utt_id in prompt_wavs:
            pt = prompt_texts.get(utt_id, "")
            samples.append((utt_id, texts[utt_id], pt, prompt_wavs[utt_id]))
        if len(samples) >= max_samples:
            break
    return samples


# ================================================================
#  数据加载 — Parquet
# ================================================================

def load_parquet_samples(list_path, max_samples):
    """加载 parquet 格式 dev 数据。

    Returns: [(utt, text, audio_data_bytes)]
    """
    import pyarrow.parquet as pq

    with open(list_path) as f:
        parquet_files = [line.strip() for line in f if line.strip()]

    samples = []
    for pf in parquet_files:
        if len(samples) >= max_samples:
            break
        table = pq.read_table(pf)
        df = table.to_pandas()
        for _, row in df.iterrows():
            if len(samples) >= max_samples:
                break

            text = row.get("text", "")
            prefix = "You are a helpful assistant.<|endofprompt|>"
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
            if not text:
                continue

            audio_data = row.get("audio_data")
            if audio_data is None:
                continue
            if isinstance(audio_data, memoryview):
                audio_data = bytes(audio_data)

            samples.append((row["utt"], text, audio_data))
    return samples


# ================================================================
#  模型加载 + 量化
# ================================================================

def _make_quant_config(w_man_bit, enable_qat_ops=False):
    return {
        "precision_mode": "aligned",
        "enable_qat_compiler_ops": enable_qat_ops,
        "w_schema": {
            "fp_mode": "sefp", "man_bit": w_man_bit,
            "hidden_bit": True, "nshare": 64, "rounding": "rne",
        },
        "act_schema": {
            "fp_mode": "sefp", "man_bit": 8,
            "hidden_bit": True, "nshare": 64, "rounding": "rne",
            "max_exp_boost": 0,
        },
    }


def _sanitize_numpy_attrs(module):
    import numpy as _np
    for name in dir(module):
        if name.startswith('_'):
            continue
        try:
            val = getattr(module, name)
        except AttributeError:
            continue
        if isinstance(val, (_np.integer, _np.floating)):
            object.__setattr__(module, name, val.item())
        elif isinstance(val, _np.ndarray) and val.ndim == 0:
            object.__setattr__(module, name, val.item())
    for child in module.children():
        _sanitize_numpy_attrs(child)


def load_model(args):
    """根据 data_source + model_type 加载模型。"""
    if args.data_source == "parquet":
        from cosyvoice.cli.cosyvoice import AutoModel
        return AutoModel(model_dir=args.model_dir)

    # cv3_eval 模式 — 使用 CosyVoice3 显式加载
    from cosyvoice.cli.cosyvoice import CosyVoice3
    cosyvoice = CosyVoice3(args.model_dir)

    if args.model_type in ("fp", "prebuilt"):
        return cosyvoice

    from xhquant.api import prepare_quanted_model_to_compile

    # XHConv1d 兼容补丁
    try:
        from xhquant.nn.modules.conv1d import XHConv1d as _XHConv1d
        if not hasattr(_XHConv1d, "kernel_size"):
            _XHConv1d.kernel_size = property(
                lambda self: self.conv2d.kernel_size[:1]
                if self.conv2d is not None else None)
    except ImportError:
        pass

    model = cosyvoice.model

    # 量化 LLM + Flow
    for mod_name in ("llm", "flow"):
        print(f"[quantize] {mod_name.upper()} w{args.w_man_bit}a8 ...")
        cfg = _make_quant_config(args.w_man_bit, enable_qat_ops=False)
        sub_mod = getattr(model, mod_name)
        setattr(model, mod_name,
                prepare_quanted_model_to_compile(
                    f"cosyvoice3_{mod_name}_qat", sub_mod, "xh2a", cfg)[0])
        _sanitize_numpy_attrs(getattr(model, mod_name))

    # 量化 HiFT (w8a8, 跳过 f0_predictor)
    print("[quantize] HiFT w8a8 ...")
    _saved_f0 = getattr(model.hift, "f0_predictor", None)
    if _saved_f0 is not None:
        del model.hift.f0_predictor
    hift_cfg = _make_quant_config(8, enable_qat_ops=False)
    model.hift, _ = prepare_quanted_model_to_compile(
        "cosyvoice3_hift_qat", model.hift, "xh2a", hift_cfg)
    if _saved_f0 is not None:
        model.hift.f0_predictor = _saved_f0
    _sanitize_numpy_attrs(model.hift)

    # 加载 QAT 权重
    if args.model_type == "qat":
        for mod_name in ("llm", "flow"):
            sub_mod = getattr(model, mod_name)
            ckpt_path = os.path.join(
                args.qat_ckpt_dir, mod_name, f"qat_steps{args.qat_steps}.pt")
            if not os.path.exists(ckpt_path):
                print(f"[WARN] QAT checkpoint 不存在: {ckpt_path}")
                continue
            print(f"[qat] loading {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu",
                                    weights_only=True)
            # 自动 strip 多余前缀
            cur_keys = set(sub_mod.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            if not cur_keys & ckpt_keys:
                parts = next(iter(ckpt_keys)).split(".")
                for i in range(1, len(parts)):
                    prefix = ".".join(parts[:i]) + "."
                    stripped = {k[len(prefix):]: v
                                for k, v in state_dict.items()
                                if k.startswith(prefix)}
                    if cur_keys & set(stripped.keys()):
                        state_dict = stripped
                        print(f"  stripped prefix '{prefix}'")
                        break
            cur_sd = sub_mod.state_dict()
            matched = {k: v for k, v in state_dict.items()
                       if k in cur_sd and cur_sd[k].shape == v.shape}
            cur_sd.update(matched)
            sub_mod.load_state_dict(cur_sd)
            print(f"  loaded {len(matched)}/{len(cur_sd)} params")

    model.eval()
    return cosyvoice


# ================================================================
#  推理
# ================================================================

@torch.no_grad()
def generate_cv3(cosyvoice, samples, output_dir, model_type, overwrite=False):
    """CV3-Eval 数据推理。"""
    wav_dir = os.path.join(output_dir, model_type, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    generated = []
    for utt_id, tts_text, prompt_text, prompt_wav in tqdm(samples, desc=model_type):
        wav_out = os.path.join(wav_dir, f"{utt_id}.wav")
        if (not overwrite) and os.path.exists(wav_out):
            generated.append((utt_id, wav_out, tts_text))
            continue
        try:
            instruct = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
            chunks = []
            for chunk in cosyvoice.inference_zero_shot(
                    tts_text, instruct, prompt_wav, stream=False):
                chunks.append(chunk["tts_speech"])
            if chunks:
                speech = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
                torchaudio.save(wav_out, speech.cpu(), cosyvoice.sample_rate)
                generated.append((utt_id, wav_out, tts_text))
        except Exception as e:
            print(f"  [WARN] {utt_id}: {e}")

    print(f"  generated {len(generated)} wavs → {wav_dir}")
    return generated, {utt: pwav for utt, _, _, pwav in samples}


@torch.no_grad()
def generate_parquet(cosyvoice, samples, output_dir, skip_inference=False):
    """Parquet 数据推理。"""
    wav_dir = os.path.join(output_dir, "wavs")
    prompt_dir = os.path.join(output_dir, "prompts")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)

    generated = []
    for utt, text, audio_data in tqdm(samples, desc="inference"):
        prompt_wav = os.path.join(prompt_dir, f"{utt}.wav")
        wav_out = os.path.join(wav_dir, f"{utt}.wav")

        # 写 prompt wav
        if not os.path.exists(prompt_wav):
            wav_tensor, sr = torchaudio.load(io.BytesIO(audio_data))
            torchaudio.save(prompt_wav, wav_tensor, sr)

        if skip_inference:
            if os.path.exists(wav_out):
                generated.append((utt, wav_out, prompt_wav, text))
            continue

        if os.path.exists(wav_out):
            generated.append((utt, wav_out, prompt_wav, text))
            continue

        try:
            instruct = f"You are a helpful assistant.<|endofprompt|>{text}"
            chunks = []
            for chunk in cosyvoice.inference_zero_shot(
                    text, instruct, prompt_wav, stream=False):
                chunks.append(chunk["tts_speech"])
            if chunks:
                speech = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
                torchaudio.save(wav_out, speech.cpu(), cosyvoice.sample_rate)
                generated.append((utt, wav_out, prompt_wav, text))
        except Exception as e:
            print(f"  [WARN] {utt}: {e}")

    return generated


# ================================================================
#  指标计算
# ================================================================

def compute_wer(generated, lang="zh"):
    """WER (Paraformer ASR + jiwer)。支持 zh/en/ja/ko。"""
    try:
        from funasr import AutoModel as FunASRModel
    except ImportError:
        print("[WARN] funasr not installed, skip WER")
        return None

    import jiwer

    all_punct = string.punctuation
    try:
        from zhon.hanzi import punctuation as zh_punc
        all_punct = zh_punc + all_punct
    except ImportError:
        pass

    print("[WER] loading Paraformer ASR ...")
    asr_model = FunASRModel(model="paraformer-zh")

    wer_list = []
    for item in tqdm(generated, desc="WER"):
        wav_path = item[1]
        ref_text = item[2]

        try:
            res = asr_model.generate(input=wav_path, batch_size_s=300)
            hyp_text = res[0]["text"] if res else ""
        except Exception:
            continue

        truth, hypo = ref_text, hyp_text
        for ch in all_punct:
            if ch == "'":
                continue
            truth = truth.replace(ch, "")
            hypo = hypo.replace(ch, "")

        if lang in ("zh", "ja", "ko"):
            try:
                import zhconv
                truth = " ".join(zhconv.convert(truth, "zh-cn"))
                hypo = " ".join(zhconv.convert(hypo, "zh-cn"))
            except ImportError:
                pass
        else:
            truth, hypo = truth.lower(), hypo.lower()

        if not truth.strip():
            continue
        wer_list.append(jiwer.wer(truth, hypo))

    return np.mean(wer_list) if wer_list else None


def compute_spk_similarity(generated, ref_wav_map, campplus_onnx_path):
    """Speaker Similarity (CAMPPlus ONNX)。"""
    import onnxruntime as ort
    import torchaudio.compliance.kaldi as kaldi

    print("[SIM] loading CAMPPlus ...")
    session = ort.InferenceSession(campplus_onnx_path,
                                   providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name

    def extract_emb(wav_path):
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        feat = kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        inp = feat.unsqueeze(0).numpy().astype(np.float32)
        return session.run(None, {inp_name: inp})[0].flatten()

    sims = []
    for item in tqdm(generated, desc="SIM"):
        utt_id = item[0]
        wav_path = item[1]
        ref_path = ref_wav_map.get(utt_id)
        if not ref_path or not os.path.exists(ref_path):
            continue
        try:
            emb_gen = extract_emb(wav_path)
            emb_ref = extract_emb(ref_path)
            cos = np.dot(emb_gen, emb_ref) / (
                np.linalg.norm(emb_gen) * np.linalg.norm(emb_ref) + 1e-8)
            sims.append(float(cos))
        except Exception:
            continue

    return np.mean(sims) if sims else None


def compute_dnsmos(generated):
    """DNSMOS (speechmos)。"""
    try:
        from speechmos import dnsmos as _dnsmos
    except ImportError:
        print("[WARN] speechmos not installed, skip DNSMOS")
        return None

    results = {"ovrl": [], "sig": [], "bak": [], "p808": []}
    for item in tqdm(generated, desc="DNSMOS"):
        wav_path = item[1]
        try:
            wav, sr = torchaudio.load(wav_path)
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            audio = wav.squeeze(0).numpy().astype(np.float32)
            if len(audio) == 0:
                continue
            score = _dnsmos.run(audio, 16000)
            for k in results:
                key = f"{k}_mos" if k != "p808" else "p808_mos"
                if key in score:
                    results[k].append(float(score[key]))
        except Exception:
            continue

    if not results["ovrl"]:
        return None
    return {k: np.mean(v) for k, v in results.items()}


# ================================================================
#  主函数
# ================================================================

def main():
    args = parse_args()

    # 自动判断语言
    if args.lang == "auto":
        args.lang = "zh" if "zh" in getattr(args, "eval_subset", "") else "en"

    model_tag = args.model_type
    print(f"\n{'=' * 60}")
    print(f"  CosyVoice3 QAT Eval — {model_tag.upper()}")
    print(f"  data_source={args.data_source}, lang={args.lang}")
    print(f"  max_samples={args.max_samples}")
    print(f"{'=' * 60}")

    # ---- 加载数据 ----
    ref_wav_map = {}
    if args.data_source == "cv3_eval":
        samples = load_cv3_eval(args.cv3_eval_dir, args.eval_subset,
                                args.max_samples)
        ref_wav_map = {utt: pwav for utt, _, _, pwav in samples}
        print(f"  loaded {len(samples)} CV3-Eval samples")
    else:
        if not args.dev_parquet:
            print("[ERROR] parquet 模式需要 --dev_parquet")
            return
        samples = load_parquet_samples(args.dev_parquet, args.max_samples)
        print(f"  loaded {len(samples)} parquet samples")

    # ---- 加载模型 + 推理 ----
    if args.data_source == "cv3_eval":
        cosyvoice = load_model(args)
        generated = generate_cv3(cosyvoice, samples, args.output_dir,
                                 model_tag, args.overwrite)
        generated, ref_wav_map = generated
    else:
        # parquet 模式
        if args.skip_inference:
            cosyvoice = None
        else:
            cosyvoice = load_model(args)
        generated = generate_parquet(cosyvoice, samples, args.output_dir,
                                     args.skip_inference)
        ref_wav_map = {item[0]: item[2] for item in generated}

    print(f"  generated {len(generated)} samples")

    # ---- WER ----
    print(f"\n[eval] WER (lang={args.lang}) ...")
    wer = compute_wer(generated, lang=args.lang)

    # ---- Speaker Similarity ----
    campplus_path = os.path.join(args.model_dir, "campplus.onnx")
    print(f"\n[eval] Speaker Similarity ...")
    sim = compute_spk_similarity(generated, ref_wav_map, campplus_path)

    # ---- DNSMOS ----
    print(f"\n[eval] DNSMOS ...")
    dnsmos = compute_dnsmos(generated)

    # ---- 汇总 ----
    print(f"\n{'=' * 60}")
    print(f"  RESULTS — {model_tag.upper()}")
    print(f"{'=' * 60}")
    print(f"  samples:      {len(generated)}")
    print(f"  WER (%):      {wer * 100:.2f}" if wer else "  WER:          N/A")
    print(f"  Spk Sim (%):  {sim * 100:.2f}" if sim else "  Spk Sim:      N/A")
    if dnsmos:
        print(f"  DNSMOS:")
        print(f"    OVRL:      {dnsmos['ovrl']:.3f}")
        print(f"    SIG:       {dnsmos['sig']:.3f}")
        print(f"    BAK:       {dnsmos['bak']:.3f}")
        print(f"    P808:      {dnsmos['p808']:.3f}")
    else:
        print("  DNSMOS:      N/A")


if __name__ == "__main__":
    main()
