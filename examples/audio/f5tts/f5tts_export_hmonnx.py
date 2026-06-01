"""
F5-TTS DiT Backbone w8a8 量化 + HMONNX 导出
===============================================

加载导出的 ONNX 模型，生成覆盖完整噪声调度的校准数据，
使用 xhquant PTQ 量化为 w8a8，转换为 HMONNX 格式。

用法:
    python f5tts_export_hmonnx.py \
        --onnx work_dirs/f5tts/export_fp32/onnx/f5tts_dit.onnx \
        --vocab /data01/nfs_shared/ASR_TTS/F5TTS_base/F5TTS_Base/vocab.txt \
        --quant-type w8a8h1_sefp \
        --calib-samples 32 \
        --out-dir work_dirs/f5tts/export_xh2a
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from f5tts_common import (
    REF_EN_WAV,
    REF_ZH_WAV,
    STATIC_N,
    STATIC_NT,
    VOCAB_PATH,
    extract_mel_spec,
    load_vocab,
    text_to_pinyin,
)


# ============================================================
# ONNX IO 工具
# ============================================================

def _onnx_io_names(onnx_path: str):
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    g = model.graph
    init_names = {i.name for i in g.initializer}
    inputs = [vi.name for vi in g.input if vi.name not in init_names]
    outputs = [vi.name for vi in g.output]
    return inputs, outputs


# ============================================================
# 校准数据生成
# ============================================================

def generate_calib_data(
    vocab_path: str,
    ref_wavs: List[str],
    n_samples: int = 32,
    static_n: int = STATIC_N,
    static_nt: int = STATIC_NT,
) -> List[torch.Tensor]:
    """
    生成覆盖完整噪声调度 [0, 1] 的校准数据。

    对每条参考音频，在不同时间步 t 均匀采样，
    构造插值加噪样本 (1-t)*noise + t*mel 作为 DiT 输入。

    返回: [x, cond, text, time, input_lengths] 五个 Tensor，每个 shape 为 (B, ...)
    """
    vocab_map, _ = load_vocab(vocab_path)

    all_x, all_cond, all_text, all_time, all_lengths = [], [], [], [], []

    # 校准文本（中英分别配对对应语言的参考音频）
    calib_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Speech synthesis technology has improved significantly.",
        "这是一段用于校准的测试文本。",
        "今天天气很好，适合出门散步。",
    ]
    pinyin_list = text_to_pinyin(calib_texts)

    for wav_idx, ref_wav in enumerate(ref_wavs):
        audio = extract_mel_spec(
            __import__("f5tts_common", fromlist=["load_audio"]).load_audio(ref_wav)
        )
        cond = F.pad(audio, (0, 0, 0, static_n - audio.shape[1]), value=0.0)
        ref_len = int(audio.shape[1])

        # 每条参考音频配对两条对应语言的校准文本
        if wav_idx == 0:  # EN ref → EN texts
            text_indices = [0, 1]
        else:             # ZH ref → ZH texts
            text_indices = [2, 3]

        samples_per_text = n_samples // len(ref_wavs) // len(text_indices)
        for ti in text_indices:
            text_tokens = __import__("f5tts_common", fromlist=["pinyin_to_tokens"]).pinyin_to_tokens(
                [pinyin_list[ti]], vocab_map, static_nt,
            )

            for j in range(samples_per_text):
                t = torch.tensor([(j + 0.5) / samples_per_text])
                x0 = torch.randn(1, static_n, 100)
                x1 = cond
                xt = (1 - t) * x0 + t * x1
                duration = ref_len + max(1, int((j + 1) * ref_len / max(samples_per_text, 1)))
                duration = min(duration, static_n)
                mask = (torch.arange(static_n).unsqueeze(0) < duration).unsqueeze(-1)
                xt = torch.where(mask, xt, torch.zeros_like(xt))

                all_x.append(xt)
                all_cond.append(cond)
                all_text.append(text_tokens)
                all_time.append(t.unsqueeze(0))
                all_lengths.append(torch.tensor([duration], dtype=torch.int32))

    return [
        torch.cat(all_x),
        torch.cat(all_cond),
        torch.cat(all_text),
        torch.cat(all_time),
        torch.cat(all_lengths),
    ]


# ============================================================
# 主流程
# ============================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--vocab", type=str, default=VOCAB_PATH)
    p.add_argument("--quant-type", type=str, default="w8a8h1_sefp")
    p.add_argument("--calib-samples", type=int, default=32)
    p.add_argument("--calib-metric", type=str, default="minmax",
                   choices=["minmax", "mse", "kl"])
    p.add_argument("--debug", action="store_true")
    p.add_argument("--out-dir", type=str, default="work_dirs/f5tts/export_xh2a")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    onnx_path = Path(args.onnx).resolve()
    out_dir = Path(args.out_dir).resolve()
    hmonnx_dir = out_dir / "hmonnx"
    hmonnx_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. ONNX IO 信息 --
    input_names, output_names = _onnx_io_names(str(onnx_path))
    print(f"ONNX inputs: {input_names}")
    print(f"ONNX outputs: {output_names}")

    # -- 2. xhquant 初始化 --
    from xhquant.api import (
        DeviceType,
        QuantScheme,
        create_quant_config,
        get_root_logger,
        xhquant_init,
    )

    log_file = out_dir / "convert.log"
    xhquant_init(str(log_file), debug=bool(args.debug))

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)
    quant_config.ops_cfg["LayerNorm"] = dict(force_fp32=True)

    if "w_cfg" in quant_config and "quantizer" in quant_config["w_cfg"]:
        quant_config["w_cfg"]["quantizer"]["calib_metric"] = args.calib_metric
    if "i_cfg" in quant_config and "quantizer" in quant_config["i_cfg"]:
        quant_config["i_cfg"]["quantizer"]["calib_metric"] = args.calib_metric

    # -- 3. 生成校准数据 --
    print(f"\n[calib] 生成 {args.calib_samples} 个校准样本...")
    input_list = generate_calib_data(
        args.vocab,
        [REF_EN_WAV, REF_ZH_WAV],
        n_samples=args.calib_samples,
    )
    batch_size = input_list[0].shape[0]
    print(f"[calib] batch_size={batch_size}")

    # -- 4. 量化 + HMONNX 导出 --
    from xhquant.api.ptq_export_hmonnx import (
        _convert_model_to_quanted_model,
        convert_quanted_model_to_hmonnx,
    )
    from xhquant.common.types import FrontendType, PrecisionMode
    from xhquant.quantization import ptq_quantize

    out_hmonnx = hmonnx_dir / f"f5tts_dit_{DeviceType.XH2a}.onnx"

    # Step 4a: ONNX → QuantGraph（不做 PTQ，后续手动校准）
    first_sample = [t[0:1].cpu() for t in input_list]
    print(f"\n[quant] ONNX → QuantGraph (enable_fuse=True)...")
    quanted = _convert_model_to_quanted_model(
        str(onnx_path),
        FrontendType.ONNX,
        first_sample,
        DeviceType.XH2a,
        quant_config,
        use_ptq=False,
        input_names=input_names,
    )

    # Step 4b: PTQ 校准
    calib_data = [
        [t[i:i + 1].cpu() for t in input_list]
        for i in range(batch_size)
    ]
    exec_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[quant] PTQ 校准 ({batch_size} 样本, device={exec_device})...")
    ptq_quantize(quanted, calib_data, PrecisionMode.ALIGNED, exec_device)

    # Step 4c: 导出 HMONNX
    print(f"[quant] 导出 HMONNX → {out_hmonnx}...")
    convert_quanted_model_to_hmonnx(
        quanted,
        first_sample,
        str(out_hmonnx),
        input_names,
        output_names,
    )

    logger = get_root_logger()
    logger.info(f"Converted to HMONNX: {out_hmonnx}")
    print(f"\nHMONNX 导出完成: {out_hmonnx}")
    print(f"  大小: {out_hmonnx.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
