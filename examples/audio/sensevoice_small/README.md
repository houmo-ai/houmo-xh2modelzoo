# SenseVoiceSmall（导出 ONNX + xhquant 量化）

本目录提供 SenseVoiceSmall 的完整示例闭环：

1. 使用 funasr 构建模型并导出 FP32 ONNX（输入为 560 维 LFR 特征）
2. 使用 xhquant 将 ONNX 量化并导出为 HMONNX
3. 对比 FP32 与量化模型的识别精度（CER/WER）

## 关键 I/O 约定

官方导出模型的输入输出固定为：

- Inputs
  - `speech`: `float32`，形状 `[B, T, 560]`
  - `speech_lengths`: `int32`，形状 `[B]`
  - `language`: `int32`，形状 `[B]`（默认 `0=auto`）
  - `textnorm`: `int32`，形状 `[B]`（默认 `15=woitn`，或 `14=withitn`）
- Outputs
  - `ctc_logits`
  - `encoder_out_lens`

其中 `560 = 80(n_mels) * 7(lfr_m)`，不是 80 维 fbank。

## 前置依赖

- 必需：`funasr`、`onnx`、`onnxruntime`、`librosa`、`torchaudio`
- 特征提取使用 `torchaudio.compliance.kaldi.fbank` + LFR + CMVN

## 默认路径

- 模型目录：`/data01/nfs_shared/ASR_TTS/SenseVoiceSmall`

## 1) 导出 FP32 ONNX

```bash
python examples/audio/sensevoice_small/sensevoice_export_onnx.py \
  --model-dir /data01/nfs_shared/ASR_TTS/SenseVoiceSmall \
  --out-dir work_dirs/sensevoice_small/export_fp32
```

产物：

- `work_dirs/sensevoice_small/export_fp32/onnx/model.onnx`

## 2) xhquant 量化导出 HMONNX

从新版开始，我们直接在导出脚本中集成校准数据生成（基于 OpenSLR LibriSpeech），无需手动生成特征文件。

```bash
python examples/audio/sensevoice_small/sensevoice_export_hmonnx.py \
  --onnx work_dirs/sensevoice_small/export_fp32/onnx/model.onnx \
  --hf-dataset openslr/librispeech_asr \
  --hf-split validation \
  --calib-samples 128 \
  --quant-type w8a8h1_sefp \
  --out-dir work_dirs/sensevoice_small/export_xh2a
```

产物：

- `work_dirs/sensevoice_small/export_xh2a/hmonnx/model_XH2a.onnx`

## 4) 直接测试浮点精度（FP32 ONNX）

浮点模型目录为：`/data01/nfs_shared/ASR_TTS/SenseVoiceSmall`。

先按“1) 导出 FP32 ONNX”导出 `model.onnx`，然后可直接跑浮点评测：

### 4.1 LibriSpeech clean（streaming）

```bash
python examples/audio/sensevoice_small/sensevoice_float_eval.py \
  --model-dir /data01/nfs_shared/ASR_TTS/SenseVoiceSmall \
  --onnx work_dirs/sensevoice_small/export_fp32/onnx/model.onnx \
  --hf-dataset openslr/librispeech_asr \
  --hf-config clean \
  --hf-split test \
  --hf-streaming \
  --limit 100 \
  --report work_dirs/sensevoice_small/report/librispeech_clean_float.json
```

### 4.2 AISHELL-1

**注意**：AISHELL-1 数据集音频字段为 `wav`，且需外部提供文本标注文件（如 `aishell_transcript_v0.8.txt`）。

```bash
python examples/audio/sensevoice_small/sensevoice_float_eval.py \
  --model-dir /data01/nfs_shared/ASR_TTS/SenseVoiceSmall \
  --onnx work_dirs/sensevoice_small/export_fp32/onnx/model.onnx \
  --hf-dataset AISHELL/AISHELL-1 \
  --hf-split train \
  --limit 500 \
  --hf-audio-field wav \
  --hf-text-path examples/audio/sensevoice_small/aishell_transcript_v0.8.txt \
  --report work_dirs/sensevoice_small/report/aishell_1_float.json
```

## 5) 测试量化精度（HMONNX）

### 5.1 LibriSpeech clean（streaming）

```bash
python examples/audio/sensevoice_small/sensevoice_quant_eval.py \
  --model-dir /data01/nfs_shared/ASR_TTS/SenseVoiceSmall \
  --hmonnx work_dirs/sensevoice_small/export_xh2a_libri_128_minmax/hmonnx/model_XH2a.onnx \
  --hf-dataset openslr/librispeech_asr \
  --hf-config clean \
  --hf-split test \
  --hf-streaming \
  --limit 100 \
  --report work_dirs/sensevoice_small/report/librispeech_clean_quant.json
```

### 5.2 AISHELL-1

```bash
python examples/audio/sensevoice_small/sensevoice_quant_eval.py \
  --model-dir /data01/nfs_shared/ASR_TTS/SenseVoiceSmall \
  --hmonnx work_dirs/sensevoice_small/export_xh2a/hmonnx/model_XH2a.onnx \
  --hf-dataset AISHELL/AISHELL-1 \
  --hf-split train \
  --limit 500 \
  --hf-audio-field wav \
  --hf-text-path examples/audio/sensevoice_small/aishell_transcript_v0.8.txt \
  --report work_dirs/sensevoice_small/report/aishell_1_quant.json
```
