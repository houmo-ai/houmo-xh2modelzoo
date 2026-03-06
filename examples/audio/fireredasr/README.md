# FireRedASR xh2a 适配指南

## 目录

- [概述](#概述)
- [架构说明](#架构说明)
- [环境准备](#环境准备)
- [模型下载](#模型下载)
- [模型转换与导出](#模型转换与导出)
  - [Audio Encoder 导出](#audio-encoder-导出)
  - [LLM 导出 - 方案1: 融合 LoRA](#llm-导出---方案1-融合-lora)
  - [LLM 导出 - 方案2: 保留 LoRA](#llm-导出---方案2-保留-lora)
- [Demo 推理](#demo-推理)
- [Eval 评测](#eval-评测)
- [量化后模型评测](#量化后模型评测)
- [文件结构](#文件结构)
- [FAQ](#faq)

---

## 概述

本项目将 [FireRedASR-LLM](https://github.com/FireRedTeam/FireRedASR) 语音识别模型适配到 xh2a 平台。FireRedASR-LLM 是一个 Encoder-Adapter-LLM 三段式语音识别模型，基于 Conformer 编码器和 Qwen2-7B-Instruct 大语言模型。

**模型架构:**
```
WAV音频 → Fbank(80维) → Conv2dSubsampling(4x↓) → Conformer×16 → Adapter(2x↓) → Qwen2-7B(+LoRA) → 文字
```

**适配方案:**
- **Audio Encoder**: 直接导出 ONNX → HMONNX（不涉及 KV Cache）
- **LLM (Qwen2-7B)**:
  - 方案1: 融合 LoRA 权重后，复用 qwen2_legacy 导出流程
  - 方案2: 保留 LoRA 分支，导出带 lora_mask 的模型

## 本次迁移交付（2026-03-05）

### 迁移落地

- 已从 `xhquant_llm/examples/fireredasr` 迁移到 `xh2modelzoo/examples/audio/fireredasr`
- 本次新增/修复重点：
  - `fireredasr_hf_forward.py`: 修复 audio hmonnx 三输入接口、prefill 分块、prefill 末块 logits 取值
  - `fireredasr_xh2a_demo.py`: 修复 embedding 接口兼容、prefill 末块 logits 取值
  - `xh_model_zoo/xh_llm/models/llm_onnx_model.py`: 补齐 `get_input_embeddings`、`pad_token` padding 逻辑、`LLMLoRAONNXModel` 兼容类

### 你要求的两条导出链路

- 链路 A（不使用 common quant，默认全 `w8a8`）：
  1. 导出 audio encoder `w8a8`（等价“vision 分支”）
  2. 导出 llm prefill/decode（`merge_lora` 和 `keep_lora`）
  3. 跑整链 demo（HF vs HMONNX）

- 链路 B（使用 common quant）：
  1. 先做 common quant（`merge_lora` / `keep_lora`）
  2. 再导出 audio encoder（resume quant ckpt + rotated adapter）
  3. 再导出 llm prefill/decode（`merge_lora` / `keep_lora`）
  4. 跑整链 demo（HF vs HMONNX）

### 本地验证产物路径

- 基线 HF：
  - `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_demo_hf_ref.json`
- 链路 A (`w8a8`)：
  - audio: `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_audio_encoder_w8a8`
  - llm merge: `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_llm_merged_w8a8`
  - llm keep: `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_llm_lora_w8a8`
- 链路 B (`common quant`)：
  - common quant ckpt:
    - `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_merge_lora`
    - `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_keep_lora`
  - audio:
    - `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_audio_encoder_merge_common_quant`
    - `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_audio_encoder_keep_common_quant`
  - llm:
    - `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_llm_merged`
    - `/data01/datasets/xh2a_model/qwen2_5_vl/fireredasr_llm_lora`

### 当前验证结论（真实跑数）

- `w8a8` 导出链路：
  - LLM `valid_wrap_vs_hf` cosine: `0.9999989` (merge), `0.9999944` (keep)
  - LLM `valid_quant_vs_hf` cosine: `0.9953805` (merge), `0.9933393` (keep)
- `common quant` 导出链路：
  - LLM `valid_wrap_vs_hf` cosine: `0.9999974` (merge), `0.9999970` (keep)
  - LLM `valid_quant_vs_hf` cosine: `0.9927503` (merge), `0.9405762` (keep)
- Demo 一致性（单条，`BAC009S0764W0121`）：
  - HF: CER `0.0`
  - HMONNX (`w8a8/common quant`, keep/merge): 目前仍出现明显文本漂移，未达到与 HF 一致

> 当前状态是：迁移、导出和分阶段验证链路已打通；`HMONNX` 端到端文本一致性仍需继续专项排查（重点在 LLM decode 路径）。
>
> 详细跑数与命令见：`examples/audio/fireredasr/VALIDATION_REPORT_20260305.md`

---

## 架构说明

### 整体数据流

```
输入: WAV 音频文件 (16kHz)
  ↓
Fbank 特征提取 (80维, 25ms窗/10ms帧移, + CMVN归一化)
  ↓ shape: [B, T_f, 80]  (T_f ≈ 100 × 秒数)
Context Padding (+6帧)
  ↓ shape: [B, T_f+6, 80]
Conv2dSubsampling (两层 Conv2d(k=3,s=2), 4x下采样)
  ↓ shape: [B, T_c, 1280]  (T_c = ((T_f+6-3)//2+1-3)//2+1)
Conformer ×16 (Macaron-style: ½FFN→RelPos-MHSA→ConvModule→½FFN→LayerNorm)
  ↓ shape: [B, T_c, 1280]  (维度不变)
Adapter (相邻2帧拼接 + 2层MLP, 2x下采样)
  ↓ shape: [B, T_c//2, 3584]
LLM Embedding 融合 (<speech> token 替换为语音特征)
  ↓ shape: [B, S, 3584]
Qwen2-7B-Instruct (+LoRA r=64, alpha=16)
  ↓
输出: 文字
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `idim` | 80 | Fbank Mel bins |
| `d_model` | 1280 | Conformer 隐藏维度 |
| `n_layers_enc` | 16 | Conformer 层数 |
| `n_head` | 20 | 注意力头数 |
| `d_k` | 64 | 每头维度 (1280/20) |
| `d_inner` | 5120 | FFN 扩展维度 (1280×4) |
| `kernel_size` | 33 | ConvModule 卷积核 |
| `llm_dim` | 3584 | Qwen2-7B hidden_size |
| `encoder_downsample_rate` | 2 | Adapter 下采样率 |
| `lora_r` | 64 | LoRA rank |
| `lora_alpha` | 16 | LoRA scaling |
| `总时间压缩比` | 8x | Conv 4x × Adapter 2x |

### ONNX 导出的输入输出

**Audio Encoder ONNX:**

| 名称 | Shape | 说明 |
|------|-------|------|
| 输入: `fbank_features` | `[B, 3000, 80]` | Pad到30s的Fbank特征 |
| 输入: `attn_mask` | `[B, 1, 1, T_conv]` | Self-Attention 加法 mask (0=valid, -65504=padding) |
| 输入: `conv_mask` | `[B, 1, T_conv]` | ConvModule 乘法 mask (1=valid, 0=padding) |
| 输出: `speech_features` | `[B, T_out, 3584]` | 送入LLM的语音特征 |

其中:
- `T_conv = ((3006-3)//2+1-3)//2+1 = 751` (30s 音频)
- `T_out = T_conv // 2 = 375` (Adapter 2x 下采样)
- `-65504` 为 fp16 可表示的最大负值，替代 `-inf`

---

## 环境准备

### 1. 克隆仓库

```bash
# 克隆 xh2modelzoo
cd /home/user/
git clone <xh2modelzoo_repo_url> xh2modelzoo
cd xh2modelzoo

# 克隆 FireRedASR (放在同级目录)
cd ..
git clone https://github.com/FireRedTeam/FireRedASR.git
cd FireRedASR
```

### 2. 安装依赖

```bash
# 安装 xh2modelzoo 依赖
cd xh2modelzoo
pip install -e .

# 安装 FireRedASR 依赖
cd ../FireRedASR
pip install -r requirements.txt
pip install kaldi_native_fbank kaldiio

# 安装 peft (LoRA 支持)
pip install peft

# 安装 ONNX 相关
pip install onnx onnxruntime onnxsim
```

### 3. 验证环境

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import xhquant; print(f'xhquant: OK')"
python -c "import fireredasr; print(f'FireRedASR: OK')"
python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')"
```

---

## 模型下载

### FireRedASR-LLM-L 模型

```bash
# 下载 FireRedASR-LLM-L
# 方式1: HuggingFace
pip install huggingface_hub
huggingface-cli download FireRedTeam/FireRedASR-LLM-L --local-dir ./pretrained_models/FireRedASR-LLM-L

# 方式2: ModelScope
pip install modelscope
modelscope download --model FireRedTeam/FireRedASR-LLM-L --local_dir ./pretrained_models/FireRedASR-LLM-L
```

### Qwen2-7B-Instruct 基座模型

```bash
# 下载到 FireRedASR 模型目录下
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir ./pretrained_models/FireRedASR-LLM-L/Qwen2-7B-Instruct
```

### 目录结构验证

```bash
ls pretrained_models/FireRedASR-LLM-L/
# 应包含:
# model.pth.tar           - 3.6GB, 含 encoder + adapter + LoRA 权重
# asr_encoder.pth.tar     - encoder 结构参数
# cmvn.ark                - CMVN 归一化参数
# config.yaml             - 配置文件
# Qwen2-7B-Instruct/     - Qwen2 基座模型目录
```

---

## 模型转换与导出

### Audio Encoder 导出

Audio Encoder 包含 Conv2dSubsampling + Conformer×16 + Adapter，不涉及 KV Cache，直接导出为 ONNX。

#### xh2a 适配优化

导出前会自动对 Conformer 进行 **Impl 替换 (patch)**，消除 xh2a 不支持的算子：

| 原始实现 | 问题算子 | 替换后实现 | 说明 |
|---------|---------|-----------|------|
| `mask.eq(0)` → `masked_fill(-inf)` | `.eq()`, `masked_fill`, `-inf` | `attn + attn_mask + attn_mask` | Self-Attention 加法 mask，-65504×2 经 softmax 后趋近于 0 |
| `mask.ne(1)` → `masked_fill_(0.0)` | `.ne()`, `masked_fill_` | `out * conv_mask` | ConvModule 乘法 mask，直接与 0/1 mask 相乘 |

**双 Mask 策略:**
- `attn_mask [B, 1, 1, T_conv]`：**加法** mask，有效位置 0，padding 位置 -65504。在 softmax 前叠加两次，使 padding 位置的注意力权重趋近于 0
- `conv_mask [B, 1, T_conv]`：**乘法** mask，有效位置 1，padding 位置 0。在 ConvModule 的 pointwise_conv1 前和 pointwise_conv2 后各乘一次，将 padding 位置归零

> 参考: `xhquant_llm/models/minicpmo/_tts_model_impl.py` (attention_mask 加两次模式)，`xhquant_llm/models/minicpmo/_tts_vocos_model_impl.py` (乘法 mask 模式)

#### Conv2dSubsampling 不需要 Mask

Conv2dSubsampling 处理 zero-padding 区域时自然产生近零输出，后续 Conformer 的 attn_mask/conv_mask 会处理无效位置。这与原始 FireRedASR 的设计一致（Conv2dSubsampling 内部不对卷积结果做 mask，只在卷积后重新计算 output_lengths 并生成新的 mask）。

#### 导出命令

```bash
cd xh2modelzoo

# 步骤1: 导出 ONNX (含精度验证)
python examples/audio/fireredasr/audio_encoder_xh2a_export.py \
    --model_dir weights/FireRedASR-LLM-L \
    --output_dir ./work_dirs/fireredasr_audio_encoder \
    --audio_seconds 30.0 \
    --batch_size 1 \
    --valid \
    --valid_asr \
    --export_hmonnx \
     --use_gpu 

# 步骤2: 同时导出 HMONNX
python examples/audio/fireredasr/audio_encoder_xh2a_export.py \
    --model_dir /path/to/FireRedASR-LLM-L \
    --output_dir ./work_dirs/fireredasr_audio_encoder \
    --audio_seconds 10.0 \
    --batch_size 1 \
    --valid \
    --export_hmonnx

# 完整验证 (含 ASR 转写对比，需 GPU)
python examples/audio/fireredasr/audio_encoder_xh2a_export.py \
    --model_dir /path/to/FireRedASR-LLM-L \
    --output_dir ./work_dirs/fireredasr_audio_encoder \
    --valid --valid_asr --export_hmonnx --use_gpu

# 与 QuaRot/GPTQ LLM 联调（加载 quant checkpoint + rotated projector）
python examples/audio/fireredasr/audio_encoder_xh2a_export.py \
    --model_dir  weights/FireRedASR-LLM-L \
    --output_dir ./work_dirs/fireredasr_audio_encoder \
    --resume_from work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_merge_lora/quarot_gptq-state-dict.safetensors \
    --rotated_adapter_path work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_merge_lora/audio_projector_rotated.safetensors \
    --valid --valid_asr --export_hmonnx --use_gpu \
    --generate_hmonnx_golden \
    --golden_dir ./work_dirs/fireredasr_audio_encoder/golden/audio_encoder

# 生成 Audio Encoder HMONNX golden
python examples/audio/fireredasr/audio_encoder_xh2a_export.py \
    --model_dir /path/to/FireRedASR-LLM-L \
    --output_dir ./work_dirs/fireredasr_audio_encoder \
    --export_hmonnx \
    --generate_hmonnx_golden \
    --golden_dir ./work_dirs/fireredasr_audio_encoder/golden/audio_encoder
```

#### 验证流程

导出脚本内置多级验证，使用 `--valid` 参数自动执行：

| 阶段 | 对比对象 | 指标 | 说明 |
|------|---------|------|------|
| Dummy 验证 | patched PyTorch vs ONNX | max_diff < 1e-3 | 验证 ONNX 导出正确性 |
| 真实音频验证 | 原始 encoder vs ONNX | max_diff / cosine_sim | 逐条对比 encoder 输出 |
| ASR 转写对比 | 原始模型 vs ONNX 替换后 | 文本完全匹配 | 验证端到端一致性 (`--valid_asr`) |
| HMONNX Dummy | ONNX vs HMONNX | max_diff < 0.5 | 验证量化精度 (`--export_hmonnx`) |
| HMONNX 真实音频 | 原始 encoder vs HMONNX | max_diff / cosine_sim | HMONNX 量化后精度 |
| HMONNX ASR | 原始模型 vs HMONNX 替换后 | 文本完全匹配 | HMONNX 端到端验证 |

**输出文件:**
```
work_dirs/fireredasr_audio_encoder/
├── audio_encoder.onnx                    # ONNX 模型
├── audio_encoder_external_data           # ONNX 外部数据
├── audio_encoder_hmonnx.onnx             # HMONNX 模型 (可选)
```

### LLM 导出 - 方案1: 融合 LoRA

将 FireRedASR 的 LoRA 权重 (r=64, alpha=16) 融合到 Qwen2-7B-Instruct 基座中，然后按标准 qwen2_legacy 流程导出。

**适用场景:** 只做语音识别任务，不需要共享基座模型。

```bash
# 导出融合 LoRA 后的 LLM
python examples/audio/fireredasr/audio_llm_xh2a_export.py \
    --config configs/fireredasr/fireredasr_llm_xh2a_4k.py \
    --fireredasr_model_dir weights/FireRedASR-LLM-L \
    --lora_mode merge_lora \
    --valid \
    --valid_asr \
    --use_gpu

# 输出到 work_dirs/fireredasr_llm_merged/
```

**输出文件:**
```
work_dirs/fireredasr_llm_merged/
├── prefill_onnx/                         # Prefill HMONNX
│   └── fireredasr_llm_merged_prefill.onnx
├── decode_onnx/                          # Decode HMONNX
│   └── fireredasr_llm_merged_decode.onnx
├── token_embedding.pt                    # Token Embedding
├── hf_config/                            # HuggingFace 配置
├── export_meta_info.json                 # 导出元信息
```

### LLM 导出 - 方案2: 保留 LoRA

保留 LoRA 分支，导出带 `lora_mask` 输入的模型。运行时通过 mask 控制 LoRA 开关。

**适用场景:** 识别和翻译共享同一个 Qwen2 基座，通过不同 LoRA 切换任务。

```bash
# 导出保留 LoRA 的 LLM
python examples/audio/fireredasr/audio_llm_xh2a_export.py \
    --config configs/fireredasr/fireredasr_llm_xh2a_4k.py \
    --fireredasr_model_dir weights/FireRedASR-LLM-L \
    --lora_mode keep_lora \
    --valid \
    --valid_asr \
    --use_gpu

# 输出到 work_dirs/fireredasr_llm_lora/
```

**LoRA 运行时控制:**
```python
# lora_mask=1.0 时启用 LoRA (语音识别模式)
# lora_mask=0.0 时禁用 LoRA (使用基座模型)
data_batch["lora_mask"] = torch.tensor([1.0])  # 启用 ASR LoRA
```

---

## Demo 推理

### 原始模型推理（基准）

```bash
# 使用 FireRedASR 原始模型推理
python examples/audio/fireredasr/fireredasr_xh2a_demo.py \
    --mode native \
    --model_dir /path/to/FireRedASR-LLM-L \
    --wav_path /path/to/test.wav \
    --use_gpu
```

### ONNX Audio Encoder + 原始 LLM

```bash
python examples/audio/fireredasr/fireredasr_xh2a_demo.py \
    --mode onnx \
    --model_dir /path/to/FireRedASR-LLM-L \
    --audio_onnx_path ./work_dirs/fireredasr_audio_encoder/audio_encoder.onnx \
    --wav_path /path/to/test.wav \
    --use_gpu
```

### 全 HMONNX 推理（xh2a 部署）

```bash
python examples/audio/fireredasr/fireredasr_xh2a_demo.py \
    --mode hmonnx \
    --model_dir /path/to/FireRedASR-LLM-L \
    --audio_hmonnx_path ./work_dirs/fireredasr_audio_encoder/audio_encoder_hmonnx.onnx \
    --llm_hmonnx_dir ./work_dirs/fireredasr_llm_merged \
    --wav_path /path/to/test.wav \
    --use_gpu
```

### 批量推理

```bash
# 多个音频文件
python examples/audio/fireredasr/fireredasr_xh2a_demo.py \
    --mode onnx \
    --model_dir /path/to/FireRedASR-LLM-L \
    --audio_onnx_path ./work_dirs/fireredasr_audio_encoder/audio_encoder.onnx \
    --wav_path audio1.wav audio2.wav audio3.wav \
    --beam_size 1 \
    --repetition_penalty 1.0
```

---

## Eval 评测

### 使用 FireRedASR 原始评测脚本

```bash
cd FireRedASR

# 原始模型评测
python fireredasr/speech2text.py \
    --asr_type llm \
    --model_dir ./pretrained_models/FireRedASR-LLM-L \
    --wav_scp /path/to/test/wav.scp \
    --result_file results.txt \
    --use_gpu \
    --batch_size 1
```

### 使用 xh2a Demo 进行评测

```bash
cd xh2modelzoo

# 带参考文本的评测（计算 WER）
python examples/audio/fireredasr/fireredasr_xh2a_demo.py \
    --mode onnx \
    --model_dir /path/to/FireRedASR-LLM-L \
    --audio_onnx_path ./work_dirs/fireredasr_audio_encoder/audio_encoder.onnx \
    --wav_path $(cat /path/to/wav.scp | awk '{print $2}') \
    --ref_file /path/to/ref.txt \
    --use_gpu
```

### 参考文本格式

```
# ref.txt 格式: uttid 文本
utt001 今天天气很好
utt002 请问你叫什么名字
utt003 我想预约明天下午三点的会议
```

---

## 量化后模型评测

### 评测流程

```bash
# 1. 先导出 Audio Encoder (无量化)
python examples/audio/fireredasr/audio_encoder_xh2a_export.py \
    --model_dir /path/to/FireRedASR-LLM-L \
    --output_dir ./work_dirs/fireredasr_audio_encoder \
    --valid --export_hmonnx

# 2. 导出量化后的 LLM
python examples/audio/fireredasr/audio_llm_xh2a_export.py \
    --config configs/fireredasr/fireredasr_llm_xh2a_4k.py \
    --fireredasr_model_dir /path/to/FireRedASR-LLM-L \
    --lora_mode merge_lora \
    --valid

# 3. 使用量化后模型进行推理评测
python examples/audio/fireredasr/fireredasr_xh2a_demo.py \
    --mode hmonnx \
    --model_dir /path/to/FireRedASR-LLM-L \
    --audio_hmonnx_path ./work_dirs/fireredasr_audio_encoder/audio_encoder_hmonnx.onnx \
    --llm_hmonnx_dir ./work_dirs/fireredasr_llm_merged \
    --wav_path /path/to/test_wav_files/*.wav \
    --ref_file /path/to/ref.txt \
    --use_gpu
```

### 精度对比

| 模型 | Audio Encoder | LLM | WER (%) |
|------|--------------|-----|---------|
| 原始 (基准) | PyTorch FP32 | Qwen2+LoRA FP16 | - |
| ONNX | ONNX FP32 | Qwen2+LoRA FP16 | - |
| HMONNX (量化) | HMONNX INT8 | HMONNX INT8 | - |

> 注: WER 数值需要在实际测试集上运行后填写。

### QuaRot + GPTQ（merge/keep 全流程）

下面给出 FireRedASR 在 `merge_lora` 和 `keep_lora` 两种模式下的统一流程。建议每个阶段都带 `--valid` 做阶段验收。

#### 0. 环境

```bash
conda activate xhquant_fireredasr2s
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=2
```

> 注意：如果你用 VSCode `launch.json`，`env.CUDA_VISIBLE_DEVICES` 会覆盖终端里的 `export`。

#### 1. 非量化导出（基线）

```bash
# merge_lora 基线导出
python examples/audio/fireredasr/audio_llm_xh2a_export.py \
    --config configs/fireredasr/fireredasr_llm_xh2a_4k.py \
    --fireredasr_model_dir weights/FireRedASR-LLM-L \
    --lora_mode merge_lora \
    --valid \
    --valid_asr \
    --use_gpu

# keep_lora 基线导出
python examples/audio/fireredasr/audio_llm_xh2a_export.py \
    --config configs/fireredasr/fireredasr_llm_xh2a_4k.py \
    --fireredasr_model_dir weights/FireRedASR-LLM-L \
    --lora_mode keep_lora \
    --valid \
    --valid_asr \
    --use_gpu
```

#### 2. common_quant（QuaRot + GPTQ）

```bash
# merge_lora 量化
python examples/audio/fireredasr/audio_llm_xh2a_common_quant.py \
    --config configs/fireredasr/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp.py \
    --fireredasr_model_dir weights/FireRedASR-LLM-L \
    --lora_mode merge_lora \
    --gptq_calib_dataset wikitext2 \
    --gptq_calib_samples 128 \
    --gptq_seqlen 2048 \
    --save_rotation_matrix \
    --rotate_audio_projector \
    --valid \
    --valid_asr \
    --use_gpu \
    --reuse_gptq_layer_cache

# keep_lora 量化
python examples/audio/fireredasr/audio_llm_xh2a_common_quant.py \
    --config configs/fireredasr/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp.py \
    --fireredasr_model_dir weights/FireRedASR-LLM-L \
    --lora_mode keep_lora \
    --gptq_calib_dataset wikitext2 \
    --gptq_calib_samples 128 \
    --gptq_seqlen 2048 \
    --save_rotation_matrix \
    --rotate_audio_projector \
    --valid \
    --valid_asr \
    --use_gpu \
    --reuse_gptq_layer_cache
```

输出目录会包含：
- `quarot_rotation_matrix.pt`
- `quarot_gptq-state-dict.safetensors`
- `audio_projector_rotated.safetensors`（启用 `--rotate_audio_projector` 时）
- 日志中的 `valid_*` 对比项（启用 `--valid` 时）
- 日志中的 `ASR-CMP-QUANT` 与 `valid_asr summary`（启用 `--valid_asr` 时）

说明：
- `audio_llm_xh2a_common_quant.py` 默认会重建 `layers_cache`，避免误复用旧 GPTQ layer cache。
- 如需复用缓存加速，请显式传 `--reuse_gptq_layer_cache`。
- `keep_lora` + `--valid_asr` 会先把运行时 LoRA buffer 临时融合到 LLM 权重，再做 ASR 对比。
- `audio_projector_rotated.safetensors` 必须使用“只旋转 `linear2`”的新逻辑生成；历史版本如果同时旋转了 `linear1/linear2`，会出现 ASR 重复字（如“我我我...”）。
- 若你怀疑用了旧的 rotated adapter，重新执行 `common_quant` 并带 `--rotate_audio_projector` 重新生成即可。

#### 3. resume 导出（量化权重回灌）

```bash

python examples/audio/fireredasr/audio_llm_xh2a_export.py \
    --config configs/fireredasr/fireredasr_llm_xh2a_4k.py \
    --fireredasr_model_dir weights/FireRedASR-LLM-L \
    --lora_mode merge_lora \
    --resume_from work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_merge_lora/quarot_gptq-state-dict.safetensors \
    --rotated_adapter_path work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_merge_lora/audio_projector_rotated.safetensors \
    --valid \
    --valid_asr \
    --use_gpu

# keep_lora 示例
python examples/audio/fireredasr/audio_llm_xh2a_export.py \
    --config configs/fireredasr/fireredasr_llm_xh2a_4k.py \
    --fireredasr_model_dir weights/FireRedASR-LLM-L \
    --lora_mode keep_lora \
    --resume_from work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_keep_lora/quarot_gptq-state-dict.safetensors \
    --rotated_adapter_path work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_keep_lora/audio_projector_rotated.safetensors \
    --valid \
    --valid_asr \
    --use_gpu
```

`--valid` 重点看 `export_meta_info.json` 中：
- `valid_wrap_vs_hf.cosine_similarity`
- `valid_quant_vs_hf.cosine_similarity`

通常 `valid_quant_vs_hf` 的 cosine 在 `0.99+` 代表量化链路正常。

`resume_from` 现在会显式把 checkpoint 里的 `*.quant_weight` 回灌到对应 Linear（`register_buffer("quant_weight")`），确保导出阶段使用 GPTQ 权重，而不是退回默认 W8 PTQ 路径。

`audio_llm_xh2a_export.py` 默认会优先使用 `--fireredasr_model_dir/Qwen2-7B-Instruct` 作为 HF 基座（除非显式传 `--hf_model_dir`）。

`keep_lora + quarot` 路径下，如果 `resume_from` 来自历史坏版本 common quant（LoRA 未按 QuaRot 旋转），导出脚本会直接报错并提示重跑 common quant，避免导出错误模型。

#### 4. HMONNX golden 导出与对比

```bash
python examples/audio/fireredasr/audio_llm_hmonnx_golden.py \
    --model-config work_dirs/fireredasr_llm_lora/export_meta_info.json \
    --max_decode_steps 5 \
    --exec_device cuda
```

输出目录：
- `work_dirs/fireredasr_llm_lora_hmonnx_golden/golden/prefill`
- `work_dirs/fireredasr_llm_lora_hmonnx_golden/golden/decode`

#### 5. 用真实语音做 ASR 结果对比

```bash
# 原始 HF 前向（基线）
python examples/audio/fireredasr/fireredasr_hf_forward.py \
    --mode hf \
    --model_dir weights/FireRedASR-LLM-L \
    --wav_path <test.wav> \
    --use_gpu

# 接入导出后的模型（对比）
python examples/audio/fireredasr/fireredasr_hf_forward.py \
    --mode hmonnx \
    --model_dir weights/FireRedASR-LLM-L \
    --llm_hmonnx_dir work_dirs/fireredasr_llm_lora \
    --rotated_adapter_path work_dirs/<your_quant_dir>/audio_projector_rotated.safetensors \
    --wav_path <test.wav> \
    --use_gpu
```

如果有标注文本，建议再跑 WER 做最终验收。

#### 6. 联合推理 Demo（encoder hmonnx + prefill hmonnx + decode hmonnx）

```bash
# 方式1：推荐，直接用 fireredasr_hf_forward.py（支持目录/通配符）
python examples/audio/fireredasr/fireredasr_hf_forward.py \
    --mode hmonnx \
    --model_dir weights/FireRedASR-LLM-L \
    --audio_hmonnx_path work_dirs/fireredasr_audio_encoder/audio_encoder_hmonnx.onnx \
    --llm_hmonnx_dir work_dirs/fireredasr_llm_merged \
    --rotated_adapter_path work_dirs/<quant_dir>/audio_projector_rotated.safetensors \
    --wav_path data/wav/*.wav \
    --use_gpu \
    --out_json work_dirs/fireredasr_joint_hmonnx_results.json

# 方式2：fireredasr_xh2a_demo.py（新增 rotated_adapter_path）
python examples/audio/fireredasr/fireredasr_xh2a_demo.py \
    --mode hmonnx \
    --model_dir weights/FireRedASR-LLM-L \
    --audio_hmonnx_path work_dirs/fireredasr_audio_encoder/audio_encoder_hmonnx.onnx \
    --llm_hmonnx_dir work_dirs/fireredasr_llm_merged \
    --rotated_adapter_path work_dirs/<quant_dir>/audio_projector_rotated.safetensors \
    --wav_path data/wav/BAC009S0764W0121.wav \
    --use_gpu
```

说明：
- `llm_hmonnx_dir` 目录下应包含 `export_meta_info.json`、prefill/decode onnx、token_embedding。
- 使用 QuaRot/GPTQ 时，建议始终传 `--rotated_adapter_path`，确保语音 projector 与 LLM 旋转基底一致。

### 已修复问题说明（keep_lora + quarot）

`fuse_layer_norms -> fuse_ln_linear` 在融合 `LayerNorm/RMSNorm` 到 Linear 时，之前只融合了基座权重 `W`，没有同步融合 LoRA 的 `A`：

- 正确做法：`W' = W ⊙ norm_w`，`A' = A ⊙ norm_w`
- 修复位置：`xh_model_zoo/xh_llm/quarot/rotation_utils.py`
- 修复函数：
  - `_fuse_ln_lora_a`（`fuse_ln_linear` 后同步处理 `weight_lora_a`）
  - `_rotate_linear_lora_input` / `_rotate_linear_lora_output`（QuaRot 时同步旋转 LoRA A/B）

修复后再跑 common_quant + resume，`keep_lora` 的量化精度会恢复并与 `merge_lora` 对齐。

### 已修复问题说明（audio projector 旋转）

FireRedASR 的 adapter 结构是 `linear1 -> ReLU -> linear2`。  
QuaRot 基底对齐时，只有最终输出投影 `linear2` 可以安全做输出侧旋转；如果把 `linear1` 也旋转，会因为中间非线性导致语音特征分布失真，表现为 `valid_asr` 大幅退化和重复字。

- 正确做法：只旋转 `linear2.weight` 和 `linear2.bias`
- 错误做法：旋转 `linear1.* + linear2.*`

---

## 文件结构

```
examples/audio/fireredasr/
├── audio_encoder_xh2a_export.py   # Audio Encoder ONNX/HMONNX 导出
├── audio_llm_xh2a_export.py       # LLM (Qwen2 + LoRA) 导出
├── fireredasr_hf_forward.py       # HF Forward 替换模块
├── fireredasr_xh2a_demo.py        # ASR Demo (支持多种后端)
└── README.md                      # 本文档

configs/qwen2/7b/
├── qwen2_7b_xh2a_4k.py           # Qwen2-7B 基础配置
├── qwen2_7b_instruct_xh2a_4k.py  # Qwen2-7B-Instruct 配置
└── qwen2_7b_instruct_xh2a_2k.py  # 2K 序列长度版本

configs/fireredasr/
├── fireredasr_llm_xh2a_4k.py                          # FireRedASR LLM 导出配置
└── fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp.py    # FireRedASR QuaRot+GPTQ(SSFP) 配置
```

### 关键模块说明

| 文件 | 用途 |
|------|------|
| `audio_encoder_xh2a_export.py` | 将 ConformerEncoder + Adapter 导出为 ONNX/HMONNX |
| `audio_llm_xh2a_export.py` | 加载 LoRA 权重并导出 Qwen2 LLM（融合/保留两种模式） |
| `fireredasr_hf_forward.py` | 将导出的模型接入 FireRedASR 的 forward 和 eval |
| `fireredasr_xh2a_demo.py` | 完整的 ASR 推理 Demo |

---

## FAQ

### Q: 为什么 Audio Encoder 使用直接导出 ONNX 而不是 wrap 方式？

A: Audio Encoder (Conformer + Adapter) 不涉及 KV Cache，是纯前向计算，没有状态管理需求。直接导出 ONNX 更简单高效，避免了 DynamicModule 替换的复杂性。这与 qwen2.5-vl 的 vision encoder 导出方式一致。

### Q: 两种 LoRA 方案如何选择？

A: 
- **融合 LoRA (merge_lora)**: 推荐用于只做 ASR 的场景。模型更小，推理更快，部署更简单。
- **保留 LoRA (keep_lora)**: 适用于多任务场景（如 ASR + 翻译共享基座）。通过 `lora_mask` 控制任务切换，但模型稍大。

### Q: 如何验证导出模型的精度？

A: 
1. 使用 `--valid` 参数在导出时进行自动验证
2. 使用 `fireredasr_xh2a_demo.py` 对比原始模型和导出模型的输出
3. 使用 `--ref_file` 参数计算 WER 进行定量评估

### Q: 支持的音频格式和长度限制？

A:
- 格式: WAV (16kHz, 单声道)
- 最大长度: 30 秒（Audio Encoder 固定 pad 到 3000 帧）
- 超过 30 秒的音频需要分段处理

### Q: attn_mask 和 conv_mask 的区别？

A: 为了避免 xh2a 不支持的 `.eq()` / `.ne()` / `masked_fill` / `-inf` 算子，我们将原始的单一 bool mask 拆分为两个 float mask：

| Mask | Shape | 类型 | 有效位置 | Padding 位置 | 用途 |
|------|-------|------|---------|-------------|------|
| `attn_mask` | `[B, 1, 1, T_conv]` | 加法 | 0 | -65504 | Self-Attention: `attn = attn + attn_mask + attn_mask` |
| `conv_mask` | `[B, 1, T_conv]` | 乘法 | 1 | 0 | ConvModule: `out = out * conv_mask` |

两者根据相同的有效长度信息生成，只是数值表示不同。`attn_mask` 的 -65504 叠加两次后 softmax 趋近于 0；`conv_mask` 直接将无效位置归零。Conv2dSubsampling 本身不需要 mask。
