# Qwen3-TTS

Qwen3-TTS 模型由 4 个子模型构成：Talker、CodePredictor、TextProjection、SpeechTokenizer。

本目录提供以下功能：

- **一键自动化脚本**：`qwen3tts_pipeline.sh` 一键完成导出、测试、评估全流程
- **Native PyTorch 推理**：统一的推理示例脚本 `native_demo.py`，支持 voice-clone/voice-design/custom-voice 三种模式
- **HMONNX 推理**：从 HF 权重到 HMONNX (XH2a) 的完整导出 + 整链路推理（生产推荐）
- **精度评估**：多 GPU 并行评估脚本 `qwen3_tts_eval.py`，支持大规模精度测试

## 快速开始：一键自动化脚本 ⭐

使用 `qwen3tts_pipeline.sh` 一键完成模型导出、测试和评估：

```bash
# 完整流程：导出 + 测试 + 评估
./qwen3tts_pipeline.sh --model 1.7b --export --test-native --test-hmonnx --eval

# 只导出模型
./qwen3tts_pipeline.sh --model 1.7b --export

# 同时处理两个模型
./qwen3tts_pipeline.sh --model 1.7b,0.6b --export --test-hmonnx
```

**支持的操作**：
- `--export`: 导出 HMONNX 模型（4 个子模型）
- `--test-native`: 测试原始浮点模型
- `--test-hmonnx`: 测试 HMONNX 模型
- `--eval`: 运行精度评估（Native + HMONNX）

详细使用说明见 [README_PIPELINE.md](README_PIPELINE.md)。

---

## 依赖

### 环境要求

```bash
# 使用 xhquant 环境
conda activate xhquant
```

### Python 包依赖

**核心依赖**:
```bash
# Python 版本
python==3.12.0

# PyTorch (CUDA 12.8)
torch==2.8.0+cu128
transformers==4.57.3

# 音频处理
torchaudio==2.8.0+cu128
soundfile==0.13.1

# 工具库
loguru==0.7.3
tqdm==4.67.3
numpy==2.2.6
```

**HMONNX 导出依赖**:
```bash
# ONNX 相关
onnx==1.16.2
onnxruntime==1.19.0  # GPU 版本
onnx-simplifier==0.4.0
onnxscript==0.4.0

# xhquant (内部依赖)
xhquant  # 通过 xh2modelzoo 安装
```

**Qwen3-TTS 原始实现**:
```bash
# 提供 Qwen3-TTS 原始 HF 实现
# 会被 XHQwen3TTSModel.from_pretrained 通过 class swap 替换为 XH 变体
pip install qwen_tts
```

### 完整安装命令

```bash
# 激活环境
conda activate xhquant

# 安装基础依赖
pip install torch==2.8.0 transformers==4.57.3 torchaudio==2.8.0
pip install soundfile==0.13.1 loguru==0.7.3 tqdm==4.67.3 numpy==2.2.6

# 安装 ONNX 相关（如需导出 HMONNX）
pip install onnx==1.16.2 onnxruntime==1.19.0 onnx-simplifier==0.4.0 onnxscript==0.4.0

# 安装 Qwen3-TTS
pip install qwen_tts
```

### 测试环境

以上版本在以下环境中测试通过：
- **操作系统**: Linux 5.15.0
- **GPU**: NVIDIA RTX A6000
- **CUDA**: 12.8
- **Python**: 3.12.0

---

## 快速开始：Native PyTorch 推理

使用统一的 `native_demo.py` 脚本进行 Native PyTorch 推理。

**支持两种模型加载方式：**
- **XH Wrapped Model（默认）**：使用 `XHQwen3TTSModel`，经过 XH 优化和适配
- **Original Model**：使用原始 `Qwen3TTSModel`（添加 `--use-original` 参数）

### Voice Clone 模式（音色克隆）

基于参考音频克隆音色，适用于 **0.6B-Base** 模型：

```bash
# 使用 XH Wrapped Model（默认）
PYTHONPATH=<xh2modelzoo path> \
python native_demo.py \
    --mode voice-clone \
    --model ./data/models/Qwen3-TTS-12Hz-0.6B-Base/ \
    --ref_audio /tmp/clone_1.wav \
    --ref_text "甚至出现交易几乎停滞的情况。" \
    --text "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈。" \
    --out output_clone.wav

# 使用原始 Qwen3TTSModel
PYTHONPATH=<xh2modelzoo path> \
python native_demo.py \
    --mode voice-clone \
    --use-original \
    --model ./data/models/Qwen3-TTS-12Hz-0.6B-Base/ \
    --ref_audio /tmp/clone_1.wav \
    --ref_text "甚至出现交易几乎停滞的情况。" \
    --text "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈。" \
    --out output_clone_original.wav
```

**准备参考音频**:
```bash
curl -sSL -o /tmp/clone_1.wav \
    https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav
```

### Voice Design 模式（音色设计）

基于文本描述生成音色，适用于 **1.7B-VoiceDesign** 模型：

```bash
PYTHONPATH=<xh2modelzoo path> \
python native_demo.py \
    --mode voice-design \
    --model ./data/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign/ \
    --text "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈。" \
    --instruct "体现温柔甜美的女声，音调适中，语速平稳。" \
    --out output_design.wav
```

### Custom Voice 模式（预定义音色）

使用预定义的 speaker，适用于 **0.6B-CustomVoice** 模型：

```bash
PYTHONPATH=<xh2modelzoo path> \
python native_demo.py \
    --mode custom-voice \
    --model ./data/models/Qwen3-TTS-12Hz-0.6B-CustomVoice/ \
    --text "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈。" \
    --speaker serena \
    --out output_custom.wav
```

**可用的 speaker**: `serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`

### 参数说明

运行 `python native_demo.py --help` 查看完整参数列表。常用参数：

- `--mode`: 推理模式（`voice-clone`/`voice-design`/`custom-voice`）
- `--model`: 模型路径
- `--text`: 要生成的文本
- `--use-original`: 使用原始 `Qwen3TTSModel`（不使用 XH wrapper）
- `--device`: 设备（默认 `cuda:0`）
- `--dtype`: 数据类型（`fp16`/`fp32`/`bf16`，默认 `fp16`）
- `--attn`: 注意力实现（`sdpa`/`flash_attention_2`/`eager`，默认 `sdpa`）

**dtype 注意事项**:

| dtype × attn | 是否可跑 |
|---|---|
| fp32 + sdpa | ✅ 验证通过 |
| bf16 + flash_attention_2 | ✅ 官方推荐 |
| fp16 + sdpa | ❌ multinomial NaN（softmax 数值溢出） |
| fp16 + flash_attention_2 | 未测 |

---

## 多 GPU 并行精度评估

使用 `qwen3_tts_eval.py` 进行大规模精度评估，支持 Native PyTorch 和 HMONNX 两种推理模式，使用多 GPU 并行加速。

### 功能特性

- ✅ 支持 Native PyTorch 和 HMONNX 两种推理模式
- ✅ 多 GPU 并行推理，自动数据分片
- ✅ 支持多种 speaker 选择模式（固定、随机、轮询）
- ✅ 基于 CV3-Eval zero_shot/zh 数据集（500 条中文样本）
- ✅ 自动生成音频文件供后续精度评估

### HMONNX 模式（推荐）

HMONNX 模式使用量化模型，推理稳定且高效。

```bash
# 测试 20 条样本，使用 4 个 GPU
PYTHONPATH=<xh2modelzoo path> \
python qwen3_tts_eval.py \
    --mode hmonnx \
    --gpus 0,1,2,3 \
    --max-samples 20 \
    --speaker-mode round-robin

# 全量测试 500 条，使用 8 个 GPU
PYTHONPATH=<xh2modelzoo path> \
python qwen3_tts_eval.py \
    --mode hmonnx \
    --gpus 0,1,2,3,4,5,6,7 \
    --speaker-mode round-robin
```

### Native 模式

Native 模式使用原生 PyTorch 模型。

**注意**: 当前版本存在 CUDA 采样错误（`probability tensor contains either inf, nan or element < 0`），建议使用 HMONNX 模式。

```bash
# 测试 20 条样本
PYTHONPATH=<xh2modelzoo path> \
python qwen3_tts_eval.py \
    --mode native \
    --gpus 0,1,2,3 \
    --max-samples 20 \
    --speaker-mode round-robin
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 推理模式：`native` 或 `hmonnx` | `native` |
| `--gpus` | GPU 列表，逗号分隔，如 `0,1,2,3` | `0` |
| `--max-samples` | 最大样本数，不指定则处理全部 | 全部 |
| `--speaker-mode` | Speaker 选择模式：`fixed`/`random`/`round-robin` | `fixed` |
| `--speaker` | 固定 speaker 名称（仅 `fixed` 模式） | `serena` |

### Speaker 模式说明

- **fixed**: 所有样本使用同一个 speaker（通过 `--speaker` 指定）
- **random**: 每个样本随机选择一个 speaker
- **round-robin**: 样本按顺序轮流使用不同 speaker（推荐用于评估）

### 输出结果

生成的音频文件保存在 `qwen3tts_eval_zh/` 目录下：

```
qwen3tts_eval_zh/
├── native_fp16/          # Native 模式输出
│   ├── uttid_1.wav
│   ├── uttid_2.wav
│   └── ...
└── hmonnx/               # HMONNX 模式输出
    ├── uttid_1.wav
    ├── uttid_2.wav
    └── ...
```

### 性能参考

基于 4 x NVIDIA RTX A6000 测试：

- **HMONNX 模式**: 20 条样本约 10-15 分钟
- **每个样本**: 约 2-3 分钟（包含模型推理和音频生成）
- **并行效率**: 4 GPU 并行，每个 GPU 处理 5 个样本
- **显存占用**: 每个 GPU 约需 10-15GB 显存

### 后续精度评估

生成音频文件后，可使用 CV3-Eval 工具进行精度评估：

```bash
# 使用 CV3-Eval 评估生成的音频
cd /data01/home/she.gao/CV3-Eval
python evaluate.py \
    --generated-dir <xh2modelzoo path>/examples/audio/qwen3_tts/qwen3tts_eval_zh/hmonnx \
    --reference-dir data/zero_shot/zh
```

---

## HMONNX 导出和推理

### 1.7B-VoiceDesign 导出流程

#### 导出 HMONNX（4 步）

```bash
# 1. Talker
python qwen3_tts_talker_xh2a_export.py \
    --config ./config/llm/qwen3_tts_12hz_1_7B_voicedesign_talker_2k_xh2a.py

# 2. CodePredictor
python qwen3_tts_code_predictor_xh2a_export.py \
    --config ./config/llm/qwen3_tts_12hz_1_7B_voicedesign_code_predictor_2k_xh2a.py

# 3. TextProjection
python qwen3_tts_text_projection_xh2a_export.py \
    --config ./config/llm/qwen3_tts_12hz_1_7B_text_projection_xh2a.py

# 4. SpeechTokenizer
python qwen3_tts_speech_tokenizer_xh2a_export.py \
    --config ./config/llm/qwen3_tts_12hz_1_7B_speech_tokenizer_xh2a.py
```

每条命令的产物落在 `./work_dirs/<config_stem>/` 下，包含 prefill/decode ONNX、`meta.json` 等。

#### XH2a Demo（整链路 HMONNX 推理）

完成上面 4 步导出后，跑整链路 demo：

```bash
python qwen3_tts_xh2a_demo.py \
    --config ./config/llm/qwen3_tts_12hz_1_7b_voicedesign_xh2a_hmonnx.py
# 期望产物：output_voice_design.wav
```

config 中的 `Qwen3TTSHMONNXInference` 会把 4 个子模型的 HMONNX 串起来，
调用 `model.generate_voice_design(text=..., language=..., instruct=...)` 走完
prefill → decode → text projection → speech tokenizer 全流程。

### 0.6B-CustomVoice 导出流程

#### 导出 HMONNX（4 步）

```bash
# 1. Talker
python qwen3_tts_0p6b_cv_talker_xh2a_export.py \
    --config ./config/llm/qwen3_tts_12hz_0_6B_customvoice_talker_2k_xh2a.py

# 2. CodePredictor
python qwen3_tts_0p6b_cv_code_predictor_xh2a_export.py \
    --config ./config/llm/qwen3_tts_12hz_0_6B_customvoice_code_predictor_2k_xh2a.py

# 3. TextProjection
python qwen3_tts_0p6b_cv_text_projection_xh2a_export.py \
    --config ./config/llm/qwen3_tts_12hz_0_6B_customvoice_text_projection_xh2a.py

# 4. SpeechTokenizer
python qwen3_tts_0p6b_cv_speech_tokenizer_xh2a_export.py \
    --config ./config/llm/qwen3_tts_12hz_0_6B_customvoice_speech_tokenizer_xh2a.py
```

#### XH2a Demo

```bash
python qwen3_tts_0p6b_cv_xh2a_demo.py \
    --config ./config/llm/qwen3_tts_12hz_0_6B_customvoice_xh2a_hmonnx.py
# 期望产物：output_custom_voice.wav
```

---

## 其他工具

### Roofline 分析

- `model_roofline.py`：基于 `XHQwen3TTSModel` 统计每个子模型的参数量/计算量
- `llm_hmonnx_roofline.py`：对导出的 HMONNX 文件做算子级 roofline 分析，输出 xlsx 报表

### 流式推理

```bash
python qwen3_tts_0p6b_cv_streaming_demo.py
```

---

## 模型信息

### 支持的模型

| 模型 | 参数量 | 类型 | 支持模式 | 路径 |
|------|--------|------|---------|------|
| Qwen3-TTS-12Hz-0.6B-Base | 0.6B | Base | voice-clone | `./data/models/Qwen3-TTS-12Hz-0.6B-Base/` |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 0.6B | CustomVoice | custom-voice | `./data/models/Qwen3-TTS-12Hz-0.6B-CustomVoice/` |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 1.7B | VoiceDesign | voice-design | `./data/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign/` |

所有模型路径均为软链接，指向 HuggingFace cache 目录。

---

## 已知问题

1. **Native 模式 CUDA 错误**: 当前 Native 模式存在 CUDA 采样错误（`probability tensor contains either inf, nan or element < 0`），建议使用 HMONNX 模式
2. **0.6B-Base 未适配 HMONNX 导出**: Base 模型走 voice clone 接口（需参考音 + 参考文本），不支持 `generate_voice_design`。完整的 0.6B 验证记录见 [SMOKE_0.6B.md](./SMOKE_0.6B.md)

---
