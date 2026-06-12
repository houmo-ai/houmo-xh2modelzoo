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
./qwen3tts_pipeline.sh --model 1_7B_voicedesign --export --test-native --test-hmonnx --eval

# 只导出模型
./qwen3tts_pipeline.sh --model 1_7B_voicedesign --export

# 导出 0.6B-Base（voice-clone，需参考音频，脚本会自动下载 clone_1.wav）
./qwen3tts_pipeline.sh --model 0_6B_base --export --test-hmonnx

# 同时处理多个模型（可选 1_7B_voicedesign / 0_6B_customvoice / 0_6B_base）
./qwen3tts_pipeline.sh --model 1_7B_voicedesign,0_6B_customvoice,0_6B_base --export --test-hmonnx
```

> `--model` 取值与 `--variant` 一致：`1_7B_voicedesign` / `0_6B_customvoice` / `0_6B_base`；
> 也接受短别名 `1.7b` / `0.6b` / `0.6b-base`。

**支持的操作**：
- `--export`: 导出 HMONNX 模型（4 个子模型）
- `--test-native`: 测试原始浮点模型
- `--test-hmonnx`: 测试 HMONNX 模型
- `--eval`: 运行精度评估（Native + HMONNX）

详细使用说明见下文「HMONNX 导出和推理」章节，或运行 `./qwen3tts_pipeline.sh --help`。

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
onnxsim==0.4.36
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

`eval/` 目录下提供两个评估脚本，对应两种测试场景：

| 脚本 | 适用模型 | 数据 |
|------|---------|------|
| `eval/qwen3_tts_eval.py` | 0.6B-CustomVoice（预定义 speaker） | CV3-Eval text（仅目标文本） |
| `eval/qwen3_tts_eval_voice_clone.py` | 0.6B-Base（voice-clone） | CV3-Eval text + prompt_text + prompt_wav.scp（含参考音频） |

两个脚本都支持 Native PyTorch 和 HMONNX 两种推理模式、多 GPU 并行、断点续传。

---

### eval/qwen3_tts_eval.py（0.6B-CustomVoice）

使用预定义 speaker 生成音频，适用于 0.6B-CustomVoice 模型的精度评估。

#### 用法

```bash
cd examples/audio/qwen3_tts
PYTHONPATH=<xh2modelzoo path>

# HMONNX 模式（推荐），500 条，4 GPU，speaker 轮询
python eval/qwen3_tts_eval.py \
    --mode hmonnx \
    --variant 0_6B_customvoice \
    --gpus 0,1,2,3 \
    --speaker-mode round-robin

# Native 模式，测试前 20 条
python eval/qwen3_tts_eval.py \
    --mode native \
    --gpus 0,1,2,3 \
    --max-samples 20 \
    --speaker-mode round-robin
```

#### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | `native` 或 `hmonnx` | 必填 |
| `--gpus` | GPU 列表，逗号分隔，如 `0,1,2,3` | 必填 |
| `--variant` | hmonnx 变体：`0_6B_customvoice` / `1_7B_voicedesign` | None |
| `--max-samples` | 最大样本数，None 表示全部 500 条 | None |
| `--speaker-mode` | `fixed` / `random` / `round-robin` | `fixed` |
| `--speaker` | 固定 speaker（仅 `fixed` 模式）| `vivian` |
| `--data-path` | CV3-Eval 数据集路径 | `/data01/home/she.gao/CV3-Eval/data/zero_shot/zh` |
| `--exp-dir` | 输出根目录 | `qwen3tts_eval_zh` |
| `--hmonnx-config` | HMONNX 配置文件 | `./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py` |
| `--hf-model` | Native 模式模型路径 | `./data/models/Qwen3-TTS-12Hz-0.6B-CustomVoice` |

#### 输出

```
qwen3tts_eval_zh/
├── native_fp16/          # Native 模式输出
│   ├── uttid_1.wav
│   └── ...
└── hmonnx/               # HMONNX 模式输出
    ├── uttid_1.wav
    └── ...
```

---

### eval/qwen3_tts_eval_voice_clone.py（0.6B-Base）

使用数据集中的参考音频克隆音色，适用于 0.6B-Base 模型的精度评估。数据集需包含 `text`、`prompt_text`、`prompt_wav.scp` 三个文件。

#### 用法

```bash
cd examples/audio/qwen3_tts
PYTHONPATH=<xh2modelzoo path>

# HMONNX 模式，500 条，4 GPU
python eval/qwen3_tts_eval_voice_clone.py \
    --mode hmonnx \
    --variant 0_6B_base \
    --gpus 0,1,2,3

# Native 模式，测试前 20 条
python eval/qwen3_tts_eval_voice_clone.py \
    --mode native \
    --gpus 0,1,2,3 \
    --max-samples 20
```

#### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | `native` 或 `hmonnx` | 必填 |
| `--gpus` | GPU 列表，逗号分隔 | 必填 |
| `--variant` | hmonnx 变体，voice-clone 用 `0_6B_base` | None |
| `--max-samples` | 最大样本数，None 表示全部 500 条 | None |
| `--xvec-only` | 仅使用 x-vector 模式（不使用参考音频细节特征） | off |
| `--data-path` | CV3-Eval 数据集路径（需包含 prompt_text / prompt_wav.scp） | `/data01/home/she.gao/CV3-Eval/data/zero_shot/zh` |
| `--exp-dir` | 输出根目录 | `qwen3tts_eval_zh_voice_clone` |
| `--hmonnx-config` | HMONNX 配置文件 | `./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py` |
| `--hf-model` | Native 模式模型路径 | `./data/models/Qwen3-TTS-12Hz-0.6B-Base` |

#### 输出

```
qwen3tts_eval_zh_voice_clone/
├── native_voice_clone/   # Native 模式输出
│   ├── uttid_1.wav
│   └── ...
└── hmonnx_voice_clone/   # HMONNX 模式输出
    ├── uttid_1.wav
    └── ...
```

---

### 性能参考

基于 4 × NVIDIA RTX A6000 测试（HMONNX 模式）：

| 脚本 | 500 条总耗时 | 单条耗时 | 显存/卡 |
|------|------------|---------|--------|
| `qwen3_tts_eval.py` | ~2 小时 | ~1 分钟 | ~3 GB |
| `qwen3_tts_eval_voice_clone.py` | ~6 小时 | ~3 分钟 | ~3 GB |

> voice-clone 每条耗时较长，因为 0.6B-Base 的 talker 需要先编码参考音频的 x-vector。

---

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

---

## HMONNX 导出和推理

三个变体（1.7B-VoiceDesign / 0.6B-CustomVoice / 0.6B-Base）共用同一套导出/推理脚本和
config，通过 `--variant {1_7B_voicedesign,0_6B_customvoice,0_6B_base}` 切换。导出脚本用 `--name` 指定产物目录名
（缺省为 config 文件名）。`--variant` 会把对应的 `hf_model` / `tts_mode` /
（ref_audio/ref_text 或 tts_speaker 或 tts_instruct）注入到解析后的 config，
具体取值集中维护在 `config/llm/_components.py` 的 `VARIANTS` / `WORKNAME` 表中。

### 统一 config

| 用途 | config 文件 |
|------|------------|
| model-level（参数化） | `config/llm/qwen3_tts_12hz_model_xh2a.py` |
| Talker 组件 | `config/llm/qwen3_tts_12hz_talker_2k_xh2a.py` |
| CodePredictor 组件 | `config/llm/qwen3_tts_12hz_code_predictor_2k_xh2a.py` |
| TextProjection 组件 | `config/llm/qwen3_tts_12hz_text_projection_xh2a.py` |
| SpeechTokenizer 组件 | `config/llm/qwen3_tts_12hz_speech_tokenizer_xh2a.py` |
| HMONNX 整链路 | `config/llm/qwen3_tts_12hz_xh2a_hmonnx.py` |
| 共享组件 / 变体表 | `config/llm/_components.py` |

### 导出 HMONNX（4 步）

以 0.6B-CustomVoice（`--variant 0_6B_customvoice`）为例。`--name` 给定的产物目录名沿用各变体既有命名。

```bash
VARIANT=cv   # 可选：voicedesign / cv / base

# 1. Talker
python qwen3_tts_talker_export.py \
    --config ./config/llm/qwen3_tts_12hz_talker_2k_xh2a.py \
    --variant ${VARIANT} --name qwen3_tts_12hz_0_6B_customvoice_talker_2k_xh2a

# 2. CodePredictor
python qwen3_tts_code_predictor_export.py \
    --config ./config/llm/qwen3_tts_12hz_code_predictor_2k_xh2a.py \
    --variant ${VARIANT} --name qwen3_tts_12hz_0_6B_customvoice_code_predictor_2k_xh2a

# 3. TextProjection
python qwen3_tts_text_projection_export.py \
    --config ./config/llm/qwen3_tts_12hz_text_projection_xh2a.py \
    --variant ${VARIANT} --name qwen3_tts_12hz_0_6B_customvoice_text_projection_xh2a

# 4. SpeechTokenizer
python qwen3_tts_speech_tokenizer_export.py \
    --config ./config/llm/qwen3_tts_12hz_speech_tokenizer_xh2a.py \
    --variant ${VARIANT} --name qwen3_tts_12hz_0_6B_customvoice_speech_tokenizer_xh2a
```

其它变体把 `--variant` 和 `--name` 换成对应值即可（产物目录名见 `_components.py` 的 `WORKNAME`）：
- **1.7B-VoiceDesign**：`--variant 1_7B_voicedesign`，`--name qwen3_tts_12hz_1_7B_voicedesign_{talker_2k,code_predictor_2k}_xh2a` 及 `qwen3_tts_12hz_1_7B_{text_projection,speech_tokenizer}_xh2a`（注意 1.7B 的 text_projection/speech_tokenizer 命名不带 `voicedesign`）。
- **0.6B-Base**：`--variant 0_6B_base`，`--name qwen3_tts_12hz_0_6B_base_{talker_2k,code_predictor_2k,text_projection,speech_tokenizer}_xh2a`。

每条命令的产物落在 `./work_dirs/<name>/` 下，包含 prefill/decode ONNX、`meta.json` 等。
推荐直接用一键脚本 `./qwen3tts_pipeline.sh --model {1.7b,0.6b,0.6b-base} --export`，已封装好上述 `--variant`/`--name`。

#### 0.6B-Base 需先准备参考音频

Base 模型走 voice-clone 接口（需参考音频 + 参考文本）。导出前先下载参考音频（缺省路径
`/tmp/clone_1.wav`，参考文本默认 `"甚至出现交易几乎停滞的情况。"`，可在
`config/llm/_components.py` 的 `VARIANTS["0_6B_base"]` 中修改）：

```bash
curl -sSL -o /tmp/clone_1.wav \
    https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav
```

### XH2a Demo（整链路 HMONNX 推理）

完成 4 步导出后，跑整链路 demo（同样用 `--variant` 切换）：

```bash
python qwen3_tts_demo.py \
    --config ./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py --variant 0_6B_customvoice
# 期望产物：work_dirs/qwen3_tts_12hz_xh2a_hmonnx_cv/output_<mode>.wav
```

`Qwen3TTSHMONNXInference` 会把 4 个子模型的 HMONNX 串起来，根据 `--variant` 自动选择
`generate_voice_design` / `generate_custom_voice` / `generate_voice_clone`，走完
prefill → decode → text projection → speech tokenizer 全流程。1.7B 用 `--variant 1_7B_voicedesign`，
Base 用 `--variant 0_6B_base`。

### Golden 导出（可选，用于硬件比对）

以上 4 个导出脚本（Talker / CodePredictor / TextProjection / SpeechTokenizer，配 `--variant` 适用于 1.7B / 0.6B-CustomVoice / 0.6B-Base）都支持 `--golden` 开关。开启后会在导出 HMONNX 之后，用 `HMONNXGoldenInference` 加载该 HMONNX 并跑一次 forward，由运行时把 golden（输入 + 输出张量）落盘，可用于和硬件结果做逐算子 / 端到端比对。导出方式与 `xh_model_zoo/xh_llm/models/qwen3_vl/qwen3_vl_converter.py` 中的 golden 导出一致。

```bash
# 在任意导出命令后追加 --golden 即可，例如：
python qwen3_tts_code_predictor_export.py \
    --config ./config/llm/qwen3_tts_12hz_code_predictor_2k_xh2a.py \
    --variant 0_6B_customvoice --name qwen3_tts_12hz_0_6B_customvoice_code_predictor_2k_xh2a \
    --golden

# 默认用当前可见的 CUDA 设备；如需固定某块卡，用 CUDA_VISIBLE_DEVICES（或 --golden-device）
CUDA_VISIBLE_DEVICES=3 python qwen3_tts_speech_tokenizer_export.py \
    --config ./config/llm/qwen3_tts_12hz_speech_tokenizer_xh2a.py \
    --variant 0_6B_customvoice --name qwen3_tts_12hz_0_6B_customvoice_speech_tokenizer_xh2a \
    --golden
```

**参数**：
- `--golden`：开启 golden 导出（默认关闭，不加则导出流程与原来完全一致）
- `--golden-device`：golden 推理设备，默认 `cuda`（落到当前可见设备；无 GPU 时回退 `cpu`）

**产物位置**：落在 `work_dirs/<config_stem>/golden/` 下，并把相对路径记录进该 work_dir 的 `meta.json`：

| 子模型 | golden 子目录 | meta.json 键 |
|--------|--------------|-------------|
| Talker / CodePredictor | `golden/<config_stem>_prefill`、`golden/<config_stem>_decode` | `prefill_golden_dir`、`decode_golden_dir` |
| TextProjection | `golden/text_projection` | `golden_dir` |
| SpeechTokenizer | `golden/speech_tokenizer` | `golden_dir` |

> 公共实现见 `_golden.py` 的 `run_hmonnx_golden(...)`：普通浮点输入会对齐到 fp16，kv-cache（`CacheTensor`）与整型输入保持不变。

---

## 其他工具

### 流式推理

严格 GGUF 对齐的流式 demo 位于 `eval/qwen3_tts_streaming_demo.py`，支持 `customvoice`、`base` 和 `voicedesign` 三个 variant。当前 live 路径是：

```text
talker/code predictor 逐帧生成 codec frame -> streamer 攒满 12 帧 -> stateful decoder -> AUDIO/FINISH
```

其中已有的 `speech_tokenizer` HMONNX 是 stateless 的整段 decoder，不能直接提供 GGUF 风格的流式状态；live 模式需要额外导出 stateful decoder。HMONNX 版 stateful decoder 使用固定 KV/history buffer，加 `kv_valid_len`、`valid_frames` 和 attention mask 来表达真实有效长度。

#### 1. 导出 stateful decoder

以 0.6B-CustomVoice 为例：

```bash
cd examples/audio/qwen3_tts

PYTHONPATH=<xh2modelzoo path> \
python qwen3_tts_stateful_decoder_export.py \
  --variant 0_6B_customvoice \
  --name qwen3_tts_12hz_0_6B_customvoice_stateful_decoder_xh2a
```

产物位于：

```text
work_dirs/qwen3_tts_12hz_0_6B_customvoice_stateful_decoder_xh2a/
├── meta.json
├── onnx/qwen3_tts_decoder_stateful_static.onnx
└── hmonnx/qwen3_tts_decoder_stateful_static_XH2a.onnx
```

其它变体只需要替换 `--variant` 和 `--name`：

```bash
# 0.6B-Base / voice-clone
PYTHONPATH=<xh2modelzoo path> \
python qwen3_tts_stateful_decoder_export.py \
  --variant 0_6B_base \
  --name qwen3_tts_12hz_0_6B_base_stateful_decoder_xh2a

# 1.7B-VoiceDesign
PYTHONPATH=<xh2modelzoo path> \
python qwen3_tts_stateful_decoder_export.py \
  --variant 1_7B_voicedesign \
  --name qwen3_tts_12hz_1_7B_voicedesign_stateful_decoder_xh2a
```

#### 2. 运行 live 流式 demo

`--stateful-decoder` 推荐直接传 stateful decoder 的 `meta.json`。`--stateful-decoder-backend auto` 会自动识别：`meta.json` / `hmonnx` 路径走 HMONNX runtime，普通 ONNX 路径走 ONNX Runtime。

```bash
cd examples/audio/qwen3_tts
```

CustomVoice：

```bash
PYTHONPATH=<xh2modelzoo path> \
python eval/qwen3_tts_streaming_demo.py \
  --config ./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py \
  --variant customvoice \
  --stateful-decoder work_dirs/qwen3_tts_12hz_0_6B_customvoice_stateful_decoder_xh2a/meta.json \
  --text "你好，这是一个流式语音合成测试。" \
  --speaker vivian \
  --mode live \
  --chunk-size 12 \
  --output output_streaming_customvoice.wav
```

Base / Voice Clone：

```bash
PYTHONPATH=<xh2modelzoo path> \
python eval/qwen3_tts_streaming_demo.py \
  --config ./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py \
  --variant base \
  --stateful-decoder work_dirs/qwen3_tts_12hz_0_6B_base_stateful_decoder_xh2a/meta.json \
  --text "你好，这是一个音色克隆流式语音合成测试。" \
  --ref-audio /tmp/clone_1.wav \
  --ref-text "甚至出现交易几乎停滞的情况。" \
  --mode live \
  --chunk-size 12 \
  --output output_streaming_base.wav
```

VoiceDesign：

```bash
PYTHONPATH=<xh2modelzoo path> \
python eval/qwen3_tts_streaming_demo.py \
  --config ./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py \
  --variant voicedesign \
  --stateful-decoder work_dirs/qwen3_tts_12hz_1_7B_voicedesign_stateful_decoder_xh2a/meta.json \
  --text "你好，这是一个音色设计流式语音合成测试。" \
  --instruct "体现温柔甜美的女声，音调适中，语速平稳。" \
  --mode live \
  --chunk-size 12 \
  --output output_streaming_voicedesign.wav
```

#### 3. 流式行为说明

- Talker / CodePredictor 侧逐帧生成 codec frame，每帧形状是 `[16]`。
- Streamer 会缓存 codec frame，满 `12` 帧后输出一个 `[12, 16]` chunk 给 decoder。
- Stateful decoder 每次解码一个 chunk，维护 `pre_conv_history`、`latent_buffer`、`conv_history`、`kv_cache`、`latent_audio` 等状态。
- 非 final chunk 必须满 `12` 帧；最后一个 chunk 可以不足 `12` 帧，内部 pad 到 `12`，再通过 `valid_frames` 裁掉 padding 对应的音频。
- 第一包通常因为 lookahead 只输出约 `8` 帧音频；后续中间包通常按 `12` 帧节奏输出。
- `--mode oneshot` 可作为非流式参考推理，不需要 `--stateful-decoder`。

#### 4. 已验证命令示例

当前仓库里已验证过的 smoke 产物示例：

```bash
PYTHONPATH=<xh2modelzoo path> \
python eval/qwen3_tts_streaming_demo.py \
  --config ./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py \
  --variant customvoice \
  --stateful-decoder work_dirs/codex_stateful_decoder_static_v2/meta.json \
  --text 你好 \
  --speaker vivian \
  --mode live \
  --chunk-size 12 \
  --max-new-tokens 48 \
  --name codex_streaming_full_smoke \
  --output smoke_streaming.wav
```

该命令会生成 `work_dirs/codex_streaming_full_smoke/smoke_streaming.wav`，日志中应看到 `AUDIO` packet 和最终 `FINISH`。

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

1. **0.6B-Base voice-clone 需准备参考音频**: Base 模型走 voice clone 接口（需参考音 + 参考文本），导出/推理前需先下载 `clone_1.wav`（见「HMONNX 导出和推理」），参考音/文本可在 `config/llm/_components.py` 的 `VARIANTS["0_6B_base"]` 中配置。HMONNX 导出已支持（统一脚本 `qwen3_tts_*_export.py` + `qwen3_tts_demo.py`，配 `--variant 0_6B_base`）。
2. **Native 推理建议用 fp32**: README 记录 fp16+sdpa 在 native 采样会出 multinomial NaN，voice-clone native 测试默认用 `--dtype fp32`（HMONNX 路径不受影响）。

---

## 精度评估结果

基于 CV3-Eval zero_shot/zh 数据集（500 条中文样本），三项指标：
- **CER**：Paraformer ASR 转写后与目标文本的字符错误率，越低越好
- **Speaker Similarity**：ERes2Net 提取 speaker embedding 与参考音频的余弦相似度，越高越好
- **DNSMOS**：DNSMOS 网络音质评分，越高越好

评估命令：
```bash
cd /data01/home/she.gao/CV3-Eval
# WER + speaker similarity + DNSMOS
bash run_infer_cv3_eval.sh

# emotion score（可选）
bash run_infer_cv3_eval_emo.sh
```

### 0.6B-CustomVoice（custom-voice，`qwen3tts_eval_zh/`）

| 模式 | 产物路径 | wav 数 |
|------|---------|--------|
| Native (bf16 + sdpa) | `qwen3tts_eval_zh/native_fp16/` | 500 |
| HMONNX (0_6B_customvoice) | `qwen3tts_eval_zh/hmonnx/` | 500 |

| 模式 | CER ↓ | Speaker Sim ↑ | DNSMOS ↑ |
|------|-------|--------------|---------|
| Native (bf16 + sdpa) | 3.38 | 21.88 | 3.915 |
| HMONNX (0_6B_customvoice) | 3.54 | 21.88 | 3.90 |

### 0.6B-Base（voice-clone，`qwen3tts_eval_zh_voice_clone/`）

| 模式 | 产物路径 | wav 数 |
|------|---------|--------|
| Native (bf16 + sdpa) | `qwen3tts_eval_zh_voice_clone/native_voice_clone/` | 500 |
| HMONNX (0_6B_base) | `qwen3tts_eval_zh_voice_clone/hmonnx_voice_clone/` | 500 |

| 模式 | CER ↓ | Speaker Sim ↑ | DNSMOS ↑ |
|------|-------|--------------|---------|
| Native (bf16 + sdpa) | 3.56 | 72.56 | 3.77 |
| HMONNX (0_6B_base) | 3.33 | 72.60 | 3.72 |
