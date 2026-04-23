# F5-TTS ONNX/HMONNX 导出与推理指南

本文档覆盖以下完整流程：
- 静态 shape ONNX 导出（含 `input_lengths` mask 机制）
- PTQ 量化并导出 HMONNX
- ORT / HMONNX demo 推理
- 质量评测（`f5tts_eval.py`）与 ASR 对比评测（`f5tts_asr_eval.py`）

## 1. 目录与脚本说明

当前目录关键脚本：
- `f5tts_common.py`: 共享常量与推理核心（含 ODE 采样、mask 逻辑）
- `f5tts_export_onnx.py`: 导出静态 ONNX
- `f5tts_export_hmonnx.py`: ONNX PTQ 并导出 HMONNX
- `f5tts_ort_demo.py`: float ONNX 端到端推理
- `f5tts_hmonnx_demo.py`: HMONNX 端到端推理
- `f5tts_eval.py`: 生成+评测（支持 pytorch/float/w8a8/w8a16）
- `f5tts_asr_eval.py`: 对已有 wav 做 Whisper WER/CER，支持多模型对比

## 2. 环境准备

使用含有 `xhquanttool` 的环境，并安装以下依赖（至少一次）：

```bash
conda install -c conda-forge ffmpeg -y
pip install opencc-python-reimplemented
```

说明：
- `ffmpeg`：Whisper 转录必需
- `opencc-python-reimplemented`：中文 CER 归一化（繁转简）必需

## 3. 必看：`f5tts_common.py` 的 hardcode 配置

请先检查 `f5tts_common.py` 的 `L26-L42` 常量区。

### 3.1 必须按你机器修改

- `MODEL_CKPT`
  - 含义：默认 PyTorch safetensors 权重路径
  - 不改会导致加载错误或误用他人路径
- `VOCAB_PATH`
  - 含义：默认 vocab 路径
  - 必须和模型版本匹配
- `F5TTS_SRC`
  - 含义：本地 `F5-TTS/src` 代码路径
  - 用于导入官方模块和示例参考音频

### 3.2 视场景修改（不是每次都要改）

- `STATIC_N`
  - 含义：静态 ONNX/HMONNX 的 mel 帧长上限
  - 一旦修改，必须重新导出 ONNX 和 HMONNX，并确保推理脚本使用同一版本模型
- `STATIC_NT`
  - 含义：静态 token 长度上限
  - 仅当你的文本显著超长、出现截断时再考虑增大

### 3.3 一般不用改（默认保持即可）

- `TARGET_SR`, `N_MEL_CHANNELS`, `HOP_LENGTH`, `WIN_LENGTH`, `N_FFT`
  - 含义：与 F5-TTS/vocos 配套的声学参数
  - 改动会直接破坏模型输入分布，不建议改
- `DEFAULT_NFE`, `DEFAULT_CFG`
  - 含义：默认采样步数和 CFG
  - 这些通常通过 CLI 参数覆盖，不需要改硬编码
- `REF_EN_WAV`, `REF_ZH_WAV`
  - 含义：HMONNX 校准默认参考音频
  - 只有你要自定义校准语料时才需要改

## 4. 导出 ONNX（静态 shape + input_lengths）

在当前目录执行：

```bash
python f5tts_export_onnx.py \
  --ckpt /your/path/model_1250000.safetensors \
  --vocab /your/path/vocab.txt \
  --n-frames 2048 \
  --n-text 256 \
  --out-dir work_dirs/f5tts_v1/static_mask/export_fp32
```

成功标志：
- 输出 `work_dirs/.../onnx/f5tts_dit.onnx`
- 日志中 `ONNX inputs` 包含 `input_lengths`
- 数值校验通过（`cos_sim` 接近 1）

## 5. 导出 HMONNX（PTQ）

```bash
python f5tts_export_hmonnx.py \
  --onnx work_dirs/f5tts_v1/static_mask/export_fp32/onnx/f5tts_dit.onnx \
  --vocab /your/path/vocab.txt \
  --quant-type w8a8h1_sefp \
  --calib-samples 32 \
  --out-dir work_dirs/f5tts_v1/static_mask/export_xh2a
```

成功标志：
- 输出 `work_dirs/.../hmonnx/f5tts_dit_XH2a.onnx`
- 日志显示 ONNX 输入包含 `x, cond, text, time, input_lengths`

## 6. Demo 推理

### 6.1 ORT（float ONNX）

```bash
python f5tts_ort_demo.py \
  --onnx work_dirs/f5tts_v1/static_mask/export_fp32/onnx/f5tts_dit.onnx \
  --ref-audio demo_refs/zh_ref.wav \
  --ref-text "至今为止，元气火箭总共发行了两张专辑。" \
  --gen-text "今天天气真不错，我真想回家睡觉，完全不想上班啊。" \
  --segment-max-chars 100 \
  --output work_dirs/f5tts_v1/static_mask/ort_demo.wav
```

### 6.2 HMONNX

```bash
python f5tts_hmonnx_demo.py \
  --hmonnx work_dirs/f5tts_v1/static_mask/export_xh2a/hmonnx/f5tts_dit_XH2a.onnx \
  --ref-audio demo_refs/zh_ref.wav \
  --ref-text "至今为止，元气火箭总共发行了两张专辑。" \
  --gen-text "今天天气真不错，我真想回家睡觉，完全不想上班啊。" \
  --output work_dirs/f5tts_v1/static_mask/hmonnx_demo.wav
```

## 7. 评测流程

## 7.1 一体化评测（生成 + WER/CER）

`f5tts_eval.py` 会生成音频并评测。最小冒烟：

```bash
python f5tts_eval.py \
  --hmonnx work_dirs/f5tts_v1/static_mask/export_xh2a/hmonnx/f5tts_dit_XH2a.onnx \
  --n-samples 1 \
  --whisper-model base \
  --device cuda \
  --out-dir work_dirs/f5tts_v1/static_mask/eval_smoke
```

输出：
- `eval_results.csv`
- `eval_summary.json`

### 7.1.1 单命令自动多卡（`--gpus`）

如果不想手动起多个命令，可以直接：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python f5tts_eval.py \
  --hmonnx work_dirs/f5tts_v1/static_mask/export_xh2a/hmonnx/f5tts_dit_XH2a.onnx \
  --n-samples 30 \
  --gpus 4 \
  --device cuda \
  --out-dir work_dirs/f5tts_v1/eval_w8a8_auto4
```

行为说明：
- 主进程会自动拉起 4 个子进程，分别绑定 4 张卡。
- 子进程输出目录为：
  - `work_dirs/f5tts_v1/eval_w8a8_auto4_shard0`
  - `work_dirs/f5tts_v1/eval_w8a8_auto4_shard1`
  - `work_dirs/f5tts_v1/eval_w8a8_auto4_shard2`
  - `work_dirs/f5tts_v1/eval_w8a8_auto4_shard3`
 - 主进程结束后会自动回收到主目录：
  - 合并音频到 `work_dirs/f5tts_v1/eval_w8a8_auto4/<model>/<lang>/*.wav`
  - 合并 CSV 到 `work_dirs/f5tts_v1/eval_w8a8_auto4/eval_results.csv`
  - 生成汇总 `work_dirs/f5tts_v1/eval_w8a8_auto4/eval_summary.json`

注意：
- `--gpus` 不能超过可见 GPU 数量（由 `CUDA_VISIBLE_DEVICES` 决定）。

## 7.2 已有音频的 ASR 对比评测（推荐对比浮点 vs 量化）

`f5tts_asr_eval.py` 已支持多模型目录对比：

```bash
python f5tts_asr_eval.py \
  --model-dir \
    pytorch=work_dirs/f5tts_v1/static_mask/eval_smoke/pytorch \
    w8a8=work_dirs/f5tts_v1/static_mask/eval_smoke/w8a8 \
  --whisper-model base \
  --n-samples 1 \
  --device cuda \
  --out work_dirs/f5tts_v1/static_mask/eval_smoke/asr_compare.csv
```

输出报告会按模型和语言分别给出 `WER/CER`。

## 8. 常见问题

- Q: 改了 `STATIC_N` 后 demo 报 shape 错误？
  - A: 这是预期。你必须重新执行 ONNX 与 HMONNX 导出，并使用新模型推理。

- Q: 中文 CER 偏高但听感正常？
  - A: 请确认已安装 `opencc`。本脚本会做繁转简和标点清洗，能显著降低“字形差异”引入的虚高 CER。

- Q: `ffmpeg not found`？
  - A: 安装 `ffmpeg` 并确保当前环境 PATH 可见，否则 Whisper 无法解码音频。
