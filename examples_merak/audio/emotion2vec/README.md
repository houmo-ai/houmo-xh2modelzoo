# emotion2vec+ Large Merak

本目录默认适配 emotion2vec 官方 GitHub README 推荐的 `iic/emotion2vec_plus_large` 权重，覆盖 ModelScope 下载、Merak 默认 W8A8 HMONNX 导出、真实音频 Golden、HMONNX 特征提取和官方 IEMOCAP 五折评估协议。plus-large 参数量约 300M，输出特征维度为 1024。

## 环境

仓库基础环境之外，模型下载和官方 PyTorch 对齐需要：

```bash
pip install funasr modelscope
```

## 1. 下载官方权重

```bash
python examples_merak/audio/emotion2vec/debug/download_model.py \
	--output-dir data/models/emotion2vec_plus_large
```

下载来源固定为 emotion2vec README 中列出的 ModelScope 模型 `iic/emotion2vec_plus_large`。

## 2. 导出默认 W8A8 HMONNX

固定输入为单声道 16 kHz、16 秒窗口，即 `[1, 256000]`。默认量化类型是 `w8a8h1_sefp`。

```bash
python examples_merak/audio/emotion2vec/export_hmonnx.py \
	--model-dir data/models/emotion2vec_plus_large \
	--config-path configs_merak/workflows/xh2a/audio_models/emotion2vec/emotion2vec_plus_large_xh2a_w8a8_16s.yaml \
	--output-dir work_dirs/emotion2vec_plus_large_xh2a_w8a8_16s \
	--dump-golden \
	--golden-audio data/models/emotion2vec_plus_large/example/test.wav
```

入口通过 `AutoLLMWorkflow.from_config()` 读取 checked-in YAML。YAML 的 `quant` 为 `null`，表示不生成独立的量化 HF 权重目录；模型导出状态链仍按 `export.model.quant_scheme.quant_type=w8a8h1_sefp` 完成全图 PTQ 和 HMONNX 导出。重复导出时可增加 `--overwrite`。

`--dump-golden` 会在导出后调用同一个 `Emotion2vecWorkflow.dump_golden()`，使用官方 FunASR FP32 模型生成 Golden。导出目录会生成 `emotion2vec_meta.json`、本次使用的 workflow YAML 副本、HMONNX 文件和 `golden/` 目录。音频解码、重采样、波形归一化、长音频分块和有效帧拼接位于图外；神经网络主体位于 HMONNX 图内。波形归一化使用 FP32，并且只统计 `valid_samples` 指定的有效区域。`valid_samples` 以 INT32 输入 HMONNX，仅用于 padding mask 和卷积输出帧长计算，不再转换为 FP16，因此每个分块可以使用完整的 256000 采样点窗口。

## 3. 比较 Golden

上一节通过 workflow 生成：

- `golden/frame_features.npy`：官方 FP32 帧级特征；
- `golden/utterance_feature.npy`：官方 FP32 句级特征；
- `golden/golden_meta.json`：音频路径、有效采样数、帧数和特征维度。

使用同一音频运行 W8A8 HMONNX 并比较：

```bash
python examples_merak/audio/emotion2vec/debug/compare_golden.py \
	--meta work_dirs/emotion2vec_plus_large_xh2a_w8a8_16s/emotion2vec_meta.json \
	--audio data/models/emotion2vec_plus_large/example/test.wav \
	--golden-dir work_dirs/emotion2vec_plus_large_xh2a_w8a8_16s/golden
```

脚本报告帧级 cosine、句级 cosine、MAE 和最大绝对误差。默认要求帧级和句级 cosine 均不低于
`0.9`，未达标时返回非零退出码。`--min-frame-cos` 和 `--min-utterance-cos` 可用于实验性调整阈值，
但降低阈值后的结果不应作为验收证据。

## 4. HMONNX 推理

```bash
python examples_merak/audio/emotion2vec/hmonnx_infer.py \
	--meta work_dirs/emotion2vec_plus_large_xh2a_w8a8_16s/emotion2vec_meta.json \
	--audio data/models/emotion2vec_plus_large/example/test.wav \
	--output work_dirs/emotion2vec_plus_large_xh2a_w8a8_16s/infer
```

输出包括帧级特征和 1024 维句级 embedding。

## 5. IEMOCAP HMONNX 评估

IEMOCAP 需要从官方网站申请授权，本仓库不下载或分发数据。脚本严格检查五个 Session 合计 5531 条四分类语音，使用官方两层线性分类头、RMSprop、CyclicLR 和 leave-one-session-out 五折协议。

先提取 HMONNX 特征：

```bash
python examples_merak/audio/emotion2vec/debug/iemocap_hmonnx_eval.py \
	--iemocap-root /path/to/IEMOCAP_full_release \
	--meta work_dirs/emotion2vec_plus_large_xh2a_w8a8_16s/emotion2vec_meta.json \
	--feature-dir work_dirs/emotion2vec_plus_large_iemocap_features \
	--extract-only
```

再运行五折分类评估：

```bash
python examples_merak/audio/emotion2vec/debug/iemocap_hmonnx_eval.py \
	--iemocap-root /path/to/IEMOCAP_full_release \
	--feature-dir work_dirs/emotion2vec_plus_large_iemocap_features \
	--feature-dim 1024 \
	--evaluate-only
```

正式报告包含每折与五折平均 WA、UA 和 weighted F1。没有完整授权数据时，不应把脚本导入或 dry-run 当成正式评估结果。
