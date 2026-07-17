# emotion2vec+ Large Merak

本目录默认适配 emotion2vec 官方 GitHub README 推荐的 `iic/emotion2vec_plus_large` 完整情感识别模型，覆盖 ModelScope 下载、Merak 默认 W8A8 HMONNX 导出、真实音频 Golden、情感分类推理和官方 IEMOCAP 五折评估协议。HMONNX 输出帧级特征、padding mask 和 1024 维句级特征；官方顶层 `proj` 作为 `hmquant/quant_embedding.pt` 独立保存，便于在合并长音频特征后自行计算分类结果。

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

`--dump-golden` 会在导出后调用同一个 `Emotion2vecWorkflow.dump_golden()`。它使用实际 FP16 waveform 和 INT32 `valid_frames` 运行 `HMONNXGoldenInference`，将每个 HMONNX 算子的输出保存到 `golden/step_0/`；同时使用官方 FunASR FP32 模型生成业务级参考结果，保存到 `golden/reference/`。导出目录还会生成 `emotion2vec_meta.json`、本次使用的 workflow YAML 副本、HMONNX 文件和 `hmquant/quant_embedding.pt`。`quant_embedding.pt` 是可直接由 `torch.load()` 读取的官方 `proj` state dict，包含 `[9, 1024]` 的 `weight` 和 `[9]` 的 `bias`。音频解码、重采样、波形归一化、长音频分块、有效帧数计算、有效帧拼接和最终分类位于图外；特征网络主体位于 HMONNX 图内。波形归一化使用 FP32，并且只统计 `valid_samples` 指定的有效区域。图外按七层卷积参数将 `valid_samples` 转换成 INT32 `valid_frames`；HMONNX 只通过一次 `Less` 生成 frame padding mask，不再包含逐层 `Sub/Div/Add` 长度计算。

## 3. 比较 Golden

上一节通过 workflow 生成：

- `golden/step_0/*.npy`：xhquant 标准 HMONNX 逐算子 Golden；
- `golden/reference/frame_features.npy`：官方 FP32 帧级特征；
- `golden/reference/utterance_feature.npy`：官方 FP32 句级特征；
- `golden/reference/logits.npy`：官方 FP32 9 类分类 logits；
- `golden/reference/probabilities.npy`：官方 FP32 9 类分类概率；
- `golden/reference/reference_meta.json`：音频路径、有效采样数、帧数和特征维度。

使用同一音频运行 W8A8 HMONNX 并比较：

```bash
python examples_merak/audio/emotion2vec/debug/compare_golden.py \
	--meta work_dirs/emotion2vec_plus_large_xh2a_w8a8_16s/emotion2vec_meta.json \
	--audio data/models/emotion2vec_plus_large/example/test.wav \
	--golden-dir work_dirs/emotion2vec_plus_large_xh2a_w8a8_16s/golden
```

脚本报告帧级 cosine、句级 cosine、分类 logits cosine、MAE 和概率误差。默认要求三项 cosine 均不低于
`0.9`，未达标时返回非零退出码。`--min-frame-cos` 和 `--min-utterance-cos` 可用于实验性调整阈值，
但降低阈值后的结果不应作为验收证据。

## 4. HMONNX 推理

```bash
python examples_merak/audio/emotion2vec/hmonnx_infer.py \
	--meta work_dirs/emotion2vec_plus_large_xh2a_w8a8_16s/emotion2vec_meta.json \
	--audio data/models/emotion2vec_plus_large/example/test.wav \
	--output work_dirs/emotion2vec_plus_large_xh2a_w8a8_16s/infer
```

运行时先合并全部有效帧并计算 1024 维句级 embedding，再加载 metadata 指向的 `hmquant/quant_embedding.pt` 执行 `linear + softmax`。输出包括帧级特征、句级 embedding、9 维 logits、9 维 probabilities，以及按官方 FunASR 逻辑过滤 `unuse_*` 后的标签、分数和预测标签。官方输出仍保留 `<unk>`。如果只需要特征，也可以忽略分类 artifact；自行分类时等价代码为 `torch.nn.functional.linear(utterance_feature, state["weight"], state["bias"])`。

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
