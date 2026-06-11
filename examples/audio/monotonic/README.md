# Monotonic Timestamp (xhquant Adaptation)

该目录用于 `funasr/Monotonic` 的时间戳模型适配，包含：
- ModelScope 原始推理验证
- ONNX 导出（含 Pow(x,2) -> Mul(x,x) 修正）
- xhquant 转 hmonnx
- FP / ONNX / HMONNX 一致性对比

## 1. 环境

建议使用 conda 环境 `xhquant`：

```bash
cd /data01/home/xuchen/xh2/xh2_model_zoo
source env.sh
```

模型目录默认：
- `/data02/datasets/funasr/Monotonic`

## 2. 原始 ModelScope 验证

```bash
/data01/home/xuchen/miniconda3/envs/xhquant/bin/python examples/audio/monotonic/monotonic.py \
  --model-dir /data02/datasets/funasr/Monotonic \
  --audio /data02/datasets/funasr/Monotonic/example/asr_example.wav \
  --text-path /data02/datasets/funasr/Monotonic/example/text.txt \
  --validate
```

## 3. 导出 ONNX

```bash
/data01/home/xuchen/miniconda3/envs/xhquant/bin/python examples/audio/monotonic/export_onnx.py \
  --model-dir /data02/datasets/funasr/Monotonic \
  --audio /data02/datasets/funasr/Monotonic/example/asr_example.wav \
  --text-path /data02/datasets/funasr/Monotonic/example/text.txt \
  --step all
```

默认生成：
- `examples/audio/monotonic/monotonic_timestamp.onnx`
- `examples/audio/monotonic/monotonic_timestamp_simplified.onnx`

## 4. 转 hmonnx 并做一致性验证

```bash
/data01/home/xuchen/miniconda3/envs/xhquant/bin/python examples/audio/monotonic/demo_xhquant.py \
  --model-dir /data02/datasets/funasr/Monotonic \
  --audio /data02/datasets/funasr/Monotonic/example/asr_example.wav \
  --text-path /data02/datasets/funasr/Monotonic/example/text.txt \
  --reuse-onnx \
  --device cuda:0
```

默认输出目录：
- `work_dirs/monotonic/hmonnx/`

日志会打印：
- `FP text / ONNX text / HMONNX text`
- `FP timestamp / ONNX timestamp / HMONNX timestamp`
- `Equal: fp-vs-onnx, fp-vs-hmonnx`
- `Max abs diff`

## 5. 导出 golden（可选）

```bash
/data01/home/xuchen/miniconda3/envs/xhquant/bin/python examples/audio/monotonic/demo_xhquant.py \
  --model-dir /data02/datasets/funasr/Monotonic \
  --audio /data02/datasets/funasr/Monotonic/example/asr_example.wav \
  --text-path /data02/datasets/funasr/Monotonic/example/text.txt \
  --reuse-onnx \
  --save-golden \
  --device cuda:0
```

golden 目录位于：
- `work_dirs/monotonic/hmonnx/monotonic_timestamp_*_golden/`
