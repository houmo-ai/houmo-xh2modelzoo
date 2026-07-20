# SenseVoiceSmall Merak 工作流

本目录将 SenseVoiceSmall 的 FP32 ONNX、XH2a HMONNX、golden、demo 和精度评测统一到
`xhmodel_merak.workflows.AutoWorkflow`。模型实现与运行时工具位于
`xhmodel_merak/xh_other_model/models/sensevoice_small`，不依赖旧的 `examples/audio/sensevoice_small`。

## 模型与依赖

- Hugging Face 模型：`FunAudioLLM/SenseVoiceSmall`
- 本地完整模型示例：`/data02/datasets/SenseVoiceSmall`
- 主要依赖：`funasr`、`torch`、`torchaudio`、`librosa`、`soundfile`、`sentencepiece`、
  `onnx`、`onnxruntime`、`onnxsim`、`datasets`、`tqdm`、`xhquant`

HF 音频列会以 `datasets.Audio(decode=False)` 读取，再由 `soundfile`/`librosa` 解码，因此校准代码
不显式依赖 FFmpeg 或 torchcodec。当前验证环境使用 `datasets==3.6.0`。

## 默认 YAML

默认配置：

```text
configs_merak/workflows/xh2a/other_models/sensevoice_small/sensevoice_small_xh2a.yaml
```

该 YAML 对齐旧 README 的实际参数：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `export.target_device` | `XH2a` | 目标芯片 |
| `export.onnx.device` | `cpu` | FunASR 模型加载及 ONNX trace 设备 |
| `export.onnx.max_seq_len` | `512` | 静态特征序列长度 |
| `export.onnx.opset` | `14` | ONNX opset |
| `export.onnx.static` | `true` | HMONNX 需要静态输入 |
| `export.onnx.simplify` | `true` | 使用 onnxsim 简化 |
| `export.onnx.layer_norm_scale` | `32.0` | LayerNorm 输入缩放，改善 FP16 稳定性 |
| `export.hmonnx.quant_type` | `w8a8h1_sefp` | HMONNX 量化精度 |
| `export.hmonnx.calib_metric` | `minmax` | 可改为 `mse` 或 `kl` |
| `export.hmonnx.force_fp32_ops` | `[LayerNorm]` | 强制保留 FP32 的算子类型 |
| `export.hmonnx.calibration.*` | LibriSpeech clean/validation、streaming、128 条 | HF 校准集 |

`quant` 固定为 `null`：SenseVoice 没有独立的量化权重模型，PTQ 在 ONNX 转 HMONNX 时完成。

## 完整导出

以下命令不传参数 override，完全使用 YAML 默认值，即旧 README 的 128 条 LibriSpeech 校准配置：

```bash
conda run -n xh2modelzoo_sensevoice python \
  examples_merak/asr/sensevoice_small/sensevoice_small_workflow.py \
  --model-dir /data02/datasets/SenseVoiceSmall \
  --overwrite \
  --dump-golden
```

快速验证可只把校准条数覆盖为 2：

```bash
conda run -n xh2modelzoo_sensevoice python \
  examples_merak/asr/sensevoice_small/sensevoice_small_workflow.py \
  --model-dir /data02/datasets/SenseVoiceSmall \
  --calib-samples 2 \
  --output-dir work_dirs/sensevoice_small_merak/smoke \
  --overwrite \
  --dump-golden
```

脚本还支持覆盖 `--max-seq-len`、`--quant-type`、`--calib-metric`、HF dataset/config/split、
本地校准 `.pth`、ONNX-only 和动态 ONNX。每次实际使用的 override 后配置都会写入导出目录。

## 产物

默认输出目录为 `work_dirs/sensevoice_small_merak/export_xh2a_w8a8h1_sefp`：

```text
export_meta_info.json
sensevoice_small_xh2a.yaml
assets/
onnx/
  model.onnx
  export_meta.json
hmonnx/
  sensevoice_small_XH2a_w8a8h1_sefp.onnx
  sensevoice_small_XH2a_w8a8h1_sefp_external_data
  golden/
golden_meta_info.json
convert.log
```

HMONNX 文件名同时包含目标芯片和量化精度。demo 和评测都从顶层 `export_meta_info.json` 解析模型及
runtime assets，不要求用户再分别传 ONNX/HMONNX 路径。

## HMONNX demo

```bash
conda run -n xh2modelzoo_sensevoice python \
  examples_merak/asr/sensevoice_small/hmonnx_demo.py \
  /data02/datasets/SenseVoiceSmall/example/en.mp3 \
  --export-dir work_dirs/sensevoice_small_merak/export_xh2a_w8a8h1_sefp \
  --device cuda:0
```

## FP32 与量化评测

```bash
conda run -n xh2modelzoo_sensevoice python \
  examples_merak/asr/sensevoice_small/float_eval.py \
  --export-dir work_dirs/sensevoice_small_merak/export_xh2a_w8a8h1_sefp \
  --hf-dataset openslr/librispeech_asr \
  --hf-config clean \
  --hf-split test \
  --hf-streaming \
  --limit 100
```

```bash
conda run -n xh2modelzoo_sensevoice python \
  examples_merak/asr/sensevoice_small/quant_eval.py \
  --export-dir work_dirs/sensevoice_small_merak/export_xh2a_w8a8h1_sefp \
  --hf-dataset openslr/librispeech_asr \
  --hf-config clean \
  --hf-split test \
  --hf-streaming \
  --limit 100 \
  --device cuda:0
```

两种评测还支持 JSONL manifest 或 Kaldi `wav.scp + text`，并生成逐条结果及 corpus-level CER/WER JSON。
