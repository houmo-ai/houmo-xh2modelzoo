# FunAudioChat Merak

该目录提供 FunAudioChat 的 Merak 多图导出、浮点推理和 HMONNX 推理入口。导出包含以下六张图：

- `audio_tower`
- `audio_encoder`
- `qwen3` prefill / decode
- `audio_decoder` prefill / decode

## 导出

```bash
python examples_merak/audio/funaudiochat/funaudiochat_workflow.py \
  --model-dir /path/to/Fun-Audio-Chat-8B \
  --audio /path/to/reference.wav \
  --output-dir work_dirs/funaudiochat_merak/export_xh2a_w8a8 \
  --device cuda:0
```

默认 workflow 配置为 [funaudiochat.yaml](../../../configs_merak/workflows/xh2a/other_models/funaudiochat/funaudiochat.yaml)。音频路径通过命令行覆盖写入实际导出的静态 audio encoder 输入，不在配置中保留机器相关绝对路径。

导出根目录会生成 `export_meta_info.json`，各组件目录保留自己的 `meta.json` 和 HMONNX 文件。

## 浮点推理

```bash
python examples_merak/audio/funaudiochat/infer_s2s.py \
  --model-dir /path/to/Fun-Audio-Chat-8B \
  --audio /path/to/input.wav \
  --device cuda:0
```

## HMONNX 推理

```bash
python examples_merak/audio/funaudiochat/hmonnx_demo.py \
  --model-dir /path/to/Fun-Audio-Chat-8B \
  --export-dir work_dirs/funaudiochat_merak/export_xh2a_w8a8 \
  --audio /path/to/input.wav \
  --device cuda:0 \
  --execution-device cpu
```

HMONNX demo 从 `export_meta_info.json` 发现六张图，并用原始 HF 模型提供 tokenizer、processor、embedding 和生成控制逻辑。

## 配置

`quant` 必须保持为 `null`。每个组件可单独设置 `enabled` 和 `quant_type`：

```yaml
components:
  audio_tower:
    enabled: true
    quant_type: w8a8h1_sefp
```

当前导出目标仅支持 `XH2a`。`mix_search` 未迁移到该 workflow，传入非空配置会明确报错。
