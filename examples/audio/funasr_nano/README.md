# FunASR-Nano XH2a 适配

本目录用于把 FunASR-Nano（SenseVoiceSmall/SenseVoiceEncoderSmall 音频编码器 + Qwen3-0.6B LLM 解码器 + audio_adaptor + CTC 分支）拆分适配到 xhquant/HMONNX。

一键流程：

```bash
python examples/audio/funasr_nano_xh2a/run_export_pipeline.py \
  --model-dir /data01/datasets/Funasr/Fun-ASR-Nano-2512 \
  --work-dir work_dirs/funasr_nano_xh2a \
  --max-frames 512 \
  --context-length 2048 \
  --input-sequence-length 256 \
  --quant-type w8a8h1_sefp
```

## 1. 导出音频侧 ONNX

```bash
python examples/audio/funasr_nano_xh2a/export_audio_modules_onnx.py \
  --model-dir /data01/datasets/Funasr/Fun-ASR-Nano-2512 \
  --work-dir work_dirs/funasr_nano_xh2a \
  --max-frames 512
```

输出：
- `Encoder/funasr_nano_encoder.onnx`
- `Adaptor/funasr_nano_audio_adaptor.onnx`
- `CTC/funasr_nano_ctc.onnx`（模型存在 CTC 分支时）
- `token_embedding.pt`
- `export_meta_info.json`

## 2. 独立导出 Qwen3-0.6B prefill/decode

LLM 解码器使用本目录的独立脚本，不复用 `qwen3_legacy` / `qwen3_asr` 示例导出流程：

```bash
python examples/audio/funasr_nano_xh2a/export_qwen3_llm_hmonnx.py \
  --model-dir /data01/datasets/Funasr/Fun-ASR-Nano-2512/Qwen3-0.6B \
  --work-dir work_dirs/funasr_nano_xh2a \
  --context-length 2048 \
  --input-sequence-length 256 \
  --quant-type w8a8h1_sefp
```

脚本会直接生成 Qwen3 prefill/decode HMONNX，并自动写入 `work_dirs/funasr_nano_xh2a/export_meta_info.json`：

```json
{
  "prefill_hmonnx_file": "Prefill/hmonnx/xxx_prefill.onnx",
  "decode_hmonnx_file": "Decoder/hmonnx/xxx_decode.onnx",
  "num_hidden_layers": 28,
  "kv_cache_shape": [1, 8, 2048, 128],
  "prefill_input_sequence_length": 256,
  "hf_config": "ConfigFiles"
}
```

## 3. ONNX 转 HMONNX

```bash
python examples/audio/funasr_nano_xh2a/convert_hmonnx.py \
  --work-dir work_dirs/funasr_nano_xh2a \
  --quant-type w8a8h1_sefp
```

该脚本会扫描 `export_meta_info.json` 中的 `*_onnx_file`，生成对应 `*_hmonnx_file` 字段。

## 4. HMONNX 推理

推理 wrapper 位于：

`xh_model_zoo/xh_llm/models/funasr_nano/funasr_nano_hmonnx_model.py`

示例：

```bash
python examples/audio/funasr_nano_xh2a/hmonnx_demo.py \
  --work-dir work_dirs/funasr_nano_xh2a \
  --audio /path/to/audio.wav
```

## 5. 精度验证建议

1. 使用 FunASR `AutoModel(...).generate()` 作为浮点基线。
2. 使用 `FunASRNanoHMONNXModel.generate()` 得到 HMONNX 结果。
3. 先对齐音频侧：比较 encoder/adaptor 输出 cosine similarity 与最大误差。
4. 再对齐 LLM：固定相同 `inputs_embeds`，比较 prefill/decode logits top-1 与误差。
5. 端到端统计 CER/WER。

端到端 FP vs HMONNX 对比脚本：

```bash
python examples/audio/funasr_nano_xh2a/eval_accuracy.py \
  --model-dir /data01/datasets/Funasr/Fun-ASR-Nano-2512 \
  --work-dir work_dirs/funasr_nano_xh2a \
  --audio /path/to/audio.wav
```
