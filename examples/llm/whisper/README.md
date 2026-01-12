# Whisper

## 依赖

1. 依赖包

```
pip install transformers==4.57
pip install torchcodec
```

## 导出HMONNX

```bash
python examples/llm/whisper/hmonnx_export_encoder_prefill.py --model data/models/whisper-medium/
python examples/llm/whisper/hmonnx_export_decoder.py --model data/models/whisper-medium/
```
