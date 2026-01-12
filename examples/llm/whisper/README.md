# Whisper

## 依赖

1. 依赖包

```
pip install transformers==4.57
pip install torchcodec
```

## 导出HMONNX

### 导出方法

encoder编码器:包含encoder + decoder的encoder注意力部分
prefill:decoder模型的prefill阶段
decode：decoder模型的decode阶段

### 导出脚本

```bash
python examples/llm/whisper/hmonnx_export_encoder_prefill.py --model data/models/whisper-medium/
python examples/llm/whisper/hmonnx_export_decoder.py --model data/models/whisper-medium/
```

### demo

```bash
python examples/llm/whisper/hm_demo.py --hf-model data/models/whisper-medium/ --hmonnx-model works/whisper-medium_XH2a/ --audio ./examples/llm/whisper/audio.mp3
```
