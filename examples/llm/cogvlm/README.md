# Qwen2

## 配置参数

```text
batch_size
context-length
input-sequence-length
quant-type: for example: w4a8h0-ssfp、w8a8h1-sefp
image_size: int = 1344
patch_size: int = 14
```

## 导出HMONNX

### w8a8

#### 1. 导出

```bash
python examples/llm/cogvlm/cogvlm2_xh2a_export_hmonnx.py --model data/models/cogvlm2-llama3-chat-19B --batch-size 1 --context-length 2048 --input-sequence-length 256 --quant-type w8a8h1_sefp
```
