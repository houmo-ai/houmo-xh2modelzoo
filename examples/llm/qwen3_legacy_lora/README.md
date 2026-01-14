# Qwen3

## 配置参数

```text
batch_size
context-length
input-sequence-length
quant-type: for example: w4a8h0-ssfp、w8a8h1-sefp
quant-weight : GPTQ、Quarot量化后的权重文件
```

## 导出HMONNX

transformers库需要升级到4.51.0以上版本，否则会报错。  

### w8a8

#### 1. 导出

```bash
python examples/llm/qwen3_legacy_lora/qwen3_legacy_lora_xh2a_export_hmonnx.py --model data/models/Qwen3-8B/ --context-length 2048 --input-sequence-length 256 --quant-type w8a8h1_sefp --lora-checkpoint work_dirs/Qwen3-8B-test.gguf
```

#### 2. GPU仿真

```bash
python examples/llm/qwen3_legacy_lora/qwen3_legacy_xh2a_hmonnx_test.py --config work_dirs/Qwen3-8B-XH2a-2k-w8a8h1_sefp-lora/meta.json
```
 