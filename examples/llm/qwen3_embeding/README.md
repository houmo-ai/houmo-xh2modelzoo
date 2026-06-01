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
python examples/llm/qwen3_embeding/qwen3_embeding_xh2a_export.py --model data/models/Qwen3-Embedding-4B/ --context-length 2048 --input-sequence-length 256 --quant-type w8a8h1_sefp
```

```bash
python examples/llm/qwen3_embeding/qwen3_embeding_xh2a_export.py --model /data01/nfs_shared/llm_models/Qwen/Qwen3-Embedding-4B-W4A16-G128 --context-length 2048 --input-sequence-length 256 --quant-type w4a8h0_ssfp
```
