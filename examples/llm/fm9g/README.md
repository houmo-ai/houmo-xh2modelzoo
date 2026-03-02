# Qwen2

## 来源：<https://www.osredm.com/jiuyuan/CPM-9G-8B>

## 地址：<https://thunlp-model.oss-cn-wulanchabu.aliyuncs.com/9g_8b_v2_thinking.tar>

## 配置参数

```text
batch_size
context-length
input-sequence-length
quant-type: for example: w4a8h0-ssfp、w8a8h1-sefp
quant-weight : GPTQ、Quarot量化后的权重文件
```

## 导出HMONNX

transformers库需要升级到4.47.0以上版本，否则会报错。  
### w8a8

#### 1. 导出

```bash
python examples/llm/fm9g/fm9g_xh2a_export_hmonnx.py --model data/models/9g_8b_thinking --context-length 2048 --input-sequence-length 256 --quant-type w8a8h1_sefp
```
