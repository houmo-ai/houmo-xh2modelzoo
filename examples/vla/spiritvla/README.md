# spirit_v1.5

## 配置参数

```text
batch_size
context-length 512
input-sequence-length 256
quant-type: w8a8h1-sefp
quant-weight : GPTQ、Quarot量化后的权重文件
```

```vision
input_size 256×320
```

## 导出HMONNX

### w8a8 sefp

#### 1. 导出llm

```bash
python qwen3_vl_xh2a_export_hmonnx.py --model model_path
```

#### 2. 导出expert

```bash
python dit_export_hmonnx.py
```


