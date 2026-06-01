# DM0

## 配置参数

```text
batch_size
context-length 3072
input-sequence-length 2555
quant-type: w8a8h1-sefp
quant-weight : GPTQ、Quarot量化后的权重文件
```

```vision
input_size 728×728
```

## 导出HMONNX

### w8a8 sefp

#### 1. 导出llm

```bash
python llm_xh2a_export_hmonnx.py --model model_path
```

#### 2. 导出expert

```bash
python expert_xh2a_export_hmonnx.py --model model_path
```

#### 3. 导出vision

```bash
python vision_xh2a_export_hmonnx.py --model model_path
```

