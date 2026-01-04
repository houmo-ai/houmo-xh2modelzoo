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
python examples_new/llm/qwen3/qwen3_export.py --model /data01/datasets/Qwen3-0.6B -context-length 2048 --input-sequence-length 256 --quant-type w8a8h1_sefp
```

#### 2. GPU仿真

```bash
python examples_new/llm/qwen3/internal/qwen3_eval_hmonnx.py --export-dir work_dirs/Qwen3-0.6B-xh2a-2k-w8a8h1_sefp
```

### w4a8

#### 1. Quarot、GPTQ量化  

lm_head 层一般不使用4bit int 量化，使用w8a8h1_sefp量化。

```bash
python examples_new/llm/qwen3/qwen3_quant.py --model /data01/datasets/Qwen3-0.6B --out-dir work_dirs/Qwen3-0.6B-gptqmodel-4bit-g64 --bits 4 --batch-size=4 --hessian-mse
默认做Quarot+GPTQ量化，weight bit = 4
```

#### 2. 导出

```bash
python examples_new/llm/qwen3/qwen3_export.py --model work_dirs/Qwen3-0.6B-gptqmodel-4-bit-64g --context-length 2048 --input-sequence-length 256 --quant-type w4a8h0_ssfp
```

#### 3. GPU仿真

```bash
python examples_new/llm/qwen3/internal/qwen3_eval_hmonnx.py --export-dir work_dirs/Qwen3-0.6B-gptqmodel-4-bit-64g-xh2a-2k-w4a8h0_ssfp
```
