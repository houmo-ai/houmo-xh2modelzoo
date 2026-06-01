# Llama

## 配置参数

```text
batch_size
context-length
input-sequence-length
quant-type: for example: w4a8h0-ssfp、w8a8h1-sefp
quant-weight : GPTQ、Quarot量化后的权重文件
```

## 导出HMONNX

### w8a8

```bash
python examples/llm/llama/llama_xh2a_export_hmonnx.py --model data/models/Meta-Llama-3.1-8B-Instruct --batch-size 1 --context-length 2048 --input-sequence-length 256 --quant-type w8a8h1_sefp
```

### GPU仿真

```bash
python examples/llm/llama/llama_xh2a_hmonnx_test.py --config work_dirs/Meta-Llama-3.1-8B-Instruct-XH2a-batch_1-2k-w8a8h1_sefp/meta.json
```

### w4a8

#### 1. Quarot、GPTQ量化  

lm_head 层一般不使用4bit int 量化，使用w8a8h1_sefp量化。
默认做Quarot+GPTQ量化，weight bit = 4

```bash
python examples/llm/llama/llama_xh2a_common_quant.py --model data/models/Meta-Llama-3.1-8B-Instruct --out-dir work_dirs/
```

#### 2. 导出

```bash
python examples/llm/llama/llama_xh2a_export_hmonnx.py --model data/models/Meta-Llama-3.1-8B-Instruct --batch-size 1 --context-length 2048 --input-sequence-length 256 --quant-type w4a8h0_ssfp --quant-weight /data01/nfs_shared/xhquant_quarot_gptq/llama3.1_8b_instruct_xh2a_2k_gptq_4bit/gptq-state-dict.safetensors
```

## GPU仿真

```bash
python examples/llm/llama/llama_xh2a_hmonnx_test.py --config work_dirs/Meta-Llama-3.1-8B-Instruct-XH2a-batch_1-2k-w4a8h0_ssfp/meta.json
```
