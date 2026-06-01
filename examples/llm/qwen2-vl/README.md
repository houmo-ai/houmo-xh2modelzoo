# Qwen2-vl

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

#### 1. 导出

```bash
python examples/llm/qwen2-vl/qwen2-vl_xh2a_export_hmonnx.py --model data/models/Qwen2-VL-2B-Instruct  --context-length 2048 --input-sequence-length 256 --quant-type w8a8h1_sefp
```

#### 2. GPU仿真

```bash
python examples/llm/qwen2-vl/qwen2-vl_xh2a_hmonnx_test.py --config work_dirs/Qwen2-VL-2B-Instruct-XH2a-batch_1-2k-w8a8h1_sefp/meta.json
```

### w4a8

#### 1. 导出Qwen2-VL-AWQ

```bash
python examples/llm/qwen2-vl/qwen2-vl_xh2a_export_hmonnx.py --model /data01/nfs_shared/llm_models/Qwen/Qwen2-VL-2B-Instruct-AWQ  --context-length 2048 --input-sequence-length 256 --quant-type w4a8h0_ssfp
```

#### 2. GPU仿真

```bash
python examples/llm/qwen2-vl/qwen2-vl_xh2a_hmonnx_test.py --config work_dirs/Qwen2-VL-2B-Instruct-AWQ-XH2a-batch_1-2k-w4a8h0_ssfp/meta.json
```
