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
python examples/llm/qwen3moe/qwen3_moe_xh2a_export_hmonnx.py --model data/datasets/QQwen3-30B-A3B --context-length 2048 --input-sequence-length 256 --quant-type w8a8h1_sefp
```

#### 2. GPU仿真

```bash
python examples/llm/qwen3moe/qwen3_moe_xh2a_hmonnx_test.py --config work_dirs/QQwen3-30B-A3B-XH2a-2k-w8a8h0_sefp/meta.json
```

### w4a8

#### 1. Quarot

lm_head 层一般不使用4bit int 量化，使用w8a8h1_sefp量化。

```bash
python examples/llm/qwen3_legacy/qwen3_xh2a_common_quant.py  --model data/datasets/QQwen3-30B-A3B  --w-bits 4 --out-dir work_dirs/
默认做Quarot+GPTQ量化，weight bit = 4
```

#### 2. 导出

```bash
python examples/llm/qwen3moe/qwen3_moe_xh2a_export_hmonnx.py --model data/datasets/QQwen3-30B-A3B --context-length 2048 --input-sequence-length 256 --quant-type w4a8h0_ssfp --quant-weight work_dirs/QQwen3-30B-A3B_quarot/quarot-state-dict.safetensors
```

#### 3. GPU仿真

```bash
python examples/llm/qwen3moe/qwen3_moe_xh2a_hmonnx_test.py --config work_dirs/QQwen3-30B-A3B-XH2a-2k-w8a8h0_sefp/meta.json
```
