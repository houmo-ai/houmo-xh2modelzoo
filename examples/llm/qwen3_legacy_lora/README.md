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

## 客户模型

```bash
rsync -avPl lihui@10.30.0.15:/data01/nfs_shared/customer_models/kylin_qwen3-8b-lora ./
```

## 量化模型

```bash
python examples/llm/qwen3_legacy_lora/qwen3_lora_xh2a_common_quant.py --model /data01/nfs_shared/customer_models/kylin_qwen3-8b-lora/Qwen3-8B-kylin-rotate --lora /data01/nfs_shared/customer_models/kylin_qwen3-8b-lora/lora_sq_intent_fp32_v1.0.1_20260123.gguf --skip-gptq
```

### w5a8

#### 1. 导出

```bash
python examples/llm/qwen3_legacy_lora/qwen3_legacy_lora_xh2a_export_hmonnx.py --model /data01/nfs_shared/customer_models/kylin_qwen3-8b-lora/Qwen3-8B-kylin-rotate --context-length 2048 --input-sequence-length 256 --quant-type w5a8h0_ssfp --quant-weight work_dirs/Qwen3-8B-kylin-rotate_quarot/quarot-state-dict.safetensors
```

#### 2. GPU仿真

```bash
python examples/llm/qwen3_legacy_lora/qwen3_legacy_xh2a_hmonnx_test.py --config work_dirs/Qwen3-8B-kylin-rotate-XH2a-2k-w5a8h0_ssfp-lora/meta.json
```
