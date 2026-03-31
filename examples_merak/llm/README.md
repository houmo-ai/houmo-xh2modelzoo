# Qwen3

## 配置参数

参考configs_merak/xh2a/llm_models/qwen3_legacy/1.7b/qwen3_1.7b_legacy_xh2a_w4a8_2k.py

## 量化

```bash

```

## 导出HMONNX

```bash
python examples_merak/llm/llm_export_hmonnx.py --config configs_merak/xh2a/llm_models/qwen3_legacy/8b/qwen3_8b_legacy_xh2a_2k.py
或者
python examples_merak/llm/llm_export_hmonnx.py --model data/models/Qwen3-8B/ --context-length 2048 --prefill-chunk-length 256 --quant-type w8a8h1_sefp
或者
python examples_merak/llm/llm_export_hmonnx.py --model data/models/Qwen3-8B/ --context-length 2048 --prefill-chunk-length 256 --quant-type w4a8h0_ssfp --quant-weight work_dirs/qwen3_8b_instruct_xh2a_2k_quarot_gptq_4bit/quarot_gptq-state-dict.safetensors
```
