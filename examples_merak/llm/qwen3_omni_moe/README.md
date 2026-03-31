# Qwen3 Omni Examples

- Export HMONNX:
  `python examples_merak/llm/qwen3_omni_moe/qwen3_omni_xh_export_hmonnx.py --model /path/to/Qwen3-Omni --model-type Qwen3OmniMoeForConditionalGeneration_text`
- HMONNX Generate:
  `python examples_merak/llm/qwen3_omni_moe/qwen3_omni_xh_hmonnx_generate.py --config /path/to/golden_meta_info.json`
- XH Debug Generate:
  `python examples_merak/llm/qwen3_omni_moe/debug_scripts/qwen3_omni_visual_xh_generate.py --model /path/to/Qwen3-Omni --model-type Qwen3OmniMoeForConditionalGeneration_text`
- Native HF Debug Generate:
  `python examples_merak/llm/qwen3_omni_moe/debug_scripts/native_qwen3_omni_generate.py --model-dir /path/to/Qwen3-Omni`

## Debug 调试

### 1. 调试原始模型

```bash
python examples_merak/llm/qwen3_omni_moe/debug_scripts/native_qwen3_omni_generate.py  --model-dir data/models/Qwen3-Omni-30B-A3B-Instruct
```

### 2. 调试视觉模型

```bash
python examples_merak/llm/qwen3_omni_moe/debug_scripts/qwen3_omni_visual_xh_generate.py --config configs_merak/xh2a/llm_models/qwen3_omni/30B-A3B-Instruct/xh2a_qwen3-omni-30b-a3b_instruct_visual_w8a8h1_sefp_256_2k_560x560.py --auto-offload
```
