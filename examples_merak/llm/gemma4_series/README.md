##gemma4 dense 和 moe的导出与demo

### 环境准备

transformers==5.5.0

### Dense 31B（mode=all：visual + prefill + decode + golden）
```
CUDA_VISIBLE_DEVICES=6 python examples_merak/llm/gemma4_series/export_hmonnx.py 
--work-dir work_dirs/gemma4_31b_dense    --variant dense --mode all --golden --force
```

### MoE 26B（mode=all：vision_encoder + prefill + decode + golden）

```
CUDA_VISIBLE_DEVICES=7 python examples_merak/llm/gemma4_series/export_hmonnx.py --work-dir work_dirs/gemma4_26b_moe   --variant moe --mode all --golden --force
```
### Dense 图文对话

```
CUDA_VISIBLE_DEVICES=6 python examples_merak/llm/gemma4_series/generate.py \
    --backend hmonnx \
    --model-config work_dirs/gemma4_31b_dense/hmquant_xh2_gemma4_31b_it_w4a8_autoround_256_4k_20260605/golden_meta_info.json \
    --image-path data/images/qwen2_vl_demo.jpeg \
    --prompt "请用中文详细描述这张图片中的内容。" \
    --max-decode-steps 96
```

### MoE 图文对话

```
CUDA_VISIBLE_DEVICES=7 python examples_merak/llm/gemma4_series/generate.py \
    --backend hmonnx \
    --model-config work_dirs/gemma4_26b_moe/hmquant_xh2_gemma4_moe_with_mask_26b_a4b_it_w4a8_256_2k_20260605/golden_meta_info.json \
    --image-path data/images/qwen2_vl_demo.jpeg \
    --prompt "请用中文详细描述这张图片中的内容。" \
    --max-decode-steps 96
```

### 纯文本对话（去掉 --image-path 即可）

```

CUDA_VISIBLE_DEVICES=6 python examples_merak/llm/gemma4_series/generate.py \
    --backend hmonnx \
    --model-config work_dirs/gemma4_31b_dense/hmquant_xh2_gemma4_31b_it_w4a8_autoround_256_4k_20260605/golden_meta_info.json \
    --prompt "用中文介绍一下你自己。" --max-decode-steps 96

CUDA_VISIBLE_DEVICES=7 python examples_merak/llm/gemma4_series/generate.py \
    --backend hmonnx \
    --model-config work_dirs/gemma4_26b_moe/hmquant_xh2_gemma4_moe_with_mask_26b_a4b_it_w4a8_256_2k_20260605/golden_meta_info.json \
    --prompt "用中文介绍一下你自己。" --max-decode-steps 96
```