# gemma4_26b_a4b的导出与demo

## 环境基础

transformers==5.5.0

## Usage

### Export

以w4a8为例

导出视觉部分：
```
CUDA_VISIBLE_DEVICES=7 python examples_merak/llm/gemma4_moe/gemma4_moe_visual_xh_export_onnx.py --config onfigs_merak/xh2a/llm_models/gemma4_moe/26b_a4b_it/gemma4_moe_visual_26b_a4b_it_xh2a_448x448.py --image data/images/qwen2_vl_demo.jpeg 
```
在config中填写正确的模型路径和量化配置

导出llm部分：
```
CUDA_VISIBLE_DEVICES=7 python examples_merak/llm/gemma4_moe/gemma4_moe_with_mask_xh_export_hmonnx.py --config configs_merak/xh2a/llm_models/gemma4_moe/26b_a4b_it/gemma4_moe_with_mask_26b_a4b_it_xh2a_w4a8_256_2k.py 
```

在config中填写正确的模型路径和量化配置，注意w4a8的配置中依然要填写官方浮点的路径

### Demo/Generate

llm回答
```
CUDA_VISIBLE_DEVICES=7 python examples_merak/llm/gemma4_moe/gemma4_moe_with_mask_xh_generate.py --backend hmonnx --model-config work_dirs/gemma4_moe_with_mask_26b_a4b_it_xh2a_w4a8_256_2k/hmquant_xh2_gemma4_moe_with_mask_26b_a4b_it_w4a8_256_2k_20260508/golden_meta_info.json --prompt "你是谁" --streaming-out
```


vision+llm回答
```
CUDA_VISIBLE_DEVICES=7 python examples_merak/llm/gemma4_moe/gemma4_moe_with_mask_xh_vlm_generate.py --llm-config work_dirs/gemma4_moe_with_mask_26b_a4b_it_xh2a_w4a8_256_2k/hmquant_xh2_gemma4_moe_with_mask_26b_a4b_it_w4a8_256_2k_20260508/golden_meta_info.json --vision-config work_dirs/gemma4_moe_26b_a4b_it_vision_xh2a_no_upsample_token_448x448/export_meta_info.json --image data/images/qwen2_vl_demo.jpeg --prompt "描述这张图片" --streaming-out
```

### 带mtp模块的export

```
CUDA_VISIBLE_DEVICES=7 python examples_merak/llm/gemma4_moe/gemma4_moe_with_mask_mtp_xh_export_hmonnx.py --config configs_merak/xh2a/llm_models/gemma4_moe/26b_a4b_it/gemma4_moe_with_mask_mtp_26b_a4b_it_xh2a_w4a8_256_2k.py 
```

### 带mtp模块的demo

```
CUDA_VISIBLE_DEVICES=7 python examples_merak/llm/gemma4_moe/gemma4_moe_with_mask_mtp_xh_generate.py --meta work_dirs/gemma4_moe_with_mask_mtp_26b_a4b_it_xh2a_w4a8_256_2k/hmquant_xh2_gemma4_moe_with_mask_26b_a4b_it_mtp_w4a8_256_2k_20260518/golden_meta_info_mtp.json --prompt "你是谁" --max-new-tokens 512
```
