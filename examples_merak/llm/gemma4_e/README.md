# Gemma4_E2B的导出与demo

## 环境基础

请确保transformers版本在5.5.0

## Usage

### Export 

以w8a8为例

导出：
```
CUDA_VISIBLE_DEVICES=7 python examples_merak/llm/gemma4_e/gemma4_e_xh_export_hmonnx.py --model ./model/gemma-4-E2B-it --quant-type w8a8h1_sefp
```

### Demo/Generate 

```
CUDA_VISIBLE_DEVICES=7 python examples_merak/llm/gemma4_e/gemma4_e_xh_hmonnx_generate.py --config ./work_dirs/gemma4_e2b_export_dir/meta.json --prompt "please describe the picture" --image-path xxx.jpg
```

