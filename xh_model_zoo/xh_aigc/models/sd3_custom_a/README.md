# sd3 2b custom a

## 依赖项

联想模型需要transformers版本4.46.0
diffusers       0.29.2
transformers    4.46.0
sentencepiece
peft            0.16.0

```text
mmdit_hypersd_fp16_v1.3.0.0.safetensors :微调后的fp16
mmdit_hypersd__non_sym_8bit_v1.3.0.1.safetensors： 量化版本
T5_non_sym_2bit_v1.3.0.1.safetensors ：量化版本, 使用该权重需要使用demo/ptq/sd3/modeling_t5.py替换transformers库的modeling_t5.py文件
pip show transformers 获取transformers路径
cp data/models/sd3_2b_custom_a/modeling_t5.py /opt/extdata/.conda/envs/xhquant/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py
vae, clip, clip_l 都使用开源版本
```

## 导出模型

```bash
python examples/aigc/sd3_custom_a/sd3_custom_a_export.py --model data/models/stable-diffusion-3-medium-diffusers --custom-a-model data/models/sd3_2b_custom_a --guidance-scale 2.5 --width 512 --height 512  
```
