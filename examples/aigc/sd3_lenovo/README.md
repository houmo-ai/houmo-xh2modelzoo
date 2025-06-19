# sd3 2b lenovo

## 依赖项

联想模型需要transformers版本4.46.2

```text
mmdit_hypersd_fp16_v1.3.0.0.safetensors :微调后的fp16
mmdit_hypersd__non_sym_8bit_v1.3.0.1.safetensors： 量化版本
T5_non_sym_2bit_v1.3.0.1.safetensors ：量化版本, 使用该权重需要使用demo/ptq/sd3/modeling_t5.py替换transformers库的modeling_t5.py文件
pip show transformers 获取transformers路径
cp data/models/sd3_2b_lenovo/modeling_t5.py /opt/extdata/.conda/envs/xhquant/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py
vae, clip, clip_l 都使用开源版本
```

## 导出模型

```bash
python examples/aigc/sd3/sd3_lenovo_export.py --model data/models/stable-diffusion-3-medium-diffusers --lenovo-model data/models/sd3_2b_lenovo --guidance-scale 2.5 --width 512 --height 512  
```
