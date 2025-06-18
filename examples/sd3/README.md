# SD3

## 导出模型

```bash
python examples/sd3/sd3_export.py --model data/models/stable-diffusion-3-medium-diffusers --guidance-scale 7 --width 512 --height 512 
```

## 验证

```bash
python examples/sd3/sd3_hmonnx_test.py --config work_dirs/stable-diffusion-3-medium-diffusers_XH2a_512x512/meta.json --hf-model data/models/stable-diffusion-3-medium-diffusers --guidance-scale 7 --width 512 --height 512 --steps 28
```
