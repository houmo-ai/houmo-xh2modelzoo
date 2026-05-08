# SD3

## 依赖

```bash
pip install transformers==4.47.0
diffusers       0.29.2
sentencepiece
peft            0.16.0
```

## 导出SD3-2b模型

```bash
python examples/aigc/sd3/sd3_export.py --model data/models/stable-diffusion-3-medium-diffusers --guidance-scale 7 --width 512 --height 512
```

## 验证

```bash
python examples/aigc/sd3/sd3_hmonnx_test.py --config work_dirs/stable-diffusion-3-medium-diffusers_XH2a_512x512/meta.json --hf-model data/models/stable-diffusion-3-medium-diffusers --steps 28 --output work_dirs/stable-diffusion-3-medium-diffusers_XH2a_512x512/hmonnx/golden
```
