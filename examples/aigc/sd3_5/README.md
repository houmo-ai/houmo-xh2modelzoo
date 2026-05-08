# SD3.5

## 依赖

```bash
pip install transformers==4.51.3
diffusers       0.31.0
sentencepiece
peft            0.16.0
```

## 导出SD3-large-turbo模型

```bash
python examples/aigc/sd3_5/sd3_5_export.py --model data/models/stable-diffusion-3.5-large-turbo --guidance-scale 0 --width 512 --height 512
```

## 验证SD3-large-turbo

```bash
python examples/aigc/sd3_5/sd3_5_hmonnx_test.py --config work_dirs/stable-diffusion-3.5-large-turbo_XH2a_512x512/meta.json --hf-model data/models/stable-diffusion-3.5-large-turbo  --steps 4 --output work_dirs/stable-diffusion-3.5-large-turbo_XH2a_512x512/hmonnx/golden//sd3.5-large-turbo-test.png
```

## 导出SD3-medium模型

```bash
python examples/aigc/sd3_5/sd3_5_export.py --model data/models/stable-diffusion-3.5-medium --guidance-scale 4.5 --width 512 --height 512
```

## 验证SD3-medium模型

```bash
python examples/aigc/sd3_5/sd3_5_hmonnx_test.py --config work_dirs/stable-diffusion-3.5-medium_XH2a_512x512/meta.json --hf-model data/models/stable-diffusion-3.5-medium  --steps 40 --output work_dirs/stable-diffusion-3.5-medium_XH2a_512x512/hmonnx/golden/sd3.5-medium-test.png
```
