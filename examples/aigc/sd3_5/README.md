# SD3.5

## 依赖

```bash
pip install transformers==4.51.3
```

## 导出SD3-large-turbo模型

```bash
python examples/aigc/sd3_5/sd3_5_export.py --model data/models/stable-diffusion-3.5-large-turbo --guidance-scale 0 --width 512 --height 512
```

## 验证

```bash
python examples/aigc/sd3_5/sd3_5_hmonnx_test.py --config work_dirs/stable-diffusion-3.5-large-turbo_XH2a_512x512/meta.json --hf-model data/models/stable-diffusion-3.5-large-turbo --guidance-scale 0 --width 512 --height 512 --steps 4
```
