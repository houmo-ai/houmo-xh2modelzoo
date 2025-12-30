# UNet

## 下载模型

```bash
bash examples/cv/unet/download.sh
```

## 导出HMONNX

```bash

python examples/cv/unet/unet_export.py --onnx data/models/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.onnx 
```

## GPU仿真

```bash
python examples/cv/unet/unet_hmonnx_test.py --hmonnx work_dirs/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes/hmonnx/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_XH2a.onnx --image data/images/berlin_000006_000019_leftImg8bit.png 
```
