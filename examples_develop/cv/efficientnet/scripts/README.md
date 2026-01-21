# efficientnet

## 下载模型

```bash
python examples/cv/efficientnet/scripts/efficientnet_export_onnx.py
python examples/cv/efficientnet/scripts/efficientnet_v2_export_onnx.py
```

## 导出HMONNX

```bash
python examples/cv/efficientnet/scripts/efficientnet_export_hmonnx.py --onnx data/models/efficientnet/efficientnet_b4.onnx
python examples/cv/efficientnet/scripts/efficientnet_export_hmonnx.py --onnx data/models/efficientnet/efficientnet_v2_m.onnx
```

## GPU仿真

```bash
python examples/cv/efficientnet/scripts/efficientnet_hmonnx_test.py --hmonnx work_dirs/efficientnet/efficientnet_b4_1x3x380x380_w8a8_sefp/efficientnet_b4_1x3x380x380_w8a8_sefp_XH2a.onnx --image data/images/ILSVRC2012_val_00002031.JPEG --input-size 380
python examples/cv/efficientnet/scripts/efficientnet_hmonnx_test.py --hmonnx work_dirs/efficientnet/efficientnet_v2_m_1x3x480x480_w8a8_sefp/efficientnet_v2_m_1x3x480x480_w8a8_sefp_XH2a.onnx --image data/images/ILSVRC2012_val_00002031.JPEG --input-size 480
```

## Imagenet 评测

### 1. 评测原浮点模型

```bash
python examples/cv/efficientnet/scripts/efficientnet_eval.py --model-type onnx --model-path data/models/efficientnet/efficientnet_b4.onnx --input-size 380 --data-dir /data01/datasets/imagenet/val
python examples/cv/efficientnet/scripts/efficientnet_eval.py --model-type onnx --model-path data/models/efficientnet/efficientnet_v2_m.onnx --input-size 480 --data-dir /data01/datasets/imagenet/val
```

### 2. 评测hmonnx模型

```bash
python examples/cv/efficientnet/scripts/efficientnet_eval.py --model-type hmonnx --model-path work_dirs/efficientnet/efficientnet_b4_w8a8_sefp/efficientnet_b4_w8a8_sefp_XH2a.onnx --input-size 380 --data-dir /data01/datasets/imagenet/val
python examples/cv/efficientnet/scripts/efficientnet_eval.py --model-type hmonnx --model-path work_dirs/efficientnet/efficientnet_b4_w8a16_sefp/efficientnet_b4_w8a16_sefp_XH2a.onnx --input-size 380 --data-dir /data01/datasets/imagenet/val
python examples/cv/efficientnet/scripts/efficientnet_eval.py --model-type hmonnx --model-path work_dirs/efficientnet/efficientnet_b4_w4a8_ssfp/efficientnet_b4_w4a8_ssfp_XH2a.onnx --input-size 380 --data-dir /data01/datasets/imagenet/val

python examples/cv/efficientnet/scripts/efficientnet_eval.py --model-type hmonnx --model-path work_dirs/efficientnet/efficientnet_v2_m_w8a8_sefp/efficientnet_v2_m_w8a8_sefp_XH2a.onnx --input-size 480 --data-dir /data01/datasets/imagenet/val
python examples/cv/efficientnet/scripts/efficientnet_eval.py --model-type hmonnx --model-path work_dirs/efficientnet/efficientnet_v2_m_w8a16_sefp/efficientnet_v2_m_w8a16_sefp_XH2a.onnx --input-size 480 --data-dir /data01/datasets/imagenet/val
python examples/cv/efficientnet/scripts/efficientnet_eval.py --model-type hmonnx --model-path work_dirs/efficientnet/efficientnet_v2_m_w4a8_ssfp/efficientnet_v2_m_w4a8_ssfp_XH2a.onnx --input-size 480 --data-dir /data01/datasets/imagenet/val
```
