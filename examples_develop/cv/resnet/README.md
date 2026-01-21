# ResNet

## 下载模型

```bash
bash examples/cv/resnet/download.sh
```

## 导出HMONNX

```bash
python examples/cv/resnet/resnet50_export.py --onnx data/models/resnet/resnet50_224x224.onnx --batch-size 1
```

## GPU仿真

```bash
python examples/cv/resnet/resnet50_hmonnx_test.py --hmonnx work_dirs/resnet50_224x224_batch_1/hmonnx/resnet50_224x224_batch_1_XH2a.onnx --image data/images/ILSVRC2012_val_00002031.JPEG  
```

## Imagenet 评测

### 1. 导出多batch的hmonnx模型

```bash
导出多batch的hmonnx模型
python examples/cv/resnet/resnet50_export.py --onnx data/models/resnet/resnet50_224x224.onnx --batch-size 16
```

### 2. 评测hmonnx模型

```bash
python examples/cv/resnet/resnet50_eval.py --hmonnx work_dirs/resnet50_224x224_batch_16/hmonnx/resnet50_224x224_batch_16_XH2a.onnx --dataset configs/datasets/imagenet_224x224.py --batch-size 16
```
