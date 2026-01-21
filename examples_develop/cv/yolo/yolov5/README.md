# yolov5m官方模型

## 导出标准onnx

```bash
bash examples/cv/yolo/yolov5/generate_yolov5_onnx.sh
```

## 导出HMONNX

```bash
python examples/cv/yolo/yolov5/yolov5_export.py --onnx data/models/yolo/yolov5m.onnx
```

## GPU仿真

```bash
python examples/cv/yolo/yolov5/yolov5_hmonnx_test.py --hmonnx work_dirs/yolov5m/hmonnx/yolov5m_XH2a.onnx --image data/images/000000001490.jpg
```

## 测试原浮点模型精度

``` bash
python examples/cv/yolo/yolov5/yolov5_eval.py --model data/models/yolo/yolov5m.onnx --model-type onnx
```

## 测试量化后模型

``` bash
python examples/cv/yolo/yolov5/yolov5_eval.py --model work_dirs/yolov5m/hmonnx/yolov5m_XH2a.onnx --model-type hmonnx
```