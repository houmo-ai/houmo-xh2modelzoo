# yolov8m官方模型

## 导出标准onnx

```bash
bash examples/cv/yolo/yolov8/generate_yolov8_onnx.sh
```

## 导出HMONNX

```bash
python examples/cv/yolo/yolov8/yolov8_export.py --onnx data/models/yolo/yolov8m.onnx
```

## GPU仿真

```bash
python examples/cv/yolo/yolov8/yolov8_hmonnx_test.py --hmonnx work_dirs/yolov8m/hmonnx/yolov8m_XH2a.onnx --image data/images/000000001490.jpg
```
