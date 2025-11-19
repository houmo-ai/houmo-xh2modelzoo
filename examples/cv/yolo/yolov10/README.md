# yolov10官方模型

## 导出标准onnx

```bash
bash examples/cv/yolo/yolov10/generate_yolov10_onnx.sh
```

## 导出HMONNX

```bash
python examples/cv/yolo/yolov10/yolov10_export.py --onnx data/models/yolo/yolov10m.onnx
```

## GPU仿真

```bash
python examples/cv/yolo/yolov10/yolov10_hmonnx_test.py --hmonnx work_dirs/yolov10m/hmonnx/yolov10m_XH2a.onnx --image data/images/000000001490.jpg
```
