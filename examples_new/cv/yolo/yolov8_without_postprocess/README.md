# YOLOv8m

## 下载模型

```bash
bash examples/cv/yolo/yolov8_without_postprocess/download.sh
```

## 导出HMONNX

```bash
python examples/cv/yolo/yolov8_without_postprocess/yolov8m_without_postprocess_export.py --onnx data/models/yolo/yolov8m_without_postprocess.onnx    
```

## GPU仿真

```bash
python examples/cv/yolo/yolov8_without_postprocess/yolov8m_without_postprocess_hmonnx_test.py --hmonnx work_dirs/yolov8m_without_postprocess/hmonnx/yolov8m_without_postprocess_XH2a.onnx --image data/images/000000001490.jpg
```
