# yolov4m官方模型

## 导出标准onnx

```bash
bash examples/cv/yolo/yolov4/generate_yolov4_onnx.sh
```

## 导出HMONNX

```bash
python examples/cv/yolo/yolov4/yolov4_export.py --onnx data/models/yolo/yolov4m.onnx
```

## GPU仿真

```bash
python examples/cv/yolo/yolov4/yolov4_hmonnx_test.py --hmonnx work_dirs/yolov4m/hmonnx/yolov4m_w8a8-sefp_XH2a.onnx --image data/images/000000001490.jpg
```

## 测试原浮点模型精度

```bash
python examples/cv/yolo/yolov4/yolov4_eval.py --model data/models/yolo/yolov4m.onnx --model-type onnx
```

## 测试量化后模型精度

```bash
python examples/cv/yolo/yolov4/yolov4_eval.py --model work_dirs/yolov4m/hmonnx/yolov4m_w8a8-sefp_XH2a.onnx --model-type hmonnx
python examples/cv/yolo/yolov4/yolov4_eval.py --model work_dirs/yolov4m/hmonnx/yolov4m_w8a16-sefp_XH2a.onnx --model-type hmonnx
python examples/cv/yolo/yolov4/yolov4_eval.py --model work_dirs/yolov4m/hmonnx/yolov4m_w4a8-ssfp_XH2a.onnx --model-type hmonnx
```