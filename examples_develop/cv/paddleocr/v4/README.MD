# PaddleOCR-V4

## Detect

### 下载模型

```bash
bash examples/cv/paddleocr/v4/download.sh
```

#### 导出HMONNX

```bash
python examples/cv/paddleocr/v4/export_det_hmonnx.py
python examples/cv/paddleocr/v4/export_rec_hmonnx.py
python examples/cv/paddleocr/v4/export_cls_hmonnx.py
```

#### GPU仿真

```bash
python examples/audio/wenet/fix_length/wenet_hmonnx_test.py --hmonnx data/models/wenet/encoder.onnx
```
