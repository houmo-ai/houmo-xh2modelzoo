# Wenet

## encoder

### 有最大长度限制的定长模型

#### 下载模型

```bash
bash examples/audio/wenet/fix_length/download.sh
```

#### 导出HMONNX

```bash
python examples/audio/wenet/fix_length/wenet_export.py --onnx data/models/wenet/encoder.onnx 
```

#### GPU仿真

```bash
python examples/audio/wenet/fix_length/wenet_hmonnx_test.py --hmonnx data/models/wenet/encoder.onnx
```
