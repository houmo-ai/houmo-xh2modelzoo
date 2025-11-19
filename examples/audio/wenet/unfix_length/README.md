# Wenet

## encoder

### 有最大长度限制的定长模型

#### 下载模型

```bash
bash examples/audio/wenet/fix_length/download.sh
```

#### 导出HMONNX

```bash
python examples/audio/wenet/unfix_lengh/wenet_export.py --onnx data/models/wenet/chunk_encoder_v2.onnx
```