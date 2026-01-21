# Openpose

## 导出标准onnx

```bash
bash examples/cv/openpose/generate_openpose_onnx.sh
```

## 导出HMONNX

```bash
python examples/cv/openpose/openpose_export.py --onnx data/models/openpose/openpose_body.onnx
```

## GPU仿真

```bash
python examples/cv/openpose/openpose_hmonnx_test.py --hmonnx work_dirs/openpose_body/hmonnx/openpose_body_XH2a.onnx --image examples/cv/openpose/demo.jpg
```

## 测试原浮点模型精度

``` bash
python examples/cv/openpose/openpose_eval.py --model-type onnx --model-path data/models/openpose/openpose_body.onnx
```

## 测试量化后模型

``` bash
python examples/cv/openpose/openpose_eval.py --model-type hmonnx --model-path work_dirs/openpose_body/hmonnx/openpose_body_XH2a.onnx
```