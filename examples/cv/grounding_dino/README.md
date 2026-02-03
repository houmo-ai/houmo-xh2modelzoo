# Grounding-Dino

## 下载模型

```bash
wget http://10.10.1.53:8082/artifactory/model_zoo2/Grounding_Dino/groundingdino.onnx # 转换hmonnx用
wget http://10.10.1.53:8082/artifactory/model_zoo2/Grounding_Dino/groundingdino_XH2a.onnx # 直接用于推理的hmonnx
wget http://10.10.1.53:8082/artifactory/model_zoo2/Grounding_Dino/groundingdino_XH2a_external_data # 直接用于推理的hmonnx权重
```
输入 1200 x 800

这里的text长度=定长256，不足的会自动补全，后续的hmonnx配置中也会固定这个长度


## 导出HMONNX

```bash
python examples/cv/grounding_dino/export_hmonnx.py --onnx ./groundingdino.onnx
```

## HMONNX推理



```bash
python examples/cv/grounding_dino/infer_hmonnx.py --hmonnx /path/to/hmonnx --image_path /path/to/single_image_path --text_prompt dog --output_dir results --box_threshold 0.35
```


