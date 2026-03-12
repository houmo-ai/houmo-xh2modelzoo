# OpenVLA-7b-libero-fintune模型

导出该模型需要同时使用xhquanttool环境。

模型主体可分为两部分：vision + llm(llama2-7b)

## 配置参数

可修改对应config文件中的参数。

## 导出HMONNX

### w8a8 sefp

#### 1. 导出 vision 部分hmonnx

替换相应的模型路径即可。

```bash
python openvla_export_vision_xh2a_libero.py
```

#### 2. 导出 llm 部分hmonnx

```bash
python openvla_export_llm_xh2a_libero.py
```


