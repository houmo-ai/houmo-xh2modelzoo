# OpenVLA-OFT模型

导出该模型需要同时使用xhquanttool环境。

模型主体可分为两部分：vision + llm(llama2-7b) + action head

## 配置参数

可修改对应config文件中的参数。

## 导出HMONNX

### w8a8 sefp

#### 1. 导出 vision 部分hmonnx

替换相应的模型路径即可。

```bash
python openvla_oft_vit_export_hmonnx.py
```

#### 2. 导出 llm 部分hmonnx

```bash
python openvla_oft_llm_export_hmonnx.py
```

#### 3. 导出 action head 部分hmonnx

具体可见[openvla_oft_action_head_export_hmonnx.py](openvla_oft_action_head_export_hmonnx.py)
