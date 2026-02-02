# PI05-libero-fintune模型

导出该模型需要同时使用xhquanttool和lerobot环境。lerobot环境配置可见 https://github.com/huggingface/lerobot;

![PI05 模型结构](/data01/home/she.gao/xh2modelzoo/examples/vla/PI05_LIBERO.png)

模型主体可分为三部分：vision , gemma-2B, gemma-300m

## 配置参数

可修改对应config文件中的参数。

```gemma-2B
context-length
input-sequence-length
quant-type: for example: w8a8h1-sefp
```

```gemma-300m
context-length 
input-sequence-length
quant-type: for example: w8a8h1-sefp
```

## 导出HMONNX

### w8a8

#### 1. 导出 vision 部分hmonnx

替换相应的模型路径即可。

```bash
python pi05_export_vision_xh2a_libero.py --model_path models--lerobot--pi05_libero_finetuned
```

#### 2. 导出 gemma2B 部分hmonnx

```bash
export PYTHONPATH=/data01/home/she.gao/xh2modelzoo:$PYTHONPATH
python pi05_export_llm_xh2a_libero.py
```

#### 3. 导出 gemma300m 部分hmonnx

```bash
python pi05_export_experts_xh2a_libero.py
```

