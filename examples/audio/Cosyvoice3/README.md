# cosyvoice3-0.5B模型

导出该模型需要同时使用xhquanttool环境。

模型主体可分为五部分：campplus, speech_tokenizer_v3, llm, flow_decoder, hift；除llm部分之外，其余模块均通过onnx可直接导出。

## 配置参数

可修改对应config文件中的参数。

```llm
context-length
input-sequence-length
quant-type: for example: w8a8h1-sefp
```

## ONNX模型预处理以及导出HMONNX

替换相应的模型路径即可。

### campplus

```bash
python campplus_export_hmonnx.py
```

### speech_tokenizer_v3

```bash
python speech_tokenizer_v3_convert.py
python speech_tokenizer_v3_export_hmonnx.py
```
### llm(qwen2 0.5B)

```bash
python qwen2_xh2a_export_0.5B.py
```

### flow decoder 

```bash
python decoder_export_hmonnx.py
```

### hift

```bash
python hift_export_hmonnx.py
```

### other module

```bash
python other_export_hmonnx.py
```

## demo运行

```bash
python demo_new.py
```

## cv3_eval 评估

```bash
python cv3_eval.py
```

