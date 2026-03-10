Qwen3-ASR

- 两种参数：架构相同，hidden dim 不同。
    - 0.6B
    - 1.7B

1. 依赖包

``` bash
pip install qwen_asr
pip install onnx-ir==0.1.14
pip install onnx==1.16.2
pip install onnxscript==0.5.7
```
2. 导出 HMONNX

    1. 导出方法：
        `python hmonnx_export_prefill_decode.py` 包含导出 encoder 阶段的代码。
        `python hmonnx_export_prefill_decode.py` 包含导出 prefill 以及 decoder 阶段的代码，其中包含 kv cache 处理。
    2. demo：
        `python hmonnx_demo.py` 推理脚本。


3. 其他
    核心依赖文件：`xh2modelzoo/xh_model_zoo/xh_llm/models/qwen3_asr`