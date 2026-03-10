Qwen3-ForceAligner-0.6B

时间戳对齐模型，分类模型

1. 依赖包

``` bash
pip install qwen_asr
pip install transformers
```
2. 导出 HMONNX

    1. 导出方法：
        `python hmonnx_export_prefill_decode.py` 包含导出 encoder 阶段的代码。
        `python hmonnx_export_prefill_forcealigner.py` 包含导出 prefill 阶段的代码。
    2. demo：
        `python hmonnx_demo.py` 推理脚本。


3. 其他
    核心依赖文件：`xh2modelzoo/xh_model_zoo/xh_llm/models/qwen3_forcealigner`