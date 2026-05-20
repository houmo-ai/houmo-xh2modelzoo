# Qwen3-VL HMONNX 图运行时设计说明

`examples/llm/qwen3vl/qwen3_vl_xh2a_multi_image_demo.py` 已统一为单一脚本，基于
`HMONNXGraphInference`（可选 `HMONNXCUDAGraphInference`）实现 vision / prefill /
decode 三阶段图运行时，并通过 `transformers` 的 `TextStreamer` 做流式输出。

要点：

- 底层模型实现位于 `xh_model_zoo/xh_llm/models/qwen3_vl/qwen3_vl_onnx_model.py`
  （`Qwen3VLONNXModel` / `Qwen3VLProcessor`）。
- 后处理风格参考 `xh_model_zoo/xh_llm/models/qwen3_5/qwen3_5_onnx_model.py`，
  统一 token 采样、停止条件与 KV cache 管理。
- 视觉特征 + deepstack token 排布严格对齐 transformers 原生实现，便于与 HF
  推理结果对比。
- 通过 `--enable-cuda-graph --cuda-graph-modules prefill decode vision` 开启
  CUDA Graph 捕获以进一步提升吞吐。

CLI 用法、输入模式（`--image-paths` / `--image-dir` / `--task-json`）、提示词
选项与多场景示例统一记录在 `examples/llm/qwen3vl/README.MD` 的"多图推理
（HMONNX 图运行时 Demo）"一节。