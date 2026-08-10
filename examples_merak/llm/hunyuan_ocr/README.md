# HunyuanOCR HMONNX 导出与推理

本目录提供 HunyuanOCR 的 XH2a HMONNX 导出、单图自回归推理和 DFlash 推测解码入口。以下命令均从仓库根目录执行。

## 环境

复用 xh2modelzoo 已安装的 Python 环境，不新增 HunyuanOCR 专用 requirements 文件。当前验证环境为：

- Python 3.12.11
- PyTorch 2.8.0+cu128
- Transformers 5.13.0
- xhquanttool 0.9.2
- safetensors 0.8.0

xhquanttool 必须与当前 xh2modelzoo 基线配套，并提供 modelzoo 公共 HMONNX 层导入的 runtime API。仅比较 `xhquanttool.__version__` 不足以证明兼容性；运行导出或推理前执行：

已验证的 QTL-428 配套源码安装方式如下。也可以使用包含同一 API 的更新 wheel 或 commit：

```bash
export XHQUANTTOOL_REPO=/path/to/xhquanttool
export XHQUANTTOOL_COMMIT=caac85f403f04216c969bc65bc446a3b0be8a7b9

CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" \
python -m pip install \
  --force-reinstall \
  --no-deps \
  --no-build-isolation \
  "git+file://$XHQUANTTOOL_REPO@$XHQUANTTOOL_COMMIT"
```

`--no-build-isolation` 用于复用当前环境中已安装的 PyTorch；xhquanttool 的构建入口在准备 wheel metadata 时需要导入 `torch`。

安装后执行：

```bash
python -c "from xhmodel_merak.xh_llm.hmonnx import AutoLLMHONNXModel; print(AutoLLMHONNXModel)"
```

若这里出现 `ModuleNotFoundError`，请先安装与当前 xh2modelzoo commit 配套的 xhquanttool build。不要通过修改 HunyuanOCR metadata 或跳过 DFlash capability gate 绕过依赖不匹配。

仓库可编辑安装方式：

```bash
python -m pip install -v -e . --no-index --no-deps --no-build-isolation
```

准备路径变量：

```bash
export MODEL_DIR=/path/to/HunyuanOCR
export IMAGE_PATH=/path/to/document.png
export WORK_DIR=$PWD/work_dirs/hunyuan_ocr
mkdir -p "$WORK_DIR"
```

普通导出要求 `$MODEL_DIR/config.json` 存在。DFlash 还要求 `$MODEL_DIR/dflash/config.json` 及对应 checkpoint 文件存在。

共享机器上先用 `nvidia-smi` 选择空闲物理 GPU，再通过 `CUDA_VISIBLE_DEVICES` 隔离。例如选择物理 GPU 4 时：

```bash
export CUDA_VISIBLE_DEVICES=4
```

此时进程内该卡会映射为 `cuda:0`，所以后续推理命令仍使用 `--device cuda:0`。

## Workflow 配置

| 配置 | 用途 |
| --- | --- |
| `configs_merak/workflows/xh2a/llm_models/hunyuan_ocr/base/hunyuan_ocr_base_xh2a_w16a16.yaml` | W16A16 普通 HMONNX bundle |
| `configs_merak/workflows/xh2a/llm_models/hunyuan_ocr/base/hunyuan_ocr_base_xh2a_w8a8.yaml` | W8A8 校准量化 bundle |
| `configs_merak/workflows/xh2a/llm_models/hunyuan_ocr/base/hunyuan_ocr_base_xh2a_w16a16_dflash.yaml` | W16A16 DFlash target/verify bundle |

默认 resolution bucket manifest 为：

```text
examples_merak/llm/hunyuan_ocr/assets/hunyuan_ocr_resolution_buckets.json
```

输入图片会路由到 manifest 中批准的静态分辨率 bucket。W8A8 workflow 还会读取 YAML 中声明的校准图片和参考轨迹，运行前需确认这些路径在当前环境中可用。

## 普通 HMONNX 导出

导出包含 visual、text prefill 和 text decode graph 的 W16A16 bundle：

```bash
export AR_BUNDLE=$WORK_DIR/hmonnx_w16a16

python examples_merak/llm/hunyuan_ocr/export_hmonnx.py \
  --model "$MODEL_DIR" \
  --config-path configs_merak/workflows/xh2a/llm_models/hunyuan_ocr/base/hunyuan_ocr_base_xh2a_w16a16.yaml \
  --output-dir "$AR_BUNDLE"
```

导出 W8A8 bundle：

```bash
export W8_BUNDLE=$WORK_DIR/hmonnx_w8a8

python examples_merak/llm/hunyuan_ocr/export_hmonnx.py \
  --model "$MODEL_DIR" \
  --config-path configs_merak/workflows/xh2a/llm_models/hunyuan_ocr/base/hunyuan_ocr_base_xh2a_w8a8.yaml \
  --output-dir "$W8_BUNDLE"
```

若只需要固定 text prefill/decode stage，可使用：

```bash
python examples_merak/llm/hunyuan_ocr/export_text_hmonnx.py \
  --model "$MODEL_DIR" \
  --config-path configs_merak/workflows/xh2a/llm_models/hunyuan_ocr/base/hunyuan_ocr_base_xh2a_w16a16.yaml \
  --output-dir "$WORK_DIR/text_w16a16"
```

导出脚本拒绝覆盖非空目录。完整 bundle 的 runtime metadata 位于 `$AR_BUNDLE/golden_meta_info.json`。

## 普通自回归推理

默认使用 HMONNXInferenceV2，在 GPU 0 对单张图片执行 OCR：

```bash
python examples_merak/llm/hunyuan_ocr/hunyuan_ocr_xh_hmonnx_generate.py \
  --hmonnx-config "$AR_BUNDLE/golden_meta_info.json" \
  --image "$IMAGE_PATH" \
  --device cuda:0 \
  --max-new-tokens 512 \
  --output "$WORK_DIR/ar_result.json"
```

默认会逐 token 输出。需要只打印最终文本时增加 `--no-stream`。`--legacy-runtime` 可切换到已弃用的 legacy HMONNX runtime，仅用于兼容性排查。

## DFlash 导出

DFlash bundle 必须按 target/verify、draft graphs 的顺序导出。第一步使用 DFlash workflow 生成 visual、target prefill/decode、target verify 和 schema v2 metadata：

```bash
export DFLASH_BUNDLE=$WORK_DIR/hmonnx_w16a16_dflash

python examples_merak/llm/hunyuan_ocr/export_hmonnx.py \
  --model "$MODEL_DIR" \
  --config-path configs_merak/workflows/xh2a/llm_models/hunyuan_ocr/base/hunyuan_ocr_base_xh2a_w16a16_dflash.yaml \
  --output-dir "$DFLASH_BUNDLE"
```

第二步在同一 artifact root 中追加 draft context、context_decode 和 decode graph，并原子升级 metadata：

```bash
python examples_merak/llm/hunyuan_ocr/export_dflash_draft_hmonnx.py \
  --metadata "$DFLASH_BUNDLE/golden_meta_info.json" \
  --target-model-dir "$MODEL_DIR" \
  --draft-model-dir "$MODEL_DIR/dflash" \
  --output-dir "$DFLASH_BUNDLE"
```

`--output-dir` 必须与 metadata 的父目录相同。成功后 metadata 的 DFlash 状态应为 `speculative_runtime_ready`，且 `target_hidden`、`target_verify`、`draft_graphs`、`speculative_runtime` capability 均为 `true`。

## DFlash 推理

DFlash 当前只支持非 streaming、确定性单序列生成，因此必须传入 `--no-stream`：

```bash
python examples_merak/llm/hunyuan_ocr/hunyuan_ocr_xh_hmonnx_generate.py \
  --hmonnx-config "$DFLASH_BUNDLE/golden_meta_info.json" \
  --image "$IMAGE_PATH" \
  --device cuda:0 \
  --max-new-tokens 512 \
  --dflash \
  --no-stream \
  --output "$WORK_DIR/dflash_result.json"
```

默认 draft token 数来自 bundle metadata。只有在 graph capacity 允许时才覆盖该值：

```bash
python examples_merak/llm/hunyuan_ocr/hunyuan_ocr_xh_hmonnx_generate.py \
  --hmonnx-config "$DFLASH_BUNDLE/golden_meta_info.json" \
  --image "$IMAGE_PATH" \
  --device cuda:0 \
  --max-new-tokens 512 \
  --dflash \
  --num-draft-tokens 15 \
  --no-stream
```

当前 speculative path 不接受 sampling、自定义 EOS、`min_new_tokens`、自定义 stopping criteria、多返回序列或 streamer。runtime 会对无法正确执行的组合直接报错，而不是静默忽略。

## CUDA Graph

`--cuda-graph` 会为兼容的 HMONNX session 请求 CUDA Graph capture。它是可选优化，不保证对所有分辨率 bucket、输出长度或 DFlash workload 加速；请在目标机器上分别对 warmup 后的普通 AR 和 DFlash 请求做延迟、吞吐和显存 benchmark，再决定是否启用。首次请求还会加载和解析 graph，不应直接作为稳态性能数据。

## 输出报告

`--output` 写入 JSON 报告，包括生成文本、token 数、耗时、runtime 名称、DFlash 开关和 request-local runtime summary。普通 AR 与 DFlash 的结果可据此比较；不要只用单次冷启动耗时判断性能。