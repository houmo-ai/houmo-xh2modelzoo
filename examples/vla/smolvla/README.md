# SmolVLA XH2 导出说明

本文给出 SmolVLA 在 XH2 上的最小可用导出流程，重点是参数清晰、替换点明确。

当前支持 4 个导出脚本：

- `smolvla_export_vision_xh2a.py`
- `smolvla_export_llm_xh2a.py`
- `smolvla_export_llm_kvcache_xh2a.py`
- `smolvla_export_action_xh2a.py`

推荐用于 LeRobot 推理替换的链路是：

- `vision + prefill + action`

`decode` 主要用于分段排查或消融，不是该链路的必选项。

## 环境准备

需要满足：

1. `xh2modelzoo` 与 `xhquant` 环境可用。
2. `lerobot/src` 可导入。
3. 安装导出依赖：

```bash
pip install onnx onnxsim
```

默认目录：

- `xh2modelzoo`: `/data01/home/chenzx/project/xh2_release/xh2modelzoo`
- `lerobot`: `/data01/home/chenzx/project/xh2_release/lerobot`

若不在默认位置，请显式传 `--lerobot_src`。

## 一、导出 vision

```bash
python examples/vla/smolvla/smolvla_export_vision_xh2a.py \
  --model_path /data02/datasets/smolvla_libero \
  --lerobot_src /data01/home/chenzx/project/xh2_release/lerobot/src \
  --device cpu \
  --image_height 512 \
  --image_width 512 \
  --output_name smolvla_vision \
  --quant_type w8a8h1_sefp
```

关键参数：

- `--model_path`: 必填，SmolVLA policy 路径或可解析模型路径。
- `--lerobot_src`: 建议显式传。
- `--image_height/--image_width`: 与运行时输入分辨率对齐。
- `--no_normalize_input`: 仅当上游已完成 `[0,1] -> [-1,1]` 时使用。

产物：

- `work_dirs/smolvla_vision/smolvla_vision_hmonnx/smolvla_vision_xh2.onnx`

## 二、导出 llm kvcache（prefill/decode）

```bash
python examples/vla/smolvla/smolvla_export_llm_kvcache_xh2a.py \
  --model_path /data02/datasets/smolvla_libero \
  --lerobot_src /data01/home/chenzx/project/xh2_release/lerobot/src \
  --device cpu \
  --prefix_length 256 \
  --suffix_length 50 \
  --output_name smolvla_llm \
  --quant_type w8a8h1_sefp
```

关键参数：

- `--prefix_length`: 必须与后续 action 导出/运行一致。
- `--suffix_length`: 一般对齐 `chunk_size`（默认 50）。
- `--quant_type`: 量化类型。

产物：

- `work_dirs/smolvla_llm_kvcache/prefill_hmonnx/smolvla_llm_prefill_xh2.onnx`
- `work_dirs/smolvla_llm_kvcache/decode_hmonnx/smolvla_llm_decode_xh2.onnx`
- `work_dirs/smolvla_llm_kvcache/meta_info.json`

说明：

- `prefill` 对应 `sample_actions()` 中 `fill_kv_cache=True` 路径。
- `decode` 对应 `denoise_step()` 中 `fill_kv_cache=False` 路径。

## 三、导出 action

```bash
python examples/vla/smolvla/smolvla_export_action_xh2a.py \
  --model_path /data02/datasets/smolvla_libero \
  --lerobot_src /data01/home/chenzx/project/xh2_release/lerobot/src \
  --device cpu \
  --prefix_length 256 \
  --prefill_meta work_dirs/smolvla_llm_kvcache/meta_info.json \
  --output_name smolvla_action \
  --quant_type w8a8h1_sefp
```

关键参数：

- `--prefix_length`: 必须与 prefill 完全一致。
- `--prefill_meta`: 推荐显式传，避免 prefix 推断错误。

产物：

- `work_dirs/smolvla_action/smolvla_action_hmonnx/smolvla_action_xh2.onnx`
- `work_dirs/smolvla_action/meta_info.json`

## 推荐替换方式（LeRobot）

对齐 `smolvla_eval.py` 的默认非 RTC 推理，推荐替换：

1. `vision`：替换 `embed_prefix()` 里的 `embed_image`。
2. `prefill`：替换 `sample_actions()` 里的 cache 生成。
3. `action`：替换 `denoise_step()`（直接输出 `v_t`）。

即主链为：`vision + prefill + action`。

## decode 什么时候用

`decode_hmonnx` 主要用于：

- 分段消融：`prefill + decode + PyTorch action_out_proj`
- 精度定位：确认误差来自 decode 还是 action 头

如果已经使用 `action_hmonnx`，通常不再需要在主链单独跑 `decode_hmonnx`。

## 常见对齐问题

1. 形状不匹配：

- `prefill` 导出常为静态 `prefix_length=256`，运行时需要补齐到 256。

2. dtype 不匹配：

- `prefill/decode` 的 `attention_mask` 和 `position_ids` 常要求 `int32`。

3. 归一化重复：

- 若 vision 导出未加 `--no_normalize_input`，运行时避免再次做同方向归一化。

## 可选：单独导出 llm text backbone

`smolvla_export_llm_xh2a.py` 用于导出纯 `text_model` 特征提取，不含 kv cache。
该脚本不用于替换 `sample_actions()/denoise_step()` 主路径。