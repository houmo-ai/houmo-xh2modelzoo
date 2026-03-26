# Qwen3.5 XH2a 导出与推理

Qwen3.5 目录当前已经支持两条链路：

1. 纯 LLM 的 XH2a 量化导出、Golden 生成与 HMONNX 推理
2. Vision 编码器的 ONNX/HMONNX 导出，以及 Vision HMONNX + LLM HMONNX 的 VL 联合推理

## 环境依赖

推荐使用 `xhquant` 环境，并在仓库根目录执行：

```bash
conda activate xhquant
export PYTHONPATH=./
```

## 支持情况

| 能力 | 状态 | 说明 |
|------|------|------|
| LLM 导出 | 已支持 | 支持 Qwen3.5 LLM 的 ONNX/HMONNX 导出与 Golden 生成 |
| Vision 导出 | 已支持 | 支持 Qwen3.5 vision encoder 的 ONNX/HMONNX 导出 |
| VL 联合推理 Demo | 已支持 | 支持 Vision HMONNX + LLM HMONNX 联合推理 |

## LLM 导出

### BF16 到 XH2a

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
python \
  examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --config configs/qwen3_5/qwen3_5_27b_xh2a.py \
  --hf_model_dir weights/Qwen3.5-27B \
  --dtype fp16 \
  --valid \
  --golden \
  --work_dir work_dirs/qwen3_5_27b_bf16_export
```

### GPTQ 到 XH2a

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
python \
  examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --config configs/qwen3_5/qwen3_5_27b_xh2a.py \
  --hf_model_dir weights/Qwen3.5-27B-GPTQModel-self-generated-4bit-hessian-mse \
  --dtype fp16 \
  --valid \
  --golden \
  --work_dir work_dirs/qwen3_5_27b_gptq_export
```

## Vision 导出

Qwen3.5 Vision 当前使用独立脚本导出 vision encoder。下面这条命令已经在当前仓真实跑通。

### 9B Vision HMONNX 导出

```bash
export PYTHONPATH=./
export LD_LIBRARY_PATH=:/data01/home/chenzx/project/houmoquantool/hmquant/ops/build/lib.linux-x86_64-cpython-38/:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
python \
  examples/llm/qwen3_5/qwen3_5_vision_xh2a_export_hmonnx.py \
  --config configs/qwen3_5/qwen3_5_instruct_vision_config.py \
  --hf_model_dir /data02/datasets/Qwen3.5-9B-rotated-fp/ \
  --model_type 9B
```

默认产物位置：

```bash
work_dirs/qwen3_5_9B/qwen3_5_instruct_vision_config_1_2_448_448_use_gptq_model_False_Qwen3/
```

关键产物：

- `onnx/visual_1.onnx`
- `vision/qwen3_5_instruct_vision_config.onnx`
- `vision/qwen3_5_9B_vision_448_448`

## LLM HMONNX 推理

### 单轮文本 Demo

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
python \
  examples/llm/qwen3_5/qwen3_5_xh2a_demo.py \
  --config work_dirs/qwen3_5_27b_bf16_export/meta.json \
  --prompt "你好呀，你是谁，中文回答" \
  --max-new-tokens 256 \
  --dtype fp16
```

### Benchmark 测试

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
python \
  examples/llm/qwen3_5/qwen3_5_xh2a_hmonnx_test.py \
  --config work_dirs/qwen3_5_27b_bf16_export/meta.json \
  --prompt "请用中文简要介绍一下混合线性注意力模型。" \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --benchmark-runs 3 \
  --dtype fp16
```

## VL Demo 使用方式

`qwen3_5_vl_hmonnx_demo.py` 用于把 Vision HMONNX 与 LLM HMONNX 串起来做图文联合推理。

当前默认组合为：

- Vision HMONNX：当前仓导出的 qwen3.5 9B vision 模型
- LLM HMONNX：`/data01/home/chenzx/project/xhquant_llm/work_dirs/qwen3_5/hmquant_xh2_qwen3.5_9b_w4_a8_256_2k_448_20260324_Qwen3.5-9B-quarot-gptq-4bit-mse24-hessian_20260324_103723/export_meta_info.json`

### 直接使用默认参数运行

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
python \
  examples/llm/qwen3_5/qwen3_5_vl_hmonnx_demo.py
```

### 指定图片和问题

```bash
export PYTHONPATH=./
export LD_LIBRARY_PATH=:/data01/home/chenzx/project/houmoquantool/hmquant/ops/build/lib.linux-x86_64-cpython-38/:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
python \
  examples/llm/qwen3_5/qwen3_5_vl_hmonnx_demo.py \
  --image-path data/images/qwen2_vl_demo.jpeg \
  --prompt "描述这张照片" \
  --max-new-tokens 64
```

### 显式指定 Vision 和 LLM 产物

```bash
export PYTHONPATH=./
export LD_LIBRARY_PATH=:/data01/home/chenzx/project/houmoquantool/hmquant/ops/build/lib.linux-x86_64-cpython-38/:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
python \
  examples/llm/qwen3_5/qwen3_5_vl_hmonnx_demo.py \
  --model-config /data01/home/chenzx/project/xhquant_llm/work_dirs/qwen3_5/hmquant_xh2_qwen3.5_9b_w4_a8_256_2k_448_20260324_Qwen3.5-9B-quarot-gptq-4bit-mse24-hessian_20260324_103723/export_meta_info.json \
  --vision-onnx /data01/home/chenzx/project/xh2modelzoo/work_dirs/qwen3_5_9B/qwen3_5_instruct_vision_config_1_2_448_448_use_gptq_model_False_Qwen3/vision/qwen3_5_instruct_vision_config.onnx \
  --image-path data/images/qwen2_vl_demo.jpeg \
  --prompt "描述这张照片"
```

## 使GPTQModel用优化量化精度

如果希望使用 GPTQModel 对 Qwen3.5 做更高精度的量化，并且让当前仓的 Vision 导出和 LLM 量化保持同一套旋转逻辑，推荐先在 GPTQModel 仓中完成旋转与量化，再回到当前仓做导出与推理。

建议先确保 GPTQModel 已经拉取到 `aeb3864e` 之后的代码，再执行下面流程。

### 第一步：先对 Qwen3.5 VL 做旋转，给 Vision 导出准备对齐后的 FP 模型

这一步的目的，是把 Qwen3.5 的 LLM 旋转逻辑和 Vision 侧输出投影对齐，得到可直接用于当前仓 Vision 导出的旋转后 FP 模型。

推荐命令：

```bash
cd gptqmodel

export CUDA_VISIBLE_DEVICES=0
python examples/quantization/examples/example_qwen35_vl_rotate_fp.py \
  --model /data02/datasets/Qwen3.5-9B \
  --out /data02/datasets/Qwen3.5-9B-rotated-fp \
  --llm-rotation hadamard \
  --vision-rotation last \
  --device cuda:0 \
  --validate
```

说明：

1. `--llm-rotation hadamard` 指定 LLM 旋转模式，后续 LLM 量化建议保持同样的旋转配置。
2. `--vision-rotation last` 做最后输出投影对齐，也可以改为 `full`，表示整个vision都做对应的旋转。
3. 这一步输出的 `/data02/datasets/Qwen3.5-9B-rotated-fp`，就是当前仓 Vision 导出脚本可直接使用的 `--hf_model_dir`。

完成后，可以在当前仓直接执行：

```bash
cd xh2modelzoo

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5/qwen3_5_vision_xh2a_export_hmonnx.py \
  --config configs/qwen3_5/qwen3_5_instruct_vision_config.py \
  --hf_model_dir /data02/datasets/Qwen3.5-9B-rotated-fp/ \
  --model_type 9B
```

### 第二步：使用 GPTQModel 对 Qwen3.5 Dense LLM 做 rotate + GPTQ

`example_qwen35dense.py` 支持在量化前先做旋转，再执行 GPTQ 量化。为了和上面的 Vision 旋转链路保持一致，建议 LLM 继续使用相同的旋转模式，比如 `hadamard`。

默认校准集可以直接使用 wikitext2，也就是 wiki，不需要额外准备数据；如果你有业务相关的校准集，也可以通过 `--calibration-jsonl` 替换。

#### 使用默认 wiki 校准集

```bash
cd ./gptqmodel

export CUDA_VISIBLE_DEVICES=0
python examples/quantization/examples/example_qwen35dense.py \
  --model /data02/datasets/Qwen3.5-9B \
  --out /data02/datasets/Qwen3.5-9B-quarot-gptq-4bit-mse24-hessian \
  --bits 4 \
  --group-size 64 \
  --rotation hadamard \
  --nsamples 256 \
  --seqlen 1024 \
  --mse 2.4 \
  --hessian-mse \
  --device cuda:0
```

说明：

1. 脚本默认使用 `wikitext-2-raw-v1` 作为校准集来源，可以把它理解成默认 wiki 校准方案。
2. `--rotation hadamard` 表示量化前先做 QuaRot 旋转。
3. `--hessian-mse` 和 `--mse 2.4` 用于量化误差优化，是当前 Qwen3.5 GPTQ 常用配置。

#### 使用自定义 JSONL 校准集

如果需要替换成自己的校准集，可以准备 JSONL 文件，每行一个 `{\"text\": ...}` 记录，再执行：

```bash
cd gptqmodel

export CUDA_VISIBLE_DEVICES=0
python examples/quantization/examples/example_qwen35dense.py \
  --model /data02/datasets/Qwen3.5-9B \
  --out /data02/datasets/Qwen3.5-9B-quarot-gptq-4bit-mse24-hessian \
  --bits 4 \
  --group-size 64 \
  --rotation hadamard \
  --nsamples 256 \
  --mse 2.4 \
  --hessian-mse \
  --calibration-jsonl /path/to/calibration.jsonl \
  --calibration-text-key text \
  --device cuda:0
```

校准集建议：

1. 没有特别需求时，默认直接使用 wiki 即可。
2. 如果业务场景比较明确，可以优先使用和业务分布接近的中英文混合文本、对话或代码数据。
3. 常用起点是 `nsamples=256`、`seqlen=1024`，如果资源允许可以继续增加。

### 推荐的整体使用顺序

如果目标是让当前仓的 Vision 导出与 LLM GPTQ 结果尽量保持同一套旋转思路，建议按下面顺序操作：

1. 在 GPTQModel 中先运行 `example_qwen35_vl_rotate_fp.py`，得到旋转后的 FP 模型目录。
2. 在当前仓中使用这个旋转后的 FP 模型目录执行 Vision 导出。
3. 在 GPTQModel 中使用 `example_qwen35dense.py` 做 LLM 的 `rotate + GPTQ` 量化。
4. 再把生成出的 LLM `export_meta_info.json` 和当前仓导出的 Vision HMONNX 一起喂给 `qwen3_5_vl_hmonnx_demo.py` 做联合推理。

### 额外提醒

1. `example_qwen35_vl_rotate_fp.py` 和 `example_qwen35dense.py` 最好使用一致的旋转模式，通常推荐都用 `hadamard`。
2. 如果 Vision 侧使用的是旋转后的 FP 模型，而 LLM 侧使用的是 GPTQModel 产出的量化模型，至少要保证两边来自同一个 Qwen3.5 基座，并且旋转逻辑一致。
3. 在使用 GPTQModel 前，建议先确认仓库代码已经包含 `aeb3864e` 之后的修复，否则 Qwen3.5 相关旋转与量化功能可能不完整。


## 注意事项

1. Qwen3.5 当前已经支持 LLM 与 Vision 两部分的独立导出，也已经支持 VL 联合推理 demo。
2. VL demo 依赖两套产物同时存在：一套是 LLM 的 `export_meta_info.json`，另一套是 Vision 的 `.onnx` HMONNX 文件，二者需要来自同一模型家族并保持 tokenizer / processor 对齐。
3. Qwen3.5 使用 M-RoPE，VL 场景下会同时使用 time、height、width 三组位置编码；demo 中已经按 vision token 布局生成对应位置索引。
4. 运行 Vision 导出或 VL demo 时，建议始终显式设置 `PYTHONPATH=./`；如果缺少 hmquant 动态库路径，`xhquant` GPU 扩展可能加载失败。
5. Vision 导出当前默认使用 `448 x 448` 输入分辨率、`patch_size=16`、`temporal_patch_size=2`，如果修改这些参数，Vision 产物与 VL demo 的输入配置也要保持一致。
6. LLM 的 Prefill 和 Decode 是两张独立图，Vision HMONNX 只负责生成图像特征，最终由 VL demo 将图像特征散射回文本 token embedding 后再调用 LLM HMONNX。