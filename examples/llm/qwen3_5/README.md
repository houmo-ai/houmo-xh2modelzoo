# Qwen3.5 XH2a 导出与推理

## MTP Head Pruning → see [MTP_HEAD_PRUNING.md](./MTP_HEAD_PRUNING.md)

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

### 长上下文导出说明

如果需要支持超过 `64K` 的上下文长度，需要在导出时额外增加 `--support_long_context_over_fp16_limit` 参数，例如：

```bash
python \
  examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  ... \
  --support_long_context_over_fp16_limit
```

说明：

1. 这个参数需要在导出阶段开启，生成的产物才会支持超过 `64K` 的上下文。
2. 开启后会带来一定性能下降，通常大约增加 `10ms` 左右的耗时。


## 六模型 8k Spec Decode 导出（命令版，不再使用 batch_export_8k.sh）

下面命令覆盖六个模型的 W4A8 GPTQ/quant-weight 8k context 导出，每个模型都有 MTP 和 DFlash 两种 spec mode。目录名包含模型、8k、W4A8、GPTQ、spec mode、draft tokens、DFlash input length、draft head weight bits 和时间戳，避免产物混淆。

推荐流程是 **target once, draft many**：

1. 每个模型、每种 spec mode 先导一次默认 W4 head 完整产物，得到 target prefill/decode、W4 draft 和 `meta.json`。
2. 如果要对比 W8 head，不再重导 target，直接用 `--draft_only/--draft-only` 复用第 1 步的 `--existing_work_dir/--existing-work-dir`，只导 W8 draft 并生成新的 `meta.json`。
3. 如果 W4/W8 效果差异可接受，后续正式产物使用默认 W4 head。

通用约定：

- 在仓库根目录运行：`export PYTHONPATH=./`，并用 `CUDA_VISIBLE_DEVICES=<gpu>` 绑定单张卡。
- MTP 默认 `--num_draft_tokens 4`；DFlash 默认 `--num_draft_tokens 9`，对应 draft decode input length 为 `10`。
- Dense 模型使用 `examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py`，GPTQ/quant weight 目录通过 `--hf_model_dir` 指定。
- 文中的 `weights/...` 和 `/data01/home/yujy/work/auto-round/output/...` 是当前批处理脚本使用的约定路径；本机目录不同则按实际位置替换。
- MoE 模型使用 `examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py`，FP base 通过 `--model` 指定，GPTQ/quant weight 通过 `--quant-weight` 指定，输出目录通过 `--work-dir` 指定。

### Dense 模型命令

先按下表选择模型变量：

| 模型 | `MODEL_KEY` | `CONFIG` | `HF_MODEL_DIR` | `DFLASH_MODEL_DIR` |
| --- | --- | --- | --- | --- |
| Qwen3.5 4B | `qwen3_5_4b` | `configs/qwen3_5/qwen3_5_4b_xh2a.py` | `/data01/home/yujy/work/auto-round/output/Qwen3.5-4B-mode1-llm-only` | `weights/Qwen3.5-4B-DFlash` |
| Qwen3.5 9B | `qwen3_5_9b` | `configs/qwen3_5/qwen3_5_9b_xh2a.py` | `/data01/home/yujy/work/auto-round/output/Qwen3.5-9B-mode1-llm-only` | `weights/Qwen3.5-9B-DFlash` |
| Qwen3.5 27B | `qwen3_5_27b` | `configs/qwen3_5/qwen3_5_27b_xh2a.py` | `/data01/home/yujy/work/auto-round/output/sym-mode1` | `weights/Qwen3.5-27B-DFlash` |
| Qwen3.6 27B | `qwen3_6_27b` | `configs/qwen3_5/qwen3_6_27b_xh2a.py` | `/data01/home/yujy/work/auto-round/output/Qwen3.6-27B-mode1-llm-only` | `weights/Qwen3.6-27B-DFlash` |

MTP：先导 W4 head 完整产物。

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_5_27b
CONFIG=configs/qwen3_5/qwen3_5_27b_xh2a.py
HF_MODEL_DIR=/data01/home/yujy/work/auto-round/output/sym-mode1
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8_gptq_spec_mtp_draft4_${TS}"
python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --config "$CONFIG" \
  --hf_model_dir "$HF_MODEL_DIR" \
  --dtype fp16 \
  --max_sequence_length 8192 \
  --spec_decode_mode mtp \
  --num_draft_tokens 4 \
  --work_dir "$WORK_DIR"

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_5_9b
CONFIG=configs/qwen3_5/qwen3_5_9b_xh2a.py
HF_MODEL_DIR=weights/Qwen3.5-9B
MTPK=81920
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8_gptq_spec_mtp_k${MTPK}_draft4_${TS}"
python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --config "$CONFIG" \
  --hf_model_dir "$HF_MODEL_DIR" \
  --dtype fp16 \
  --max_sequence_length 8192 \
  --spec_decode_mode mtp \
  --num_draft_tokens 4 \
  --work_dir "$WORK_DIR"

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=5
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_5_9b
CONFIG=configs/qwen3_5/qwen3_5_9b_xh2a.py
HF_MODEL_DIR=/data01/home/yujy/work/auto-round/output/Qwen3.5-9B-mode1-llm-only
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8_gptq_spec_mtp_draft4_${TS}"
python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --config "$CONFIG" \
  --hf_model_dir "$HF_MODEL_DIR" \
  --dtype fp16 \
  --max_sequence_length 8192 \
  --spec_decode_mode mtp \
  --num_draft_tokens 4 \
  --work_dir "$WORK_DIR" \
  --golden \
  --package_release \
  --release_xh_version xh2a \
  --support_long_context_over_fp16_limit

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=6
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_6_27b
CONFIG=configs/qwen3_5/qwen3_6_27b_xh2a.py
HF_MODEL_DIR=/data01/home/yujy/work/auto-round/output/Qwen3.6-27B-mode1-llm-only
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8_gptq_spec_mtp_draft4_${TS}"
python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --config "$CONFIG" \
  --hf_model_dir "$HF_MODEL_DIR" \
  --dtype fp16 \
  --max_sequence_length 8192 \
  --spec_decode_mode mtp \
  --num_draft_tokens 4 \
  --work_dir "$WORK_DIR" \
  --golden \
  --package_release \
  --release_xh_version xh2a \
  --support_long_context_over_fp16_limit

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=7
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_5_9b
CONFIG=configs/qwen3_5/qwen3_5_9b_xh2a.py
HF_MODEL_DIR=/data01/home/yujy/work/auto-round/output/Qwen3.5-9B-mode1-llm-only
MTPK=81920
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8_gptq_spec_mtp_k${MTPK}_draft4_${TS}"
python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --config "$CONFIG" \
  --hf_model_dir "$HF_MODEL_DIR" \
  --dtype fp16 \
  --max_sequence_length 8192 \
  --spec_decode_mode mtp \
  --num_draft_tokens 4 \
  --work_dir "$WORK_DIR" \
  --golden \
  --package_release \
  --release_xh_version xh2a \
  --support_long_context_over_fp16_limit

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=4
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_6_27b
CONFIG=configs/qwen3_5/qwen3_6_27b_xh2a.py
HF_MODEL_DIR=/data01/home/yujy/work/auto-round/output/Qwen3.6-27B-mode1-llm-only
MTPK=81920
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8_gptq_spec_mtp_k${MTPK}_draft4_${TS}"
python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --config "$CONFIG" \
  --hf_model_dir "$HF_MODEL_DIR" \
  --dtype fp16 \
  --max_sequence_length 8192 \
  --spec_decode_mode mtp \
  --num_draft_tokens 4 \
  --work_dir "$WORK_DIR" \
  --golden \
  --package_release \
  --release_xh_version xh2a \
  --support_long_context_over_fp16_limit
```

MTP：复用上面的 target，只重导 W8 draft 生成对比 `meta.json`。

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_5_27b
CONFIG=configs/qwen3_5/qwen3_5_27b_xh2a.py
HF_MODEL_DIR=/data01/home/yujy/work/auto-round/output/sym-mode1
EXISTING_WORK_DIR=work_dirs/qwen3_5_27b_xh2a_8k_w4a8_gptq_spec_mtp_draft4_headw4_<timestamp>
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8_gptq_spec_mtp_draft4_headw8_${TS}"
python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --draft_only \
  --existing_work_dir "$EXISTING_WORK_DIR" \
  --config "$CONFIG" \
  --hf_model_dir "$HF_MODEL_DIR" \
  --dtype fp16 \
  --max_sequence_length 8192 \
  --spec_decode_mode mtp \
  --num_draft_tokens 4 \
  --work_dir "$WORK_DIR"
```

DFlash：先导 W4 head 完整产物。

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_5_27b
CONFIG=configs/qwen3_5/qwen3_5_27b_xh2a.py
HF_MODEL_DIR=/data01/home/yujy/work/auto-round/output/sym-mode1
DFLASH_MODEL_DIR=weights/Qwen3.5-27B-DFlash
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8_gptq_spec_dflash_draft9_input10_${TS}"
python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --config "$CONFIG" \
  --hf_model_dir "$HF_MODEL_DIR" \
  --dtype fp16 \
  --max_sequence_length 8192 \
  --spec_decode_mode dflash \
  --num_draft_tokens 9 \
  --dflash_model_dir "$DFLASH_MODEL_DIR" \
  --work_dir "$WORK_DIR"
```

DFlash：复用上面的 target，只重导 W8 draft 生成对比 `meta.json`。

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_5_27b
CONFIG=configs/qwen3_5/qwen3_5_27b_xh2a.py
HF_MODEL_DIR=/data01/home/yujy/work/auto-round/output/sym-mode1
DFLASH_MODEL_DIR=weights/Qwen3.5-27B-DFlash
EXISTING_WORK_DIR=work_dirs/qwen3_5_27b_xh2a_8k_w4a8_gptq_spec_dflash_draft9_input10_headw4_<timestamp>
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8_gptq_spec_dflash_draft9_input10_headw8_${TS}"
python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --draft_only \
  --existing_work_dir "$EXISTING_WORK_DIR" \
  --config "$CONFIG" \
  --hf_model_dir "$HF_MODEL_DIR" \
  --dtype fp16 \
  --max_sequence_length 8192 \
  --spec_decode_mode dflash \
  --num_draft_tokens 9 \
  --dflash_model_dir "$DFLASH_MODEL_DIR" \
  --work_dir "$WORK_DIR"

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=5
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_5_9b
CONFIG=configs/qwen3_5/qwen3_5_9b_xh2a.py
HF_MODEL_DIR=/data01/home/yujy/work/gptqmodel/output/Qwen3.5-9B-mode1-llm-only
DFLASH_MODEL_DIR=weights/Qwen3.5-9B-DFlash
EXISTING_WORK_DIR=work_dirs/qwen3_5_9b_xh2a_8k_w4a8_gptq_spec_dflash_draft9_input10_headw4_<timestamp>
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8_gptq_spec_dflash_draft9_input10_headw4_fix_${TS}"
python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --draft_only \
  --existing_work_dir "$EXISTING_WORK_DIR" \
  --config "$CONFIG" \
  --hf_model_dir "$HF_MODEL_DIR" \
  --dtype fp16 \
  --max_sequence_length 8192 \
  --spec_decode_mode dflash \
  --num_draft_tokens 9 \
  --dflash_model_dir "$DFLASH_MODEL_DIR" \
  --work_dir "$WORK_DIR"
```

### MoE 模型命令

先按下表选择模型变量：

| 模型 | `MODEL_KEY` | `MODEL_DIR` | `QUANT_WEIGHT` | `DFLASH_MODEL_DIR` |
| --- | --- | --- | --- | --- |
| Qwen3.5 35B-A3B | `qwen3_5_35b_a3b` | `weights/Qwen3.5-35B-A3B` | `/data01/home/yujy/work/auto-round/output/Qwen3.5-35B-A3B-mode1-llm-only` | `weights/Qwen3.5-35B-A3B-DFlash` |
| Qwen3.6 35B-A3B | `qwen3_6_35b_a3b` | `weights/Qwen3.6-35B-A3B` | `/data01/home/yujy/work/auto-round/output/Qwen3.6-35B-A3B-mode1-llm-only` | `weights/Qwen3.6-35B-A3B-DFlash` |

MTP：先导 W4 head 完整产物。

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_5_35b_a3b
MODEL_DIR=weights/Qwen3.5-35B-A3B
QUANT_WEIGHT=/data01/home/yujy/work/auto-round/output/Qwen3.5-35B-A3B-mode1-llm-only
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8h1_ssfp_gptq_spec_mtp_draft4_${TS}"
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model "$MODEL_DIR" \
  --context-length 8192 \
  --input-sequence-length 256 \
  --quant-type w4a8h1_ssfp \
  --spec-decode-mode mtp \
  --num-draft-tokens 4 \
  --quant-weight "$QUANT_WEIGHT" \
  --work-dir "$WORK_DIR"
```

MTP：复用上面的 target，只重导 W8 draft 生成对比 `meta.json`。

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_5_35b_a3b
MODEL_DIR=weights/Qwen3.5-35B-A3B
QUANT_WEIGHT=/data01/home/yujy/work/auto-round/output/Qwen3.5-35B-A3B-mode1-llm-only
EXISTING_WORK_DIR=work_dirs/qwen3_5_35b_a3b_xh2a_8k_w4a8h1_ssfp_gptq_spec_mtp_draft4_headw4_<timestamp>
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8h1_ssfp_gptq_spec_mtp_draft4_headw8_${TS}"
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --draft-only \
  --existing-work-dir "$EXISTING_WORK_DIR" \
  --model "$MODEL_DIR" \
  --context-length 8192 \
  --input-sequence-length 256 \
  --quant-type w4a8h1_ssfp \
  --spec-decode-mode mtp \
  --num-draft-tokens 4 \
  --quant-weight "$QUANT_WEIGHT" \
  --work-dir "$WORK_DIR"
```

DFlash：先导 W4 head 完整产物。

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_5_35b_a3b
MODEL_DIR=weights/Qwen3.5-35B-A3B
QUANT_WEIGHT=/data01/home/yujy/work/auto-round/output/Qwen3.5-35B-A3B-mode1-llm-only
DFLASH_MODEL_DIR=weights/Qwen3.5-35B-A3B-DFlash
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8h1_ssfp_gptq_spec_dflash_draft9_input10_${TS}"
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model "$MODEL_DIR" \
  --context-length 8192 \
  --input-sequence-length 256 \
  --quant-type w4a8h1_ssfp \
  --spec-decode-mode dflash \
  --num-draft-tokens 9 \
  --quant-weight "$QUANT_WEIGHT" \
  --dflash-model-dir "$DFLASH_MODEL_DIR" \
  --work-dir "$WORK_DIR"
```

DFlash：复用上面的 target，只重导 W8 draft 生成对比 `meta.json`。

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_5_35b_a3b
MODEL_DIR=weights/Qwen3.5-35B-A3B
QUANT_WEIGHT=/data01/home/yujy/work/auto-round/output/Qwen3.5-35B-A3B-mode1-llm-only
DFLASH_MODEL_DIR=weights/Qwen3.5-35B-A3B-DFlash
EXISTING_WORK_DIR=work_dirs/qwen3_5_35b_a3b_xh2a_8k_w4a8h1_ssfp_gptq_spec_dflash_draft9_input10_headw4_<timestamp>
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8h1_ssfp_gptq_spec_dflash_draft9_input10_headw8_${TS}"
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --draft-only \
  --existing-work-dir "$EXISTING_WORK_DIR" \
  --model "$MODEL_DIR" \
  --context-length 8192 \
  --input-sequence-length 256 \
  --quant-type w4a8h1_ssfp \
  --spec-decode-mode dflash \
  --num-draft-tokens 9 \
  --quant-weight "$QUANT_WEIGHT" \
  --dflash-model-dir "$DFLASH_MODEL_DIR" \
  --work-dir "$WORK_DIR"
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


## 六模型 8k Spec Decode Benchmark（命令版，不再使用 batch_bench_8k.sh）

`batch_bench_8k.sh` 的核心行为等价于下面的单任务命令：从导出目录选择 `meta.json`，把 Markdown/JSON/log 写入 `output/qwen35_bench`，按 `think-mode`、`max-new-tokens`、`limit` 和 `shards` 控制评测范围。多任务并行时，一个 benchmark 进程绑定一张卡：`CUDA_VISIBLE_DEVICES=0` 后，进程内部仍使用 `cuda:0` / `exec-device cuda:0`。

默认数据集：

```bash
examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl
```

参数对齐：

- `--think-mode on|off|both`：默认建议 `both`；快速 smoke test 可用 `off`。
- `--max-new-tokens`：批处理脚本默认 `8192`；快速验证可降到 `32` 或 `128`。
- `--limit 0` 表示全量；`--limit 1`/`10` 适合 smoke test。
- `--shard-index N --num-shards M` 把数据集切成 M 份，N 从 0 开始；输出文件名建议带 `.shardNofM`。
- 批处理脚本默认开启 CUDA Graph：手写命令中显式加 `--enable-cuda-graph --cuda-graph-warmup-runs 3 --cuda-graph-graph-warmup-runs 6`。
- Dense 额外可加 `--auto-offload-max-memory`、`--prefill-auto-offload-max-memory`、`--decode-auto-offload-max-memory`；MoE 可用 `--enable-auto-offload` / `--disable-auto-offload`。

### 单任务模板

Dense 模型（Qwen3.5 4B / 9B / 27B，Qwen3.6 27B）：

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
mkdir -p output/qwen35_bench
python examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_bench.py \
  --meta work_dirs/<dense-export-dir>/meta.json \
  --dataset examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl \
  --output-md output/qwen35_bench/<tag>.md \
  --output-json output/qwen35_bench/<tag>.json \
  --think-mode both \
  --max-new-tokens 8192 \
  --dtype fp16 \
  --device cuda:0 \
  --exec-device cuda:0 \
  --enable-cuda-graph \
  --cuda-graph-warmup-runs 3 \
  --cuda-graph-graph-warmup-runs 6 \
  --limit 0 \
  2>&1 | tee output/qwen35_bench/<tag>.log
```

MoE 模型（35B-A3B）：

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
mkdir -p output/qwen35_bench
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_bench.py \
  --meta work_dirs/<moe-export-dir>/meta.json \
  --dataset examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl \
  --output-md output/qwen35_bench/<tag>.md \
  --output-json output/qwen35_bench/<tag>.json \
  --think-mode both \
  --max-new-tokens 8192 \
  --dtype fp16 \
  --device cuda:0 \
  --exec-device cuda:0 \
  --enable-cuda-graph \
  --cuda-graph-warmup-runs 3 \
  --cuda-graph-graph-warmup-runs 6 \
  --limit 0 \
  2>&1 | tee output/qwen35_bench/<tag>.log
```

分片示例（第 1/4 片，文件名同步带 shard 后缀）：

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
mkdir -p output/qwen35_bench
python examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_bench.py \
  --meta work_dirs/<dense-export-dir>/meta.json \
  --dataset examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl \
  --output-md output/qwen35_bench/<tag>.shard0of4.md \
  --output-json output/qwen35_bench/<tag>.shard0of4.json \
  --think-mode both \
  --max-new-tokens 8192 \
  --dtype fp16 \
  --device cuda:0 \
  --exec-device cuda:0 \
  --enable-cuda-graph \
  --cuda-graph-warmup-runs 3 \
  --cuda-graph-graph-warmup-runs 6 \
  --shard-index 0 \
  --num-shards 4 \
  2>&1 | tee output/qwen35_bench/<tag>.shard0of4.log
```

### 六模型并行示例（MTP，每个任务一张卡）

把 `<TS_...>` 替换为上面导出目录中的实际时间戳。DFlash 评测同理，把 meta 目录和输出 tag 中的 `spec_mtp_draft4` 换成 `spec_dflash_draft9_input10`；如果同时测 MTP + DFlash，请把 12 个 meta 当作 12 个独立任务，或在同一张卡上串行执行。

```bash
export PYTHONPATH=./
mkdir -p output/qwen35_bench

CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_bench.py \
  --meta work_dirs/qwen3_5_4b_xh2a_8k_w4a8_gptq_spec_mtp_draft4_headw4_<TS_Q35_4B>/meta.json \
  --dataset examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl \
  --output-md output/qwen35_bench/qwen3_5_4b_mtp_8k.md \
  --output-json output/qwen35_bench/qwen3_5_4b_mtp_8k.json \
  --think-mode both --max-new-tokens 8192 --dtype fp16 --device cuda:0 --exec-device cuda:0 \
  --enable-cuda-graph --cuda-graph-warmup-runs 3 --cuda-graph-graph-warmup-runs 6 \
  2>&1 | tee output/qwen35_bench/qwen3_5_4b_mtp_8k.log &

CUDA_VISIBLE_DEVICES=1 python examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_bench.py \
  --meta work_dirs/qwen3_5_9b_xh2a_8k_w4a8_gptq_spec_mtp_draft4_headw4_<TS_Q35_9B>/meta.json \
  --dataset examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl \
  --output-md output/qwen35_bench/qwen3_5_9b_mtp_8k.md \
  --output-json output/qwen35_bench/qwen3_5_9b_mtp_8k.json \
  --think-mode both --max-new-tokens 8192 --dtype fp16 --device cuda:0 --exec-device cuda:0 \
  --enable-cuda-graph --cuda-graph-warmup-runs 3 --cuda-graph-graph-warmup-runs 6 \
  2>&1 | tee output/qwen35_bench/qwen3_5_9b_mtp_8k.log &

CUDA_VISIBLE_DEVICES=2 python examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_bench.py \
  --meta work_dirs/qwen3_5_27b_xh2a_8k_w4a8_gptq_spec_mtp_draft4_headw4_<TS_Q35_27B>/meta.json \
  --dataset examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl \
  --output-md output/qwen35_bench/qwen3_5_27b_mtp_8k.md \
  --output-json output/qwen35_bench/qwen3_5_27b_mtp_8k.json \
  --think-mode both --max-new-tokens 8192 --dtype fp16 --device cuda:0 --exec-device cuda:0 \
  --enable-cuda-graph --cuda-graph-warmup-runs 3 --cuda-graph-graph-warmup-runs 6 \
  2>&1 | tee output/qwen35_bench/qwen3_5_27b_mtp_8k.log &

CUDA_VISIBLE_DEVICES=3 python examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_bench.py \
  --meta work_dirs/qwen3_6_27b_xh2a_8k_w4a8_gptq_spec_mtp_draft4_headw4_<TS_Q36_27B>/meta.json \
  --dataset examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl \
  --output-md output/qwen35_bench/qwen3_6_27b_mtp_8k.md \
  --output-json output/qwen35_bench/qwen3_6_27b_mtp_8k.json \
  --think-mode both --max-new-tokens 8192 --dtype fp16 --device cuda:0 --exec-device cuda:0 \
  --enable-cuda-graph --cuda-graph-warmup-runs 3 --cuda-graph-graph-warmup-runs 6 \
  2>&1 | tee output/qwen35_bench/qwen3_6_27b_mtp_8k.log &

CUDA_VISIBLE_DEVICES=4 python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_bench.py \
  --meta work_dirs/qwen3_5_35b_a3b_xh2a_8k_w4a8h1_ssfp_gptq_spec_mtp_draft4_headw4_<TS_Q35_35B_A3B>/meta.json \
  --dataset examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl \
  --output-md output/qwen35_bench/qwen3_5_35b_a3b_mtp_8k.md \
  --output-json output/qwen35_bench/qwen3_5_35b_a3b_mtp_8k.json \
  --think-mode both --max-new-tokens 8192 --dtype fp16 --device cuda:0 --exec-device cuda:0 \
  --enable-cuda-graph --cuda-graph-warmup-runs 3 --cuda-graph-graph-warmup-runs 6 \
  2>&1 | tee output/qwen35_bench/qwen3_5_35b_a3b_mtp_8k.log &

CUDA_VISIBLE_DEVICES=5 python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_bench.py \
  --meta work_dirs/qwen3_6_35b_a3b_xh2a_8k_w4a8h1_ssfp_gptq_spec_mtp_draft4_headw4_<TS_Q36_35B_A3B>/meta.json \
  --dataset examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl \
  --output-md output/qwen35_bench/qwen3_6_35b_a3b_mtp_8k.md \
  --output-json output/qwen35_bench/qwen3_6_35b_a3b_mtp_8k.json \
  --think-mode both --max-new-tokens 8192 --dtype fp16 --device cuda:0 --exec-device cuda:0 \
  --enable-cuda-graph --cuda-graph-warmup-runs 3 --cuda-graph-graph-warmup-runs 6 \
  2>&1 | tee output/qwen35_bench/qwen3_6_35b_a3b_mtp_8k.log &

wait
```

快速 smoke test 时，把每条命令里的 `--think-mode` / `--max-new-tokens` 改成下面的取值；已有 `--limit 0` 的模板将其改为 `--limit 1`，没有 `--limit` 的并行示例则追加 `--limit 1`：

```bash
--think-mode off --max-new-tokens 32 --limit 1
```

## 注意事项

1. Qwen3.5 当前已经支持 LLM 与 Vision 两部分的独立导出，也已经支持 VL 联合推理 demo。
2. VL demo 依赖两套产物同时存在：一套是 LLM 的 `export_meta_info.json`，另一套是 Vision 的 `.onnx` HMONNX 文件，二者需要来自同一模型家族并保持 tokenizer / processor 对齐。
3. Qwen3.5 使用 M-RoPE，VL 场景下会同时使用 time、height、width 三组位置编码；demo 中已经按 vision token 布局生成对应位置索引。
4. 运行 Vision 导出或 VL demo 时，建议始终显式设置 `PYTHONPATH=./`；如果缺少 hmquant 动态库路径，`xhquant` GPU 扩展可能加载失败。
5. Vision 导出当前默认使用 `448 x 448` 输入分辨率、`patch_size=16`、`temporal_patch_size=2`，如果修改这些参数，Vision 产物与 VL demo 的输入配置也要保持一致。
6. LLM 的 Prefill 和 Decode 是两张独立图，Vision HMONNX 只负责生成图像特征，最终由 VL demo 将图像特征散射回文本 token embedding 后再调用 LLM HMONNX。



python examples/llm/qwen3_5/qwen3_5_xh2a_mtp_demo_benchmark.py \
      --config work_dirs/qwen3_5_9b_xh2a_8k_w4a8_gptq_spec_mtp_draft4_headw4_20260519_004608/meta.json \
      --per-category 100 \
      --out-dir tmp/mtp_demo_xh2a_benchmark \
      --enable_cuda_graph \
      --cuda_graph_modules prefill,decode,draft_prefill,draft_context,draft_context_decode,draft_decode

python examples/llm/qwen3_5/qwen3_5_xh2a_mtp_demo_benchmark.py \
      --config work_dirs/qwen3_5_9b_xh2a_8k_w4a8_gptq_spec_mtp_k81920_draft4_headw4_20260519_004651/meta.json \
      --per-category 100 \
      --out-dir tmp/mtp_demo_xh2a_benchmark1 \
      --enable_cuda_graph \
      --cuda_graph_modules prefill,decode,draft_prefill,draft_context,draft_context_decode,draft_decode