# MiniCPM-V-4.6 Merak Workflow

本目录提供 MiniCPM-V-4.6 的 Merak 导出、Golden 和 HMONNX 推理入口。模型实现位于
`xhmodel_merak/xh_llm/models/minicpm_v_4_6`。

## 环境

MiniCPM-V-4.6 的原生实现要求 `transformers>=5.7.0`。安装仓库和运行时后，从仓库
根目录执行以下命令：

```bash
pip install "transformers==5.7.0"
pip install -v -e . --no-build-isolation
export TOKENIZERS_PARALLELISM=false
```

默认 YAML 是：

```text
configs_merak/workflows/xh2a/llm_models/minicpm_v_4_6/0_8b/minicpm_v_4_6_xh2a_w8a8.yaml
```

它导出三个组件：1536 patch-token capacity 的 4x Vision、同容量的 16x Vision，以及
8K context、256-token Prefill 的 Qwen3.5-0.8B Prefill/Decode。每个组件的量化精度可在
`export.components` 中独立配置。

## 浮点基线

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/llm/minicpm_v_4_6/float_demo.py \
  --model-dir <minicpm_v_4_6_model_dir> \
  --image <image_file> \
  --device cuda:0 \
  --downsample-mode 4x
```

重复传入 `--image` 或 `--video` 可验证官方多图、视频和动态切片预处理。该入口直接使用
Transformers 原生模型，作为 HMONNX demo 的浮点基线。

## 导出与 Golden

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/llm/minicpm_v_4_6/minicpm_v_4_6_workflow.py \
  --model-dir <minicpm_v_4_6_model_dir> \
  --device cuda:0 \
  --export-output-dir work_dirs/minicpm_v_4_6_merak_export \
  --overwrite
```

同一次工作流生成 Golden：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/llm/minicpm_v_4_6/minicpm_v_4_6_workflow.py \
  --model-dir <minicpm_v_4_6_model_dir> \
  --device cuda:0 \
  --dump-golden \
  --overwrite
```

Golden 会覆盖生成 `vision_4x`、`vision_16x`、`prefill` 和 `decode` 四部分；LLM 只生成
一套，不因 Vision profile 重复。每部分的 `step_N/` 直接位于对应 HMONNX 目录，满足
xhquant 的相对软链接布局；图像 Prompt 超过 256-token Prefill 长度时会产生多个 Prefill
step。`export()` 本身不生成 Golden，`dump_golden()` 会先清理所有图的旧 step。

导出目录的唯一外部运行入口是根 `export_meta_info.json`。它记录有效 YAML、三个组件、
量化精度、静态 ABI 和相对产物路径；LLM 子目录中的元数据只供 Merak 的 Qwen3.5
运行时内部解析。

## HMONNX 推理

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/llm/minicpm_v_4_6/hmonnx_demo.py \
  --export-meta work_dirs/minicpm_v_4_6_merak_export/export_meta_info.json \
  --image data/images/qwen2_vl_demo.jpeg \
  --downsample-mode 4x \
  --prompt "请描述这张图片。"
```

重复传入 `--image` 或 `--video` 可执行多图/视频推理。宿主预处理保持官方动态切片：
每个原生 patch grid 先按 2x2 merger 层级重排，再在 token 维补齐到固定容量；图内不再
依赖动态高宽或动态 reshape。超过容量会明确报错。

## 已有精度基线

使用相同 processor、动态切片、4x、最多 9 个切片和 greedy decoding 的配对结果如下：

| 数据集 | 样本数 | Torch | HMONNX | 差值 |
|---|---:|---:|---:|---:|
| OCRBench | 1000 | 833 | 841 | +8 |
| ChartQA_TEST | 2500 | 76.60% | 76.72% | +0.12 pp |
| AI2D_TEST（Direct） | 3088 | 80.38% | 79.95% | -0.42 pp |

OCRBench 官方浮点协议使用 beam=3，复现分数为 839，官方公布值为 838。当前 Merak
HMONNX demo 采用 greedy；beam search 不属于导出图能力，不应把 greedy 结果与官方
beam=3 结果直接比较。
