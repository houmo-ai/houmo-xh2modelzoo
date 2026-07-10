# PI05 Merak Workflow 示例

本目录提供 PI05 迁移到 `xhmodel_merak/xh_other_model` 后的 workflow 示例入口。模型适配、导出辅助代码位于 `xhmodel_merak/xh_other_model/models/pi05`。所有命令默认从仓库根目录执行。

## 环境安装

环境名称和 Python 版本可按实际工程要求调整。示例安装步骤如下：

```bash
conda create -n <env_name> python=3.12
conda activate <env_name>

pip install -v -e . --no-build-isolation
pip install lerobot==0.4.4
```

PI05 导出还需要提供 Paligemma 配置目录，用于 tokenizer 和相关配置加载。该路径通过 `--config-dir` 传入。

## 配置

脚本按 `--variant` 选择默认 YAML 和默认导出目录：

- `libero`: `configs_merak/workflows/xh2a/other_models/pi05/libero/pi05_libero.yaml`
- `droid`: `configs_merak/workflows/xh2a/other_models/pi05/droid/pi05_droid.yaml`

`--model-dir` 必须显式传入。默认输出目录是相对当前工作目录的字符串路径：

- 量化目录：`work_dirs/pi05_quant`
- Libero 导出目录：`work_dirs/pi05_libero_XH2a`
- Droid 导出目录：`work_dirs/pi05_droid_XH2a`

## 导出

Libero:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/vla/pi05/pi05_workflow.py \
  --variant libero \
  --model-dir <pi05_libero_model_dir> \
  --config-dir <paligemma_config_dir> \
  --device cuda:0 \
  --overwrite
```

Droid:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/vla/pi05/pi05_workflow.py \
  --variant droid \
  --model-dir <pi05_droid_model_dir> \
  --config-dir <paligemma_config_dir> \
  --device cuda:0 \
  --overwrite
```

可使用 `--components vision,gemma,expert,other` 只导出部分子模型，使用 `--quant-type w8a8h1_sefp` 覆盖所有子模型导出精度。

## Golden 生成

golden 数据只通过 `dump_golden` 生成：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/vla/pi05/pi05_workflow.py \
  --variant libero \
  --model-dir <pi05_libero_model_dir> \
  --config-dir <paligemma_config_dir> \
  --device cuda:0 \
  --dump-golden \
  --overwrite
```

导出目录包含顶层 `export_meta_info.json`。Gemma 子模型目录也会保留各自的 `export_meta_info.json`，用于兼容原 LLM 导出约定。
