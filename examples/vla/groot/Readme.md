# GROOT (N1.6-3B) 示例说明

本目录提供 GROOT 模型在 XH2a 侧的导出与联调示例，包含三部分：

- Vision 导出：`vision_export.py`
- Text Encoder 导出：`text_encoder_export.py`
- Action Head 导出：`head_export.py`
- 端到端联调：`hm_demo.py`

## 1. 环境准备

建议在仓库根目录执行，并先配置本项目运行环境。

### 1.1 Python 依赖

```bash
pip install transformers==4.51.3
```
### 1.2 Clone the Repository

The Gr00t model repository is already cloned at:
```
/data01/home/xuchen/xh2/xh2_model_zoo/xh_model_zoo/xh_llm/models/groot/gr00t
```

If you need to re-clone or update the repository:

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
```

### 1.3 Soft Link Setup

The Gr00t model is already soft-linked to the correct location:
```
ln -s Isaac-GR00T/gr00t xh_model_zoo/xh_llm/models/groot/gr00t
```

The current repository is already in the correct location, so no additional soft linking is needed.

如未安装本仓库依赖，请先在仓库根目录完成基础安装（按项目既有方式，如 `pip install -e .` 或 `poetry install`）。

### 1.4 模型与路径

当前脚本中存在硬编码路径，使用前请替换为你本机路径：

- GROOT 权重路径：`/data02/datasets/GROOT-N1.6-3B`
- `hm_demo.py` 中 HMONNX 路径：
	- `work_dirs/groot/hmonnx/groot_vision-*.onnx`
	- `work_dirs/groot_head/hmonnx/groot_head_pre-*.onnx`
	- `work_dirs/groot_head/hmonnx/groot_head-*.onnx`

## 2. 导出 HMONNX

以下命令默认在仓库根目录执行：

```bash
python examples/vla/groot/vision_export.py --quant-type w8a8h1_sefp
python examples/vla/groot/text_encoder_export.py --quant-type w8a8h1_sefp
python examples/vla/groot/head_export.py --quant-type w8a8h1_sefp
```

导出后通常会在以下目录产物：

- `work_dirs/groot`
- `work_dirs/groot_head`

## 3. 联调推理

运行端到端示例：

```bash
python examples/vla/groot/hm_demo.py
```

脚本会构造一份随机观测（视频、机器人状态、语言指令），并打印动作输出 shape。成功时可看到：

```text
GR00T Inference Success!
```

## 4. dataset.py 说明

`dataset.py` 演示了 `lerobot` 数据集读取方式：

```bash
python examples/vla/groot/dataset.py
```

若本机未安装 `lerobot`，请先安装其依赖并确保可访问对应 Hub 数据集。

## 5. 常见问题

- **找不到模型文件**：优先检查 `model_path` 与 ONNX 路径是否替换为本机可访问路径。
- **导出成功但 demo 失败**：确认 `hm_demo.py` 中读取的 ONNX 文件名与实际导出产物一致。
- **CUDA/显存问题**：当前脚本默认 `device='cuda'`，请确保 GPU 环境可用。
- **路径相对问题**：建议固定在仓库根目录执行命令，避免 `work_dirs/...` 相对路径错位。

## 6. 推荐执行顺序

1. 修改脚本中的硬编码路径。
2. 依次运行 `vision_export.py`、`text_encoder_export.py`、`head_export.py`。
3. 运行 `hm_demo.py` 验证端到端推理。