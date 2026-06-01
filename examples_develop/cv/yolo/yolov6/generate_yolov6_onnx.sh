#!/bin/env bash

# 目标：从 YOLOv6 官方仓库下载 .pt 模型，使用官方工具导出为 yolov6m.onnx
# 并确保最终的 yolov6m.onnx 放置在 data/models/yolo/ 目录下，
# 以便 xhquant 的导出脚本能够直接使用其默认路径。

# 1. 切换到脚本所在的目录，确保所有相对路径正确
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || { echo "错误: 无法切换到脚本目录"; exit 1; }

# 确定项目根目录 (xh2_model_zoo 目录)
# 从脚本所在目录向上查找 '.git' 或 'examples' 目录
PROJECT_ROOT=""
CURRENT_DIR="$PWD"
while [[ "$CURRENT_DIR" != "/" ]]; do
    if [[ -d "$CURRENT_DIR/.git" ]]; then
        PROJECT_ROOT="$CURRENT_DIR"
        break
    fi
    if [[ -d "$CURRENT_DIR/examples/cv/yolo" ]]; then
        PROJECT_ROOT="$CURRENT_DIR"
        break
    fi
    CURRENT_DIR=$(dirname "$CURRENT_DIR")
done

if [[ -z "$PROJECT_ROOT" ]]; then
    echo "错误: 无法找到项目根目录 (未找到 .git 目录或顶层 examples/cv/yolo 目录)。"
    exit 1
fi
echo "项目根目录: $PROJECT_ROOT"

# --- 配置 ---
# YOLOv6 ONNX 文件的最终目标存放目录 (与 xhquant export.py 的默认路径一致)
FINAL_ONNX_DIR="$PROJECT_ROOT/data/models/yolo"
YOLOV6_OFFICIAL_REPO_PATH="$PROJECT_ROOT/yolov6_official_repo"
YOLOV6_REPO_URL="https://github.com/meituan/YOLOv6.git"
YOLOV6M_PT_URL="https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m.pt" # YOLOv6m 的下载链接
YOLOV6M_PT_FILE="$YOLOV6_OFFICIAL_REPO_PATH/yolov6m.pt"
YOLOV6_EXPORT_SCRIPT_PATH="deploy/ONNX/export_onnx.py" # 官方导出脚本相对于其仓库根目录的路径
# --- 修正这里: 导出的 ONNX 文件会在官方仓库的根目录生成 ---
YOLOV6M_GENERATED_ONNX_TEMP_PATH="$YOLOV6_OFFICIAL_REPO_PATH/yolov6m.onnx"
YOLOV6M_FINAL_ONNX_FILE="$FINAL_ONNX_DIR/yolov6m.onnx"


echo "开始准备 YOLOv6m ONNX 模型..."
echo "==========================================================="

# 1. 检查 ONNX 模型的最终目标目录是否存在，不存在则创建
echo "检查并创建 ONNX 最终目标目录: $FINAL_ONNX_DIR"
mkdir -p "$FINAL_ONNX_DIR"

# 2. 检查 yolov6m.onnx 是否已存在于最终目标位置
if [ -f "$YOLOV6M_FINAL_ONNX_FILE" ]; then
    echo "yolov6m.onnx already exists at $YOLOV6M_FINAL_ONNX_FILE. Exiting."
    exit 0
fi

# 3. 克隆 YOLOv6 官方代码 (如果还没有的话)
if [ ! -d "$YOLOV6_OFFICIAL_REPO_PATH" ]; then
    echo "Cloning YOLOv6 official repository to $YOLOV6_OFFICIAL_REPO_PATH..."
    git clone "$YOLOV6_REPO_URL" "$YOLOV6_OFFICIAL_REPO_PATH"
    if [ $? -ne 0 ]; then
        echo "错误: 克隆 YOLOv6 仓库失败！请检查网络或权限。脚本将终止。"
        exit 1
    fi
else
    echo "YOLOv6 official repository already exists at $YOLOV6_OFFICIAL_REPO_PATH."
fi

# 4. 进入官方仓库并安装 Python 依赖
echo "进入 $YOLOV6_OFFICIAL_REPO_PATH 并安装 Python 依赖..."
(cd "$YOLOV6_OFFICIAL_REPO_PATH" && pip install -r requirements.txt)
if [ $? -ne 0 ]; then
    echo "错误: YOLOv6 官方仓库依赖安装失败！脚本将终止。"
    exit 1
fi

# 5. 下载 YOLOv6m.pt 权重文件
if [ ! -f "$YOLOV6M_PT_FILE" ]; then
    echo "Downloading YOLOv6m.pt from $YOLOV6M_PT_URL to $YOLOV6M_PT_FILE..."
    wget -O "$YOLOV6M_PT_FILE" "$YOLOV6M_PT_URL" # -O 指定输出文件路径和文件名
    if [ $? -ne 0 ]; then
        echo "错误: YOLOv6m.pt 下载失败！请检查网络或 URL。脚本将终止。"
        exit 1
    fi
else
    echo "YOLOv6m.pt already exists at $YOLOV6M_PT_FILE."
fi

# 6. 使用官方脚本导出 ONNX 模型
echo "使用官方脚本导出 YOLOv6m.pt 为 ONNX 格式..."
# PyTorch 2.6+ 默认为 torch.load(..., weights_only=True)，
# 会导致 YOLOv6 官方脚本无法直接加载发布的 .pt checkpoint。
(cd "$YOLOV6_OFFICIAL_REPO_PATH" && \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python "$YOLOV6_EXPORT_SCRIPT_PATH" \
    --weights "$YOLOV6M_PT_FILE" \
    --img 640 \
    --batch 1 \
    --simplify)
if [ $? -ne 0 ]; then
    echo "错误: YOLOv6m ONNX 导出失败！脚本将终止。"
    exit 1
fi
# --- 修正这里: 打印实际生成的位置 ---
echo "YOLOv6m ONNX 成功导出到临时位置: $YOLOV6M_GENERATED_ONNX_TEMP_PATH"

# 7. 将导出的 ONNX 文件移动到最终目标位置
if [ -f "$YOLOV6M_GENERATED_ONNX_TEMP_PATH" ]; then
    echo "Moving exported ONNX from $YOLOV6M_GENERATED_ONNX_TEMP_PATH to $YOLOV6M_FINAL_ONNX_FILE..."
    mv "$YOLOV6M_GENERATED_ONNX_TEMP_PATH" "$YOLOV6M_FINAL_ONNX_FILE"
    if [ $? -ne 0 ]; then
        echo "错误: 移动 ONNX 文件失败！脚本将终止。"
        exit 1
    fi
    echo "YOLOv6m ONNX 成功移动到最终位置。"
else
    # 理论上这里不应该再发生，因为上面已经判断导出成功
    echo "严重错误: ONNX 文件导出成功但仍未在预期临时位置找到 ($YOLOV6M_GENERATED_ONNX_TEMP_PATH)。请手动检查。"
    exit 1
fi

echo "==========================================================="
echo "YOLOv6m ONNX 准备完成，文件位于: $YOLOV6M_FINAL_ONNX_FILE"
