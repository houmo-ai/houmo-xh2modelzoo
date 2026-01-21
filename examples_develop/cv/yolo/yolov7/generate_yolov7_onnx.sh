#!/bin/env bash

# 目标：从 YOLOv7 官方仓库 (WongKinYiu) 下载 .pt 模型，使用官方工具导出为 yolov7.onnx
# 并确保最终的 yolov7.onnx 放置在 data/models/yolo/ 目录下，
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

# --- 配置 (YOLOv7 Specific) ---
# YOLOv7 ONNX 文件的最终目标存放目录 (与 xhquant export.py 的默认路径一致)
FINAL_ONNX_DIR="$PROJECT_ROOT/data/models/yolo"
YOLOV7_OFFICIAL_REPO_PATH="$PROJECT_ROOT/yolov7_official_repo"
# YOLOv7 官方仓库的下载链接
YOLOV7_REPO_URL="https://github.com/WongKinYiu/yolov7.git" 
# YOLOv7 模型名称 (这里使用 yolov7，也可以是 yolov7-tiny, yolov7-x 等)
YOLOV7_MODEL_NAME="yolov7" 
# YOLOv7 .pt 模型的下载链接 (来自 WongKinYiu 仓库的 releases)
YOLOV7_PT_URL="https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt" 
YOLOV7_PT_FILE="$YOLOV7_OFFICIAL_REPO_PATH/$YOLOV7_MODEL_NAME.pt"
# YOLOv7 官方导出脚本相对于其仓库根目录的路径 (通常是 export.py)
YOLOV7_EXPORT_SCRIPT_PATH="export.py" 
# 导出的 ONNX 文件会在官方仓库的根目录生成 (默认行为，因为没有 --output 参数)
YOLOV7_GENERATED_ONNX_TEMP_PATH="$YOLOV7_OFFICIAL_REPO_PATH/$YOLOV7_MODEL_NAME.onnx"
YOLOV7_FINAL_ONNX_FILE="$FINAL_ONNX_DIR/$YOLOV7_MODEL_NAME.onnx"
# YOLOv7 的输入图片尺寸
YOLOV7_IMG_SIZE=640 


echo "开始准备 YOLOv7 ONNX 模型..."
echo "==========================================================="

# 1. 检查 ONNX 模型的最终目标目录是否存在，不存在则创建
echo "检查并创建 ONNX 最终目标目录: $FINAL_ONNX_DIR"
mkdir -p "$FINAL_ONNX_DIR"

# 2. 检查 yolov7.onnx 是否已存在于最终目标位置
if [ -f "$YOLOV7_FINAL_ONNX_FILE" ]; then
    echo "$YOLOV7_MODEL_NAME.onnx already exists at $YOLOV7_FINAL_ONNX_FILE. Exiting."
    exit 0
fi

# 3. 克隆 YOLOv7 官方代码 (如果还没有的话)
if [ ! -d "$YOLOV7_OFFICIAL_REPO_PATH" ]; then
    echo "Cloning YOLOv7 official repository to $YOLOV7_OFFICIAL_REPO_PATH..."
    git clone "$YOLOV7_REPO_URL" "$YOLOV7_OFFICIAL_REPO_PATH"
    if [ $? -ne 0 ]; then
        echo "错误: 克隆 YOLOv7 仓库失败！请检查网络或权限。脚本将终止。"
        exit 1
    fi
else
    echo "YOLOv7 official repository already exists at $YOLOV7_OFFICIAL_REPO_PATH."
fi

# 4. 进入官方仓库并安装 Python 依赖
echo "进入 $YOLOV7_OFFICIAL_REPO_PATH 并安装 Python 依赖..."
# 注意: YOLOv7 的 requirements.txt 可能需要一些特殊处理 (例如安装 torch 等)
# 确保在安装 torch 后再安装 requirements，或者按照其官方指南
(cd "$YOLOV7_OFFICIAL_REPO_PATH" && pip install -r requirements.txt)
if [ $? -ne 0 ]; then
    echo "错误: YOLOv7 官方仓库依赖安装失败！脚本将终止。"
    exit 1
fi

# 5. 下载 YOLOv7.pt 权重文件
if [ ! -f "$YOLOV7_PT_FILE" ]; then
    echo "Downloading $YOLOV7_MODEL_NAME.pt from $YOLOV7_PT_URL to $YOLOV7_PT_FILE..."
    wget -O "$YOLOV7_PT_FILE" "$YOLOV7_PT_URL" # -O 指定输出文件路径和文件名
    if [ $? -ne 0 ]; then
        echo "错误: $YOLOV7_MODEL_NAME.pt 下载失败！请检查网络或 URL。脚本将终止。"
        exit 1
    fi
else
    echo "$YOLOV7_MODEL_NAME.pt already exists at $YOLOV7_PT_FILE."
fi

# 6. 使用官方脚本导出 ONNX 模型
echo "使用官方脚本导出 $YOLOV7_MODEL_NAME.pt 为 ONNX 格式..."
# --- 关键修正: 移除 --output 参数，确保行尾没有空格 ---
(cd "$YOLOV7_OFFICIAL_REPO_PATH" && \
python "$YOLOV7_EXPORT_SCRIPT_PATH" \
    --weights "$YOLOV7_PT_FILE" \
    --img-size "$YOLOV7_IMG_SIZE" \
    --batch-size 1 \
    --simplify \
    &> /dev/null) # 将所有输出重定向到 /dev/null，避免干扰
# 检查 python 命令的退出状态
if [ $? -ne 0 ]; then
    echo "错误: YOLOv7 ONNX 导出失败！(Python 命令返回非零状态)。脚本将终止。"
    exit 1
fi

# 7. 将导出的 ONNX 文件移动到最终目标位置
if [ -f "$YOLOV7_GENERATED_ONNX_TEMP_PATH" ]; then
    echo "Moving exported ONNX from $YOLOV7_GENERATED_ONNX_TEMP_PATH to $YOLOV7_FINAL_ONNX_FILE..."
    mv "$YOLOV7_GENERATED_ONNX_TEMP_PATH" "$YOLOV7_FINAL_ONNX_FILE"
    if [ $? -ne 0 ]; then
        echo "错误: 移动 ONNX 文件失败！脚本将终止。"
        exit 1
    fi
    echo "$YOLOV7_MODEL_NAME ONNX 成功移动到最终位置。"
else
    # 只有当 python 导出命令成功但文件仍不存在时才报告这个严重错误
    echo "严重错误: ONNX 导出脚本成功执行，但未在预期位置找到文件 ($YOLOV7_GENERATED_ONNX_TEMP_PATH)。请手动检查。"
    exit 1
fi

echo "==========================================================="
echo "$YOLOV7_MODEL_NAME ONNX 准备完成，文件位于: $YOLOV7_FINAL_ONNX_FILE"