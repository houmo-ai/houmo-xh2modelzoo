#!/bin/env bash
# generate_yolov5_onnx.sh

# 定义模型目录和文件名
MODEL_DIR="data/models/yolo"
ONNX_FILE="$MODEL_DIR/yolov5m.onnx"
DOWNLOAD_URL="http://10.10.1.53:8082/artifactory/model_zoo2/houmo/yolov5/yolov5s_650x650.simplify.onnx"

# 创建模型目录
mkdir -p "$MODEL_DIR"

# 检查最终的 ONNX 文件是否存在
if [ ! -f "$ONNX_FILE" ]; then
    echo "yolov5m.onnx not found. Downloading from Artifactory..."
    
    # 使用 wget 下载文件，并将其重命名为我们期望的 yolov5.onnx
    # -O 参数指定输出文件名
    wget -O "$ONNX_FILE" "$DOWNLOAD_URL"

    # 检查下载是否成功
    if [ $? -eq 0 ]; then
        echo "yolov5m.onnx downloaded successfully."
    else
        echo "Error: Failed to download yolov5.onnx. Please check the URL and your network connection."
        # 如果下载失败，删除可能创建的空文件并退出
        rm -f "$ONNX_FILE"
        exit 1
    fi
else
    echo "yolovm.onnx already exists."
fi