#!/bin/env bash
# generate_openpose_onnx.sh

# 定义模型目录和文件名
MODEL_DIR="data/models/openpose"
ONNX_FILE="$MODEL_DIR/openpose_body.onnx"
DOWNLOAD_URL="http://10.10.1.53:8082/artifactory/model_zoo2/houmo/openpose/openpose_body.onnx"

# 创建模型目录 (-p 确保父目录不存在时也会被创建)
mkdir -p "$MODEL_DIR"

# 检查最终的 ONNX 文件是否存在
if [ ! -f "$ONNX_FILE" ]; then
    echo "openpose_body.onnx not found. Downloading from Artifactory..."
    
    # 使用 wget 下载文件，并将其保存到指定路径
    # -O 参数指定输出文件名和路径
    wget -O "$ONNX_FILE" "$DOWNLOAD_URL"

    # 检查下载是否成功
    if [ $? -eq 0 ]; then
        echo "openpose_body.onnx downloaded successfully to $ONNX_FILE"
    else
        echo "Error: Failed to download openpose_body.onnx. Please check the URL and your network connection."
        # 如果下载失败，删除可能创建的空文件并退出
        rm -f "$ONNX_FILE"
        exit 1
    fi
else
    echo "openpose_body.onnx already exists in $MODEL_DIR"
fi