#!/bin/env bash
# generate_detr_onnx.sh

# 定义模型目录和文件名
MODEL_DIR="data/models/detr"
ONNX_FILE="$MODEL_DIR/detr_1312.onnx"
ONNX_FILE_NEW="$MODEL_DIR/detr_1312_prepared.onnx"
# 使用你同事脚本中提供的DETR模型链接
DOWNLOAD_URL="http://10.10.1.53:8082/artifactory/model_zoo2/houmo/detr/127001_modified_detr_simple.onnx"

# 创建模型目录
mkdir -p "$MODEL_DIR"

# 检查最终的 ONNX 文件是否存在
if [ ! -f "$ONNX_FILE" ]; then
    echo "detr_1312.onnx not found. Downloading from Artifactory..."
    
    # 使用 wget 下载文件，并保存为指定的文件名
    # -O 参数指定输出文件名
    wget -O "$ONNX_FILE" "$DOWNLOAD_URL"

    # 检查下载是否成功
    if [ $? -eq 0 ]; then
        echo "detr_1312.onnx downloaded successfully."
    else
        echo "Error: Failed to download detr_1312.onnx. Please check the URL and your network connection."
        # 如果下载失败，删除可能创建的空文件并退出
        rm -f "$ONNX_FILE"
        exit 1
    fi
else
    echo "detr_1312.onnx already exists."
fi

# 准备 ONNX 模型， 转化多头注意力算子
echo "Preparing ONNX model..."
python examples/cv/detr/detr_prepare_onnx.py --input-onnx "$ONNX_FILE" --output-onnx "$ONNX_FILE_NEW"