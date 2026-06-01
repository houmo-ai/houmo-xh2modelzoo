#!/bin/env bash
if [ ! -d data/models/baidu ]; then
    mkdir -p data/models/baidu
fi

# 下载模型
wget http://10.10.1.53:8082/artifactory/model_zoo/baidu/modified_model_c3_camera_v2_960x544_si.onnx -O data/models/baidu/modified_model_c3_camera_v2_960x544_si.onnx
 