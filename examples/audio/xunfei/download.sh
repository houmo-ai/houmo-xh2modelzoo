#!/bin/env bash
if [ ! -d data/models/xunfei ]; then
    mkdir -p data/models/xunfei
fi

wget http://10.10.1.53:8082/artifactory/model_zoo2/houmo/xunfei/encoder1.onnx -O data/models/xunfei/encoder1d_0.onnx