#!/bin/env bash
if [ ! -d data/models/resnet ]; then
    mkdir -p data/models/resnet
fi
wget http://10.10.1.53:8082/artifactory/model_zoo2/houmo/resnet/resnet50_224x224.onnx -O data/models/resnet/resnet50_224x224.onnx