#!/bin/env bash
if [ ! -d data/models/unet ]; then
    mkdir -p data/models/unet
fi
wget http://10.10.1.53:8082/artifactory/model_zoo2/houmo/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.onnx -O data/models/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.onnx