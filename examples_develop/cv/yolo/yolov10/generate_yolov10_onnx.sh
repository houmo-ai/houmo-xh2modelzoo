#!/bin/env bash
if [ ! -d data/models/yolo ]; then
    mkdir -p data/models/yolo
fi

if [ ! -f data/models/yolo/yolov10m.onnx ]; then
    pip install ultralytics
    cd data/models/yolo/
    yolo export model=yolov10m.pt format=onnx
    rm -rf yolov10m.pt
else
    echo "yolov10m.onnx already exists"
fi