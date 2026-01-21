#!/bin/env bash
if [ ! -d data/models/yolo ]; then
    mkdir -p data/models/yolo
fi

if [ ! -f data/models/yolo/yolov8m.onnx ]; then
    pip install ultralytics
    cd data/models/yolo/
    yolo export model=yolov8m.pt format=onnx
    rm -rf yolov8m.pt
else
    echo "yolov8m.onnx already exists"
fi