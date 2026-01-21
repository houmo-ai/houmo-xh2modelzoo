#!/bin/env bash
if [ ! -d data/models/paddleocr/v4 ]; then
    mkdir -p data/models/paddleocr/v4
fi

wget http://10.10.1.53:8082/artifactory/model_zoo2/houmo/paddleocr/ocrpaddle/paddleocrv4_det-sim.onnx -O data/models/paddleocr/v4/paddleocrv4_det-sim.onnx
wget http://10.10.1.53:8082/artifactory/model_zoo2/houmo/paddleocr/ocrpaddle/paddleocrv4_rec-sim.onnx -O data/models/paddleocr/v4/paddleocrv4_rec-sim.onnx
wget http://10.10.1.53:8082/artifactory/model_zoo2/houmo/paddleocr/ocrpaddle/paddleocr_cls-sim.onnx -O data/models/paddleocr/v4/paddleocr_cls-sim.onnx
wget http://10.10.1.53:8082/artifactory/model_zoo2/houmo/paddleocr/ocrpaddle/ppocr_keys_v1.txt -O data/models/paddleocr/v4/ppocr_keys_v1.txt
wget http://10.10.1.53:8082/artifactory/model_zoo2/houmo/paddleocr/ocrpaddle/test_images.zip -O data/test_images.zip
cd data
unzip test_images.zip
rm test_images.zip