#!/bin/env bash
if [ ! -d data/models/wenet ]; then
    mkdir -p data/models/wenet
fi

wget http://10.10.1.53:8081/artifactory/model_zoo2/houmo/wenet/encoder.onnx -O data/models/wenet/encoder.onnx
wget http://10.10.1.53:8081/artifactory/model_zoo2/houmo/wenet/train.yaml -O data/models/wenet/train.yaml
wget http://10.10.1.53:8081/artifactory/model_zoo2/houmo/wenet/units.txt -O data/models/wenet/units.txt
wget http://10.10.1.53:8081/artifactory/model_zoo2/houmo/wenet/wenet_input_sample.zip -O data/
unzip data/wenet_input_sample.zip
rm data/wenet_input_sample.zip