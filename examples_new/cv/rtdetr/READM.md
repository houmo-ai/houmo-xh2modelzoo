# RTDetr

## 1. 导出ONNX模型(<https://github.com/nanmi/RT-DETR-Deploy>)

### 1.1 安装PaddleDetection

```bash
conda create -n paddle python=3.10 -y
conda activate paddle
python -m pip install paddlepaddle-gpu==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
pip install -r requirements.txt
python setup.py install
```

### 1.2 导出Paddle模型

```bash
cd PaddleDetection/
python tools/export_model.py -c configs/rtdetr/rtdetr_hgnetv2_l_6x_coco.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_l_6x_coco.pdparams trt=True --output_dir=output_inference
```

### 1.3 安装Paddle2ONNX

```bash
pip install paddle2onnx 或者
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git && cd Paddle2ONNX
git checkout v1.3.1
git submodule update --init
export PIP_EXTRA_INDEX_URL="https://www.paddlepaddle.org.cn/packages/nightly/cpu/"
pip install -e .
```

### 1.4 导出ONNX模型

```bash
paddle2onnx --model_dir=./output_inference/rtdetr_hgnetv2_l_6x_coco/ --model_filename model.pdmodel  --params_filename model.pdiparams --opset_version 16 --save_file rtdetr_hgnetv2_l_6x_coco.onnx
onnxsim  rtdetr_hgnetv2_l_6x_coco.onnx rtdetr_hgnetv2_l_6x_coco-sim.onnx --overwrite-input-shape im_shape:1,2 image:1,3,640,640 scale_factor:1,2
```

### 1.5 验证导出模型

```bash
python examples/cv/rtdetr/scripts/onnx_run_original_onnx_post_process.py --onnx data/models/rtdetr/rtdetr_hgnetv2_l_6x_coco-sim.onnx --image data/images/dog.jpg
```

### 1.6 移除onnx的后处理部分

```bash
python examples/cv/rtdetr/scripts/modify_onnx.py --onnx data/models/rtdetr/rtdetr_hgnetv2_l_6x_coco-sim.onnx --image data/images/dog.jpg
```

## 2.导出HMONNX

### 2.1 导出不带后处理的模型

```bash
python examples/cv/rtdetr/export_hmonnx_no_post_process.py --onnx data/models/rtdetr/rtdetr_hgnetv2_l_6x_coco-no-post_process.onnx
```

### 2.2 验证模型

```bash
python examples/cv/rtdetr/rtdetr_no_post_process_test.py --hmonnx work_dirs/rtdetr_hgnetv2_l_6x_coco-no-post_process/hmonnx/rtdetr_hgnetv2_l_6x_coco-no-post_process_XH2a.onnx  --image data/images/dog.jpg
```
