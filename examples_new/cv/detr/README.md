# detr模型

## 导出onnx

```bash
bash examples/cv/detr/generate_detr_onnx.sh
```

## 导出HMONNX

```bash
python examples/cv/detr/detr_export.py --onnx data/models/detr/detr_1312_prepared.onnx
```

## GPU仿真

```bash
python examples/cv/detr/detr_hmonnx_test.py --hmonnx work_dirs/detr_1312_prepared/hmonnx/detr_1312_prepared_w8a8_sefp_XH2a.onnx --image data/images/000000001490.jpg
```

## 测试原浮点模型精度

可通过修改yaml文件修改到你的数据集位置

``` bash
python examples/cv/detr/detr_eval.py --model data/models/detr/detr_1312_prepared.onnx --model-type onnx
```

## 测试量化后模型

``` bash
python examples/cv/detr/detr_eval.py --model work_dirs/detr_1312_prepared/hmonnx/detr_1312_prepared_w8a8_sefp_XH2a.onnx --model-type hmonnx
python examples/cv/detr/detr_eval.py --model work_dirs/detr_1312_prepared/hmonnx/detr_1312_prepared_w8a16_sefp_XH2a.onnx --model-type hmonnx
python examples/cv/detr/detr_eval.py --model work_dirs/detr_1312_prepared/hmonnx/detr_1312_prepared_w4a8_ssfp_XH2a.onnx --model-type hmonnx
```

``` bash
#  浮点模型
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.395
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.589
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.419
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.217
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.424
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.558
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.321
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.514
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.555
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.323
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.593
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.767

#  w8a8sefp
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.332
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.557
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.332
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.357
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.520
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.285
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.451
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.489
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.230
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.525
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.731
 
#  w8a16sefp
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.560
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.338
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.143
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.361
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.522
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.286
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.454
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.492
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.233
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.529
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.733

#  w4a8ssfp
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.057
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.149
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.037
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.013
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.123
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.089
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.123
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.136
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.002
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.047
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.349
```