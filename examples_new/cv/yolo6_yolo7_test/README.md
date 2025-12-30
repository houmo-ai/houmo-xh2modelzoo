### YOLOv6 & YOLOv7 量化评测说明

此目录包含对YOLOv6和YOLOv7模型进行量化和COCO数据集评测的脚本。

#### 步骤一：准备原始ONNX模型

在执行量化评测前，请先运行以下脚本，用于从官方仓库生成原始的FP32 ONNX模型。

1. `./examples/cv/yolo/yolov6/generate_yolov6_onnx.sh`
2. `./examples/cv/yolo/yolov7/generate_yolov7_onnx.sh`

脚本会自动将生成的ONNX模型存放在 `./data/models/` 目录下。

#### 步骤二：执行量化与评测

原始模型准备就绪后，即可执行本目录下的评测脚本：

1. `run_yolov6_evaluation_parallel.sh`
2. `run_yolov7_evaluation_parallel.sh`

**脚本功能**

脚本会自动完成模型的量化、导出，并在COCO数据集上进行并行测试。

**相关路径**

* **输入模型**:
    `./xh2_model_zoo/data/models/`

* **量化后模型**:
    `./xh2_model_zoo/work_dirs/` (分别在 `yolov6m` 和 `yolov7` 子目录下)

* **评测结果**:
    `./xh2_model_zoo/yolo6_yolo7_test/quantization_coco_evaluation_results/`

**结果示例**

由于完整评测耗时较长，提供以下结果文件作为参考：

* `./xh2_model_zoo/yolo6_yolo7_test/final_coco_evaluation_results/`
    > 这是一个完整的评测结果文件夹示例，包含所有日志和JSON。因其体积较大，运行脚本并确认流程无误后，**可以删除此文件夹**。

* `./xh2_model_zoo/yolo6_yolo7_test/6&7_results.txt`
    > 这是精简后的核心mAP指标汇总，可供快速查阅和参考。
