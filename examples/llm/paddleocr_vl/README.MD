# 1. 环境安装

    ```shell
    pip install transformers==4.57.1

# —————————— —————————— ———————— ——————— 不带版面分析功能（PP-DocLayoutV2) 只有OCR —————————— —————————— ———————— ———————
```
# 2. 浮点golden推理
float_demo.py：gpu 浮点golden

    ```shell
    python examples/llm/paddleocr_vl/float_demo.py --image_path /data01/home/linxiang.wang/xh2modelzoo/data/images/layout.png

```
# 3. 导出vision模块 ONNX
paddleocr_vl_vision_xh2a_export_hmonnx.py：

    ```shell
    python examples/llm/paddleocr_vl/paddleocr_vl_vision_xh2a_export_hmonnx.py --valid --image_path /data01/home/linxiang.wang/xh2modelzoo/data/images/layout.png --min_pixels 147384 --max_pixels 2822400 --config /data01/home/linxiang.wang/xh2modelzoo/xh_model_zoo/xh_llm/models/paddleocr_vl/paddleocr_vl_vision_config.py

```
# 4. 导出llm模块(prefill/decode) ONNX
paddleocr_vl_llm_xh2a_export_hmonnx.py：

    ```shell
    python examples/llm/paddleocr_vl/paddleocr_vl_llm_xh2a_export_hmonnx.py --valid --image_path /data01/home/linxiang.wang/xh2modelzoo/data/images/layout.png --min_pixels 147384 --max_pixels 2822400 --config /data01/home/linxiang.wang/xh2modelzoo/xh_model_zoo/xh_llm/models/paddleocr_vl/paddleocr_vl_llm_config.py
 
```
# 5. 加载导出的hmonnx进行前向推理验证
paddleocr_vl_demo.py: 

    ```shell
    python examples/llm/paddleocr_vl/paddleocr_vl_demo.py --hf_model_config_dir /data01/datasets/PaddleOCR-VL --visual_onnx_path /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl/paddleocr_vl_vision_config_1_use_gptq_model_False_PaddleOCR-VL/vision/paddleocr_vl_vision_config.onnx --prefill_onnx_path /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl/paddleocr_vl_llm_config_PaddleOCR-VL/prefill_onnx/paddleocr_vl_llm_prefill.onnx  --decode_onnx_path /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl/paddleocr_vl_llm_config_PaddleOCR-VL/decode_onnx/paddleocr_vl_instruct_llm_decode.onnx  --meta_info /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl/paddleocr_vl_llm_config_PaddleOCR-VL/export_meta_info.json --min_pixels 147384 --max_pixels 2822400 --image /data01/home/linxiang.wang/xh2modelzoo/data/images/layout.png

# —————————— —————————— ———————— ——————— 带版面分析功能（PP-DocLayoutV2) —————————— —————————— ———————— ———————
```    
# 1. 使用paddle2onnx导出 layout部分的onnx(需要安装paddlex)

    ```shell
    paddlex --paddle2onnx --paddle_model_dir /data01/datasets/PaddleOCR-VL/PP-DocLayoutV2/ --onnx_model_dir /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl --opset_version 7

```    
# 2. 导出layout模块 ONNX
doclayoutv2_xh2a_export_hmonnx.py：

    ```shell
    python examples/llm/paddleocr_vl/doclayoutv2_xh2a_export_hmonnx.py --config /data01/home/linxiang.wang/xh2modelzoo/xh_model_zoo/xh_llm/models/paddleocr_vl/paddleocr_vl_doclayoutv2_config.py --image /data01/home/linxiang.wang/xh2modelzoo/data/images/layout.png --src_onnx /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl/inference.onnx


```
# 3. （带版面分析功能）加载导出的hmonnx进行前向推理验证
paddleocr_vl_with_layoutv2_demo.py:

    ```shell
    python examples/llm/paddleocr_vl/paddleocr_vl_with_layoutv2_demo.py  --image /data01/home/linxiang.wang/xh2modelzoo/data/images/layout.png  --layout_onnx  /data01/home/linxiang.wang/xhquant_llm/work_dirs/paddleocr_vl/doclayoutv2_hmonnx_export/PP-DocLayoutV2hm.onnx  --hf_model_config_dir  /data01/datasets/PaddleOCR-VL  --visual_onnx_path   /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl/paddleocr_vl_vision_config_1_use_gptq_model_False_PaddleOCR-VL/vision/paddleocr_vl_vision_config.onnx --prefill_onnx_path  /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl/paddleocr_vl_llm_config_PaddleOCR-VL/prefill_onnx/paddleocr_vl_llm_prefill.onnx --decode_onnx_path  /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl/paddleocr_vl_llm_config_PaddleOCR-VL/decode_onnx/paddleocr_vl_instruct_llm_decode.onnx  --meta_info  /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl/paddleocr_vl_llm_config_PaddleOCR-VL/export_meta_info.json --min_pixels 147384 --max_pixels 2822400 


