# 1. 环境安装

    ```shell
    pip install transformers==4.57.1

# —————————— —————————— ———————— ——————— 不带版面分析功能（PP-DocLayoutV3) 只有OCR —————————— —————————— ———————— ———————
```
# 2. 浮点golden推理(本过程需要"transformers>=5.0.0")
float_demo.py：gpu 浮点golden

    ```shell
    python examples/llm/paddleocr_vl/float_demo.py 

```
# 3. 导出vision模块 ONNX
paddleocr_vl_1_5_vision_xh2a_export_hmonnx.py：

    ```shell
    python examples/llm/paddleocr_vl_1_5/paddleocr_vl_1_5_vision_xh2a_export_hmonnx.py --valid --min_pixels 147384 --max_pixels 2822400 --config /data01/home/linxiang.wang/xh2modelzoo/xh_model_zoo/xh_llm/models/paddleocr_vl_1_5/paddleocr_vl_1_5_vision_config.py --image /data01/home/linxiang.wang/xh2modelzoo/data/images/ocr_img.png

```
# 4. 导出llm模块(prefill/decode) ONNX
paddleocr_vl_1_5_llm_xh2a_export_hmonnx.py：

    ```shell
    python examples/llm/paddleocr_vl_1_5/paddleocr_vl_1_5_llm_xh2a_export_hmonnx.py --valid --min_pixels 147384 --max_pixels 2822400 --config /data01/home/linxiang.wang/xh2modelzoo/xh_model_zoo/xh_llm/models/paddleocr_vl_1_5/paddleocr_vl_1_5_llm_config.py  --image /data01/home/linxiang.wang/xh2modelzoo/data/images/ocr_img.png

```
# 4. 加载导出的hmonnx进行前向推理验证
paddleocr_vl_demo.py: 

    ```shell
    python examples/llm/paddleocr_vl_1_5/paddleocr_vl_1_5_demo.py --hf_model_config_dir /data01/datasets/PaddleOCR-VL-1.5 --image /data01/home/linxiang.wang/xh2modelzoo/data/images/ocr_img.png --visual_onnx_path /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl_1_5/paddleocr_vl_1_5_vision_config_1_use_gptq_model_False_PaddleOCR-VL-1/vision/paddleocr_vl_1_5_vision_config.onnx --prefill_onnx_path /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl_1_5/paddleocr_vl_1_5_llm_config_PaddleOCR-VL-1/prefill_onnx/paddleocr_vl_instruct_llm_prefill.onnx  --decode_onnx_path /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl_1_5/paddleocr_vl_1_5_llm_config_PaddleOCR-VL-1/decode_onnx/paddleocr_vl_1_5_llm_decode.onnx  --meta_info /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl_1_5/paddleocr_vl_1_5_llm_config_PaddleOCR-VL-1/export_meta_info.json --min_pixels 147384 --max_pixels 2822400 

# —————————— —————————— ———————— ——————— 带版面分析功能（PP-DocLayoutV3) —————————— —————————— ———————— ———————
```    
# 1. 使用paddle2onnx导出 layout部分的onnx(需要安装paddlex)

    ```shell
    paddlex --paddle2onnx --paddle_model_dir /data01/datasets/PaddleOCR-VL-1.5/PP-DocLayoutV3/ --onnx_model_dir /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl_1_5/ --opset_version 7

```    
# 2. 导出layout模块 ONNX
doclayoutv3_xh2a_export_hmonnx.py：

    ```shell
    python examples/llm/paddleocr_vl_1_5/doclayoutv3_xh2a_export_hmonnx.py --config /data01/home/linxiang.wang/xh2modelzoo/xh_model_zoo/xh_llm/models/paddleocr_vl_1_5/paddleocr_vl_1_5_doclayoutv3_config.py --image /data01/home/linxiang.wang/xh2modelzoo/data/images/ocr_img.png --src_onnx /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl_1_5/inference.onnx


```
# 3. （带版面分析功能）加载导出的hmonnx进行前向推理验证
paddleocr_vl_1_5_with_layoutv3_demo.py:

    ```shell
    pip install shapely-2.1.2
    python examples/llm/paddleocr_vl_1_5/paddleocr_vl_1_5_with_layoutv3_demo.py --hf_model_config_dir /data01/datasets/PaddleOCR-VL-1.5 --image /data01/home/linxiang.wang/xh2modelzoo/data/images/ocr_img.png  --layout_onnx  /data01/home/linxiang.wang/xhquant_llm/work_dirs/paddleocr_vl_1_5/doclayoutv3_hmonnx_export/PP-DocLayoutV3hm.onnx --visual_onnx_path /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl_1_5/paddleocr_vl_1_5_vision_config_1_use_gptq_model_False_PaddleOCR-VL-1/vision/paddleocr_vl_1_5_vision_config.onnx --prefill_onnx_path /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl_1_5/paddleocr_vl_1_5_llm_config_PaddleOCR-VL-1/prefill_onnx/paddleocr_vl_instruct_llm_prefill.onnx  --decode_onnx_path /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl_1_5/paddleocr_vl_1_5_llm_config_PaddleOCR-VL-1/decode_onnx/paddleocr_vl_1_5_llm_decode.onnx  --meta_info /data01/home/linxiang.wang/xh2modelzoo/work_dirs/paddleocr_vl_1_5/paddleocr_vl_1_5_llm_config_PaddleOCR-VL-1/export_meta_info.json --min_pixels 147384 --max_pixels 2822400 

    


   