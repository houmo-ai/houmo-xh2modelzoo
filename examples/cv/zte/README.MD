# 带loop的模型

## 1. 固定batch为32
    ```shell
        python -m onnxsim weights/zte/model_hexiaoxi.onnx weights/zte/model_hexiaoxi_sim.onnx --input-shape serving_default_lstm_input:0:32,9,13
    ```
## 2. 将loop模型拆开，lstm拆开
    ```shell
        python examples/cv/zte/convert_loop.py --in weights/zte/model_hexiaoxi_sim.onnx --out weights/zte/model_hexiaoxi_sim_unloop.onnx --loop_name while_loop --trip_count 9
    ```
## 3. onnxsim下
    ```shell 
        python -m onnxsim weights/zte/model_hexiaoxi_sim_unloop.onnx weights/zte/model_hexiaoxi_sim_unloop_sim.onnx --skip-optimization fuse_matmul_add_bias_into_gemm
    ```
## 4. 量化生成golden
    ```shell
        python examples/cv/zte/loop_model_quant.py
    ```

# model_timi_cjb

## 1. 固定src和src_key_padding_mask的batch_size为1
    ```shell
    python -m onnxsim weights/zte/model_timi_cjb.onnx weights/zte/model_timi_cjb_sim.onnx --input-shape src:1,512,64 src_key_padding_mask:1,512
    ```
## 2. 量化生成golden
    ```shell
        python examples/cv/zte/timo_cjb_quant.py
    ```

# 带LSTM的模型

## 1. 固定batch为32
    ```shell
        python -m onnxsim weights/zte/model_v4_hxx.onnx weights/zte/model_v4_hxx_sim.onnx --input-shape lstm_input:32,9,13
    ```
