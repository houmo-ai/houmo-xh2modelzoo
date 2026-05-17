Qwen3-ASR

- 两种参数：架构相同，hidden dim 不同。
    - 0.6B
    - 1.7B

1. 依赖包

``` bash
pip install qwen_asr
pip install onnx-ir==0.1.14
pip install onnx==1.16.2
pip install onnxscript==0.5.7
```

2. 导出 HMONNX

    1. 导出方法：
        `python hmonnx_export_encoder.py` 包含导出 encoder 阶段的代码。
        `python hmonnx_export_prefill_decode.py` 包含导出 prefill 以及 decoder 阶段的代码，其中包含 kv cache 处理。

    > 注意： `max_audio_length` 参数指定导出时限制最大音频长度，大致可认为 100=1s，即 `--max_audio_length=1500` 表示 encoder 最大处理约 15 秒音频。对于累计音频的 streaming 推理，`max_audio_length` 需要覆盖整段音频时长。

    导出长度相关参数：

    - `--max_audio_length`：固定 encoder 的音频特征时间维度，约 `100 = 1s`。
    - `--prefix_token_budget`：只影响 prefill/decode 导出，为 streaming 文本 prefix 额外预留 token 长度。
    - `prefill_input_sequence_length`：prefill 固定输入长度，约等于：

      ```text
      audio_embed_lengths + 21 + prefix_token_budget
      ```

      其中 `audio_embed_lengths = _get_feat_extract_output_lengths(max_audio_length)`，`21` 是基础 prompt 中除音频 embedding 外的固定 token 预算。

    示例长度：

    | 目标累计音频时长 | `max_audio_length` | `audio_embed_lengths` | `prefix_token_budget` | `prefill_input_sequence_length` |
    | --- | ---: | ---: | ---: | ---: |
    | 15s | 1500 | 195 | 512 | 728 |
    | 45s | 4500 | 585 | 512 | 1118 |
    | 60s | 6000 | 780 | 512 | 1313 |

    选择规则：

    - 普通 `hmonnx_demo.py` 分段离线推理：`max_audio_length` 只需要覆盖单个 chunk 长度。
    - `hmonnx_demo_chunk_prefix.py` 对齐 vLLM streaming：因为每一步输入的是累计音频 `audio_accum`，`max_audio_length` 需要覆盖整段音频总时长。
    - 如果 `max_audio_length` 太小，encoder 会截断累计音频，结果会偏离 vLLM streaming。
    - 如果 `prefix_token_budget` 太小，文本 prefix 会在 prefill 输入末尾被截断，表现为每个 chunk 反复从头生成、结果重复。
    - `max_audio_length` 和 `prefix_token_budget` 越大，prefill 固定输入越长，推理和导出开销也越大。

    推荐导出顺序：

    ```bash
    # 1. 导出 encoder。示例为约 60 秒累计音频。
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data01/home/binghu.ji/nnew/xh2modelzoo \
    conda run -n xhquant python hmonnx_export_encoder.py \
      --model /data01/home/binghu.ji/models/Qwen/Qwen3-ASR-0.6B \
      --quant-type w8a8_sefp \
      --max_audio_length 6000
    ```

    如果需要在 streaming 推理中复用文本 prefix，导出 prefill/decode 时需要额外预留 prefix 长度：

    ```bash
    # 2. 导出 prefill/decode。max_audio_length 必须与 encoder 保持一致。
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data01/home/binghu.ji/nnew/xh2modelzoo \
    conda run -n xhquant python hmonnx_export_prefill_decode.py \
      --model /data01/home/binghu.ji/models/Qwen/Qwen3-ASR-0.6B \
      --config ./config/llm/qwen3_asr_decode_xh2a.py \
      --quant-type w8a8_sefp \
      --max_audio_length 6000 \
      --prefix_token_budget 512
    ```

    导出完成后，可以检查 `work_dirs/Qwen3-ASR-0.6B_XH2a/export_meta_info.json`：

    ```json
    {
      "max_audio_length": 6000,
      "audio_embed_lengths": 780,
      "prefix_token_budget": 512,
      "prefill_input_sequence_length": 1313
    }
    ```

3. 推理 Demo

    当前提供两种 HMONNX 推理方式：

    1. `hmonnx_demo.py`：分段离线推理

        - 每个音频 chunk 独立执行 `encoder -> prefill -> decode`。
        - chunk 内部使用 KV cache 做自回归 decode。
        - chunk 之间不传 KV cache，也不传文本 prefix。
        - 长音频通过 overlap 和文本去重合并最终结果。
        - 适合普通离线长音频切片识别。

        示例：

        ```bash
        CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data01/home/binghu.ji/nnew/xh2modelzoo \
        conda run -n xhquant python hmonnx_demo.py \
          --max_audio_length 1500 \
          --chunk_overlap_seconds 1.0 \
          --boundary_search_seconds 0.0
        ```

    2. `hmonnx_demo_chunk_prefix.py`：对齐 vLLM streaming 的累计音频 + 文本 prefix 推理

        - 每次收到新 chunk 后，将已知音频拼成 `audio_accum`。
        - chunk 只按 `chunk_seconds` 固定步长切分，不使用 overlap 或低能量边界搜索。
        - 每一步重新对 `audio_accum` 执行 encoder。
        - 第 `unfixed_chunk_num` 个 chunk 之后，将上一轮识别文本 rollback `rollback_tokens` 后作为文本 prefix 拼进 prompt。
        - 每一步重新执行 prefill/decode，chunk 之间不传 KV cache。
        - prefix 是文本约束复用，不是计算 cache 复用。
        - 适合验证和复现 Qwen3-ASR vLLM streaming 的行为。

        示例：

        ```bash
        CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data01/home/binghu.ji/nnew/xh2modelzoo \
        conda run -n xhquant python hmonnx_demo_chunk_prefix.py \
          --audio /data01/home/binghu.ji/new/xh2modelzoo/examples/audio/qwen3_asr/1output.wav \
          --max_audio_length 6000 \
          --chunk_seconds 20.0 \
          --max_new_tokens 128
        ```

        如果已经激活 `xhquant` 环境，建议使用 `python -u` 获得实时日志：

        ```bash
        CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data01/home/binghu.ji/nnew/xh2modelzoo \
        python -u hmonnx_demo_chunk_prefix.py \
          --audio /data01/home/binghu.ji/new/xh2modelzoo/examples/audio/qwen3_asr/1output.wav \
          --max_audio_length 6000 \
          --chunk_seconds 20.0 \
          --max_new_tokens 128
        ```

        英文音频需要显式切换语言，并关闭中文过滤：

        ```bash
        CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data01/home/binghu.ji/nnew/xh2modelzoo \
        python -u hmonnx_demo_chunk_prefix.py \
          --audio /data01/home/binghu.ji/new/xh2modelzoo/examples/audio/qwen3_asr/output.wav \
          --language English \
          --no_reject_non_chinese_hallucination
        ```

4. 其他
    依赖文件：`xh2modelzoo/xh_model_zoo/xh_llm/models/qwen3_asr`
