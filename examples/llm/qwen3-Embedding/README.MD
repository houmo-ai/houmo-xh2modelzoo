# Qwen3-Embedding


## 文件说明
- `native_qwen3_embeding.py`：原生 FP16 推理示例。
- `qwen3_embedding_common_quant.py`：QUAROT + GPTQ 量化主脚本。
- `gen_calib_mteb.py`：从 MTEB / C‑MTEB 生成 `jsonl` 校准集（检索或全任务，`--all-tasks`）。

- `qwen3_embedding_quarot_demo.py`：加载 QUAROT 权重的推理示例。
- `qwen3_embedding_quarot_gptq_demo.py`：加载 QUAROT+GPTQ 权重的推理示例。
- `qwen3_embedding_quarot_compare.py`：对比原始 vs QUAROT 输出（FP16）。
- `qwen3_embedding_quarot_compare_fp32.py`：对比原始 vs QUAROT 输出（FP32）。

- `qwen3_embedding_ptq_quant.py`：0.6B 的 PTQ 量化流程（xhquant_llm），产出 `ptq-state-dict.safetensors`。
- `qwen3_embedding_xh2a_export_hmonnx.py`：导出 XH2a / HMONNX（支持 PTQ、量化权重加载）。
- `qwen3_embedding_xh2a_hmonnx_test.py`：HMONNX 推理与对比（可与 HF 模型比较）。


## 1) 生成校准集（MTEB）
中文检索（C‑MTEB）示例：
```bash
python xh2modelzoo/examples/llm/qwen3-Embedding/gen_calib_mteb.py \
  --mteb-benchmark C-MTEB \
  --mteb-split test \
  --mteb-max-per-task 64 \
  --use-instruction \
  --instruction-template "Instruct: {}\nQuery:" \
  --task-prompts /data01/home/feilong.kong/xhquant_llm/examples/qwen3_embeding/qwen3_embeding_eval/task_prompts.json \
  --out xh2modelzoo/work_dirs/calib_mteb.jsonl
```

中文全任务（C‑MTEB）示例：
```bash
python examples/llm/qwen3-Embedding/gen_calib_mteb.py \
  --mteb-benchmark C-MTEB \
  --mteb-split test \
  --mteb-max-per-task 64 \
  --all-tasks \
  --use-instruction \
  --instruction-template "Instruct: {}\nQuery:" \
  --task-prompts /data01/home/feilong.kong/xhquant_llm/examples/qwen3_embeding/qwen3_embeding_eval/task_prompts.json \
  --out xh2modelzoo/work_dirs/calib_mteb_all.jsonl
```

英文检索（MTEB）示例：
```bash
python examples/llm/qwen3-Embedding/gen_calib_mteb.py \
  --mteb-benchmark "MTEB(eng, v2)" \
  --mteb-split test \
  --mteb-max-per-task 64 \
  --use-instruction \
  --instruction-template "Instruct: {}\nQuery:" \
  --task-prompts /data01/home/feilong.kong/xhquant_llm/examples/qwen3_embeding/qwen3_embeding_eval/task_prompts.json \
  --out xh2modelzoo/work_dirs/calib_mteb_en.jsonl
```

## 2) QUAROT + GPTQ 量化
```bash
python examples/llm/qwen3-Embedding/qwen3_embedding_common_quant.py \
  --model /data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-4B \
  --calib-dataset /data01/home/feilong.kong/xh2modelzoo/xh2modelzoo/work_dirs/calib_mteb_all.jsonl

python examples/llm/qwen3-Embedding/qwen3_embedding_common_quant.py \
  --model /data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-8B \
  --calib-dataset /data01/home/feilong.kong/xh2modelzoo/xh2modelzoo/work_dirs/calib_mteb_all.jsonl
  
```


## 3) 推理验证
QUAROT 权重推理：
```bash
python xh2modelzoo/examples/llm/qwen3-Embedding/qwen3_embedding_quarot_demo.py \
  --model-dir /data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-4B \
  --rot-ckpt /data01/home/feilong.kong/xh2modelzoo/work_dirs/Qwen3-Embedding-4B_quarot_gptq_transformers-4.57.3/quarot-state-dict.safetensors
```

QUAROT+GPTQ 权重推理：
```bash
python xh2modelzoo/examples/llm/qwen3-Embedding/qwen3_embedding_quarot_gptq_demo.py \
  --model-dir /data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-4B \
  --ckpt /data01/home/feilong.kong/xh2modelzoo/work_dirs/Qwen3-Embedding-4B_quarot_gptq_transformers-4.57.3/quarot_gptq-state-dict.safetensors
```

## 4) 输出对比
```bash
python xh2modelzoo/examples/llm/qwen3-Embedding/qwen3_embedding_quarot_compare.py \
  --model-dir /data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-4B \
  --rot-ckpt /data01/home/feilong.kong/xh2modelzoo/work_dirs/Qwen3-Embedding-4B_quarot_gptq_transformers-4.57.3/quarot-state-dict.safetensors
```

## 5) 0.6B PTQ 量化（xhquant_llm）
`qwen3_embedding_ptq_quant.py` 用于 **Qwen3‑Embedding‑0.6B** 的 PTQ 流程（W8A8）。示例：
```bash
python xh2modelzoo/examples/llm/qwen3-Embedding/qwen3_embedding_ptq_quant.py \
  --model /data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-0.6B \
  --calib-dataset /data01/home/feilong.kong/xh2modelzoo/work_dirs/calib_mteb_all.jsonl \
  --calib-samples 128 \
  --context-length 2048 \
  --input-seq-len 256 \
  --out-dir work_dirs/qwen3_embedding_0.6b_ptq
```
输出：`ptq-state-dict.safetensors`

## 6) XH2a/HMONNX 导出与测试
### 6.1 导出（XH2a）
```bash
python xh2modelzoo/examples/llm/qwen3-Embedding/qwen3_embedding_xh2a_export_hmonnx.py \
  --model /data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-4B \
  --context-length 2048 \
  --input-sequence-length 256 \
  --quant-type w4a8h0_ssfp \
  --quant-weight /data01/home/feilong.kong/xh2modelzoo/work_dirs/Qwen3-Embedding-4B_quarot_gptq_transformers-4.57.3/quarot_gptq-state-dict.safetensors

python examples/llm/qwen3-Embedding/qwen3_embedding_xh2a_export_hmonnx.py \
  --model /data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-4B \
  --context-length 2048 \
  --input-sequence-length 256 \
  --quant-type w4a8h0_ssfp \
  --quant-weight /data01/home/feilong.kong/xh2modelzoo/work_dirs/Qwen3-Embedding-4B_quarot_gptq_transformers-4.57.3/quarot_gptq-state-dict.safetensors

python examples/llm/qwen3-Embedding/qwen3_embedding_xh2a_export_hmonnx.py \
  --model /data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-8B \
  --quant-weight /data01/home/feilong.kong/xh2modelzoo/work_dirs/Qwen3-Embedding-8B_quarot_gptq_transformers-4.57.3/quarot_gptq-state-dict.safetensors \
  --context-length 2048 \
  --input-sequence-length 256 \
  --quant-type w4a8h0_ssfp \
  --export-date 20260309
  python examples/llm/qwen3-Embedding/qwen3_embedding_xh2a_export_hmonnx.py \
  --model /data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-4B \
  --context-length 2048 \
  --input-sequence-length 256 \
  --quant-type w4a8h0_ssfp \
  --quant-weight /data01/home/feilong.kong/xh2modelzoo/work_dirs/Qwen3-Embedding-4B_quarot_gptq_transformers-4.57.3/quarot_gptq-state-dict.safetensors \
  --export-date 20260309
  python /data01/home/feilong.kong/xh2modelzoo/examples/llm/qwen3-Embedding/qwen3_embedding_xh2a_export_hmonnx.py   --model /data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-0.6B   --context-length 2048   --input-sequence-length 256   --ptq --export-date 20260309
```


### 6.2 HMONNX 推理/对比
```bash
python xh2modelzoo/examples/llm/qwen3-Embedding/qwen3_embedding_xh2a_hmonnx_test.py \
  --config work_dirs/Qwen3-Embedding-4B-XH2a-2k-w4a8h0_ssfp/meta.json \
  --compare \
  --quant-weight /data01/home/feilong.kong/xh2modelzoo/work_dirs/Qwen3-Embedding-4B_quarot_gptq_transformers-4.57.3/quarot_gptq-state-dict.safetensors
```

