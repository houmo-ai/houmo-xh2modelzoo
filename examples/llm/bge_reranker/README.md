# BGE-Reranker

## 配置参数

```text
batch_size
context-length 
quant-type: for example: w8a8-sefp
```

## 导出HMONNX

### w8a8

#### 1. 导出

```bash
python examples/llm/bge_reranker/bge_reranker_xh2a_export_hmonnx.py --model data/models/bge-large-zh-v1.5 --context-length 512 --batch-size 10 --quant-type w8a8_sefp
```

#### 2. GPU仿真

```bash
python examples/llm/bge_reranker/bge_reranker_xh2a_test.py --config work_dirs/bge-large-zh-v1.5-XH2a/meta.json
```
