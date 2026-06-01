# 校准集选择与生成

## 当前量化模型参考校准集

已支持并提供的有：

* Qwen2-7B-Instruct
* Qwen2.5-7B-Instruct
* Qwen2.5-14B-Instruct
* Deepseek-Qwen-7B
* Qwen3-8B
* Qwen3-30B-A3B
* Qwen3-Coder-30B-A3B

## 使用自己的模型生成校准集

推荐使用参数化命令（支持你自己提供问题）：

```bash
python examples/xh_gen_data/gen_data.py \
	--prefix-name example_data \
	--model-path ./model/Qwen3-8B/ \
	--gen-seqlen 512 \
	--num-samples 512 \
	--questions-file ./my_questions.txt
```

其中 `--questions-file` 支持：

* `.txt`：每行一个问题
* `.jsonl`：每行一个 JSON，字段可为 `question` / `prompt` / `text`

脚本会生成固定 token 长度（`--gen-seqlen`）的数据，输出路径为：

* `./gen_data/<prefix-name>/<gen-seqlen>.jsonl`

兼容旧命令（仍可用）：

```
python examples/xh_gen_data/gen_data.py ${文件名} ${浮点大模型对应路径} ${校准集数量，默认512}
```

举例：
```
python examples/xh_gen_data/gen_data.py example_data ./model/Qwen3-8B/ 512
```

则会在当前目录 `./gen_data/example_data/512.jsonl` 看到输出文件，里面每一行都是一条字典，以 `text` 为 key。


## 校准集的使用

目前提供两种使用方式 

### 利用当前modelzoo的量化工具

在校准集选项中改为校准集路径即可，默认为wikitext2
```
python examples/llm/qwen3/qwen3_xh2a_common_quant.py --model /data02/datasets/Qwen3-8B --calib_data xxx/gen_qwen3_8b.jsonl
```


### 利用gptqmodel

直接参考gptqmodel中的example示例，将生成数据路径填入对应的位置即可