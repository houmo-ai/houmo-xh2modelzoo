# VoxCPM2 Merak Workflow 示例

本目录提供 VoxCPM2 迁移到 `xhmodel_merak/xh_other_model` 后的导出入口、HMONNX 推理 demo、流式对齐脚本和 demo suite。所有命令默认从仓库根目录执行。

## 导出

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/voxcpm2/voxcpm2_workflow.py \
  --model-dir <model_dir> \
  --config-path configs_merak/workflows/xh2a/other_models/voxcpm2/default/voxcpm2_xh2a.yaml \
  --cal-wav <audio_file> \
  --device cuda \
  --overwrite
```

`--export-output-dir` 表示最终产物目录，不会再追加一层自动生成的目录。例如传入：

```text
--export-output-dir work_dirs/hmquant_xh2_voxcpm2_wmix_amix_256_1k_<date>
```

workflow 会先在同级临时目录完成各 exporter 的原始导出，再执行一次 release 整理，
以延续历史发布产物的目录和文件命名。最终根目录使用 `baselm_prefill`、
`baselm_decode`、`residuallm_prefill`、`residuallm_decode`、`locenc`、`locdit` 以及
各 AudioVAE 组件目录；图文件继续使用 `hmquant_*_<component>_with_act.onnx` 命名。
embedding 保存为 `quant_embedding.pt`，其余 host 权重按 release prefix 放在根目录。
临时原始导出目录会在 release 完成后删除。

发布根目录的固定入口仍为 `export_meta_info.json`，其中使用相对路径完整索引实际
YAML、HF config、embedding、host 权重、10 张 HMONNX 图、external data 和可选
golden。demo/eval 只根据该文件装配组件，不猜测目录名。发布目录不生成带前缀的
`*_manifest.json` 或 `golden_meta_info.json`；golden 状态直接记录在
`export_meta_info.json` 中。

目录名由调用方一次性确定，不再需要单独的 release 模式参数。
`--dump-golden` 是可选项。添加后 workflow 会先完成 HMONNX 导出，再调用标准
`dump_golden(export_result, device)` 接口，根据最终图的静态输入契约构造确定性输入并为各组件
生成 `step_0`。该阶段只运行已有 HMONNX，不重新加载原始模型，也不重新量化或导出。

`--device` 是统一的导出执行设备，会传给所有组件，并用于模型初始化、KV cache、导出张量
和 golden 生成。可使用 `cpu`、`cuda` 或 `cuda:N`；`cuda:N` 还会被设为当前 CUDA device，
因此依赖内部未带编号的 `cuda` 分配也会落到同一张卡。`N` 是当前进程可见的逻辑 GPU 编号。
完整精度导出应传入 `--cal-wav`，该音频会同时用于 LocEnc 和 AudioVAE Encoder 校准。

默认配置会导出以下组件：

- `lm`: BaseLM/ResidualLM prefill 和 decode 图。
- `locenc`: 文本/audio 条件编码图。
- `locdit`: flow matching step 图。
- `audiovae_encoder`: reference/prompt wav 编码图。
- `audiovae_decoder_full`: 非流式整段 decoder 图。
- `audiovae_decoder_stream`: 旧 overlap/crop 流式 decoder 图。
- `audiovae_decoder_stateful`: 真流式 stateful decoder 图。

`enc_to_lm_proj` 已融合在 `locenc` 图中，因此不再重复保存对应 `.pt`。其余 host
侧模块仍以根目录 `.pt` 形式发布，包括 `lm_to_dit_proj`、`res_to_dit_proj`、
`fusion_concat_proj`、`fsq_layer`、`stop_proj` 和 `stop_head`。

`export.components` 必须显式配置，写哪些组件就导出哪些；列出全部组件时生成完整发布目录，
未配置或配置为空会直接报错。命令行可用 `--components lm,locenc` 临时覆盖为部分导出，
也可用 `--quant-type w8a8_sefp` 覆盖支持量化的组件类型。入口统一使用顶层
`AutoWorkflow`。

同时生成 golden 的完整示例：

```bash
PYTHONPATH=$PWD python examples_merak/tts/voxcpm2/voxcpm2_workflow.py \
  --model-dir <model_dir> \
  --config-path configs_merak/workflows/xh2a/other_models/voxcpm2/default/voxcpm2_xh2a.yaml \
  --cal-wav <audio_file> \
  --device cuda:0 \
  --dump-golden \
  --overwrite
```

当前默认配置中，AudioVAE Encoder 使用 `w16a16`，其余组件使用 `w8a8`。因此发布
根目录使用 `wmix_amix`，各组件内部文件名使用该组件对应的固定比特 `w16a16` 或 `w8a8`。

## HMONNX Demo

非流式推理使用 `AudioVAE_Decoder_np128`：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/voxcpm2/hmonnx_demo.py \
  --work_dir <output_dir> \
  --text "你好，这是 VoxCPM2 HMONNX 非流式 demo 验证。" \
  --output <demo_output_dir>/non_streaming.wav \
  --audio_encoder_backend hmonnx \
  --inference_timesteps 4 \
  --min_len 2 \
  --max_len 24
```

带参考音频的推理会额外使用 `AudioVAE_Encoder_np128`：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/voxcpm2/hmonnx_demo.py \
  --work_dir <output_dir> \
  --text "这是一段使用参考音色合成的语音。" \
  --reference_wav <audio_file> \
  --output <demo_output_dir>/reference.wav \
  --audio_encoder_backend hmonnx \
  --inference_timesteps 4 \
  --min_len 2 \
  --max_len 24
```

## 流式推理

`hmonnx_demo.py` 支持两种流式后端：

- `--streaming_backend stateful`: 推荐路径。通过顶层 metadata 定位 stateful decoder，每次只输入最新 latent patch，并在 HMONNX 图输入/输出间传递 decoder states，对齐原生 `audio_vae.streaming_decode()` 的接口。
- `--streaming_backend overlap`: 兼容路径。使用 `AudioVAE_Decoder_np3`，每步输入带重叠窗口的 latent，再做 overlap/crop 拼接；实现简单，但不是真正的 state cache 流式。

真流式 demo：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/voxcpm2/hmonnx_demo.py \
  --work_dir <output_dir> \
  --text "你好，这是 VoxCPM2 HMONNX 真流式 stateful demo 验证。" \
  --output <demo_output_dir>/streaming_stateful.wav \
  --streaming \
  --streaming_backend stateful \
  --audio_encoder_backend hmonnx \
  --torch_audio_model_dir <model_dir> \
  --inference_timesteps 4 \
  --min_len 2 \
  --max_len 24
```

旧 overlap 流式 demo：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/voxcpm2/hmonnx_demo.py \
  --work_dir <output_dir> \
  --text "你好，这是 VoxCPM2 HMONNX overlap 流式 demo 验证。" \
  --output <demo_output_dir>/streaming_overlap.wav \
  --streaming \
  --streaming_backend overlap \
  --audio_encoder_backend hmonnx \
  --inference_timesteps 4 \
  --min_len 2 \
  --max_len 24
```

stateful decoder 对齐脚本会比较 HMONNX stateful decoder 和 PyTorch `audio_vae.streaming_decode()` 的逐 chunk 输出，并保存报告：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/voxcpm2/voxcpm2_stateful_streaming_align.py \
  --work_dir <output_dir> \
  --stateful_decoder_dir <output_dir> \
  --torch_audio_model_dir <model_dir> \
  --audio_encoder_backend hmonnx \
  --output_dir <demo_output_dir>/stateful_align \
  --inference_timesteps 4 \
  --min_len 2 \
  --max_len 24 \
  --streaming_prefix_len 4
```

报告文件为 `stateful_streaming_alignment_report.json`，其中 `metrics.hmonnx_stateful_streaming_vs_torch_true_streaming` 会给出整段音频的长度、`mean_abs` 和 `cosine`。

## Demo Suite

`run_demo_suite.py` 保留完整的 01 到 07 用例：zero-shot、reference clone、prompt continuation、reference+prompt、stateful streaming、HMONNX audio encoder reference，以及 PyTorch 端到端 align。套件还会运行 `08_stateful_streaming_align/`，逐 chunk 对齐 HMONNX stateful decoder 与 PyTorch 原生流式 decoder，并将报告写入顶层 `summary.json` 的 `stateful_align` 字段。

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> \
PYTHONPATH=$PWD/examples_merak/tts/voxcpm2:$PWD \
python examples_merak/tts/voxcpm2/run_demo_suite.py \
  --work-dir <output_dir> \
  --model-dir <model_dir> \
  --reference-wav <audio_file> \
  --output-dir <demo_output_dir>/full_demo_suite \
  --audio-encoder-backend hmonnx \
  --inference-timesteps 4 \
  --min-len 2 \
  --max-len 24
```

短 smoke test 可把 `--inference-timesteps` 和 `--max-len` 调小，只用于确认各入口能加载图并落盘，不用于判断最终音质。

## 注意事项

- YAML 中 `lm.skip_verify: true` 只跳过导出阶段的 LM PyTorch/HMONNX 对齐验证，不影响 demo 推理。
- `locenc.cal_wav` 为空时会使用随机校准数据，适合快速打通导出链路；正式精度评估建议指定真实音频校准集。
- `lm.wrap_cfg` 是 BaseLM 和 ResidualLM 共用的唯一长度配置源：`input_sequence_length` 控制 prefill 图长度，`max_sequence_length` 直接控制 KV cache 总容量。当前两者分别为 256 和 1024，prefill 和 decode 共用这 1024 个位置。
