# VoxCPM2 Merak Workflow 示例

本目录提供 VoxCPM2 迁移到 `xhmodel_merak/xh_other_model` 后的导出入口、HMONNX 推理 demo、流式对齐脚本和 demo suite。所有命令默认从仓库根目录执行。

## 导出

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/voxcpm2/voxcpm2_workflow.py \
  --model-dir /data01/nfs_shared/ASR_TTS/VoxCPM2 \
  --config-path configs_merak/workflows/xh2a/other_models/voxcpm2/default/voxcpm2_xh2a.yaml \
  --device cuda \
  --overwrite
```

导出入口始终一次生成符合《HM模型版本发布命名规则》的完整目录。`--export-output-dir`
表示规范发布目录的父目录，默认是 `work_dirs`，因此上面的命令会直接生成：

```text
work_dirs/hmquant_xh2_voxcpm2_wmix_amix_256_1k_<date>/
```

其中四个 LM 图分别使用根目录下的 `baselm_prefill`、`baselm_decode`、
`residuallm_prefill` 和 `residuallm_decode` 目录；其他组件同样各自使用独立目录。
所有 host 侧 `.pt` 文件都直接位于发布根目录，embedding 文件名为 `quant_embedding.pt`，
不再生成 `host_modules` 中间目录。
`hf_config` 不复制模型仓库的 README；各组件目录只保留 HMONNX、external_data 和可选
`step_0`，不再生成组件级 `*_meta_info.json`，组件信息统一记录在根目录 manifest 中。

如需修改父目录可传 `--export-output-dir <dir>`；不再需要单独的 release 模式参数。
`--dump-golden` 是可选项。添加后 workflow 会先完成一次规范 HMONNX 导出，再调用标准
`dump_golden(export_result, device)` 接口直接加载发布目录中的 HMONNX，为各组件生成 `step_0`；
不会重新量化或重新导出模型。也可以在已有 `ExportResult` 上单独调用该接口刷新 golden。

`--device` 是统一的导出执行设备，会传给所有组件，并用于模型初始化、KV cache、导出张量
和 golden 生成。可使用 `cpu`、`cuda` 或 `cuda:N`；`cuda:N` 还会被设为当前 CUDA device，
因此依赖内部未带编号的 `cuda` 分配也会落到同一张卡。`N` 是当前进程可见的逻辑 GPU 编号。

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
也可用 `--quant-type w8a8_sefp` 覆盖支持量化的组件类型。入口会优先走标准 `AutoWorkflow`，
如果本地环境缺少可选 `xh_llm` 依赖，会自动回退到 `AutoOtherModelWorkflow`。

指定日期并同时生成 golden 的完整示例：

```bash
PYTHONPATH=$PWD python examples_merak/tts/voxcpm2/voxcpm2_workflow.py \
  --model-dir /data01/nfs_shared/ASR_TTS/VoxCPM2 \
  --config-path configs_merak/workflows/xh2a/other_models/voxcpm2/default/voxcpm2_xh2a.yaml \
  --device cuda:6 \
  --dump-golden \
  --release-date 20260715
```

当前默认配置中，AudioVAE Encoder 使用 `w16a16`，其余组件使用 `w8a8`。因此发布
根目录使用 `wmix_amix`，各组件内部文件名使用该组件对应的固定比特 `w16a16` 或 `w8a8`。

## HMONNX Demo

非流式推理使用 `AudioVAE_Decoder_np128`：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/voxcpm2/hmonnx_demo.py \
  --work_dir work_dirs/VoxCPM2_XH2a \
  --text "你好，这是 VoxCPM2 HMONNX 非流式 demo 验证。" \
  --output work_dirs/VoxCPM2_XH2a/demo_outputs/non_streaming.wav \
  --audio_encoder_backend hmonnx \
  --inference_timesteps 4 \
  --min_len 2 \
  --max_len 24
```

带参考音频的推理会额外使用 `AudioVAE_Encoder_np128`：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/voxcpm2/hmonnx_demo.py \
  --work_dir work_dirs/VoxCPM2_XH2a \
  --text "这是一段使用参考音色合成的语音。" \
  --reference_wav /data01/nfs_shared/ASR_TTS/CAM++/examples/speaker1_b_cn_16k.wav \
  --output work_dirs/VoxCPM2_XH2a/demo_outputs/reference.wav \
  --audio_encoder_backend hmonnx \
  --inference_timesteps 4 \
  --min_len 2 \
  --max_len 24
```

## 流式推理

`hmonnx_demo.py` 支持两种流式后端：

- `--streaming_backend stateful`: 推荐路径。使用 `AudioVAE_Decoder_StreamState_np1`，每次只输入最新 1 个 latent patch，并在 HMONNX 图输入/输出间传递 decoder states，对齐原生 `audio_vae.streaming_decode()` 的接口。
- `--streaming_backend overlap`: 兼容路径。使用 `AudioVAE_Decoder_np3`，每步输入带重叠窗口的 latent，再做 overlap/crop 拼接；实现简单，但不是真正的 state cache 流式。

真流式 demo：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/voxcpm2/hmonnx_demo.py \
  --work_dir work_dirs/VoxCPM2_XH2a \
  --text "你好，这是 VoxCPM2 HMONNX 真流式 stateful demo 验证。" \
  --output work_dirs/VoxCPM2_XH2a/demo_outputs/streaming_stateful.wav \
  --streaming \
  --streaming_backend stateful \
  --audio_encoder_backend hmonnx \
  --torch_audio_model_dir /data01/nfs_shared/ASR_TTS/VoxCPM2 \
  --inference_timesteps 4 \
  --min_len 2 \
  --max_len 24
```

旧 overlap 流式 demo：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/voxcpm2/hmonnx_demo.py \
  --work_dir work_dirs/VoxCPM2_XH2a \
  --text "你好，这是 VoxCPM2 HMONNX overlap 流式 demo 验证。" \
  --output work_dirs/VoxCPM2_XH2a/demo_outputs/streaming_overlap.wav \
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
  --work_dir work_dirs/VoxCPM2_XH2a \
  --stateful_decoder_dir work_dirs/VoxCPM2_XH2a/AudioVAE_Decoder_StreamState_np1 \
  --torch_audio_model_dir /data01/nfs_shared/ASR_TTS/VoxCPM2 \
  --audio_encoder_backend hmonnx \
  --output_dir work_dirs/VoxCPM2_XH2a/demo_outputs/stateful_align \
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
  --work-dir work_dirs/VoxCPM2_XH2a \
  --model-dir /data01/nfs_shared/ASR_TTS/VoxCPM2 \
  --reference-wav /data01/nfs_shared/ASR_TTS/CAM++/examples/speaker1_b_cn_16k.wav \
  --output-dir work_dirs/VoxCPM2_XH2a/demo_outputs/full_demo_suite \
  --audio-encoder-backend hmonnx \
  --inference-timesteps 4 \
  --min-len 2 \
  --max-len 24
```

短 smoke test 可把 `--inference-timesteps` 和 `--max-len` 调小，只用于确认各入口能加载图并落盘，不用于判断最终音质。

## 注意事项

- YAML 中 `lm.skip_verify: true` 只跳过导出阶段的 LM PyTorch/HMONNX 对齐验证，不影响 demo 推理。
- `locenc.cal_wav` 为空时会使用随机校准数据，适合快速打通导出链路；正式精度评估建议指定真实音频校准集。
- `model.wrap_cfg` 是 BaseLM 和 ResidualLM 共用的唯一长度配置源：`input_sequence_length` 控制 prefill 图长度，`max_sequence_length` 直接控制 KV cache 总容量。当前两者分别为 256 和 1024，prefill 和 decode 共用这 1024 个位置。
