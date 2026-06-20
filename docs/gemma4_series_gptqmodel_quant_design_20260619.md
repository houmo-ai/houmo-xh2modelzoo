# Gemma4 Series GPTQModel 量化与导出方案（2026-06-19）

## 结论

Gemma4 Series 的 Merak workflow 固定为与 Qwen3.5 一致的边界：

```text
workflow YAML
  -> Gemma4SeriesWorkflow.quant()
  -> xh2modelzoo 薄 adapter
  -> GPTQModel Gemma4 recipe
  -> GPTQModel HF quant dir + provenance
  -> Gemma4SeriesWorkflow.export()
  -> to_quanted_aligned / HMONNX / generate 验收
```

- 默认推荐 YAML 不再是 base-only；`quant.algorithm=gptqmodel`。
- 底层默认方法为 `method=gptq`，产物格式固定 `gptqmodel_hf`。
- xh2modelzoo 不重写 GPTQ 量化算法，只做配置校验、recipe 调用和 provenance。
- base 导出只允许显式传 `config_overrides={"quant": None}`，并且 export config 的 `quant_scheme` 仍保持 `w8a8h1_sefp`。

## 公共量化配置

默认配置：

```yaml
quant:
  algorithm: gptqmodel
  method: gptq
  preset: full_multimodal
  rotation: null
  artifact_format: gptqmodel_hf
  output_format: gptqmodel_hf
  bits: 4
  group_size: 64
  sym: true
  iters: 200
  seed: 42
  quant_nontext_module: false
  calibration:
    dataset: wikitext
    split: train
    nsamples: 256
    seqlen: 2048
  runtime:
    batch_size: 1
    device_map: auto
    trust_remote_code: true
    offload_to_disk: false
  validation:
    check_quant_text_demo: true
    check_quant_image_demo: true
    check_quant_video_demo: true
    check_quant_audio_demo: true
```

支持入口：

1. `algorithm=gptqmodel`：主路径，调用 `gptqmodel.recipes.gemma4:quantize_gemma4`。
2. `algorithm=gptq`：兼容旧写法，映射到 `method=gptq` 后仍走 GPTQModel recipe。
3. `algorithm=existing_hf`：跳过量化，直接复用已有 GPTQModel HF 量化目录。
4. `config_overrides={"quant": None}`：只允许用于 base 模型验证。

## Gemma4 特有传参

adapter 会从 HF `config.json` 和 export YAML 推断：

- `variant`: `e4b` / `31b` / `26b_a4b` / `unknown`
- `topology`: `dense` / `moe`
- `capabilities`: text/image/video/audio
- `export_subgraphs`: image visual、独立 video visual、audio、per-layer input
- `context_max_length=2048`
- `prefill_chunk_length=256`

这些参数传给 GPTQModel recipe，由 recipe 决定 E4B audio/video、31B dense、26B-A4B MoE 的量化细节。

## 约束

- `artifact_format/output_format` 必须统一为 `gptqmodel_hf`。
- `group_size` 固定 64。
- rotated quant 不允许 `offload_to_disk=true`。
- 默认 recipe 入口为 `gptqmodel.recipes.gemma4:quantize_gemma4`；如 GPTQModel 临时入口不同，可在 quant YAML 中设置 `recipe_entrypoint`。
- HMONNX 验收仍必须按 AGENTS.md 的 Gemma4 全量规则执行：三模型、text+image、E4B audio/video、长 prompt、真实 generate、独立 video ViT。
