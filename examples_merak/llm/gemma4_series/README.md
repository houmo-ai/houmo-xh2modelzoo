# Gemma4 Series unified workflow demo

本目录只推荐新的统一 workflow API。E2B、E4B、31B dense、26B-A4B MoE 都从
同一个入口进入：

```text
Gemma4ForConditionalGeneration
  -> AutoLLMWorkflow.from_config(...)
  -> xhmodel_merak.xh_llm.models.gemma4_series.workflow.Gemma4SeriesWorkflow
  -> xhmodel_merak.xh_llm.models.gemma4_series.XHGemma4SeriesModel
```

旧 `gemma4/`、`gemma4e/`、`gemma4_moe/` 目录只作为历史兼容面；新的
Gemma4 Series 导出/生成不要再把它们当实现入口。

## 详细文档

完整结构差异、输入输出 shape、HMONNX 产物说明、量化/导出/demo/e2e 命令见：

- [Gemma4 Series Merak 统一 Workflow 使用与模型说明](../../../docs/gemma4_series_merak_workflow_guide_20260621.md)
- [Gemma4 ViT padded 输入定版方案](../../../docs/gemma4_vit_padded_input_design_20260616.md)

README 只保留最小入口和常用命令，避免和长期文档重复。

## 推荐配置

| preset | HF checkpoint | workflow YAML |
| --- | --- | --- |
| `e2b` | `/data01/datasets/gemma-4-E2B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml` |
| `e4b` | `/data01/datasets/gemma-4-E4B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml` |
| `31b` | `/data01/datasets/gemma-4-31B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full.yaml` |
| `26b-a4b` | `/data01/datasets/gemma-4-26B-A4B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml` |

四份 YAML 都使用同一个 public model type：

```yaml
model_type: Gemma4ForConditionalGeneration
prefill_chunk_length: 256
sliding_kv_cache_input_mode: slice_window
```

正式 QTL-384 集成导出请用 `--context-max-length 8192` 覆盖 YAML 默认值。

## 当前推荐最优权重

| preset | 推荐量化 | 环境变量 |
| --- | --- | --- |
| `e2b` | AutoRound | `GEMMA4_E2B_AUTOROUND_HF` |
| `e4b` | GPTQModel | `GEMMA4_E4B_GPTQMODEL_HF` |
| `26b-a4b` | AutoRound | `GEMMA4_26B_A4B_AUTOROUND_HF` |
| `31b` | AutoRound | `GEMMA4_31B_AUTOROUND_HF` |

## 轻量解析验证

不会加载大模型权重，不会量化/导出：

```bash
python examples_merak/llm/gemma4_series/gemma4_workflow_demo.py \
  --preset e2b --dry-run
python examples_merak/llm/gemma4_series/gemma4_workflow_demo.py \
  --preset e4b --dry-run
python examples_merak/llm/gemma4_series/gemma4_workflow_demo.py \
  --preset 31b --dry-run
python examples_merak/llm/gemma4_series/gemma4_workflow_demo.py \
  --preset 26b-a4b --dry-run
```

## 一键从头量化、导出、dump golden、跑 demo

当前重型任务脚本就是下面这套：从原始 HF checkpoint 出发，生成 GPTQModel / AutoRound
两套 workflow config，按“单 GPU 单任务”调度 4 个模型 × 2 种量化，然后对每个产物跑
text / image / video demo；E2B/E4B 额外跑 audio demo。导出规格固定为
`context_max_length=8192`、`prefill_chunk_length=256`、`slice_window` KV cache、`--golden`。

> 如果只想跑当前最优组合，把 `TASKS` 改成：
> `('e2b','autoround'), ('e4b','gptq'), ('26b-a4b','autoround'), ('31b','autoround')`。

```bash
cat > /tmp/run_gemma4_quant_export_demo.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import yaml

REPO = Path('/data01/home/yujy/work/xh2modelzoo')
GPTQMODEL_REPO = REPO.parent / 'gptqmodel'
PYTHON = '/data01/home/yujy/miniconda3/envs/gemma4/bin/python'
ROOT = REPO / 'work_dirs' / f'qtl384_gemma4_quant_export_demo_{time.strftime("%Y%m%d_%H%M%S")}'
CONFIG_DIR = ROOT / 'configs'
LOG_DIR = ROOT / 'logs'

# 单 GPU 单任务；按机器空闲情况改这里即可。
GPUS = [0, 1, 5, 6, 7]
TASKS = [
    ('e2b', 'gptq'), ('e2b', 'autoround'),
    ('e4b', 'gptq'), ('e4b', 'autoround'),
    ('31b', 'gptq'), ('31b', 'autoround'),
    ('26b-a4b', 'gptq'), ('26b-a4b', 'autoround'),
]

PRESET_CONFIGS = {
    'e2b': REPO / 'configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml',
    'e4b': REPO / 'configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml',
    '31b': REPO / 'configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full.yaml',
    '26b-a4b': REPO / 'configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml',
}

DENSE_AUTOROUND_QUANT = {
    'algorithm': 'autoround',
    'method': 'autoround',
    'preset': 'mode1',
    'rotation': None,
    'artifact_format': 'gptqmodel_hf',
    'output_format': 'gptqmodel_hf',
    'bits': 4,
    'group_size': 64,
    'sym': True,
    'iters': 200,
    'seed': 42,
    'format': 'auto_gptq',
    'calibration': {
        'jsonl': 'gptqmodel://quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl',
        'text_key': 'text',
        'nsamples': 128,
        'seqlen': 512,
    },
    'runtime': {'batch_size': 8, 'trust_remote_code': True},
}
MOE_AUTOROUND_QUANT = {
    **DENSE_AUTOROUND_QUANT,
    'calibration': {
        'jsonl': 'gptqmodel://quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl',
        'text_key': 'text',
        'nsamples': 128,
        'seqlen': 512,
    },
    'runtime': {'batch_size': 8, 'dtype': 'bfloat16', 'trust_remote_code': True},
    'validation': {'prompt': '你是谁', 'max_new_tokens': 128},
}

LONG_PROMPT = (REPO / 'work_dirs/qtl384_gemma4_exports_8192_20260620/long_prompt.txt').read_text(encoding='utf-8')
IMAGE = REPO / '.omx/fixtures/gemma4_strict/strict_image.png'
VIDEO = REPO / '.omx/fixtures/gemma4_strict/video_frames'
AUDIO = REPO / '.omx/fixtures/gemma4_strict/secret_orange.wav'


def q(x: object) -> str:
    return shlex.quote(str(x))


def write_configs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for model, algo in TASKS:
        data = yaml.safe_load(PRESET_CONFIGS[model].read_text(encoding='utf-8'))
        if algo == 'autoround':
            data['quant'] = MOE_AUTOROUND_QUANT if model == '26b-a4b' else DENSE_AUTOROUND_QUANT
        cfg = CONFIG_DIR / f'{model}_{algo}.yaml'
        cfg.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')
        manifest.append({'model': model, 'algorithm': algo, 'config': str(cfg.relative_to(REPO))})
    (ROOT / 'manifest_configs.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def task_script(model: str, algo: str, gpu: int) -> tuple[str, Path, Path]:
    task = f'{model}_{algo}'.replace('-', '_')
    task_dir = ROOT / task
    task_dir.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG_DIR / f'{model}_{algo}.yaml'
    prompt_path = task_dir / 'long_prompt.txt'
    prompt_path.write_text(LONG_PROMPT, encoding='utf-8')
    supports_audio = model in {'e2b', 'e4b'}

    workflow_cmd = ' '.join([
        q(PYTHON), 'examples_merak/llm/gemma4_series/gemma4_workflow_demo.py',
        '--preset', q(model), '--action', 'quant-export', '--config-path', q(cfg),
        '--work-dir', q(task_dir), '--device', 'cuda:0',
        '--context-max-length', '8192', '--prefill-chunk-length', '256',
        '--sliding-kv-cache-input-mode', 'slice_window', '--golden', '--force',
        '--prompt', '"$(cat ' + q(prompt_path) + ')"',
    ])
    gen_base = f'{q(PYTHON)} examples_merak/llm/gemma4_series/generate.py --backend hmonnx --model-config "$META" --device cuda:0'
    lines = [
        '#!/usr/bin/env bash',
        'set -uo pipefail',
        f'cd {q(REPO)}',
        f'export CUDA_VISIBLE_DEVICES={gpu}',
        'export CUDA_DEVICE_ORDER=PCI_BUS_ID',
        'export TOKENIZERS_PARALLELISM=false',
        f'export GPTQMODEL_REPO={q(GPTQMODEL_REPO)}',
        f'export PYTHONPATH={q(REPO)}:{q(GPTQMODEL_REPO)}:${{PYTHONPATH:-}}',
        f'echo START $(date -Is) model={model} algo={algo} gpu={gpu}',
        workflow_cmd,
        'rc=$?',
        f'echo "$rc" > {q(task_dir / "workflow.rc")}',
        'echo WORKFLOW_RC=$rc $(date -Is)',
        'if [ "$rc" -ne 0 ]; then exit "$rc"; fi',
        f'META=$(find {q(task_dir)} -name golden_meta_info.json | sort | tail -1)',
        'echo META=$META',
        f'echo "$META" > {q(task_dir / "meta_path.txt")}',
        f'{gen_base} --prompt "$(cat {q(prompt_path)})" --max-decode-steps 128 > {q(task_dir / "demo_text.log")} 2>&1',
        f'echo $? > {q(task_dir / "demo_text.rc")}',
        f'{gen_base} --image-path {q(IMAGE)} --prompt "请描述图片中的文字、颜色、形状和布局，并回答图片主要表达什么。" --max-decode-steps 128 > {q(task_dir / "demo_image.log")} 2>&1',
        f'echo $? > {q(task_dir / "demo_image.rc")}',
        f'{gen_base} --video-path {q(VIDEO)} --video-num-frames 16 --prompt "请总结这段多帧视频画面的变化，并指出其中可见的文字、颜色或物体。" --max-decode-steps 128 > {q(task_dir / "demo_video.log")} 2>&1',
        f'echo $? > {q(task_dir / "demo_video.rc")}',
    ]
    if supports_audio:
        lines += [
            f'{gen_base} --audio-path {q(AUDIO)} --prompt "请转写并概括这段音频内容。" --max-decode-steps 128 > {q(task_dir / "demo_audio.log")} 2>&1',
            f'echo $? > {q(task_dir / "demo_audio.rc")}',
        ]
    lines += [f'echo 0 > {q(task_dir / "rc")}', 'echo DONE $(date -Is)']
    script = task_dir / 'run.sh'
    script.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    script.chmod(0o755)
    return task, task_dir, script


def launch(model: str, algo: str, gpu: int) -> dict:
    task, task_dir, script = task_script(model, algo, gpu)
    log = open(task_dir / 'run.log', 'ab', buffering=0)
    proc = subprocess.Popen(['bash', str(script)], cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT,
                            env=os.environ.copy(), start_new_session=True)
    (task_dir / 'pid').write_text(str(proc.pid), encoding='utf-8')
    return {'task': task, 'model': model, 'algo': algo, 'gpu': gpu, 'proc': proc,
            'dir': str(task_dir), 'start': time.time()}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_configs()
    pending = list(TASKS)
    running = []
    events = []
    print(f'ROOT={ROOT}', flush=True)
    while pending or running:
        for gpu in GPUS:
            if not pending:
                break
            if any(item['gpu'] == gpu for item in running):
                continue
            model, algo = pending.pop(0)
            item = launch(model, algo, gpu)
            running.append(item)
            events.append({'event': 'start', 'task': item['task'], 'gpu': gpu, 'time': time.strftime('%F %T')})
            (ROOT / 'scheduler_events.json').write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding='utf-8')
        time.sleep(30)
        still = []
        for item in running:
            rc = item['proc'].poll()
            if rc is None:
                still.append(item)
                continue
            events.append({'event': 'finish', 'task': item['task'], 'gpu': item['gpu'], 'rc': rc,
                           'elapsed': round(time.time() - item['start'], 1), 'time': time.strftime('%F %T')})
            Path(item['dir'], 'rc').write_text(str(rc), encoding='utf-8')
            (ROOT / 'scheduler_events.json').write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding='utf-8')
        running = still
    (ROOT / 'DONE').write_text(time.strftime('%F %T'), encoding='utf-8')
    print(f'DONE={ROOT}', flush=True)


if __name__ == '__main__':
    main()
PY

cd /data01/home/yujy/work/xh2modelzoo
/data01/home/yujy/miniconda3/envs/gemma4/bin/python /tmp/run_gemma4_quant_export_demo.py
```

进度查看：

```bash
ROOT=$(ls -td work_dirs/qtl384_gemma4_quant_export_demo_* | head -1)
cat "$ROOT/scheduler_events.json"
find "$ROOT" -name golden_meta_info.json -print | sort
for d in "$ROOT"/*_{gptq,autoround}; do
  [ -d "$d" ] || continue
  echo "=== ${d#$ROOT/} ==="
  for f in workflow.rc demo_text.rc demo_image.rc demo_video.rc demo_audio.rc rc; do
    [ -f "$d/$f" ] && printf '%s=' "$f" && cat "$d/$f"
  done
done
```

## 复用已有量化 HF 目录重新导出 HMONNX

如果量化权重已经存在，才使用这个入口跳过 quant 阶段。示例：E4B GPTQModel，8192 context，256 prefill，并为所有模块 dump golden。

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/gemma4_series/gemma4_workflow_demo.py \
  --preset e4b \
  --action existing-hf \
  --existing-hf-model-dir "$GEMMA4_E4B_GPTQMODEL_HF" \
  --work-dir ./work_dirs/qtl384_gemma4_best_exports/e4b_gptqmodel \
  --context-max-length 8192 \
  --prefill-chunk-length 256 \
  --sliding-kv-cache-input-mode slice_window \
  --golden \
  --force
```

完整四模型命令见详细文档。

## 真实 generate demo

Text：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/gemma4_series/generate.py \
  --backend hmonnx \
  --model-config /path/to/golden_meta_info.json \
  --prompt "请阅读下面长资料卡并回答最后的问题：..." \
  --max-decode-steps 128
```

Image：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/gemma4_series/generate.py \
  --backend hmonnx \
  --model-config /path/to/golden_meta_info.json \
  --image-path /path/to/real_image.png \
  --prompt "请用中文描述图片中的文字、颜色、形状和布局。" \
  --max-decode-steps 128
```

Video / Audio：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/gemma4_series/generate.py \
  --backend hmonnx \
  --model-config /path/to/golden_meta_info.json \
  --video-path /path/to/video_or_frames \
  --prompt "请总结视频中多帧画面的变化。" \
  --max-decode-steps 128

CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/gemma4_series/generate.py \
  --backend hmonnx \
  --model-config /path/to/golden_meta_info.json \
  --audio-path /path/to/audio.wav \
  --prompt "请转写并概括音频内容。" \
  --max-decode-steps 128
```

Audio 只适用于 E2B/E4B。

## 严格验收规则摘要

- text prompt token 必须大于 1024，且 `prompt_tokens + max_new_tokens <= context`。
- 每个模型至少覆盖 text 和 image generate；E2B/E4B 还要覆盖 audio/video。
- image/video/audio 必须是真实多模态问答，不能用无关问题替代。
- video 必须经过独立 `video_visual` HMONNX。
- audio 必须经过 `audio` HMONNX。
- `py_compile`、dry-run、processor-only validation、meta 文件存在都不能替代 e2e generate。
- 单 GPU 同时只跑一个 Gemma4 重型导出/生成任务。
