"""Gradio-based web UI for the hm_eval unified evaluation platform.

Launch: python -m hm_eval --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from .core.model_registry import ModelRegistry
from .core.dataset_registry import DatasetRegistry
from .core.task_manager import TaskManager, TaskStatus
from .core.report import enrich_report_from_outputs, format_report_html, format_report_text

logger = logging.getLogger(__name__)

_model_registry: Optional[ModelRegistry] = None
_dataset_registry: Optional[DatasetRegistry] = None
_task_manager: Optional[TaskManager] = None
_report_render_cache: dict[str, tuple[float, str, str]] = {}

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_MULTIMODAL_DATASETS = {"cmmmu", "mmmu", "mmmu_pro", "math_vision", "omnidoc_bench"}
_PREDICT_PROGRESS_RE = re.compile(
    r"Predicting\[(?P<label>[^\]]+)\]:\s*(?P<pct>\d+)%\|.*?\|\s*"
    r"(?P<done>\d+)/(?P<total>\d+)\s*\[(?P<elapsed>[^<,\]]+)"
    r"(?:<(?P<eta>[^,\]]+))?,\s*(?P<rate>[^\]]+)\]"
)
_LOADING_PROGRESS_RE = re.compile(
    r"Loading weights:\s*(?P<pct>\d+)%\|.*?\|\s*(?P<done>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[^<,\]]+)(?:<(?P<eta>[^,\]]+))?,\s*(?P<rate>[^\]]+)\]"
)
_PROCESSING_SAMPLES_RE = re.compile(r"Processing (?P<total>\d+) samples")
_FAILURE_LINE_RE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*Error|Exception|RuntimeError|Traceback)(?::|\b)")
_STALLED_LOG_SECONDS = 600
_GPU_IDLE_USED_MB_THRESHOLD = 2048
_GPU_IDLE_UTIL_THRESHOLD = 10
_GPU_PARTIAL_USED_RATIO = 0.4
_GPU_PARTIAL_USED_MB_THRESHOLD = 24576
_GPU_PARTIAL_UTIL_THRESHOLD = 40

# ──────────────────────────────────────────────────────────
# CSS for a polished look
# ──────────────────────────────────────────────────────────
_CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; margin: auto; }
.header-bar { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
              padding: 24px 32px; border-radius: 12px; margin-bottom: 16px; color: white; }
.header-bar h1 { margin: 0; font-size: 28px; }
.header-bar p  { margin: 4px 0 0; opacity: 0.85; font-size: 14px; }
.status-badge  { display: inline-block; padding: 2px 10px; border-radius: 12px;
                 font-size: 13px; font-weight: 600; }
.st-pending    { background: #fff3cd; color: #856404; }
.st-exporting  { background: #d1ecf1; color: #0c5460; }
.st-running    { background: #cce5ff; color: #004085; }
.st-completed  { background: #d4edda; color: #155724; }
.st-abnormal   { background: #fff3cd; color: #8a5a00; }
.st-failed     { background: #f8d7da; color: #721c24; }
.report-card   { border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;
                 margin: 8px 0; background: #fafafa; }
.metric-box    { text-align: center; padding: 16px; border-radius: 8px;
                 background: white; border: 1px solid #e9ecef; }
.metric-value  { font-size: 28px; font-weight: 700; color: #495057; }
.metric-label  { font-size: 12px; color: #6c757d; margin-top: 4px; }
.monitor-status-bar { display: flex; justify-content: space-between; align-items: center;
                      gap: 12px; padding: 10px 14px; border-radius: 10px; margin: 8px 0 12px;
                      border: 1px solid transparent; font-size: 14px; }
.monitor-status-message { font-weight: 600; }
.monitor-status-time { font-size: 12px; opacity: 0.8; white-space: nowrap; }
.monitor-status-loading { background: #fff8e1; border-color: #f6c445; color: #8a5a00; }
.monitor-status-ready { background: #e8f5e9; border-color: #81c784; color: #1b5e20; }
.monitor-status-warning { background: #fdecea; border-color: #ef9a9a; color: #8e1c1c; }
.monitor-status-info { background: #e3f2fd; border-color: #90caf9; color: #0d47a1; }
.report-task-table-wrap { border: 1px solid #dee2e6; border-radius: 12px; overflow: hidden; background: #ffffff; }
.report-task-table { width: 100%; border-collapse: collapse; table-layout: auto; }
.report-task-table th, .report-task-table td { padding: 14px 16px; border-bottom: 1px solid #e9ecef; text-align: left; vertical-align: middle; }
.report-task-table th { background: #f8f9fa; color: #212529; font-size: 15px; font-weight: 700; }
.report-task-table tr:last-child td { border-bottom: none; }
.report-task-table tr:hover td { background: #fcfcfd; }
.report-task-id-btn { background: none; border: none; padding: 0; color: #0d6efd; font: inherit; font-weight: 700; cursor: pointer; }
.report-task-id-btn:hover { text-decoration: underline; }
.report-task-action-btn { border: 1px solid #ced4da; border-radius: 8px; background: #ffffff; color: #212529; padding: 6px 12px; font-size: 13px; font-weight: 600; cursor: pointer; }
.report-task-action-btn:hover { background: #f8f9fa; }
.report-task-action-btn.delete { border-color: #f1b0b7; background: #fff5f5; color: #b42318; }
.report-task-empty { text-align: center; padding: 28px 16px; color: #6c757d; }
.report-task-empty strong { color: #212529; display: block; margin-bottom: 8px; }
fieldset#dataset-checkbox-group,
fieldset#dataset-search-results {
    max-height: 16rem;
    overflow-y: auto;
    overscroll-behavior: contain;
    padding-right: 8px;
}
fieldset#dataset-checkbox-group label,
fieldset#dataset-search-results label {
    line-height: 1.45;
}
"""


def _init_registries() -> None:
    global _model_registry, _dataset_registry, _task_manager
    _model_registry = ModelRegistry()
    _model_registry.scan()
    _dataset_registry = DatasetRegistry()
    _task_manager = TaskManager()
    logger.info("Initialized: %d models, %d datasets",
                len(_model_registry.list_models()), len(_dataset_registry.list_all()))


def _query_gpu_info() -> list[dict]:
    """Query GPU status via nvidia-smi. Returns list of {id, name, mem_total, mem_used, mem_free, util}."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "id": parts[0], "name": parts[1],
                    "mem_total": parts[2], "mem_used": parts[3],
                    "mem_free": parts[4], "util": parts[5],
                })
        return gpus
    except Exception:
        return []


def _classify_gpu_status(gpu: dict[str, str]) -> tuple[str, str, int]:
    mem_total = max(float(gpu["mem_total"]), 1.0)
    mem_used = max(float(gpu["mem_used"]), 0.0)
    mem_free = max(float(gpu["mem_free"]), 0.0)
    util = max(float(gpu["util"]), 0.0)
    free_pct = int(mem_free / mem_total * 100)

    # Large-memory servers need an absolute-used-memory threshold; percentage-only heuristics
    # will mislabel a card with 10+ GiB occupied as "free" on an 80 GiB GPU.
    if mem_used <= _GPU_IDLE_USED_MB_THRESHOLD and util <= _GPU_IDLE_UTIL_THRESHOLD:
        return "🟢 空闲", "free", free_pct
    if mem_used <= min(mem_total * _GPU_PARTIAL_USED_RATIO, _GPU_PARTIAL_USED_MB_THRESHOLD) and util <= _GPU_PARTIAL_UTIL_THRESHOLD:
        return "🟡 部分占用", "partial", free_pct
    return "🔴 繁忙", "busy", free_pct


def _build_gpu_choices_from_info(gpus: list[dict]) -> list[str]:
    if not gpus:
        return ["auto — 自动选择"]
    choices = ["auto — 自动选择空闲 GPU"]
    for g in gpus:
        status, _, free_pct = _classify_gpu_status(g)
        choices.append(
            f"GPU {g['id']} — {g['name']} | {status} | 显存 {g['mem_used']}/{g['mem_total']} MiB ({free_pct}% 空闲) | 利用率 {g['util']}%"
        )
    # Multi-GPU options
    if len(gpus) > 1:
        free_ids = [g["id"] for g in gpus if _classify_gpu_status(g)[1] == "free"]
        if len(free_ids) >= 2:
            choices.append(f"GPU {','.join(free_ids)} — 所有空闲卡 ({len(free_ids)} 张)")
        choices.append(f"GPU {','.join(g['id'] for g in gpus)} — 全部 {len(gpus)} 张卡")
    return choices


def _build_gpu_choices() -> list[str]:
    """Build GPU dropdown choices with real-time status."""
    return _build_gpu_choices_from_info(_query_gpu_info())


def _render_monitor_status(message: str, tone: str = "info") -> str:
    tone_class = {
        "loading": "monitor-status-loading",
        "ready": "monitor-status-ready",
        "warning": "monitor-status-warning",
        "info": "monitor-status-info",
    }.get(tone, "monitor-status-info")
    timestamp = datetime.now().strftime("%H:%M:%S")
    return (
        f'<div class="monitor-status-bar {tone_class}">'
        f'<span class="monitor-status-message">{message}</span>'
        f'<span class="monitor-status-time">{timestamp}</span>'
        f'</div>'
    )


def _build_status_html_js(message: str, tone: str = "info") -> str:
    html = _render_monitor_status(message, tone)
    return f"() => [{json.dumps(html, ensure_ascii=False)}]"


def get_gradio_launch_kwargs() -> dict[str, Any]:
    return {
        "theme": gr.themes.Default(
            font=["ui-sans-serif", "system-ui", "Segoe UI", "Arial", "sans-serif"],
            font_mono=["ui-monospace", "SFMono-Regular", "Consolas", "Liberation Mono", "monospace"],
        ),
        "css": _CUSTOM_CSS,
    }


def _build_model_root_status_markdown() -> str:
    if not _model_registry:
        return "系统尚未初始化模型注册表。"

    active_root = _model_registry.get_active_model_root()
    available_count = len(_model_registry.get_model_choices())
    if active_root:
        return (
            f"当前模型目录：`{active_root}`  \n"
            f"发现可用模型 **{available_count}** 个"
        )
    return (
        "当前使用配置文件中的默认模型路径。  \n"
        f"发现可用模型 **{available_count}** 个"
    )


def _default_dataset_choices() -> list[str]:
    if not _dataset_registry:
        return []
    return [d.name for d in _dataset_registry.list_all() if d.category != "discovered"]


def _reset_submit_model_state(model_choices: list[str]):
    dataset_choices = _default_dataset_choices()
    return (
        gr.update(choices=model_choices, value=None),
        "*选择模型后显示详情*",
        gr.update(choices=dataset_choices, value=[]),
        dataset_choices,
        gr.update(visible=False),
        "",
        gr.update(choices=[], value=None),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=""),
    )


def _shorten_dataset_description(text: str, limit: int = 42) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip().replace("|", "/")
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _build_selected_dataset_summary_markdown(dataset_names: list[str]) -> str:
    if not _dataset_registry or not dataset_names:
        return ""

    rows = _dataset_registry.build_selected_summaries(dataset_names)
    lines = [
        "| 数据集 | 题型 | 题量 | 简介 |",
        "|:--|:--|:--|:--|",
    ]
    for row in rows:
        title = row["title"].replace("|", "/")
        question_type = row["question_type"].replace("|", "/")
        question_count = row["question_count"].replace("|", "/")
        description = _shorten_dataset_description(row["description"])
        lines.append(f"| {title} | {question_type} | {question_count} | {description} |")
    return "\n".join(lines)


def _build_selected_dataset_summary_updates(dataset_names: list[str]):
    selected = [name for name in (dataset_names or []) if name]
    if not selected:
        return gr.update(visible=False), ""
    return gr.update(visible=True), _build_selected_dataset_summary_markdown(selected)


def _selected_multimodal_datasets(dataset_names: list[str]) -> list[str]:
    selected = [name for name in (dataset_names or []) if name]
    return sorted(set(selected) & _MULTIMODAL_DATASETS)


def on_apply_model_root(model_root: str):
    if not _model_registry:
        return (
            "⚠️ 系统未初始化，暂时无法刷新模型列表。",
            *_reset_submit_model_state([]),
        )

    normalized_root = (model_root or "").strip()
    if normalized_root:
        root_path = Path(normalized_root).expanduser()
        if not root_path.is_dir():
            model_choices = _model_registry.get_model_choices()
            return (
                f"⚠️ 模型目录不存在：`{normalized_root}`",
                *_reset_submit_model_state(model_choices),
            )

    _model_registry.set_model_root(normalized_root or None)
    model_choices = _model_registry.get_model_choices()
    status_text = _build_model_root_status_markdown()
    if not model_choices:
        status_text += "\n\n⚠️ 当前目录下没有匹配到已配置模型，请检查目录是否包含对应模型子目录。"

    return (status_text, *_reset_submit_model_state(model_choices))


# ──────────────────────────────────────────────────────────
# Tab 1: Submit task — callbacks
# ──────────────────────────────────────────────────────────
def on_model_selected(model_name: str):
    """When a model is selected, populate datasets, backends, and hmonnx section."""
    if not _model_registry or not model_name:
        return (
            gr.update(),
            [],
            gr.update(visible=False),
            "",
            gr.update(),
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
        )

    model = _model_registry.get_model_by_display_name(model_name)
    if model is None:
        return (
            gr.update(choices=[], value=[]),
            [],
            gr.update(visible=False),
            "",
            gr.update(choices=[]),
            "模型未找到",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
        )

    # Datasets
    rec = model.recommended_datasets
    ds_choices = list(dict.fromkeys(rec + _default_dataset_choices()))
    dataset_summary_group, dataset_summary_md = _build_selected_dataset_summary_updates(rec)

    # Backends
    backends = list(model.backends.keys())
    backend_choices = []
    for b in backends:
        if b == "float":
            status = "✅ 权重就绪" if model.hf_model_available else "❌ 权重不存在"
            backend_choices.append(f"float — HF 官方浮点推理 ({status})")
        elif b == "hmonnx":
            hmonnx_meta = model.backends["hmonnx"].export_meta_info
            status = "✅ 已配置默认 meta" if hmonnx_meta and Path(hmonnx_meta).is_file() else "✍️ 手动填写 golden_meta_info.json / export_meta_info.json / meta.json 绝对路径"
            backend_choices.append(f"hmonnx — ONNX 量化推理 ({status})")

    hmonnx_meta = model.backends["hmonnx"].export_meta_info if "hmonnx" in model.backends else ""
    hmonnx_status = "✅ 已配置默认 meta" if hmonnx_meta and Path(hmonnx_meta).is_file() else "✍️ 提交时手动填写 `golden_meta_info.json`、`export_meta_info.json` 或 `meta.json` 绝对路径"

    # Info card
    info = f"""### 📋 模型信息
| 属性 | 值 |
|:-----|:---|
| **模型** | {model.display_name} |
| **架构** | `{model.model_class}` |
| **路径** | `{model.hf_model_dir}` |
| **transformers** | {model.transformers_version} |
| **HF 权重** | {'✅ 就绪' if model.hf_model_available else '❌ 不存在'} |
| **HMONNX 导出** | {hmonnx_status} |"""

    has_hmonnx = "hmonnx" in model.backends

    return (
        gr.update(choices=ds_choices, value=rec),
        ds_choices,
        dataset_summary_group,
        dataset_summary_md,
        gr.update(choices=backend_choices, value=backend_choices[0] if backend_choices else None),
        info,
        gr.update(visible=has_hmonnx),
        gr.update(visible=has_hmonnx),
        gr.update(visible=has_hmonnx),
        gr.update(value=hmonnx_meta if has_hmonnx else ""),
        gr.update(value=model.backends["hmonnx"].vision_export_meta_info if has_hmonnx else ""),
    )


def _hmonnx_meta_has_embedded_vision(meta_info_path: Path) -> bool:
    from hm_eval.core.hmonnx_meta import hmonnx_meta_has_embedded_vision

    return hmonnx_meta_has_embedded_vision(meta_info_path)


def on_backend_changed(backend_label: str):
    """Show/hide hmonnx export section based on backend selection."""
    if not backend_label:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    is_hmonnx = backend_label.startswith("hmonnx")
    return gr.update(visible=is_hmonnx), gr.update(visible=is_hmonnx), gr.update(visible=is_hmonnx)


def on_search_datasets(query: str):
    if not _dataset_registry or not query.strip():
        return gr.update()
    results = _dataset_registry.search(query.strip())
    choices = [f"{d.name} — {d.display_name}" for d in results]
    return gr.update(choices=choices, value=[])


def on_add_searched(search_sel: list[str], current: list[str], current_choices: list[str]):
    new = [item.split(" — ")[0].strip() for item in (search_sel or [])]
    updated = list(current or []) + [n for n in new if n and n not in (current or [])]
    choices = list(dict.fromkeys(list(current_choices or []) + updated))
    summary_group, summary_md = _build_selected_dataset_summary_updates(updated)
    return gr.update(choices=choices, value=updated), choices, summary_group, summary_md


def on_dataset_selection_changed(dataset_names: list[str]):
    return _build_selected_dataset_summary_updates(dataset_names)


def on_refresh_gpu(current_gpu_choice: str):
    """Refresh GPU dropdown choices."""
    gpus = _query_gpu_info()
    choices = _build_gpu_choices_from_info(gpus)
    next_value = current_gpu_choice if current_gpu_choice in choices else choices[0]
    return gr.update(choices=choices, value=next_value)


def on_submit_task(
    model_name: str, datasets: list[str], backend_label: str,
    limit: int, max_tokens: int,
    gpu_choice: str,
    meta_info_choice: str,
    vision_meta_info_choice: str,
):
    """Validate and submit an evaluation task, then refresh the report center."""
    _no_report_updates = (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        _render_monitor_status("任务提交未开始，请先修正当前输入。", "warning"),
    )
    if not _model_registry or not _task_manager:
        return ("⚠️ 系统未初始化", *_no_report_updates)
    if not model_name:
        return ("⚠️ 请选择模型", *_no_report_updates)
    if not datasets:
        return ("⚠️ 请选择至少一个数据集", *_no_report_updates)
    if not backend_label:
        return ("⚠️ 请选择后端", *_no_report_updates)

    model = _model_registry.get_model_by_display_name(model_name)
    if model is None:
        return (f"⚠️ 未找到模型: {model_name}", *_no_report_updates)

    backend_type = "hmonnx" if backend_label.startswith("hmonnx") else "float"

    # Parse GPU selection → cuda_devices string
    cuda_devices = "auto"
    if gpu_choice and not gpu_choice.startswith("auto"):
        gpu_part = gpu_choice.split("—")[0].strip()
        cuda_devices = gpu_part.replace("GPU", "").strip()

    # For hmonnx, require a user-provided runtime meta file (export done offline)
    export_meta_info_path = ""
    vision_export_meta_info_path = ""
    if backend_type == "hmonnx":
        meta_info_input = (meta_info_choice or "").strip()
        if not meta_info_input:
            return (
                "⚠️ hmonnx 后端需要手动填写 `golden_meta_info.json`、`export_meta_info.json` 或 `meta.json` 的绝对路径。\n\n"
                "请先在离线环境完成模型导出（参考 `examples_merak/llm/` 目录下的导出脚本），"
                "然后在此粘贴完整的绝对路径。",
                *_no_report_updates,
            )
        meta_info_path = Path(meta_info_input)
        if not meta_info_path.is_absolute():
            return (
                "⚠️ hmonnx 后端只接受 meta 文件的绝对路径。\n\n"
                "请填写类似 `/path/to/golden_meta_info.json`、`/path/to/export_meta_info.json` 或 `/path/to/meta.json` 的完整绝对路径。",
                *_no_report_updates,
            )
        if meta_info_path.name not in {"golden_meta_info.json", "export_meta_info.json", "meta.json"}:
            return (
                "⚠️ hmonnx 后端需要填写名为 `golden_meta_info.json`、`export_meta_info.json` 或 `meta.json` 的文件绝对路径。",
                *_no_report_updates,
            )
        if not meta_info_path.is_file():
            return (
                "⚠️ 提供的 meta 文件路径不存在，或当前进程不可读。\n\n"
                f"请检查路径是否正确：`{meta_info_input}`",
                *_no_report_updates,
            )
        export_meta_info_path = str(meta_info_path)

        multimodal_datasets = _selected_multimodal_datasets(datasets)
        vision_meta_input = (vision_meta_info_choice or "").strip()
        if multimodal_datasets and not vision_meta_input and not _hmonnx_meta_has_embedded_vision(meta_info_path):
            return (
                "⚠️ 多模态 HMONNX 评测需要额外填写 vision `export_meta_info.json` 的绝对路径。\n\n"
                f"当前选择的数据集包含：`{', '.join(multimodal_datasets)}`。如果 LLM `golden_meta_info.json` 已经内嵌 vision 子图，"
                "可以直接使用该 unified meta。",
                *_no_report_updates,
            )
        if vision_meta_input:
            vision_meta_path = Path(vision_meta_input)
            if not vision_meta_path.is_absolute():
                return (
                    "⚠️ vision meta 文件只接受绝对路径。\n\n"
                    "请填写类似 `/path/to/export_meta_info.json` 的完整绝对路径。",
                    *_no_report_updates,
                )
            if vision_meta_path.name != "export_meta_info.json":
                return (
                    "⚠️ vision meta 文件应为导出目录中的 `export_meta_info.json`。",
                    *_no_report_updates,
                )
            if not vision_meta_path.is_file():
                return (
                    "⚠️ 提供的 vision meta 文件路径不存在，或当前进程不可读。\n\n"
                    f"请检查路径是否正确：`{vision_meta_input}`",
                    *_no_report_updates,
                )
            vision_export_meta_info_path = str(vision_meta_path)

    task = _task_manager.create_task(
        model_config_id=model.config_id,
        model_display_name=model.display_name,
        backend=backend_type,
        datasets=datasets,
        limit=int(limit) if limit and limit > 0 else 0,
        max_tokens=int(max_tokens) if max_tokens else 512,
        transformers_version=model.transformers_version,
        export_meta_info_path=export_meta_info_path,
        vision_export_meta_info_path=vision_export_meta_info_path,
        hf_model_dir=model.hf_model_dir,
        cuda_devices=cuda_devices,
    )

    ok = _task_manager.start_task(task.task_id)
    if ok:
        gpu_display = f"GPU {cuda_devices}" if cuda_devices != "auto" else "自动选择"
        rows, summary = refresh_task_table()
        selector_update = _build_report_task_selector_update(rows, task.task_id)
        compare_selector_update = _build_report_compare_selector_update([])
        diagnosis, log_text, result_html, result_text = _empty_report_center_details()
        return (
            _build_submit_success_markdown(
                task.task_id,
                model.display_name,
                backend_type,
                gpu_display,
                int(limit) if limit else 0,
            ),
            summary,
            rows,
            selector_update,
            diagnosis,
            log_text,
            result_html,
            result_text,
            compare_selector_update,
            _render_monitor_status(
                f"任务 {task.task_id} 已提交成功，可自行切换到评测报告页查看。",
                "ready",
            ),
        )
    else:
        return (
            f"❌ 任务启动失败 (ID: {task.task_id})，请查看日志",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            _render_monitor_status(f"任务 {task.task_id} 启动失败，请检查日志。", "warning"),
        )


# ──────────────────────────────────────────────────────────
# Tab 2: Monitor — callbacks
# ──────────────────────────────────────────────────────────
_STATUS_MAP = {
    TaskStatus.PENDING:   ("⏳", "等待中"),
    TaskStatus.ENV_SETUP: ("🔧", "环境配置中"),
    TaskStatus.EXPORTING: ("📦", "ONNX 导出中"),
    TaskStatus.RUNNING:   ("🔄", "评测中"),
    TaskStatus.COMPLETED: ("✅", "已完成"),
    TaskStatus.ABNORMAL:  ("⚠️", "结果异常"),
    TaskStatus.FAILED:    ("❌", "失败"),
}


def _clean_log_text(log_text: str) -> str:
    cleaned = _ANSI_ESCAPE_RE.sub("", log_text or "").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()


def _format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "未知"
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分钟"
    if minutes:
        return f"{minutes}分钟{secs}秒"
    return f"{secs}秒"


def _find_latest_match(pattern: re.Pattern[str], log_text: str) -> Optional[dict[str, str]]:
    for line in reversed(log_text.splitlines()):
        match = pattern.search(line)
        if match:
            return match.groupdict()
    return None


def _get_runtime_seconds(task) -> float | None:
    started_at = task.started_at or task.created_at
    if not started_at:
        return None
    try:
        started_dt = datetime.fromisoformat(started_at)
    except ValueError:
        return None

    if task.finished_at:
        try:
            end_dt = datetime.fromisoformat(task.finished_at)
        except ValueError:
            end_dt = datetime.now()
    else:
        end_dt = datetime.now()

    return (end_dt - started_dt).total_seconds()


def _get_log_age_seconds(task) -> float | None:
    if not task.log_file:
        return None
    log_path = Path(task.log_file)
    if not log_path.exists():
        return None
    return time.time() - log_path.stat().st_mtime


def _last_meaningful_log_line(log_text: str) -> str:
    for line in reversed(log_text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{") or stripped.startswith("}") or stripped.startswith('"'):
            continue
        return stripped
    return ""


def _extract_failure_reason(task, log_text: str) -> str:
    combined = f"{task.error}\n{log_text}".strip()
    checks: list[tuple[str, str]] = [
        (r"KeyboardInterrupt", "任务被外部中断，常见于终端 Ctrl+C、父进程退出或手动停止任务。"),
        (r"terminate called without an active exception", "底层运行时异常终止，通常是 C++/CUDA 侧 abort，常与外部中断或底层算子异常相关。"),
        (r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED", "GPU 显存不足，进程被中断。"),
        (r"transformers switch failed", "transformers 版本切换失败，请检查 pip 安装输出。"),
        (r"pip install timed out after 300s", "transformers 版本切换超时。"),
        (r"Backend creation failed", "后端初始化失败，请检查模型路径或依赖。"),
        (r"No module named ['\"]([^'\"]+)['\"]", "缺少 Python 依赖模块。"),
        (r"ModuleNotFoundError", "缺少 Python 依赖模块。"),
        (r"FileNotFoundError", "模型、导出产物或中间文件不存在。"),
        (r"PermissionError|Permission denied", "权限不足，无法读取模型或写入输出目录。"),
        (r"hmonnx 后端需要提供 .*meta_info\.json .*路径", "缺少 HMONNX meta 文件，导出产物未准备好。"),
        (r"Model config not found", "模型配置不存在或未正确加载。"),
        (r"Killed\b|SIGKILL|oom-kill|Out of memory", "进程被系统强制终止，常见原因是内存或显存不足。"),
    ]
    for pattern, message in checks:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            return message

    for line in reversed(log_text.splitlines()):
        stripped = line.strip()
        if not stripped or stripped == "Traceback (most recent call last):":
            continue
        if _FAILURE_LINE_RE.search(stripped):
            return stripped

    if task.error and task.error != "进程异常退出":
        return task.error
    if task.error == "进程异常退出":
        return "worker 进程异常退出，日志里没有明确 traceback，常见于系统 OOM/SIGKILL。"
    return "未从日志中提取到明确错误，请结合最后几行日志继续排查。"


def _build_task_diagnosis(task, log_text: str) -> str:
    icon, label = _STATUS_MAP.get(task.status, ("?", "未知"))
    lines = ["### 任务诊断", f"- 当前状态：{icon} {label}"]

    runtime_seconds = _get_runtime_seconds(task)
    if runtime_seconds is not None:
        lines.append(f"- 已运行：{_format_seconds(runtime_seconds)}")

    log_age_seconds = _get_log_age_seconds(task)
    if log_age_seconds is not None:
        lines.append(f"- 最近日志更新：{_format_seconds(log_age_seconds)}前")

    if task.status == TaskStatus.ENV_SETUP:
        lines.append(f"- 当前阶段：正在切换 transformers 到 {task.transformers_version}。")
        lines.append("- 判断：这一步可能持续几分钟，完成后会自动进入模型加载和评测。")
        return "\n".join(lines)

    if task.status == TaskStatus.FAILED:
        lines.append(f"- 失败原因：{_extract_failure_reason(task, log_text)}")
        last_line = _last_meaningful_log_line(log_text)
        if last_line:
            lines.append(f"- 最后一条有效日志：{last_line}")
        return "\n".join(lines)

    if task.status == TaskStatus.ABNORMAL:
        lines.append(f"- 异常原因：{task.error or '报告或日志存在异常，请结合下方日志与报告回溯。'}")
        lines.append("- 判断：任务流程已结束，但结果不可直接采信。请优先检查日志中的异常记录和报告细节。")
        return "\n".join(lines)

    if task.status == TaskStatus.COMPLETED:
        lines.append("- 判断：任务已完成，可直接查看下方实时报告或切换到评测报告页。")
        return "\n".join(lines)

    loading = _find_latest_match(_LOADING_PROGRESS_RE, log_text)
    if loading:
        lines.append(
            f"- 当前阶段：正在加载模型权重 {loading['done']}/{loading['total']} ({loading['pct']}%)。"
        )
        if loading.get("eta"):
            lines.append(f"- 预计剩余：{loading['eta']}")
        if loading.get("rate"):
            lines.append(f"- 当前速度：{loading['rate']}")
        lines.append("- 判断：任务仍在启动阶段，这不是卡死。")
        return "\n".join(lines)

    progress = _find_latest_match(_PREDICT_PROGRESS_RE, log_text)
    if progress:
        total = int(progress["total"])
        done = int(progress["done"])
        percent = round(done / total * 100, 1) if total else 0.0
        lines.append(f"- 当前进度：{progress['label']} {done}/{total} ({percent}%)。")
        if progress.get("eta"):
            lines.append(f"- 预计剩余：{progress['eta']}")
        if progress.get("rate"):
            lines.append(f"- 当前速度：{progress['rate']}")
        if task.limit == 0 and total >= 200:
            lines.append("- 判断：这是全量评测，不是卡死。float 后端在大数据集上可能需要数小时。")
            lines.append("- 建议：如果只是验证链路，先把每子集题数设为 5 或 20。")
        return "\n".join(lines)

    samples_match = _PROCESSING_SAMPLES_RE.search(log_text)
    if samples_match:
        lines.append(f"- 当前阶段：数据已准备完成，待处理样本数约 {samples_match.group('total')}。")

    if "Model loaded successfully." in log_text:
        lines.append("- 当前阶段：模型已加载完成，正在等待评测输出落盘。")
    elif "Loading model for prediction..." in log_text:
        lines.append("- 当前阶段：正在构建评测模型。")
    elif "Start loading benchmark dataset" in log_text:
        lines.append("- 当前阶段：正在准备评测数据集。")

    if log_age_seconds is not None and log_age_seconds >= _STALLED_LOG_SECONDS:
        lines.append("- 判断：最近较久没有新日志，任务可能卡住，建议检查 GPU、内存和 worker 进程状态。")
    else:
        lines.append("- 判断：任务仍在正常运行，监控页会自动刷新。")

    return "\n".join(lines)


def _build_task_detail(task_id: str):
    if not _task_manager or not task_id:
        return "", "", "", gr.update(visible=False, value="")

    task = _task_manager.update_task_status(task_id)
    if task is None:
        return task_id, "", "", gr.update(visible=False, value="")

    log_text = _clean_log_text(_task_manager.get_task_log(task.task_id, tail=200))
    diagnosis = _build_task_diagnosis(task, log_text)
    report_update = gr.update(visible=False, value="")

    if task.status in (TaskStatus.COMPLETED, TaskStatus.ABNORMAL):
        rendered_report = _load_cached_report_render(task)
        if rendered_report is not None:
            report_update = gr.update(visible=True, value=rendered_report[0])

    return task.task_id, log_text, diagnosis, report_update


def _extract_table_rows(table_data: Any) -> list[list[Any]]:
    if isinstance(table_data, dict):
        rows = table_data.get("data")
        return rows if isinstance(rows, list) else []
    if isinstance(table_data, list):
        return table_data
    return []


def _normalize_select_index(index: Any) -> tuple[Optional[int], Optional[int]]:
    if index is None:
        return None, None
    if isinstance(index, (list, tuple)):
        if len(index) >= 2:
            return index[0], index[1]
        if len(index) == 1:
            return index[0], None
    if isinstance(index, int):
        return index, None
    return None, None


def _refresh_monitor_view(selected_task_id: str, ready_message: str):
    rows, summary = refresh_task_table()

    active_task_id = selected_task_id
    if not active_task_id and _task_manager:
        tasks = _task_manager.list_tasks()
        if tasks:
            active_task_id = tasks[0].task_id

    task_id, log_text, diagnosis, report_update = _build_task_detail(active_task_id)
    if task_id:
        status_message = f"{ready_message} 当前选中任务：{task_id}。"
    elif rows:
        status_message = f"{ready_message} 已同步任务列表。"
    else:
        status_message = "任务监控已就绪，当前暂无任务。"
    return (
        rows,
        summary,
        task_id,
        log_text,
        diagnosis,
        report_update,
        _render_monitor_status(status_message, "ready"),
    )


def refresh_monitor_view(selected_task_id: str):
    return _refresh_monitor_view(selected_task_id, "任务状态刷新完成。")


def refresh_monitor_view_on_enter(selected_task_id: str):
    return _refresh_monitor_view(selected_task_id, "已进入任务监控。")


def refresh_monitor_view_auto(selected_task_id: str):
    return _refresh_monitor_view(selected_task_id, "自动刷新已完成。")


def refresh_task_table():
    if not _task_manager:
        return [], ""

    tasks = _task_manager.list_tasks()
    rows = []
    for t in tasks:
        icon, label = _STATUS_MAP.get(t.status, ("?", "未知"))
        rows.append([
            t.task_id,
            t.model_display_name,
            t.backend,
            ", ".join(t.datasets[:3]) + ("..." if len(t.datasets) > 3 else ""),
            f"{icon} {label}",
            t.created_at[:19] if t.created_at else "",
        ])

    # Summary stats
    total = len(tasks)
    running = sum(
        1 for t in tasks if t.status in (TaskStatus.RUNNING, TaskStatus.EXPORTING, TaskStatus.ENV_SETUP)
    )
    done = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
    abnormal = sum(1 for t in tasks if t.status == TaskStatus.ABNORMAL)
    failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
    summary = (
        f"共 **{total}** 个任务 | 🔄 运行中 **{running}** | ✅ 完成 **{done}** | "
        f"⚠️ 结果异常 **{abnormal}** | ❌ 失败 **{failed}**"
    )

    return rows, summary


def _build_report_task_selector_update(rows: list[list[Any]], selected_task_id: str):
    task_ids = [row[0] for row in rows if row]
    normalized_task_id = selected_task_id if selected_task_id in task_ids else None
    return gr.update(choices=task_ids, value=normalized_task_id)


def on_task_row_select(evt: gr.SelectData, table_data):
    if not _task_manager or evt.index is None:
        return "", "", "", gr.update(visible=False, value=""), _render_monitor_status("未选中有效任务。", "warning")

    row_idx, _ = _normalize_select_index(evt.index)
    table_rows = _extract_table_rows(table_data)
    if row_idx is None or row_idx >= len(table_rows):
        return "", "", "", gr.update(visible=False, value=""), _render_monitor_status("未选中有效任务。", "warning")

    task_id = table_rows[row_idx][0]
    selected_task_id, log_text, diagnosis, report_update = _build_task_detail(task_id)
    return (
        selected_task_id,
        log_text,
        diagnosis,
        report_update,
        _render_monitor_status(f"已加载任务 {task_id} 的最新日志与诊断。", "ready"),
    )


def refresh_task_detail(task_id: str):
    _, log_text, diagnosis, report_update = _build_task_detail(task_id)
    if task_id:
        status_message = f"任务 {task_id} 的日志与诊断已刷新。"
        status_tone = "ready"
    else:
        status_message = "当前没有可刷新的任务。"
        status_tone = "warning"
    return log_text, diagnosis, report_update, _render_monitor_status(status_message, status_tone)


# ──────────────────────────────────────────────────────────
# Report center — callbacks
# ──────────────────────────────────────────────────────────
def _build_task_message_html(icon: str, title: str, subtitle: str = "", details: Optional[list[str]] = None) -> str:
    detail_items = details or []
    detail_html = "".join(
        (
            "<p style=\"color:#6c757d; margin: 6px 0;\">"
            f"{html.escape(detail)}"
            "</p>"
        )
        for detail in detail_items
        if detail
    )
    subtitle_html = ""
    if subtitle:
        subtitle_html = (
            "<p style=\"color:#495057; margin: 10px 0 12px; font-size: 15px;\">"
            f"{html.escape(subtitle)}"
            "</p>"
        )
    return f"""
<div style="text-align:center; padding: 44px 20px; border: 1px solid #e9ecef; border-radius: 12px; background: #fafafa;">
    <div style="font-size: 56px; margin-bottom: 12px;">{html.escape(icon)}</div>
    <h2 style="color: #212529; margin-bottom: 0;">{html.escape(title)}</h2>
    {subtitle_html}
    <div style="max-width: 760px; margin: 0 auto;">{detail_html}</div>
</div>"""


def _empty_report_center_details():
    return (
        "选择任务后可直接查看状态、失败诊断、实时日志和评测报告。",
        "",
        _build_task_message_html("🗂️", "当前暂无可展示任务", "提交任务后，所有状态、诊断、日志和报告都会集中显示在这里。"),
        "",
    )


def _empty_report_compare_details():
    return (
        _build_task_message_html(
            "📊",
            "尚未生成任务对比",
            "请选择至少两个已完成任务，然后点击“生成对比”。",
            ["支持 2 个、3 个、4 个或更多任务同时对比。"],
        ),
        "",
    )


def _format_compare_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_compare_table_html(title: str, headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""

    header_html = "".join(
        f"<th style=\"padding:10px 12px; border:1px solid #dee2e6; background:#f8f9fa; text-align:left;\">{html.escape(header)}</th>"
        for header in headers
    )
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td style=\"padding:10px 12px; border:1px solid #dee2e6; vertical-align:top;\">{html.escape(str(cell))}</td>"
            for cell in row
        )
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        "<div style=\"margin-top:18px;\">"
        f"<h3 style=\"margin:0 0 10px;\">{html.escape(title)}</h3>"
        "<div style=\"overflow-x:auto;\">"
        "<table style=\"width:100%; border-collapse:collapse; background:#fff;\">"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
        "</div>"
    )


def _render_compare_table_text(title: str, headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""

    matrix = [headers] + rows
    widths = [max(len(str(row[idx])) for row in matrix) for idx in range(len(headers))]

    def _fmt(row: list[str]) -> str:
        return " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row))

    divider = "-+-".join("-" * width for width in widths)
    lines = [title, _fmt(headers), divider]
    lines.extend(_fmt(row) for row in rows)
    return "\n".join(lines)


def _get_report_compare_candidate_task_ids() -> list[str]:
    if not _task_manager:
        return []

    candidate_task_ids = []
    for task in _task_manager.list_tasks():
        report_path = Path(task.report_path or Path(task.work_dir) / "report.json")
        if report_path.exists():
            candidate_task_ids.append(task.task_id)
    return candidate_task_ids


def _build_report_compare_selector_update(selected_task_ids: Optional[list[str]] = None):
    task_ids = _get_report_compare_candidate_task_ids()
    normalized_ids = []
    for task_id in selected_task_ids or []:
        if task_id in task_ids and task_id not in normalized_ids:
            normalized_ids.append(task_id)
    return gr.update(choices=task_ids, value=normalized_ids)


def _build_report_comparison_view(selected_task_ids: Optional[list[str]]):
    normalized_ids = []
    for task_id in selected_task_ids or []:
        if task_id and task_id not in normalized_ids:
            normalized_ids.append(task_id)

    if len(normalized_ids) < 2:
        compare_html, compare_text = _empty_report_compare_details()
        return normalized_ids, compare_html, compare_text, "请至少选择两个已完成任务再生成对比。", "warning"

    comparable_entries = []
    ignored_task_ids = []
    for task_id in normalized_ids:
        if not _task_manager:
            ignored_task_ids.append(task_id)
            continue
        report = _task_manager.get_task_report(task_id)
        task = _task_manager.get_task(task_id)
        if task is None or report is None:
            ignored_task_ids.append(task_id)
            continue
        comparable_entries.append({"task_id": task_id, "task": task, "report": report})

    normalized_ids = [entry["task_id"] for entry in comparable_entries]
    if len(normalized_ids) < 2:
        compare_html, compare_text = _empty_report_compare_details()
        message = "所选任务中可用于对比的报告不足两个，请确认任务已经产出 report.json。"
        if ignored_task_ids:
            message += f" 已忽略：{', '.join(ignored_task_ids)}。"
        return normalized_ids, compare_html, compare_text, message, "warning"

    headers = ["指标"] + normalized_ids
    overview_rows = [
        ["模型", *[entry["task"].model_display_name or entry["report"].get("model", "-") for entry in comparable_entries]],
        ["后端", *[entry["task"].backend or entry["report"].get("backend", "-") for entry in comparable_entries]],
        ["状态", *[entry["task"].status.value for entry in comparable_entries]],
        ["数据集", *[", ".join(entry["task"].datasets) for entry in comparable_entries]],
        ["创建时间", *[(entry["task"].created_at[:19] if entry["task"].created_at else "-") for entry in comparable_entries]],
        ["总耗时(s)", *[_format_compare_value(entry["report"].get("total_elapsed_seconds")) for entry in comparable_entries]],
    ]

    summary_metric_priority = ["status", "macro_acc", "accuracy", "score", "exact_match", "pass@1", "elapsed_seconds", "error"]
    dataset_names = sorted(
        {
            dataset_name
            for entry in comparable_entries
            for dataset_name in entry["report"].get("summary", {}).keys()
        }
    )
    summary_rows: list[list[str]] = []
    for dataset_name in dataset_names:
        metric_keys = set()
        for entry in comparable_entries:
            dataset_summary = entry["report"].get("summary", {}).get(dataset_name, {})
            if isinstance(dataset_summary, dict):
                metric_keys.update(dataset_summary.keys())
        ordered_metric_keys = [key for key in summary_metric_priority if key in metric_keys]
        ordered_metric_keys.extend(sorted(metric_keys - set(ordered_metric_keys)))
        for metric_key in ordered_metric_keys:
            row = [f"{dataset_name} / {metric_key}"]
            for entry in comparable_entries:
                dataset_summary = entry["report"].get("summary", {}).get(dataset_name, {})
                value = dataset_summary.get(metric_key) if isinstance(dataset_summary, dict) else None
                row.append(_format_compare_value(value))
            summary_rows.append(row)

    detail_tables_html = []
    detail_tables_text = []
    detail_dataset_names = sorted(
        {
            dataset_name
            for entry in comparable_entries
            for dataset_name in entry["report"].get("details", {}).keys()
        }
    )
    for dataset_name in detail_dataset_names:
        subset_names = sorted(
            {
                subset_name
                for entry in comparable_entries
                for subset_name in entry["report"].get("details", {}).get(dataset_name, {}).keys()
            }
        )
        if not subset_names:
            continue

        detail_rows = []
        for subset_name in subset_names:
            row = [subset_name]
            for entry in comparable_entries:
                subset_data = entry["report"].get("details", {}).get(dataset_name, {}).get(subset_name, {})
                cell_value = "-"
                if isinstance(subset_data, dict):
                    accuracy = subset_data.get("accuracy")
                    correct = subset_data.get("correct")
                    total = subset_data.get("total")
                    total_predictions = subset_data.get("total_predictions")
                    if isinstance(accuracy, (int, float)):
                        cell_value = f"{accuracy:.4f}"
                        if isinstance(correct, int) and isinstance(total, int):
                            cell_value = f"{cell_value} ({correct}/{total})"
                    elif isinstance(correct, int) and isinstance(total, int):
                        cell_value = f"{correct}/{total}"
                    elif isinstance(total_predictions, int):
                        cell_value = f"pred={total_predictions}"
                row.append(cell_value)
            detail_rows.append(row)

        if detail_rows:
            title = f"{dataset_name} 子项准确率对比"
            detail_tables_html.append(_render_compare_table_html(title, headers, detail_rows))
            detail_tables_text.append(_render_compare_table_text(title, headers, detail_rows))

    compare_sections_html = [
        "<div style=\"border:1px solid #dee2e6; border-radius:12px; background:#fafafa; padding:18px;\">",
        f"<h2 style=\"margin:0;\">任务对比</h2><p style=\"margin:8px 0 0; color:#495057;\">当前共对比 {len(normalized_ids)} 个任务。</p>",
    ]
    if ignored_task_ids:
        compare_sections_html.append(
            "<p style=\"margin:12px 0 0; color:#8a5a00;\">"
            f"以下任务因缺少可用报告已被忽略：{html.escape(', '.join(ignored_task_ids))}"
            "</p>"
        )
    compare_sections_html.append(_render_compare_table_html("任务概览", headers, overview_rows))
    if summary_rows:
        compare_sections_html.append(_render_compare_table_html("数据集摘要指标对比", headers, summary_rows))
    compare_sections_html.extend(detail_tables_html)
    compare_sections_html.append("</div>")

    compare_sections_text = [
        f"任务对比（共 {len(normalized_ids)} 个任务）",
        _render_compare_table_text("任务概览", headers, overview_rows),
    ]
    if summary_rows:
        compare_sections_text.append(_render_compare_table_text("数据集摘要指标对比", headers, summary_rows))
    compare_sections_text.extend(detail_tables_text)
    if ignored_task_ids:
        compare_sections_text.append(f"已忽略无报告任务: {', '.join(ignored_task_ids)}")

    return (
        normalized_ids,
        "\n".join(compare_sections_html),
        "\n\n".join(section for section in compare_sections_text if section),
        f"已生成 {len(normalized_ids)} 个任务的对比结果。",
        "ready",
    )


def compare_selected_report_tasks(selected_task_ids: Optional[list[str]]):
    normalized_ids, compare_html, compare_text, status_message, status_tone = _build_report_comparison_view(selected_task_ids)
    return (
        _build_report_compare_selector_update(normalized_ids),
        compare_html,
        compare_text,
        _render_monitor_status(status_message, status_tone),
    )


def clear_report_comparison():
    compare_html, compare_text = _empty_report_compare_details()
    return (
        _build_report_compare_selector_update([]),
        compare_html,
        compare_text,
        _render_monitor_status("已清空任务对比。", "ready"),
    )

def _build_submit_success_markdown(
    task_id: str,
    model_display_name: str,
    backend_type: str,
    gpu_display: str,
    limit: int,
) -> str:
    limit_note = "全量评测" if not limit else f"limit={int(limit)}"
    return (
        "### ✅ 提交成功\n"
        f"- 任务 ID: `{task_id}`\n"
        f"- 模型: {model_display_name}\n"
        f"- 后端: {backend_type}\n"
        f"- GPU: {gpu_display}\n"
        f"- 范围: {limit_note}\n\n"
        "可随时切换到 **评测报告** 页面查看任务状态、日志和最终报告。"
    )


def _load_cached_report_render(task) -> Optional[tuple[str, str]]:
    report_path = Path(task.report_path or Path(task.work_dir) / "report.json")
    if not report_path.exists():
        return None

    report_key = str(report_path)
    report_mtime = report_path.stat().st_mtime
    cached = _report_render_cache.get(report_key)
    if cached and cached[0] == report_mtime:
        return cached[1], cached[2]

    report = json.loads(report_path.read_text(encoding="utf-8"))
    enrich_report_from_outputs(report, task.work_dir)
    report_html = format_report_html(report)
    report_text = format_report_text(report)
    _report_render_cache[report_key] = (report_mtime, report_html, report_text)
    return report_html, report_text


def _purge_report_render_cache(task) -> None:
    candidate_paths = []
    if getattr(task, "report_path", ""):
        candidate_paths.append(str(Path(task.report_path)))
    if getattr(task, "work_dir", ""):
        candidate_paths.append(str(Path(task.work_dir) / "report.json"))
    for report_key in candidate_paths:
        _report_render_cache.pop(report_key, None)


def _preserve_report_detail_state(
    selected_task_id: str,
    rows: list[list[Any]],
    current_diagnosis: str,
    current_log_text: str,
    current_result_html: str,
    current_result_text: str,
):
    available_task_ids = {row[0] for row in rows if row}
    if selected_task_id and selected_task_id in available_task_ids:
        empty_diagnosis, _, empty_result_html, _ = _empty_report_center_details()
        return (
            selected_task_id,
            current_diagnosis or empty_diagnosis,
            current_log_text or "",
            current_result_html or empty_result_html,
            current_result_text or "",
        )
    diagnosis, log_text, result_html, result_text = _empty_report_center_details()
    return "", diagnosis, log_text, result_html, result_text


def _build_task_result_view(task_id: str):
    if not _task_manager or not task_id:
        diagnosis, log_text, result_html, result_text = _empty_report_center_details()
        return "", diagnosis, log_text, result_html, result_text

    task = _task_manager.update_task_status(task_id)
    if task is None:
        missing_html = _build_task_message_html("⚠️", "任务不存在", f"任务 {task_id} 已不存在或被移除。")
        return task_id, "未找到该任务。", "", missing_html, ""

    log_text = _clean_log_text(_task_manager.get_task_log(task.task_id, tail=200))
    diagnosis = _build_task_diagnosis(task, log_text)
    started_at = task.started_at[:19] if task.started_at else (task.created_at[:19] if task.created_at else "未知")

    if task.status == TaskStatus.COMPLETED:
        rendered_report = _load_cached_report_render(task)
        if rendered_report is None:
            warning_html = _build_task_message_html(
                "⚠️",
                "任务已完成，但暂未找到报告文件",
                "请检查报告产物是否已正确写入。",
                [f"任务 ID: {task.task_id}", f"模型: {task.model_display_name}", f"后端: {task.backend}"],
            )
            return task.task_id, diagnosis, log_text, warning_html, ""
        report_html, report_text = rendered_report
        return task.task_id, diagnosis, log_text, report_html, report_text

    if task.status == TaskStatus.ABNORMAL:
        abnormal_reason = task.error or "报告或日志中检测到异常，结果不可直接采信。"
        abnormal_header = _build_task_message_html(
            "⚠️",
            "结果异常",
            abnormal_reason,
            [
                f"任务 ID: {task.task_id}",
                f"模型: {task.model_display_name}",
                f"后端: {task.backend}",
                f"数据集: {', '.join(task.datasets)}",
                f"开始时间: {started_at}",
            ],
        )
        rendered_report = _load_cached_report_render(task)
        if rendered_report is None:
            return task.task_id, diagnosis, log_text, abnormal_header, ""
        report_html, report_text = rendered_report
        return (
            task.task_id,
            diagnosis,
            log_text,
            abnormal_header + report_html,
            f"结果异常\n{abnormal_reason}\n\n{report_text}",
        )

    if task.status == TaskStatus.FAILED:
        reason = _extract_failure_reason(task, log_text)
        failed_html = _build_task_message_html(
            "❌",
            "评测失败",
            reason,
            [
                f"任务 ID: {task.task_id}",
                f"模型: {task.model_display_name}",
                f"后端: {task.backend}",
                f"数据集: {', '.join(task.datasets)}",
                f"开始时间: {started_at}",
            ],
        )
        return task.task_id, diagnosis, log_text, failed_html, ""

    if task.status in (TaskStatus.RUNNING, TaskStatus.EXPORTING, TaskStatus.ENV_SETUP):
        icon, label = _STATUS_MAP.get(task.status, ("🔄", "进行中"))
        running_html = _build_task_message_html(
            icon,
            f"任务{label}",
            "该任务仍在执行中，完成后会直接在这里展示报告。",
            [
                f"任务 ID: {task.task_id}",
                f"模型: {task.model_display_name}",
                f"后端: {task.backend}",
                f"数据集: {', '.join(task.datasets)}",
                f"开始时间: {started_at}",
                "下方日志会持续刷新，可直接用来判断当前进展。",
            ],
        )
        return task.task_id, diagnosis, log_text, running_html, ""

    pending_html = _build_task_message_html(
        "⏳",
        "任务等待中",
        "任务已进入队列，调度开始后这里会自动显示实时日志。",
        [
            f"任务 ID: {task.task_id}",
            f"模型: {task.model_display_name}",
            f"后端: {task.backend}",
            f"数据集: {', '.join(task.datasets)}",
        ],
    )
    return task.task_id, diagnosis, log_text, pending_html, ""


def _refresh_report_overview(
    selected_task_id: str,
    current_diagnosis: str,
    current_log_text: str,
    current_result_html: str,
    current_result_text: str,
    selected_compare_task_ids: Optional[list[str]],
    ready_message: str,
):
    rows, summary = refresh_task_table()
    available_task_ids = {row[0] for row in rows if row}
    normalized_task_id = selected_task_id if selected_task_id in available_task_ids else ""
    selector_update = _build_report_task_selector_update(rows, normalized_task_id)
    compare_selector_update = _build_report_compare_selector_update(selected_compare_task_ids)

    if normalized_task_id:
        selected_task_id, diagnosis, log_text, result_html, result_text = _build_task_result_view(normalized_task_id)
        status_message = f"{ready_message} 已同步任务列表，并刷新当前任务 {selected_task_id} 的详情。"
    elif rows:
        diagnosis, log_text, result_html, result_text = _empty_report_center_details()
        status_message = f"{ready_message} 已同步任务列表。选择任务后再加载详情。"
    else:
        diagnosis, log_text, result_html, result_text = _empty_report_center_details()
        status_message = "评测报告页已就绪，当前暂无任务。"

    return (
        summary,
        rows,
        selector_update,
        diagnosis,
        log_text,
        result_html,
        result_text,
        compare_selector_update,
        _render_monitor_status(status_message, "ready"),
    )


def _load_report_task(
    task_id: str,
    selected_compare_task_ids: Optional[list[str]],
    success_message_template: str,
    empty_message: str,
):
    rows, summary = refresh_task_table()
    available_task_ids = {row[0] for row in rows if row}

    if not task_id:
        diagnosis, log_text, result_html, result_text = _empty_report_center_details()
        return (
            summary,
            rows,
            _build_report_task_selector_update(rows, ""),
            diagnosis,
            log_text,
            result_html,
            result_text,
            _build_report_compare_selector_update(selected_compare_task_ids),
            _render_monitor_status(empty_message, "warning"),
        )

    if task_id not in available_task_ids:
        diagnosis, log_text, result_html, result_text = _empty_report_center_details()
        return (
            summary,
            rows,
            _build_report_task_selector_update(rows, ""),
            diagnosis,
            log_text,
            result_html,
            result_text,
            _build_report_compare_selector_update(selected_compare_task_ids),
            _render_monitor_status(f"任务 {task_id} 已不存在或已被删除。", "warning"),
        )

    selected_task_id, diagnosis, log_text, result_html, result_text = _build_task_result_view(task_id)
    return (
        summary,
        rows,
        _build_report_task_selector_update(rows, selected_task_id),
        diagnosis,
        log_text,
        result_html,
        result_text,
        _build_report_compare_selector_update(selected_compare_task_ids),
        _render_monitor_status(success_message_template.format(task_id=selected_task_id), "ready"),
    )


def refresh_report_center(
    selected_task_id: str,
    current_diagnosis: str,
    current_log_text: str,
    current_result_html: str,
    current_result_text: str,
    selected_compare_task_ids: Optional[list[str]],
):
    return _refresh_report_overview(
        selected_task_id,
        current_diagnosis,
        current_log_text,
        current_result_html,
        current_result_text,
        selected_compare_task_ids,
        "任务状态刷新完成。",
    )


def refresh_report_center_on_enter(
    selected_task_id: str,
    current_diagnosis: str,
    current_log_text: str,
    current_result_html: str,
    current_result_text: str,
    selected_compare_task_ids: Optional[list[str]],
):
    return _refresh_report_overview(
        selected_task_id,
        current_diagnosis,
        current_log_text,
        current_result_html,
        current_result_text,
        selected_compare_task_ids,
        "已进入评测报告页。",
    )


def refresh_report_center_auto(
    selected_task_id: str,
    current_diagnosis: str,
    current_log_text: str,
    current_result_html: str,
    current_result_text: str,
    selected_compare_task_ids: Optional[list[str]],
):
    return _refresh_report_overview(
        selected_task_id,
        current_diagnosis,
        current_log_text,
        current_result_html,
        current_result_text,
        selected_compare_task_ids,
        "自动刷新已完成。",
    )


def enter_report_tab(
    selected_task_id: str,
    current_diagnosis: str,
    current_log_text: str,
    current_result_html: str,
    current_result_text: str,
    selected_compare_task_ids: Optional[list[str]],
):
    overview = refresh_report_center_on_enter(
        selected_task_id,
        current_diagnosis,
        current_log_text,
        current_result_html,
        current_result_text,
        selected_compare_task_ids,
    )
    return (*overview, gr.update(active=True))


def pause_report_timer():
    return gr.update(active=False)


def on_report_task_row_select(
    evt: gr.SelectData,
    table_data,
    current_selected_task_id: str,
    current_diagnosis: str,
    current_log_text: str,
    current_result_html: str,
    current_result_text: str,
    current_compare_task_ids: Optional[list[str]],
):
    rows, summary = refresh_task_table()
    row_idx, col_idx = _normalize_select_index(evt.index)
    table_rows = _extract_table_rows(table_data)

    if row_idx is None or row_idx >= len(table_rows):
        selected_task_id, diagnosis, log_text, result_html, result_text = _preserve_report_detail_state(
            current_selected_task_id,
            rows,
            current_diagnosis,
            current_log_text,
            current_result_html,
            current_result_text,
        )
        return (
            summary,
            rows,
            _build_report_task_selector_update(rows, selected_task_id),
            diagnosis,
            log_text,
            result_html,
            result_text,
            _build_report_compare_selector_update(current_compare_task_ids),
            _render_monitor_status("未选中有效任务。", "warning"),
        )

    row = table_rows[row_idx]
    if not row:
        selected_task_id, diagnosis, log_text, result_html, result_text = _preserve_report_detail_state(
            current_selected_task_id,
            rows,
            current_diagnosis,
            current_log_text,
            current_result_html,
            current_result_text,
        )
        return (
            summary,
            rows,
            _build_report_task_selector_update(rows, selected_task_id),
            diagnosis,
            log_text,
            result_html,
            result_text,
            _build_report_compare_selector_update(current_compare_task_ids),
            _render_monitor_status("未选中有效任务。", "warning"),
        )

    task_id = row[0]
    return _load_report_task(
        task_id,
        current_compare_task_ids,
        "已加载任务 {task_id} 的结果、日志和诊断。",
        "未选中有效任务。",
    )


def refresh_selected_report_task(task_id: str, selected_compare_task_ids: Optional[list[str]]):
    return _load_report_task(
        task_id,
        selected_compare_task_ids,
        "任务 {task_id} 的结果、日志和诊断已刷新。",
        "当前没有可刷新的任务。",
    )


def view_selected_report_task(task_id: str, selected_compare_task_ids: Optional[list[str]]):
    return _load_report_task(
        task_id,
        selected_compare_task_ids,
        "已查看任务 {task_id} 的结果、日志和诊断。",
        "请先在任务列表中选中一个任务。",
    )


def delete_selected_report_task(task_id: str, selected_compare_task_ids: Optional[list[str]]):
    rows, summary = refresh_task_table()
    available_task_ids = {row[0] for row in rows if row}
    if not task_id:
        diagnosis, log_text, result_html, result_text = _empty_report_center_details()
        return (
            summary,
            rows,
            _build_report_task_selector_update(rows, ""),
            diagnosis,
            log_text,
            result_html,
            result_text,
            _build_report_compare_selector_update(selected_compare_task_ids),
            _render_monitor_status("请先在任务列表中选中一个任务。", "warning"),
        )

    if task_id not in available_task_ids:
        diagnosis, log_text, result_html, result_text = _empty_report_center_details()
        return (
            summary,
            rows,
            _build_report_task_selector_update(rows, ""),
            diagnosis,
            log_text,
            result_html,
            result_text,
            _build_report_compare_selector_update(selected_compare_task_ids),
            _render_monitor_status(f"任务 {task_id} 已不存在或已被删除。", "warning"),
        )

    deleted_task = _task_manager.delete_task(task_id) if _task_manager else None
    rows, summary = refresh_task_table()
    if deleted_task is None:
        return _load_report_task(
            task_id,
            selected_compare_task_ids,
            "已查看任务 {task_id} 的结果、日志和诊断。",
            "请先在任务列表中选中一个任务。",
        )[:-1] + (_render_monitor_status(f"任务 {task_id} 删除失败或已不存在。", "warning"),)

    _purge_report_render_cache(deleted_task)
    diagnosis, log_text, result_html, result_text = _empty_report_center_details()
    return (
        summary,
        rows,
        _build_report_task_selector_update(rows, ""),
        diagnosis,
        log_text,
        result_html,
        result_text,
        _build_report_compare_selector_update(selected_compare_task_ids),
        _render_monitor_status(f"已删除任务 {task_id}，本地评测结果已清理。", "ready"),
    )


def open_submitted_task_results(task_id: str, selected_compare_task_ids: Optional[list[str]] = None):
    rows, summary = refresh_task_table()
    selected_task_id, diagnosis, log_text, result_html, result_text = _build_task_result_view(task_id)
    if selected_task_id:
        status_message = f"已进入评测报告页并加载任务 {selected_task_id}。"
    elif rows:
        diagnosis, log_text, result_html, result_text = _empty_report_center_details()
        status_message = "已进入评测报告页，但未找到目标任务，当前仅同步任务列表。"
    else:
        diagnosis, log_text, result_html, result_text = _empty_report_center_details()
        status_message = "已进入评测报告页，当前暂无任务。"
    return (
        gr.Tabs(selected="reports"),
        summary,
        rows,
        _build_report_task_selector_update(rows, selected_task_id),
        diagnosis,
        log_text,
        result_html,
        result_text,
        _build_report_compare_selector_update(selected_compare_task_ids),
        _render_monitor_status(status_message, "ready"),
    )


# ──────────────────────────────────────────────────────────
# Build the Gradio app
# ──────────────────────────────────────────────────────────
def create_app() -> gr.Blocks:
    _init_registries()
    model_choices = _model_registry.get_model_choices() if _model_registry else []
    dataset_choices = _default_dataset_choices()

    with gr.Blocks(
        title="HM-Eval 统一评测平台",
    ) as app:

        # ── Header ──
        gr.HTML("""
<div class="header-bar">
    <h1>🔬 HM-Eval 统一评测平台</h1>
    <p>基于 evalscope · 支持 float / hmonnx 双后端 → 评测 → 报告一站式流程</p>
</div>""")

        with gr.Tabs() as tabs:

            # ═══════════════════ Tab 1: 提交任务 ═══════════════════
            with gr.Tab("📋 提交评测任务", id="submit") as submit_tab:
                with gr.Row(equal_height=False):
                    # Left column: model + backend
                    with gr.Column(scale=1):
                        gr.Markdown("### 🤖 选择模型")
                        model_root_tb = gr.Textbox(
                            value=_model_registry.get_active_model_root() if _model_registry and _model_registry.get_active_model_root() else "",
                            label="官方模型目录",
                            info="可填写当前服务器上的模型根目录；留空则回退到 YAML 配置里的默认路径",
                            placeholder="/data01/datasets",
                        )
                        model_root_apply_btn = gr.Button("📁 应用目录并刷新模型", size="sm")
                        model_root_status_md = gr.Markdown(_build_model_root_status_markdown())
                        model_dd = gr.Dropdown(
                            choices=model_choices, label="模型",
                            info="基于 hm_eval/model_configs/ 与当前模型目录共同决定可选项",
                        )
                        model_info_md = gr.Markdown("*选择模型后显示详情*")

                        gr.Markdown("### ⚙️ 推理后端")
                        backend_dd = gr.Dropdown(choices=[], label="后端")

                        gr.Markdown("### 🖥️ GPU 选择")
                        gpu_choices = _build_gpu_choices()
                        gpu_dd = gr.Dropdown(
                            choices=gpu_choices,
                            value=gpu_choices[0],
                            label="CUDA 设备",
                            info="选择运行任务的 GPU，提交时自动设置 CUDA_VISIBLE_DEVICES",
                        )
                        gpu_refresh_btn = gr.Button("🔄 刷新 GPU 状态", size="sm")

                    # Right column: datasets + params
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 评测数据集")
                        dataset_cg = gr.CheckboxGroup(
                            choices=dataset_choices,
                            label="数据集（选模型后自动推荐）",
                            elem_id="dataset-checkbox-group",
                        )
                        dataset_choices_state = gr.State(dataset_choices)
                        with gr.Accordion("📝 已选数据集摘要", open=True, visible=False) as dataset_summary_acc:
                            dataset_summary_md = gr.Markdown("")
                        with gr.Row():
                            limit_num = gr.Number(value=0, label="每子集题数 (0=全量)",
                                                  precision=0, minimum=0)
                            max_tok_num = gr.Number(value=512, label="最大生成 tokens",
                                                    info="float 与 hmonnx 均表示 max_new_tokens，不包含 prompt/few-shot 输入长度",
                                                    precision=0, minimum=1)

                        gr.Markdown("#### 🔍 搜索更多数据集")
                        with gr.Row():
                            search_input = gr.Textbox(label="关键词", placeholder="gpqa, bbh, math...")
                            search_btn = gr.Button("搜索")
                        search_cg = gr.CheckboxGroup(
                            choices=[],
                            label="搜索结果",
                            elem_id="dataset-search-results",
                        )
                        add_btn = gr.Button("➕ 添加选中数据集")

                # ── HMONNX export section ──
                with gr.Group(visible=False) as hmonnx_group:
                    gr.Markdown("### 📦 HMONNX 评测配置")
                    gr.Markdown(
                        "> ⚠️ HMONNX 模型导出请在**离线环境**中完成。旧式分离多模态评测需要同时填写 LLM 和 Vision 两个导出 JSON；"
                        "如果 unified `golden_meta_info.json` 或兼容 `meta.json` 已内嵌 vision/audio 子图，只需要填写 LLM meta。"
                    )
                    with gr.Row(equal_height=True):
                        with gr.Group() as llm_meta_group:
                            gr.Markdown("#### LLM 部分")
                            meta_info_dd = gr.Textbox(
                                label="LLM HMONNX meta 文件路径（必填）",
                                info="填写 LLM 导出的 golden_meta_info.json、export_meta_info.json 或兼容 meta.json 的绝对路径。",
                                placeholder="/abs/path/to/llm/golden_meta_info.json",
                            )
                        with gr.Group() as vision_meta_group:
                            gr.Markdown("#### Vision 部分")
                            vision_meta_info_dd = gr.Textbox(
                                label="Vision HMONNX export_meta_info.json 路径（旧式多模态可填）",
                                info="旧式分离导出物需要填写 vision export_meta_info.json；如果 LLM unified meta 已内嵌 vision 子图，这里可留空。",
                                placeholder="/abs/path/to/vision/export_meta_info.json",
                            )

                # ── Submit ──
                submit_btn = gr.Button("🚀 提交评测任务", variant="primary", size="lg")
                submit_result = gr.Markdown("")

                # ── Wiring (within submit tab) ──
                model_root_apply_btn.click(
                    on_apply_model_root,
                    inputs=[model_root_tb],
                    outputs=[model_root_status_md, model_dd, model_info_md, dataset_cg, dataset_choices_state, dataset_summary_acc, dataset_summary_md, backend_dd, hmonnx_group, llm_meta_group, vision_meta_group, meta_info_dd, vision_meta_info_dd],
                    queue=False,
                    show_progress="hidden",
                )
                model_dd.change(
                    on_model_selected, inputs=[model_dd],
                    outputs=[dataset_cg, dataset_choices_state, dataset_summary_acc, dataset_summary_md, backend_dd, model_info_md, hmonnx_group, llm_meta_group, vision_meta_group, meta_info_dd, vision_meta_info_dd],
                    queue=False,
                    show_progress="hidden",
                )
                dataset_cg.change(
                    on_dataset_selection_changed,
                    inputs=[dataset_cg],
                    outputs=[dataset_summary_acc, dataset_summary_md],
                    queue=False,
                    show_progress="hidden",
                )
                backend_dd.change(
                    on_backend_changed,
                    inputs=[backend_dd],
                    outputs=[hmonnx_group, llm_meta_group, vision_meta_group],
                    queue=False,
                    show_progress="hidden",
                )
                gpu_refresh_btn.click(
                    on_refresh_gpu,
                    inputs=[gpu_dd],
                    outputs=[gpu_dd],
                    queue=False,
                    show_progress="hidden",
                )
                search_btn.click(
                    on_search_datasets,
                    inputs=[search_input],
                    outputs=[search_cg],
                    queue=False,
                    show_progress="hidden",
                )
                add_btn.click(
                    on_add_searched,
                    inputs=[search_cg, dataset_cg, dataset_choices_state],
                    outputs=[dataset_cg, dataset_choices_state, dataset_summary_acc, dataset_summary_md],
                    queue=False,
                    show_progress="hidden",
                )

            # ═══════════════════ Tab 2: 评测报告 ═══════════════════
            with gr.Tab("📈 评测报告", id="reports") as reports_tab:
                with gr.Row():
                    report_refresh_btn = gr.Button("🔄 刷新任务状态")
                    report_summary_md = gr.Markdown("自动每 5 秒刷新任务状态；优先用下面的下拉框直接选择任务，查看 / 删除 / 刷新都作用于当前选中任务")

                report_status_html = gr.HTML(
                    _render_monitor_status("评测报告页已就绪。请先用下拉框选择任务；表格仅用于浏览列表。已选任务会随自动刷新同步最新状态。", "info")
                )

                report_task_table = gr.Dataframe(
                    headers=["ID", "模型", "后端", "数据集", "状态", "创建时间"],
                    label="任务结果列表",
                    interactive=False,
                    wrap=True,
                )

                with gr.Row():
                    report_selected_task_id = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="当前任务",
                        info="直接从这里选择要查看的任务，不依赖点表格",
                        allow_custom_value=False,
                        interactive=True,
                    )
                    report_view_btn = gr.Button("👁 查看当前选中任务")
                    report_delete_btn = gr.Button("🗑 删除当前选中任务", variant="stop")
                    report_detail_refresh_btn = gr.Button("🔄 刷新当前任务")

                gr.Markdown("### 📊 多任务对比")
                report_compare_task_ids = gr.Dropdown(
                    choices=[],
                    value=[],
                    multiselect=True,
                    label="选择要对比的任务",
                    info="支持 2 个、3 个、4 个或更多已完成任务同时对比",
                    allow_custom_value=False,
                    interactive=True,
                )
                with gr.Row():
                    report_compare_btn = gr.Button("📊 生成对比")
                    report_compare_clear_btn = gr.Button("清空对比")
                report_compare_html = gr.HTML(_empty_report_compare_details()[0])
                with gr.Accordion("对比文本（可复制）", open=False):
                    report_compare_text = gr.Code(label="Compare Report", language="shell", lines=24)

                report_task_diagnosis_md = gr.Markdown("从下拉框选择任务后会自动显示运行诊断、失败原因或完成状态。")
                report_result_html = gr.HTML(
                    _build_task_message_html("🗂️", "当前暂无可展示任务", "提交任务后，所有状态、诊断、日志和报告都会集中显示在这里。")
                )
                with gr.Accordion("任务日志", open=False):
                    report_task_log_tb = gr.Code(label="任务日志", language="shell", lines=18)

                with gr.Accordion("纯文本报告（可复制）", open=False):
                    report_text = gr.Code(label="Text Report", language="shell", lines=30)

                report_timer = gr.Timer(value=5, active=False)
                report_timer.tick(
                    refresh_report_center_auto,
                    inputs=[report_selected_task_id, report_task_diagnosis_md, report_task_log_tb,
                            report_result_html, report_text, report_compare_task_ids],
                    outputs=[report_summary_md, report_task_table, report_selected_task_id, report_task_diagnosis_md,
                             report_task_log_tb, report_result_html, report_text, report_compare_task_ids, report_status_html],
                    queue=False,
                    show_progress="hidden",
                )

                report_refresh_btn.click(
                    refresh_report_center,
                    inputs=[report_selected_task_id, report_task_diagnosis_md, report_task_log_tb,
                            report_result_html, report_text, report_compare_task_ids],
                    outputs=[report_summary_md, report_task_table, report_selected_task_id, report_task_diagnosis_md,
                             report_task_log_tb, report_result_html, report_text, report_compare_task_ids, report_status_html],
                    queue=False,
                    show_progress="hidden",
                )
                report_task_table.select(
                    on_report_task_row_select,
                    inputs=[report_task_table, report_selected_task_id, report_task_diagnosis_md,
                            report_task_log_tb, report_result_html, report_text, report_compare_task_ids],
                    outputs=[report_summary_md, report_task_table, report_selected_task_id, report_task_diagnosis_md,
                             report_task_log_tb, report_result_html, report_text, report_compare_task_ids, report_status_html],
                    queue=False,
                    show_progress="hidden",
                )
                report_selected_task_id.change(
                    view_selected_report_task,
                    inputs=[report_selected_task_id, report_compare_task_ids],
                    outputs=[report_summary_md, report_task_table, report_selected_task_id, report_task_diagnosis_md,
                             report_task_log_tb, report_result_html, report_text, report_compare_task_ids, report_status_html],
                    queue=False,
                    show_progress="hidden",
                )
                report_view_btn.click(
                    view_selected_report_task,
                    inputs=[report_selected_task_id, report_compare_task_ids],
                    outputs=[report_summary_md, report_task_table, report_selected_task_id, report_task_diagnosis_md,
                             report_task_log_tb, report_result_html, report_text, report_compare_task_ids, report_status_html],
                    queue=False,
                    show_progress="hidden",
                )
                report_delete_btn.click(
                    delete_selected_report_task,
                    inputs=[report_selected_task_id, report_compare_task_ids],
                    outputs=[report_summary_md, report_task_table, report_selected_task_id, report_task_diagnosis_md,
                             report_task_log_tb, report_result_html, report_text, report_compare_task_ids, report_status_html],
                    queue=False,
                    show_progress="hidden",
                )
                report_detail_refresh_btn.click(
                    refresh_selected_report_task,
                    inputs=[report_selected_task_id, report_compare_task_ids],
                    outputs=[report_summary_md, report_task_table, report_selected_task_id, report_task_diagnosis_md,
                             report_task_log_tb, report_result_html, report_text, report_compare_task_ids, report_status_html],
                    queue=False,
                    show_progress="hidden",
                )
                report_compare_btn.click(
                    compare_selected_report_tasks,
                    inputs=[report_compare_task_ids],
                    outputs=[report_compare_task_ids, report_compare_html, report_compare_text, report_status_html],
                    queue=False,
                    show_progress="hidden",
                )
                report_compare_clear_btn.click(
                    clear_report_comparison,
                    outputs=[report_compare_task_ids, report_compare_html, report_compare_text, report_status_html],
                    queue=False,
                    show_progress="hidden",
                )

                reports_tab.select(
                    enter_report_tab,
                    inputs=[report_selected_task_id, report_task_diagnosis_md, report_task_log_tb,
                            report_result_html, report_text, report_compare_task_ids],
                    outputs=[report_summary_md, report_task_table, report_selected_task_id, report_task_diagnosis_md,
                             report_task_log_tb, report_result_html, report_text, report_compare_task_ids, report_status_html, report_timer],
                    queue=False,
                    show_progress="hidden",
                )

                submit_tab.select(
                    pause_report_timer,
                    outputs=[report_timer],
                    queue=False,
                    show_progress="hidden",
                )

            # ═══════════════════ Tab 3: Case 分析 ═══════════════════
            with gr.Tab("🔍 产品级 Case 分析", id="case_analysis"):
                from .case_analysis_tab import build_case_analysis_tab
                build_case_analysis_tab()

        # ── Cross-tab wiring (submit → reports) ──
        submit_btn.click(
            on_submit_task,
            inputs=[model_dd, dataset_cg, backend_dd, limit_num, max_tok_num,
                    gpu_dd, meta_info_dd, vision_meta_info_dd],
            outputs=[submit_result, report_summary_md, report_task_table,
                     report_selected_task_id, report_task_diagnosis_md, report_task_log_tb,
                     report_result_html, report_text, report_compare_task_ids, report_status_html],
            queue=False,
            show_progress="hidden",
        )

    return app
