"""Gradio tab for product-level case analysis (qualitative evaluation).

This module builds the third tab in the HM-Eval app. It provides:
- Model path inputs (float HF dir or HMONNX meta path)
- GPU selection
- Multimodal prompt input (text + image + audio + PDF)
- Inference execution and result display
- Case history with badcase tagging (for future DB integration)
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import gradio as gr

logger = logging.getLogger(__name__)

_inference_lock = threading.Lock()


def _query_gpu_choices() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return ["0"]
        choices = []
        free_ids = []
        all_ids = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpu_id = parts[0]
                name = parts[1]
                mem_used = parts[3]
                mem_total = parts[2]
                util = parts[5]
                mem_free = parts[4]
                free_pct = int(float(mem_free) / max(float(mem_total), 1) * 100)
                all_ids.append(gpu_id)
                if float(mem_used) <= 2048:
                    status = "🟢 空闲"
                    free_ids.append(gpu_id)
                elif float(mem_used) <= float(mem_total) * 0.4:
                    status = "🟡 部分占用"
                else:
                    status = "🔴 繁忙"
                choices.append(
                    f"GPU {gpu_id} — {name} | {status} | 显存 {mem_used}/{mem_total} MiB ({free_pct}% 空闲) | 利用率 {util}%"
                )
        if len(free_ids) >= 2:
            choices.append(f"GPU {','.join(free_ids)} — 所有空闲卡 ({len(free_ids)} 张，推荐大模型多模态)")
        if len(all_ids) > 1:
            choices.append(f"GPU {','.join(all_ids)} — 全部 {len(all_ids)} 张卡")
        return choices if choices else ["0"]
    except Exception:
        return ["0"]


def _parse_gpu_id(gpu_choice: str) -> str:
    if not gpu_choice:
        return "0"
    if gpu_choice.startswith("GPU "):
        return gpu_choice.split("—")[0].replace("GPU", "").strip()
    return gpu_choice.split()[0] if gpu_choice else "0"


def _collect_file_paths(file_list: Optional[list]) -> list[str]:
    if not file_list:
        return []
    paths = []
    for f in file_list:
        if isinstance(f, str):
            paths.append(f)
        elif isinstance(f, dict):
            paths.append(f.get("name", "") or f.get("path", ""))
        elif hasattr(f, "name"):
            paths.append(f.name)
    return [p for p in paths if p]


def _classify_files(file_paths: list[str]) -> tuple[list[str], list[str], list[str]]:
    images, audios, pdfs = [], [], []
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    for p in file_paths:
        ext = Path(p).suffix.lower()
        if ext == ".pdf":
            pdfs.append(p)
        elif ext in img_exts:
            images.append(p)
        elif ext in audio_exts:
            audios.append(p)
        else:
            images.append(p)
    return images, audios, pdfs


def on_run_inference(
    backend_type: str,
    float_model_path: str,
    hmonnx_meta_path: str,
    vision_meta_path: str,
    gpu_choice: str,
    prompt_text: str,
    uploaded_files: Optional[list],
    max_tokens: int,
    max_pdf_pages: int,
):
    """Execute case inference and return results."""
    if not backend_type:
        return "⚠️ 请选择推理后端", "", gr.update()

    if backend_type == "float":
        model_path = (float_model_path or "").strip()
        if not model_path:
            return "⚠️ 请填写浮点模型路径", "", gr.update()
        if not Path(model_path).is_dir():
            return f"⚠️ 浮点模型路径不存在: {model_path}", "", gr.update()
    else:
        model_path = (hmonnx_meta_path or "").strip()
        if not model_path:
            return "⚠️ 请填写 HMONNX meta 文件路径", "", gr.update()
        if not Path(model_path).is_file():
            return f"⚠️ HMONNX meta 文件不存在: {model_path}", "", gr.update()

    if not prompt_text.strip() and not uploaded_files:
        return "⚠️ 请至少输入文本 prompt 或上传文件", "", gr.update()

    gpu_id = _parse_gpu_id(gpu_choice)

    file_paths = _collect_file_paths(uploaded_files)
    images, audios, pdfs = _classify_files(file_paths)

    if not _inference_lock.acquire(blocking=False):
        return "⚠️ 当前有推理任务正在执行，请等待完成后再试", "", gr.update()

    try:
        import tempfile
        import time

        request_data = {
            "backend_type": backend_type,
            "model_path": model_path,
            "gpu_id": gpu_id,
            "prompt_text": prompt_text,
            "image_paths": images,
            "audio_paths": audios,
            "pdf_paths": pdfs,
            "max_tokens": int(max_tokens) if max_tokens else 2048,
            "vision_meta_path": (vision_meta_path or "").strip(),
            "max_pdf_pages": int(max_pdf_pages) if max_pdf_pages else 5,
        }

        tmp_dir = Path(tempfile.mkdtemp(prefix="hm_case_"))
        request_path = tmp_dir / "request.json"
        request_path.write_text(json.dumps(request_data, ensure_ascii=False), encoding="utf-8")

        result = subprocess.run(
            ["python", "-m", "hm_eval.case_worker", str(request_path)],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            timeout=600,
        )

        result_path = request_path.with_suffix(".result.json")
        if not result_path.exists():
            stderr_tail = (result.stderr or "")[-500:]
            return f"❌ 推理子进程失败\n\n```\n{stderr_tail}\n```", "", gr.update()

        record_data = json.loads(result_path.read_text(encoding="utf-8"))

        status_md = (
            f"### ✅ 推理完成\n"
            f"- Case ID: `{record_data['case_id']}`\n"
            f"- 模型: `{record_data['model_path']}`\n"
            f"- 后端: {record_data['backend_type']}\n"
            f"- GPU: {gpu_id}\n"
            f"- 耗时: {record_data['elapsed_seconds']:.2f}s\n"
            f"- 输入文件: {len(record_data['input_files'])} 个"
        )
        return status_md, record_data["response"], _build_case_history_update()
    except subprocess.TimeoutExpired:
        return "❌ 推理超时（超过 10 分钟）", "", gr.update()
    except Exception as e:
        logger.exception("Case inference error")
        return f"❌ 推理异常: {type(e).__name__}: {e}", "", gr.update()
    finally:
        _inference_lock.release()


def _build_case_history_update():
    from .core.case_analysis import list_cases
    cases = list_cases(limit=50)
    if not cases:
        return gr.update(value=[], visible=True)
    rows = []
    for c in cases:
        label_display = {"good": "✅ 好", "bad": "❌ 差", "uncertain": "❓ 待定"}.get(c.label, "—")
        rows.append([
            c.case_id,
            c.backend_type,
            Path(c.model_path).name if c.model_path else "",
            c.prompt_text[:40] + ("..." if len(c.prompt_text) > 40 else ""),
            f"{c.elapsed_seconds:.1f}s",
            label_display,
            c.created_at[:19] if c.created_at else "",
        ])
    return gr.update(value=rows, visible=True)


def on_tag_case(case_id: str, label: str):
    """Tag a case as good/bad/uncertain for badcase collection."""
    from .core.case_analysis import _ensure_case_store
    import json

    if not case_id:
        return "请先选择一个 case"

    store_dir = _ensure_case_store()
    path = store_dir / f"{case_id}.json"
    if not path.exists():
        return f"Case {case_id} 不存在"

    data = json.loads(path.read_text(encoding="utf-8"))
    data["label"] = label
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"已标记 {case_id} 为 {label}"


def on_refresh_case_history():
    return _build_case_history_update()


def on_backend_type_changed(backend_type: str):
    if backend_type == "float":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)


def build_case_analysis_tab():
    """Build and return the case analysis tab content. Call inside gr.Tab context."""

    gr.Markdown(
        "> 产品级 Case 分析：用于定性评估模型在具体场景下的表现。"
        "支持文本、图像、音频、PDF 等多模态输入，直接调用浮点或 HMONNX 模型进行推理。"
        "结果会自动保存，后续可标记 badcase 用于搜集和分析。"
    )

    with gr.Row(equal_height=False):
        # Left: model config
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 模型配置")

            backend_type_radio = gr.Radio(
                choices=["float", "hmonnx"],
                value="float",
                label="推理后端",
                info="选择浮点推理或 HMONNX 量化推理",
            )

            with gr.Group(visible=True) as float_group:
                gr.Markdown("#### 浮点模型")
                float_model_path_tb = gr.Textbox(
                    label="HF 模型目录路径",
                    placeholder="/data01/datasets/Qwen3.5-35B-A3B",
                    info="填写 HuggingFace 格式模型的本地绝对路径",
                )

            with gr.Group(visible=False) as hmonnx_group:
                gr.Markdown("#### HMONNX 模型")
                hmonnx_meta_path_tb = gr.Textbox(
                    label="LLM meta 文件路径（必填）",
                    placeholder="/path/to/golden_meta_info.json",
                    info="填写 golden_meta_info.json / export_meta_info.json / meta.json 的绝对路径",
                )

            with gr.Group(visible=False) as vision_group:
                vision_meta_path_tb = gr.Textbox(
                    label="Vision meta 文件路径（多模态可选）",
                    placeholder="/path/to/vision/export_meta_info.json",
                    info="旧式分离导出需要填写；unified meta 已内嵌 vision 子图时可留空",
                )

            gr.Markdown("### 🖥️ GPU 选择")
            gpu_choices = _query_gpu_choices()
            case_gpu_dd = gr.Dropdown(
                choices=gpu_choices,
                value=gpu_choices[0] if gpu_choices else "0",
                label="CUDA 设备",
                info="选择推理使用的 GPU",
            )
            case_gpu_refresh_btn = gr.Button("🔄 刷新 GPU 状态", size="sm")

            gr.Markdown("### ⚙️ 推理参数")
            max_tokens_num = gr.Number(
                value=2048, label="最大生成 tokens",
                precision=0, minimum=1, maximum=8192,
            )
            max_pdf_pages_num = gr.Number(
                value=5, label="PDF 最大页数",
                info="PDF 文件最多读取的页数（每页转为一张图片输入模型）",
                precision=0, minimum=1, maximum=20,
            )

        # Right: prompt input
        with gr.Column(scale=1):
            gr.Markdown("### 💬 输入 Prompt")
            prompt_tb = gr.Textbox(
                label="文本 Prompt",
                placeholder="请输入你的问题或指令...",
                lines=6,
            )

            gr.Markdown("### 📎 上传文件")
            file_upload = gr.File(
                label="上传图像 / 音频 / PDF 文件",
                file_count="multiple",
                file_types=[".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp",
                            ".wav", ".mp3", ".flac", ".ogg", ".m4a",
                            ".pdf"],
                type="filepath",
            )
            gr.Markdown(
                "*支持格式：图像 (png/jpg/bmp/gif/webp)、音频 (wav/mp3/flac)、PDF*\n\n"
                "*PDF 会按页转为图片输入模型，适合文档理解类测试*"
            )

    # Submit
    run_btn = gr.Button("🚀 执行推理", variant="primary", size="lg")
    case_status_md = gr.Markdown("")

    # Output
    gr.Markdown("### 📝 模型输出")
    case_output_tb = gr.Textbox(
        label="模型回答",
        lines=12,
        interactive=False,
    )

    # Case history
    gr.Markdown("### 📋 Case 历史记录")
    gr.Markdown(
        "> 所有推理结果自动保存。可标记 badcase 用于后续搜集分析。"
    )
    case_history_table = gr.Dataframe(
        headers=["Case ID", "后端", "模型", "Prompt", "耗时", "标签", "时间"],
        label="历史记录",
        interactive=False,
        wrap=True,
    )
    with gr.Row():
        tag_case_id_tb = gr.Textbox(label="Case ID", placeholder="从上方表格复制 Case ID")
        tag_label_dd = gr.Dropdown(
            choices=["good", "bad", "uncertain"],
            label="标记",
            info="标记该 case 的质量",
        )
        tag_btn = gr.Button("🏷️ 标记", size="sm")
        refresh_history_btn = gr.Button("🔄 刷新历史", size="sm")
    tag_result_md = gr.Markdown("")

    # Wiring
    backend_type_radio.change(
        on_backend_type_changed,
        inputs=[backend_type_radio],
        outputs=[float_group, hmonnx_group, vision_group],
        queue=False,
        show_progress="hidden",
    )

    case_gpu_refresh_btn.click(
        lambda: gr.update(choices=_query_gpu_choices()),
        outputs=[case_gpu_dd],
        queue=False,
        show_progress="hidden",
    )

    run_btn.click(
        on_run_inference,
        inputs=[
            backend_type_radio, float_model_path_tb, hmonnx_meta_path_tb,
            vision_meta_path_tb, case_gpu_dd, prompt_tb, file_upload,
            max_tokens_num, max_pdf_pages_num,
        ],
        outputs=[case_status_md, case_output_tb, case_history_table],
    )

    tag_btn.click(
        on_tag_case,
        inputs=[tag_case_id_tb, tag_label_dd],
        outputs=[tag_result_md],
        queue=False,
        show_progress="hidden",
    )

    refresh_history_btn.click(
        on_refresh_case_history,
        outputs=[case_history_table],
        queue=False,
        show_progress="hidden",
    )
