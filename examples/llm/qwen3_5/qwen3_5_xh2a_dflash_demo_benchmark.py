#!/usr/bin/env python3
"""Categorized batch benchmark runner for Qwen3.5/3.6 DFlash spec-decode on hmonnx (xh2a).

Adapted from qwen3_5_xh2a_mtp_demo_benchmark.py for DFlash speculative decoding.
Measures acceptance rate of quantized DFlash models.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qwen3_5_spec_decode_metrics import run_dense_spec_generate_once  # noqa: E402
from qwen3_5_xh2a_spec_decode_test import (  # noqa: E402
    load_spec_decode_runtime,
    parse_auto_offload_max_memory,
    parse_dtype,
)


def _common_sense_questions(count: int) -> List[str]:
    topics = [
        "睡眠", "饮水", "天气", "交通", "做饭", "运动", "学习", "工作", "理财", "旅行",
        "购物", "看病", "用电", "手机", "电脑", "社交", "时间管理", "食品安全", "家务", "急救",
    ]
    questions = []
    templates = [
        "请用三句话回答一个常识问题：日常生活中，关于{topic}最容易被忽略的一点是什么？",
        "请给出简短常识建议：如果要改善{topic}相关体验，最先应该注意什么？",
        "请判断并解释：关于{topic}，普通人常见的误区是什么？",
        "请面向初学者说明：{topic}相关的一个安全注意事项是什么？",
        "请用通俗语言回答：为什么{topic}需要提前规划？",
    ]
    for index in range(count):
        topic = topics[index % len(topics)]
        questions.append(templates[index % len(templates)].format(topic=topic) + f" 编号{index + 1}。")
    return questions


def _reasoning_questions(count: int) -> List[str]:
    questions = []
    for index in range(count):
        base = index + 3
        mod = index % 5
        if mod == 0:
            questions.append(
                f"逻辑推理题：甲、乙、丙三人排队，甲不在第一位，乙在丙前面。若队伍只有三人，请说明所有可能顺序。题号{index + 1}。"
            )
        elif mod == 1:
            questions.append(
                f"逻辑推理题：一个数加上{base}后再乘以2，结果比原数的3倍少{base - 1}。请列式求这个数。题号{index + 1}。"
            )
        elif mod == 2:
            questions.append(
                f"逻辑推理题：如果所有A都是B，有些B是C，能否推出有些A是C？请给出理由。题号{index + 1}。"
            )
        elif mod == 3:
            questions.append(
                f"逻辑推理题：四盏灯编号1到4，每次切换相邻两盏灯。初始全灭，怎样让第{(index % 4) + 1}盏灯单独亮起？若不能，请说明原因。题号{index + 1}。"
            )
        else:
            questions.append(
                f"逻辑推理题：有{base}个球，其中一个偏重。只有天平且最多称两次，在哪些{base}值下能保证找出偏重球？请简要分析。题号{index + 1}。"
            )
    return questions


def _summary_questions(count: int) -> List[str]:
    passages = [
        "某城市推动绿色出行，新增公交专用道，优化地铁接驳，并鼓励企业错峰通勤。三个月后，主城区早高峰拥堵指数下降，公交准点率提升，但部分居民反映换乘距离仍偏长。",
        "一家制造企业引入智能质检系统，将图像识别用于缺陷检测。系统上线后漏检率下降，人工复核压力减轻，但早期数据标注不足导致少量特殊缺陷识别不稳定。",
        "学校试行项目式学习，让学生围绕真实问题分组调研、实验和展示。学生参与度明显提高，表达能力有所提升，但教师需要投入更多时间设计任务和评价标准。",
        "医院上线线上复诊平台，慢病患者可以提交指标、获取用药建议并预约检查。平台减少了排队时间，但老年用户在注册和上传资料时仍需要家属协助。",
        "社区建立共享工具库，居民可以预约借用电钻、梯子等低频用品。该项目降低了重复购买，也促进邻里互动，但维护和归还管理需要更细致的规则。",
    ]
    questions = []
    for index in range(count):
        passage = passages[index % len(passages)]
        questions.append(
            f"请将下面文本总结为三点，并指出一个潜在问题。文本：{passage} 题号{index + 1}。"
        )
    return questions


def build_questions(per_category: int) -> Dict[str, List[str]]:
    return {
        "常识回答": _common_sense_questions(per_category),
        "逻辑推理": _reasoning_questions(per_category),
        "文本总结": _summary_questions(per_category),
    }


def mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    categories = sorted({row["category"] for row in rows})
    for category in categories:
        group = [row for row in rows if row["category"] == category and row["status"] == "ok"]
        failed = len([row for row in rows if row["category"] == category and row["status"] != "ok"])
        total_decode_tokens = sum(float(row.get("decode_tokens", 0.0)) for row in group)
        total_verify_calls = sum(float(row.get("verify_calls", 0.0)) for row in group)
        total_drafts = sum(float(row.get("draft_tokens", 0.0)) for row in group)
        total_accepted = sum(float(row.get("accepted_drafts", 0.0)) for row in group)
        decode_tps = [float(row.get("decode_tps", 0.0)) for row in group]
        decode_model_avg_ms = [float(row.get("asic_decode_model_avg_ms", 0.0)) for row in group]
        total_decode_model_ms = sum(float(row.get("asic_decode_model_total_ms", 0.0)) for row in group)
        total_decode_model_calls = sum(float(row.get("asic_decode_model_calls", 0.0)) for row in group)
        summaries.append(
            {
                "category": category,
                "ok": len(group),
                "failed": failed,
                "weighted_speedup_tokens_per_verify": total_decode_tokens / total_verify_calls if total_verify_calls else 0.0,
                "avg_decode_fps": mean(decode_tps),
                "fastest_decode_fps": max(decode_tps) if decode_tps else 0.0,
                "slowest_decode_fps": min(decode_tps) if decode_tps else 0.0,
                "avg_acceptance_rate": mean(float(row.get("acceptance_rate", 0.0)) for row in group),
                "avg_asic_decode_model_ms_per_run": mean(decode_model_avg_ms),
                "weighted_asic_decode_model_ms_per_run": total_decode_model_ms / total_decode_model_calls
                if total_decode_model_calls
                else 0.0,
                "fastest_asic_decode_model_ms_per_run": min(decode_model_avg_ms) if decode_model_avg_ms else 0.0,
                "slowest_asic_decode_model_ms_per_run": max(decode_model_avg_ms) if decode_model_avg_ms else 0.0,
                "avg_asic_decode_model_total_ms_per_question": total_decode_model_ms / len(group) if group else 0.0,
                "total_asic_decode_model_ms": total_decode_model_ms,
                "total_asic_decode_model_calls": total_decode_model_calls,
                "avg_decode_tokens": mean(float(row.get("decode_tokens", 0.0)) for row in group),
                "total_decode_tokens": total_decode_tokens,
                "total_draft_tokens": total_drafts,
                "total_accepted_drafts": total_accepted,
            }
        )
    return summaries


def write_outputs(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize(rows)
    raw_json = out_dir / "dflash_demo_benchmark_raw.json"
    raw_csv = out_dir / "dflash_demo_benchmark_raw.csv"
    summary_json = out_dir / "dflash_demo_benchmark_summary.json"
    summary_md = out_dir / "dflash_demo_benchmark_summary.md"
    raw_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with raw_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# DFlash Demo Benchmark Summary",
        "",
        "Acceptance rate benchmark for DFlash speculative decoding (quantized model).",
        "",
        "| 类型 | 成功 | 失败 | tokens/verify | avg acceptance rate | avg decode tps | total drafts | total accepted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {category} | {ok} | {failed} | {weighted_speedup_tokens_per_verify:.3f} | "
            "{avg_acceptance_rate:.2%} | {avg_decode_fps:.2f} | "
            "{total_draft_tokens:.0f} | {total_accepted_drafts:.0f} |".format(**item)
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_cuda_graph_modules(modules_arg: str) -> Optional[Tuple[str, ...]]:
    if not modules_arg or not modules_arg.strip():
        return None
    modules = tuple(part.strip().lower() for part in modules_arg.split(",") if part.strip())
    return modules or None


def _build_row(
    *,
    category: str,
    index: int,
    question: str,
    result: Dict[str, Any],
    block_size: int,
    hidden_output_name: Optional[str],
    draft_capacity: int,
) -> Dict[str, Any]:
    text = result.get("text", "") or ""
    output_tokens = int(result.get("output_tokens", 0))
    target_decoder_calls = int(result.get("target_decoder_calls", 0))
    latency_s = float(result.get("latency_s", 0.0))

    decode_calls = int(result.get("dflash_decode_calls", 0))
    draft_tokens_calc = decode_calls * draft_capacity
    draft_tokens_total = int(result.get("draft_tokens_total", draft_tokens_calc))
    draft_tokens = draft_tokens_calc if draft_tokens_calc > 0 else draft_tokens_total

    accepted_drafts_total = int(result.get("accepted_drafts_total", max(0, output_tokens - target_decoder_calls)))
    accepted_drafts_calc = output_tokens - target_decoder_calls
    accepted_drafts = accepted_drafts_calc if accepted_drafts_calc >= 0 else accepted_drafts_total

    acceptance_rate = (accepted_drafts_total / draft_tokens_total) if draft_tokens_total > 0 else 0.0
    decode_tps = (output_tokens / latency_s) if latency_s > 0 else 0.0
    asic_decode_model_total_ms = latency_s * 1000.0
    asic_decode_model_avg_ms = (
        asic_decode_model_total_ms / target_decoder_calls if target_decoder_calls > 0 else 0.0
    )

    row: Dict[str, Any] = {
        "category": category,
        "index": index,
        "status": "ok",
        "question": question,
        "output_chars": len(text),
        "decode_tokens": output_tokens,
        "verify_calls": target_decoder_calls,
        "draft_tokens": draft_tokens,
        "accepted_drafts": accepted_drafts,
        "acceptance_rate": acceptance_rate,
        "decode_tps": decode_tps,
        "asic_decode_model_total_ms": asic_decode_model_total_ms,
        "asic_decode_model_calls": target_decoder_calls,
        "asic_decode_model_avg_ms": asic_decode_model_avg_ms,
        "spec_decode_mode": "dflash",
        "block_size": block_size,
        "hidden_output_name": hidden_output_name,
        "avg_accepted_per_round": float(result.get("avg_accepted_per_round", 0.0)),
        "num_rounds": int(result.get("num_rounds", target_decoder_calls)),
    }
    return row


def run_benchmark(args: argparse.Namespace) -> None:
    cuda_graph_modules = parse_cuda_graph_modules(args.cuda_graph_modules)
    runtime, tokenizer, meta_info = load_spec_decode_runtime(
        meta_path=args.config,
        dtype=parse_dtype(args.dtype),
        device=args.device,
        exec_device=args.exec_device,
        auto_offload=not args.disable_auto_offload,
        auto_offload_max_memory=parse_auto_offload_max_memory(args.auto_offload_max_memory),
        prefill_auto_offload_max_memory=parse_auto_offload_max_memory(args.prefill_auto_offload_max_memory),
        decode_auto_offload_max_memory=parse_auto_offload_max_memory(args.decode_auto_offload_max_memory),
        resource_tight_mode=args.resource_tight_mode,
        num_draft_tokens_override=(args.num_draft_tokens if args.num_draft_tokens > 0 else None),
        enable_cuda_graph=args.enable_cuda_graph,
        cuda_graph_modules=cuda_graph_modules,
        cuda_graph_warmup_runs=args.cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=args.cuda_graph_graph_warmup_runs,
    )

    spec_decode = meta_info.get("spec_decode", {})
    spec_mode = spec_decode.get("mode")
    if spec_mode != "dflash":
        raise ValueError(f"Expected spec_decode mode 'dflash', got '{spec_mode}'. Use mtp benchmark for MTP models.")

    block_size = int(getattr(runtime, "block_size", spec_decode.get("block_size", 0) or 0))
    hidden_output_name = getattr(runtime, "hidden_output_name", spec_decode.get("hidden_output_name"))
    # NOTE: For DFlash ONNX, meta.json's `block_size` already equals num_drafts per round
    # (verify_length = block_size + 1 = seed + drafts).  The previous `block_size - 1`
    # under-counted draft capacity by 1, making the displayed `draft_tokens` inconsistent
    # with the runtime's `draft_tokens_total` (which uses actual #drafts emitted).
    draft_capacity = block_size

    print(f"spec_decode_mode: {spec_mode}", flush=True)
    print(f"block_size: {block_size}", flush=True)
    print(f"hidden_output_name: {hidden_output_name}", flush=True)
    print(f"draft_capacity: {draft_capacity}", flush=True)

    rows: List[Dict[str, Any]] = []
    raw_json = args.out_dir / "dflash_demo_benchmark_raw.json"
    if args.resume and raw_json.exists():
        rows = json.loads(raw_json.read_text(encoding="utf-8"))
    done = {(row["category"], int(row["index"])) for row in rows if row.get("status") == "ok"}

    try:
        for category, questions in build_questions(args.per_category).items():
            for index, question in enumerate(questions, 1):
                if (category, index) in done:
                    continue
                print(
                    f"BENCH_PROGRESS category={category} index={index}/{len(questions)}",
                    flush=True,
                )
                try:
                    result = run_dense_spec_generate_once(
                        runtime=runtime,
                        tokenizer=tokenizer,
                        prompt=question,
                        max_new_tokens=args.max_new_tokens,
                        enable_thinking=args.enable_thinking,
                        repetition_penalty=args.repetition_penalty,
                        presence_penalty=args.presence_penalty,
                    )
                    row = _build_row(
                        category=category,
                        index=index,
                        question=question,
                        result=result,
                        block_size=block_size,
                        hidden_output_name=hidden_output_name,
                        draft_capacity=draft_capacity,
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    row = {
                        "category": category,
                        "index": index,
                        "status": "failed",
                        "question": question,
                        "error": repr(exc),
                    }
                rows.append(row)
                write_outputs(args.out_dir, rows)
        write_outputs(args.out_dir, rows)
    finally:
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"BENCH_DONE out_dir={args.out_dir}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.5/3.6 xh2a hmonnx DFlash demo benchmark (3 categories x N questions)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to meta.json from export (must contain spec_decode section with mode=dflash)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--exec_device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--disable_auto_offload", action="store_true")
    parser.add_argument("--auto_offload_max_memory", type=str, default=None)
    parser.add_argument("--prefill_auto_offload_max_memory", type=str, default=None)
    parser.add_argument("--decode_auto_offload_max_memory", type=str, default=None)
    parser.add_argument("--resource_tight_mode", action="store_true")
    parser.add_argument(
        "--num_draft_tokens",
        type=int,
        default=0,
        help="Override number of draft tokens per round (0 = use meta.json value)",
    )
    parser.add_argument("--enable_cuda_graph", action="store_true")
    parser.add_argument(
        "--cuda_graph_modules",
        type=str,
        default="",
        help="Comma-separated session names: prefill,decode,draft_prefill,draft_context,draft_context_decode,draft_decode",
    )
    parser.add_argument("--cuda_graph_warmup_runs", type=int, default=3)
    parser.add_argument("--cuda_graph_graph_warmup_runs", type=int, default=6)
    parser.add_argument("--per-category", dest="per_category", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=Path,
        default=Path("./tmp/dflash_demo_xh2a_benchmark"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    args.out_dir = Path(args.out_dir).resolve()
    run_benchmark(args)
