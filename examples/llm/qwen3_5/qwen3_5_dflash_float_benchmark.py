#!/usr/bin/env python3
"""Float model DFlash acceptance rate benchmark for Qwen3.5/3.6.

Runs categorized questions through the float target + DFlash draft model
and reports acceptance rate statistics. Uses the same question set as
qwen3_5_xh2a_mtp_demo_benchmark.py for comparability.

Usage:
    PYTHONPATH=/data01/home/yujy/work/dflash python qwen3_5_dflash_float_benchmark.py \
        --model weights/Qwen3.6-27B \
        --draft-model weights/Qwen3.6-27B-DFlash \
        --dtype fp16 \
        --per-category 20 \
        --max-new-tokens 512 \
        --out-dir ./tmp/dflash_float_27b_benchmark
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen3_5_spec_decode_metrics import (  # noqa: E402
    resolve_dflash_block_size,
    finalise_spec_metrics,
    _is_multi_gpu_dispatched,
    _generate_dflash_qwen3_5_forced_multigpu,
)

try:
    import dflash.qwen3_5_transformers_split_benchmark as dflash_split_benchmark
    from dflash.model import DFlashDraftModel
    from dflash.qwen3_5_transformers_benchmark import (
        generate_dflash_qwen3_5_forced,
    )
    from dflash.qwen3_5_transformers_split_benchmark import (
        generate_dflash_qwen3_5_forced_split,
    )
except ImportError:
    raise RuntimeError("dflash package is not importable. Set PYTHONPATH to /data01/home/yujy/work/dflash.")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

FLOAT_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _common_sense_questions(count: int) -> List[str]:
    topics = [
        "睡眠", "饮水", "天气", "交通", "做饭", "运动", "学习", "工作", "理财", "旅行",
        "购物", "看病", "用电", "手机", "电脑", "社交", "时间管理", "食品安全", "家务", "急救",
    ]
    templates = [
        "请用三句话回答一个常识问题：日常生活中，关于{topic}最容易被忽略的一点是什么？",
        "请给出简短常识建议：如果要改善{topic}相关体验，最先应该注意什么？",
        "请判断并解释：关于{topic}，普通人常见的误区是什么？",
        "请面向初学者说明：{topic}相关的一个安全注意事项是什么？",
        "请用通俗语言回答：为什么{topic}需要提前规划？",
    ]
    questions = []
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
        total_output_tokens = sum(int(row.get("output_tokens", 0)) for row in group)
        total_verify_calls = sum(int(row.get("target_decoder_calls", 0)) for row in group)
        total_drafts = sum(int(row.get("draft_tokens_total", 0)) for row in group)
        total_accepted = sum(int(row.get("accepted_drafts_total", 0)) for row in group)
        acceptance_rates = [float(row.get("acceptance_rate", 0.0)) for row in group]
        avg_accepted_per_round = [float(row.get("avg_accepted_per_round", 0.0)) for row in group]
        summaries.append(
            {
                "category": category,
                "ok": len(group),
                "failed": failed,
                "tokens_per_verify": total_output_tokens / total_verify_calls if total_verify_calls else 0.0,
                "avg_acceptance_rate": mean(acceptance_rates),
                "avg_accepted_per_round": mean(avg_accepted_per_round),
                "total_output_tokens": total_output_tokens,
                "total_draft_tokens": total_drafts,
                "total_accepted_drafts": total_accepted,
                "overall_acceptance_rate": total_accepted / total_drafts if total_drafts else 0.0,
            }
        )
    return summaries


def write_outputs(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize(rows)
    raw_json = out_dir / "dflash_float_benchmark_raw.json"
    raw_csv = out_dir / "dflash_float_benchmark_raw.csv"
    summary_json = out_dir / "dflash_float_benchmark_summary.json"
    summary_md = out_dir / "dflash_float_benchmark_summary.md"
    raw_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with raw_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# DFlash Float Benchmark Summary",
        "",
        "Acceptance rate benchmark for DFlash speculative decoding (float model).",
        "",
        "| 类型 | 成功 | 失败 | tokens/verify | avg acceptance rate | avg accepted/round | overall acceptance rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {category} | {ok} | {failed} | {tokens_per_verify:.3f} | "
            "{avg_acceptance_rate:.2%} | {avg_accepted_per_round:.2f} | "
            "{overall_acceptance_rate:.2%} |".format(**item)
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_row(
    *,
    category: str,
    index: int,
    question: str,
    result: Dict[str, Any],
    block_size: int,
) -> Dict[str, Any]:
    spec = result.get("speculative", {})
    draft_capacity = block_size - 1
    output_tokens = int(spec.get("output_tokens", 0))
    target_decoder_calls = int(spec.get("target_decoder_calls", 0))
    accepted_total = int(spec.get("accepted_drafts_total", 0))
    draft_tokens_total = int(spec.get("draft_tokens_total", 0))
    acceptance_rate = float(spec.get("overall_acceptance_rate", 0.0))
    avg_accepted = float(spec.get("avg_accepted_per_round", 0.0))
    text_match = result.get("text_match", False)

    return {
        "category": category,
        "index": index,
        "status": "ok",
        "question": question,
        "output_tokens": output_tokens,
        "target_decoder_calls": target_decoder_calls,
        "accepted_drafts_total": accepted_total,
        "draft_tokens_total": draft_tokens_total,
        "acceptance_rate": acceptance_rate,
        "avg_accepted_per_round": avg_accepted,
        "text_match": text_match,
        "block_size": block_size,
        "draft_capacity": draft_capacity,
        "spec_text_preview": spec.get("text", "")[:200],
    }


@torch.inference_mode()
def _run_single_dflash_metrics(
    target,
    draft,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    block_size: int,
    enable_thinking: bool,
    model_path: str,
    use_split: bool = False,
) -> Dict[str, Any]:
    """Run DFlash metrics for a single prompt without reloading models."""
    use_multigpu = _is_multi_gpu_dispatched(target) or _is_multi_gpu_dispatched(draft)

    draft_decode_calls = 0
    original_forward = draft.forward

    def counted_forward(*args, **kwargs):
        nonlocal draft_decode_calls
        draft_decode_calls += 1
        return original_forward(*args, **kwargs)

    draft.forward = counted_forward
    try:
        if use_multigpu:
            spec = _generate_dflash_qwen3_5_forced_multigpu(
                draft, target, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                enable_thinking=enable_thinking,
                block_size=block_size,
            )
        elif use_split:
            spec, _diag = generate_dflash_qwen3_5_forced_split(
                draft, target, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                enable_thinking=enable_thinking,
                block_size=block_size,
            )
        else:
            spec = generate_dflash_qwen3_5_forced(
                draft, target, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                enable_thinking=enable_thinking,
                block_size=block_size,
            )
    finally:
        draft.forward = original_forward

    accepted_drafts_per_round = [max(int(v) - 1, 0) for v in spec.acceptance_lengths]
    return finalise_spec_metrics(
        prompt=prompt,
        draft_mode="dflash",
        model_name=model_path,
        baseline_text=None,
        baseline_decoder_calls=None,
        baseline_prefill_calls=None,
        spec_text=spec.text,
        spec_output_tokens=int(spec.num_output_tokens),
        target_prefill_calls=1,
        target_decoder_calls=len(spec.acceptance_lengths),
        accepted_drafts_per_round=accepted_drafts_per_round,
        draft_capacity=max(block_size - 1, 0),
        extra_counts={
            "dflash_prefill_calls": 0,
            "dflash_decode_calls": draft_decode_calls,
        },
    )


def run_benchmark(args: argparse.Namespace) -> None:
    model_path = str(Path(args.model).resolve()) if not args.model.startswith("/") else args.model
    draft_model_path = str(Path(args.draft_model).resolve()) if not args.draft_model.startswith("/") else args.draft_model
    block_size = resolve_dflash_block_size(draft_model_path, args.block_size)
    print(f"model: {model_path}", flush=True)
    print(f"draft_model: {draft_model_path}", flush=True)
    print(f"block_size: {block_size}", flush=True)
    print(f"draft_capacity: {block_size - 1}", flush=True)
    print(f"dtype: {args.dtype}", flush=True)

    torch_dtype = FLOAT_DTYPE_MAP[args.dtype]
    device = args.device
    print(f"Loading target model to {device}...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
    ).eval()
    print(f"Loading draft model to {device}...", flush=True)
    draft = DFlashDraftModel.from_pretrained(
        draft_model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Models loaded.", flush=True)

    rows: List[Dict[str, Any]] = []
    raw_json = args.out_dir / "dflash_float_benchmark_raw.json"
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
                    result = _run_single_dflash_metrics(
                        target, draft, tokenizer, question,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        enable_thinking=args.enable_thinking,
                        model_path=model_path,
                        use_split=args.use_split,
                    )
                    row = _build_row(
                        category=category,
                        index=index,
                        question=question,
                        result=result,
                        block_size=block_size,
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    import traceback
                    traceback.print_exc()
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
        del target, draft
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"BENCH_DONE out_dir={args.out_dir}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.5/3.6 float DFlash acceptance rate benchmark (3 categories x N questions)"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to target model (e.g. weights/Qwen3.6-27B)")
    parser.add_argument("--draft-model", type=str, required=True, help="Path to DFlash draft model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to load models on (single GPU)")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--block-size", type=int, default=None, help="Override block size (auto from draft config)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--per-category", dest="per_category", type=int, default=20)
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=Path,
        default=Path("./tmp/dflash_float_benchmark"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--use-split", action="store_true",
        help="Use split ctx-KV mode (qwen3_5_transformers_split_benchmark) instead of standard forced decode",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    args.out_dir = Path(args.out_dir).resolve()
    run_benchmark(args)
