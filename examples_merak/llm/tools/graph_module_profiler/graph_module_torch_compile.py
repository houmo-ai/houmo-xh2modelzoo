import argparse
import csv
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
import torch.fx as fx

from xhmodel_merak.xh_llm import (
    AutoLLMConfig,
    AutoLLMModel,
    LLMInferenceContextManager,
    LLMModelState,
)
from xhquant.api import Config, FXInterpreter, get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers


def synchronize_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def clone_call_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_call_value(item) for item in value)
    if isinstance(value, list):
        return [clone_call_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_call_value(item) for key, item in value.items()}
    return value


def clone_call_args(args: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(clone_call_value(arg) for arg in args)


def warmup_runner(
    name: str,
    runner: Callable[..., Any],
    warmup_runs: int,
    device: str,
    logger,
    *args,
    **kwargs,
) -> None:
    if warmup_runs <= 0:
        return

    logger.info(f"Start warmup for {name}: {warmup_runs} run(s).")
    for warmup_index in range(warmup_runs):
        runner(*clone_call_args(args), **kwargs)
        synchronize_device(device)
        logger.info(f"Finished {name} warmup run {warmup_index + 1}/{warmup_runs}.")


def measure_runner_total_latency_ms(
    runner: Callable[..., Any],
    device: str,
    *args,
    **kwargs,
) -> float:
    call_args = clone_call_args(args)
    synchronize_device(device)
    start_time = time.perf_counter()
    runner(*call_args, **kwargs)
    synchronize_device(device)
    return (time.perf_counter() - start_time) * 1000.0


def measure_runner_latency_series_ms(
    name: str,
    runner: Callable[..., Any],
    measure_runs: int,
    device: str,
    logger,
    *args,
    **kwargs,
) -> List[float]:
    if measure_runs <= 0:
        raise ValueError("measure_runs must be greater than 0.")

    latencies_ms: List[float] = []
    logger.info(f"Start latency measurement for {name}: {measure_runs} run(s).")
    for measure_index in range(measure_runs):
        latency_ms = measure_runner_total_latency_ms(runner, device, *args, **kwargs)
        latencies_ms.append(latency_ms)
        logger.info(f"Finished {name} measure run {measure_index + 1}/{measure_runs}: {latency_ms:.3f} ms.")
    return latencies_ms


def summarize_latencies(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        raise ValueError("latencies_ms must not be empty.")

    average_ms = sum(latencies_ms) / len(latencies_ms)
    return {
        "run_count": len(latencies_ms),
        "avg_latency_ms": average_ms,
        "min_latency_ms": min(latencies_ms),
        "median_latency_ms": statistics.median(latencies_ms),
        "max_latency_ms": max(latencies_ms),
        "std_latency_ms": statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "first_latency_ms": latencies_ms[0],
        "last_latency_ms": latencies_ms[-1],
    }


def build_summary_row(
    eval_type: str,
    benchmark: str,
    latencies_ms: List[float],
    compiled: bool,
    includes_compile_overhead: bool,
    compile_backend: str,
    compile_mode: str,
    compile_fullgraph: bool,
    compile_dynamic: bool,
) -> Dict[str, Any]:
    row = {
        "eval_type": eval_type,
        "benchmark": benchmark,
        "compiled": compiled,
        "includes_compile_overhead": includes_compile_overhead,
        "compile_backend": compile_backend,
        "compile_mode": compile_mode,
        "compile_fullgraph": compile_fullgraph,
        "compile_dynamic": compile_dynamic,
    }
    row.update(summarize_latencies(latencies_ms))
    return row


def build_run_rows(
    eval_type: str,
    benchmark: str,
    latencies_ms: List[float],
    compiled: bool,
    includes_compile_overhead: bool,
    compile_backend: str,
    compile_mode: str,
    compile_fullgraph: bool,
    compile_dynamic: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_index, latency_ms in enumerate(latencies_ms, start=1):
        rows.append(
            {
                "eval_type": eval_type,
                "benchmark": benchmark,
                "compiled": compiled,
                "includes_compile_overhead": includes_compile_overhead,
                "compile_backend": compile_backend,
                "compile_mode": compile_mode,
                "compile_fullgraph": compile_fullgraph,
                "compile_dynamic": compile_dynamic,
                "run_index": run_index,
                "latency_ms": latency_ms,
            }
        )
    return rows


def build_comparison_row(
    eval_type: str,
    comparison: str,
    baseline_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_avg_latency_ms = float(baseline_row["avg_latency_ms"])
    candidate_avg_latency_ms = float(candidate_row["avg_latency_ms"])
    latency_diff_ms = candidate_avg_latency_ms - baseline_avg_latency_ms

    candidate_vs_baseline_ratio = ""
    speedup_ratio = ""
    speedup_percent = ""
    if baseline_avg_latency_ms > 0:
        candidate_vs_baseline_ratio = candidate_avg_latency_ms / baseline_avg_latency_ms
        if candidate_avg_latency_ms > 0:
            speedup_ratio = baseline_avg_latency_ms / candidate_avg_latency_ms
        speedup_percent = ((baseline_avg_latency_ms - candidate_avg_latency_ms) / baseline_avg_latency_ms) * 100.0

    return {
        "eval_type": eval_type,
        "comparison": comparison,
        "baseline_benchmark": baseline_row["benchmark"],
        "candidate_benchmark": candidate_row["benchmark"],
        "baseline_avg_latency_ms": baseline_avg_latency_ms,
        "candidate_avg_latency_ms": candidate_avg_latency_ms,
        "latency_diff_ms": latency_diff_ms,
        "candidate_vs_baseline_ratio": candidate_vs_baseline_ratio,
        "speedup_ratio": speedup_ratio,
        "speedup_percent": speedup_percent,
    }


def dump_latency_runs_csv(work_dir: Path, eval_type: str, rows: List[Dict[str, Any]]) -> Path:
    csv_path = work_dir / f"graph_module_compile_latency_runs_{eval_type}.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "eval_type",
                "benchmark",
                "compiled",
                "includes_compile_overhead",
                "compile_backend",
                "compile_mode",
                "compile_fullgraph",
                "compile_dynamic",
                "run_index",
                "latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def dump_latency_summary_csv(work_dir: Path, eval_type: str, rows: List[Dict[str, Any]]) -> Path:
    csv_path = work_dir / f"graph_module_compile_latency_summary_{eval_type}.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "eval_type",
                "benchmark",
                "compiled",
                "includes_compile_overhead",
                "compile_backend",
                "compile_mode",
                "compile_fullgraph",
                "compile_dynamic",
                "run_count",
                "avg_latency_ms",
                "min_latency_ms",
                "median_latency_ms",
                "max_latency_ms",
                "std_latency_ms",
                "first_latency_ms",
                "last_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def dump_latency_comparison_csv(work_dir: Path, eval_type: str, rows: List[Dict[str, Any]]) -> Path:
    csv_path = work_dir / f"graph_module_compile_latency_comparison_{eval_type}.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "eval_type",
                "comparison",
                "baseline_benchmark",
                "candidate_benchmark",
                "baseline_avg_latency_ms",
                "candidate_avg_latency_ms",
                "latency_diff_ms",
                "candidate_vs_baseline_ratio",
                "speedup_ratio",
                "speedup_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def normalize_call_args(inputs: Any) -> Tuple[Any, ...]:
    if isinstance(inputs, tuple):
        return inputs
    if isinstance(inputs, list):
        return tuple(inputs)
    return (inputs,)


def resolve_graph_module(inference_model: Any) -> fx.GraphModule:
    if isinstance(inference_model, FXInterpreter):
        graph_module = inference_model.module
    else:
        graph_module = inference_model

    if not isinstance(graph_module, fx.GraphModule):
        raise TypeError(
            f"Expected xh_model._inference_model to be a torch.fx.GraphModule, but got {type(graph_module).__name__}."
        )

    return graph_module


def normalize_compile_mode(compile_mode: str) -> str:
    if not compile_mode or compile_mode == "default":
        return "default"
    return compile_mode


def maybe_reset_torch_compile_state() -> None:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "reset"):
        dynamo.reset()


def compile_graph_module(graph_module: fx.GraphModule, args, logger):
    if not hasattr(torch, "compile"):
        raise RuntimeError("Current PyTorch build does not expose torch.compile.")

    maybe_reset_torch_compile_state()
    compile_kwargs = {
        "backend": args.compile_backend,
        "fullgraph": args.compile_fullgraph,
        "dynamic": args.compile_dynamic,
        "mode": "reduce-overhead",
    }
    if args.compile_mode and args.compile_mode != "default":
        compile_kwargs["mode"] = args.compile_mode

    logger.info(f"torch.compile options: {compile_kwargs}")
    return torch.compile(graph_module, **compile_kwargs)


def prepare_work_dir(config_path: str, eval_type: str) -> Path:
    cfg_name = Path(config_path).stem
    cfg_name = f"{cfg_name}_graph_profiler"
    work_dir = Path("./work_dirs") / cfg_name / eval_type / "torch_compile"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def prepare_graph_module_runtime(
    args,
    work_dir: Path,
    logger,
    device: str,
    dtype,
) -> Tuple[List[Any], Any, Tuple[Any, ...]]:
    cfg = Config.fromfile(args.config)
    cfg.model.context_max_length = args.prefill_length + 1
    cfg.model.prefill_chunk_length = args.prefill_length
    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = work_dir / Path(args.config).name
    cfg.dump(config_file)

    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    model_cfg.num_logits_to_keep = 1
    model_cfg.use_cache = False

    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    model_state = LLMModelState.from_string(args.eval_type)
    xh_model.set_state(model_state)
    xh_model.to(device)
    xh_model.to(dtype)
    xh_model.eval()
    data_preprocessor = xh_model.get_data_preprocessor()
    data_batch = xh_model.get_prefill_dummy_inputs()
    inputs = normalize_call_args(data_preprocessor(data_batch))
    inference_contexts = [
        torch.no_grad(),
        LLMInferenceContextManager(xh_model),
    ]

    return inference_contexts, xh_model, inputs


def log_latency_summary(logger, row: Dict[str, Any]) -> None:
    logger.info(
        f"{row['benchmark']}: avg={row['avg_latency_ms']:.3f} ms, min={row['min_latency_ms']:.3f} ms, "
        f"median={row['median_latency_ms']:.3f} ms, max={row['max_latency_ms']:.3f} ms, "
        f"std={row['std_latency_ms']:.3f} ms, runs={row['run_count']}"
    )


def log_comparison_summary(logger, row: Dict[str, Any]) -> None:
    speedup_ratio = row["speedup_ratio"]
    speedup_percent = row["speedup_percent"]
    speedup_ratio_text = f"{speedup_ratio:.3f}x" if isinstance(speedup_ratio, float) else "n/a"
    speedup_percent_text = f"{speedup_percent:.2f}%" if isinstance(speedup_percent, float) else "n/a"
    logger.info(
        f"{row['comparison']}: baseline={row['baseline_avg_latency_ms']:.3f} ms, "
        f"candidate={row['candidate_avg_latency_ms']:.3f} ms, diff={row['latency_diff_ms']:.3f} ms, "
        f"speedup={speedup_ratio_text}, speedup_percent={speedup_percent_text}"
    )


def main(args):
    eval_type = args.eval_type
    work_dir = prepare_work_dir(args.config, eval_type)
    log_file = str(work_dir / "graph_torch_compile.log")
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xhquant_init(log_file, debug=False)
    os.environ["USE_TRITON_MATMUL"] = "1"
    logger = get_xhquant_logger()
    compile_mode = normalize_compile_mode(args.compile_mode)
    logger.info(
        f"Benchmark settings: warmup_runs={args.warmup_runs}, measure_runs={args.measure_runs}, "
        f"compile_warmup_runs={args.compile_warmup_runs}, compile_backend={args.compile_backend}, "
        f"compile_mode={compile_mode}, compile_fullgraph={args.compile_fullgraph}, "
        f"compile_dynamic={args.compile_dynamic}"
    )

    inference_contexts, xh_model, inputs = prepare_graph_module_runtime(args, work_dir, logger, device, dtype)

    with ContextManagers(inference_contexts):
        graph_module = resolve_graph_module(xh_model._inference_model)

        run_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []

        eager_cold_latencies_ms = [measure_runner_total_latency_ms(graph_module, device, *inputs)]
        logger.info(f"eager_cold first run latency: {eager_cold_latencies_ms[0]:.3f} ms")
        eager_cold_summary = build_summary_row(
            eval_type,
            "eager_cold",
            eager_cold_latencies_ms,
            compiled=False,
            includes_compile_overhead=False,
            compile_backend="",
            compile_mode="",
            compile_fullgraph=False,
            compile_dynamic=False,
        )
        summary_rows.append(eager_cold_summary)
        run_rows.extend(
            build_run_rows(
                eval_type,
                "eager_cold",
                eager_cold_latencies_ms,
                compiled=False,
                includes_compile_overhead=False,
                compile_backend="",
                compile_mode="",
                compile_fullgraph=False,
                compile_dynamic=False,
            )
        )

        warmup_runner("graph_module_eager", graph_module, args.warmup_runs, device, logger, *inputs)
        eager_steady_latencies_ms = measure_runner_latency_series_ms(
            "graph_module_eager",
            graph_module,
            args.measure_runs,
            device,
            logger,
            *inputs,
        )
        eager_steady_summary = build_summary_row(
            eval_type,
            "eager_steady",
            eager_steady_latencies_ms,
            compiled=False,
            includes_compile_overhead=False,
            compile_backend="",
            compile_mode="",
            compile_fullgraph=False,
            compile_dynamic=False,
        )
        summary_rows.append(eager_steady_summary)
        run_rows.extend(
            build_run_rows(
                eval_type,
                "eager_steady",
                eager_steady_latencies_ms,
                compiled=False,
                includes_compile_overhead=False,
                compile_backend="",
                compile_mode="",
                compile_fullgraph=False,
                compile_dynamic=False,
            )
        )

        compiled_graph_module = compile_graph_module(graph_module, args, logger)
        compile_cold_latencies_ms = [measure_runner_total_latency_ms(compiled_graph_module, device, *inputs)]
        logger.info(f"compile_cold first run latency: {compile_cold_latencies_ms[0]:.3f} ms")
        compile_cold_summary = build_summary_row(
            eval_type,
            "compile_cold",
            compile_cold_latencies_ms,
            compiled=True,
            includes_compile_overhead=True,
            compile_backend=args.compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=args.compile_fullgraph,
            compile_dynamic=args.compile_dynamic,
        )
        summary_rows.append(compile_cold_summary)
        run_rows.extend(
            build_run_rows(
                eval_type,
                "compile_cold",
                compile_cold_latencies_ms,
                compiled=True,
                includes_compile_overhead=True,
                compile_backend=args.compile_backend,
                compile_mode=compile_mode,
                compile_fullgraph=args.compile_fullgraph,
                compile_dynamic=args.compile_dynamic,
            )
        )

        warmup_runner(
            "graph_module_compiled",
            compiled_graph_module,
            args.compile_warmup_runs,
            device,
            logger,
            *inputs,
        )
        compile_steady_latencies_ms = measure_runner_latency_series_ms(
            "graph_module_compiled",
            compiled_graph_module,
            args.measure_runs,
            device,
            logger,
            *inputs,
        )
        compile_steady_summary = build_summary_row(
            eval_type,
            "compile_steady",
            compile_steady_latencies_ms,
            compiled=True,
            includes_compile_overhead=False,
            compile_backend=args.compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=args.compile_fullgraph,
            compile_dynamic=args.compile_dynamic,
        )
        summary_rows.append(compile_steady_summary)
        run_rows.extend(
            build_run_rows(
                eval_type,
                "compile_steady",
                compile_steady_latencies_ms,
                compiled=True,
                includes_compile_overhead=False,
                compile_backend=args.compile_backend,
                compile_mode=compile_mode,
                compile_fullgraph=args.compile_fullgraph,
                compile_dynamic=args.compile_dynamic,
            )
        )

        comparison_rows = [
            build_comparison_row(eval_type, "compile_cold_vs_eager_cold", eager_cold_summary, compile_cold_summary),
            build_comparison_row(
                eval_type,
                "compile_steady_vs_eager_steady",
                eager_steady_summary,
                compile_steady_summary,
            ),
        ]

        runs_csv = dump_latency_runs_csv(work_dir, eval_type, run_rows)
        summary_csv = dump_latency_summary_csv(work_dir, eval_type, summary_rows)
        comparison_csv = dump_latency_comparison_csv(work_dir, eval_type, comparison_rows)

        for row in summary_rows:
            log_latency_summary(logger, row)
        for row in comparison_rows:
            log_comparison_summary(logger, row)

        logger.info(f"Saved latency runs to: {runs_csv}")
        logger.info(f"Saved latency summary to: {summary_csv}")
        logger.info(f"Saved latency comparison to: {comparison_csv}")


if __name__ == "__main__":
    eval_types = LLMModelState.get_all_values()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/qwen3_legacy/8b/qwen3_8b_legacy_xh2a_2k.py",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="number of eager warmup runs before steady-state measurement",
    )
    parser.add_argument(
        "--compile-warmup-runs",
        type=int,
        default=1,
        help="number of extra compiled warmup runs after the first compile+cold run",
    )
    parser.add_argument("--measure-runs", type=int, default=5, help="number of steady-state measurement runs")
    parser.add_argument("--compile-backend", type=str, default="inductor", help="backend passed to torch.compile")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        help="mode passed to torch.compile; use 'default' for the PyTorch default behavior",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="pass fullgraph=True to torch.compile",
    )
    parser.add_argument(
        "--compile-dynamic",
        action="store_true",
        help="pass dynamic=True to torch.compile",
    )
    parser.add_argument(
        "--prefill-length",
        type=int,
        default=2048,
        help="the sequence length for the prefill phase",
    )
    parser.add_argument(
        "--only-prefill",
        action="store_true",
        help="only run the prefill phase for evaluation, skipping the decode phase",
    )
    parser.add_argument("--eval-type", type=str, default="quanted_disable", choices=eval_types)
    args = parser.parse_args()
    main(args)
