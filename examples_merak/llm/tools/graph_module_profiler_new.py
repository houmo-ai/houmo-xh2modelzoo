import argparse
import csv
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.fx as fx

from xhmodel_merak.xh_llm import (
    AutoLLMConfig,
    AutoLLMModel,
    LLMInferenceContextManager,
    LLMModelState,
)
from xhquant.api import Config, FXInterpreter
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


from xhquant.api import xhquant_init, get_xhquant_logger  # isort:skip


def synchronize_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


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
        runner(*args, **kwargs)
        synchronize_device(device)
        logger.info(f"Finished {name} warmup run {warmup_index + 1}/{warmup_runs}.")


def measure_runner_total_latency_ms(
    runner: Callable[..., Any],
    device: str,
    *args,
    **kwargs,
) -> float:
    synchronize_device(device)
    start_time = time.perf_counter()
    runner(*args, **kwargs)
    synchronize_device(device)

    return (time.perf_counter() - start_time) * 1000.0


def measure_graph_module_total_latency_ms(
    graph_module: fx.GraphModule,
    device: str,
    *args,
    **kwargs,
) -> float:
    return measure_runner_total_latency_ms(
        graph_module,
        device,
        *args,
        **kwargs,
    )


def measure_fx_interpreter_total_latency_ms(
    graph_module: fx.GraphModule,
    device: str,
    *args,
    **kwargs,
) -> float:
    interpreter = FXInterpreter(graph_module)
    return measure_runner_total_latency_ms(
        interpreter.run,
        device,
        *args,
        **kwargs,
    )


def dump_total_latency_csv(
    work_dir: Path,
    eval_type: str,
    graph_module_total_latency_ms: float,
    fx_interpreter_total_latency_ms: float,
) -> Path:
    latency_diff_ms = fx_interpreter_total_latency_ms - graph_module_total_latency_ms
    slowdown_ratio = ""
    slowdown_percent = ""
    if graph_module_total_latency_ms > 0:
        slowdown_ratio = fx_interpreter_total_latency_ms / graph_module_total_latency_ms
        slowdown_percent = (latency_diff_ms / graph_module_total_latency_ms) * 100.0

    total_latency_csv = work_dir / f"graph_module_total_latency_{eval_type}.csv"
    with total_latency_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "eval_type",
                "graph_module_total_latency_ms",
                "fx_interpreter_total_latency_ms",
                "latency_diff_ms",
                "fx_interpreter_slowdown_ratio",
                "fx_interpreter_slowdown_percent",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "eval_type": eval_type,
                "graph_module_total_latency_ms": graph_module_total_latency_ms,
                "fx_interpreter_total_latency_ms": fx_interpreter_total_latency_ms,
                "latency_diff_ms": latency_diff_ms,
                "fx_interpreter_slowdown_ratio": slowdown_ratio,
                "fx_interpreter_slowdown_percent": slowdown_percent,
            }
        )

    return total_latency_csv


class GraphModuleNodeProfiler(FXInterpreter):
    def __init__(self, graph_module: fx.GraphModule, logger, device: str):
        super().__init__(graph_module)
        self.logger = logger
        self.device = device
        self.records: List[Dict[str, Any]] = []
        self._record_enabled = False
        self._current_run_index = -1
        self._sync_cuda = device.startswith("cuda") and torch.cuda.is_available()

    def run_node(self, node: fx.Node) -> Any:
        if self._sync_cuda:
            # Exclude the previous node's queued CUDA work from this node's timing.
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        output = super().run_node(node)

        if self._sync_cuda:
            # Wait until this node's CUDA work is fully completed before stopping the timer.
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if self._record_enabled:
            self.records.append(
                {
                    "run_index": self._current_run_index,
                    "node_name": node.name,
                    "node_op": node.op,
                    "node_target": str(node.target),
                    "module_type": self._get_module_type(node),
                    "elapsed_ms": elapsed_ms,
                }
            )

        return output

    def warmup(self, warmup_runs: int, *args, **kwargs) -> None:
        if warmup_runs <= 0:
            return

        self.logger.info(f"Start warmup for graph module: {warmup_runs} run(s).")
        self._record_enabled = False
        for warmup_index in range(warmup_runs):
            self.run(*args, **kwargs)
            self.logger.info(f"Finished warmup run {warmup_index + 1}/{warmup_runs}.")

    def profile(self, profile_runs: int, *args, **kwargs) -> Any:
        if profile_runs <= 0:
            raise ValueError("profile_runs must be greater than 0.")

        self.records.clear()
        last_output = None
        self.logger.info(f"Start node profiling for graph module: {profile_runs} run(s).")
        for run_index in range(profile_runs):
            self._current_run_index = run_index
            self._record_enabled = True
            last_output = self.run(*args, **kwargs)
            self.logger.info(f"Finished profile run {run_index + 1}/{profile_runs}.")
        self._record_enabled = False
        return last_output

    def build_summary(self) -> List[Dict[str, Any]]:
        summary_by_node: Dict[str, Dict[str, Any]] = {}
        for record in self.records:
            node_name = record["node_name"]
            if node_name not in summary_by_node:
                summary_by_node[node_name] = {
                    "node_name": node_name,
                    "node_op": record["node_op"],
                    "node_target": record["node_target"],
                    "module_type": record["module_type"],
                    "total_ms": 0.0,
                    "call_count": 0,
                }

            summary_item = summary_by_node[node_name]
            summary_item["total_ms"] += record["elapsed_ms"]
            summary_item["call_count"] += 1

        summary = list(summary_by_node.values())
        for item in summary:
            item["avg_ms"] = item["total_ms"] / item["call_count"]

        summary.sort(key=lambda item: item["total_ms"], reverse=True)
        return summary

    def build_log_summary_rows(self, topk: int) -> List[Dict[str, Any]]:
        summary = self.build_summary()
        if topk <= 0:
            topk = len(summary)

        rows: List[Dict[str, Any]] = []
        for index, item in enumerate(summary[:topk], start=1):
            rows.append(
                {
                    "rank": index,
                    "node_name": item["node_name"],
                    "node_op": item["node_op"],
                    "module_type": item["module_type"] or "-",
                    "node_target": item["node_target"],
                    "call_count": item["call_count"],
                    "avg_ms": item["avg_ms"],
                    "total_ms": item["total_ms"],
                }
            )
        return rows

    def dump_csv(self, work_dir: Path, eval_type: str, topk: int) -> Tuple[Path, Path, Path]:
        raw_csv = work_dir / f"node_profile_raw_{eval_type}.csv"
        summary_csv = work_dir / f"node_profile_summary_{eval_type}.csv"
        log_summary_csv = work_dir / f"node_profile_topk_{topk}_{eval_type}.csv"
        with raw_csv.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "run_index",
                    "node_name",
                    "node_op",
                    "node_target",
                    "module_type",
                    "elapsed_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(self.records)

        summary = self.build_summary()
        with summary_csv.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["node_name", "node_op", "node_target", "module_type", "call_count", "avg_ms", "total_ms"],
            )
            writer.writeheader()
            writer.writerows(summary)

        log_summary_rows = self.build_log_summary_rows(topk)
        with log_summary_csv.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "rank",
                    "node_name",
                    "node_op",
                    "module_type",
                    "node_target",
                    "call_count",
                    "avg_ms",
                    "total_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(log_summary_rows)

        return summary_csv, raw_csv, log_summary_csv

    def log_summary(self, topk: int) -> None:
        summary = self.build_summary()
        total_node_time_ms = sum(item["total_ms"] for item in summary)
        self.logger.info(
            f"Node profile summary: {len(summary)} unique node(s), {len(self.records)} total node execution(s), "
            f"accumulated {total_node_time_ms:.3f} ms."
        )

        for item in self.build_log_summary_rows(topk):
            self.logger.info(
                f"[{item['rank']:03d}] {item['node_name']:<48} op={item['node_op']:<12} "
                f"module={item['module_type']:<32} avg={item['avg_ms']:.3f} ms "
                f"total={item['total_ms']:.3f} ms target={item['node_target']}"
            )

    def _get_module_type(self, node: fx.Node) -> str:
        if node.op != "call_module":
            return ""
        try:
            return type(self.module.get_submodule(node.target)).__name__
        except AttributeError:
            return ""


def prepare_work_dir(config_path: str, eval_type: str, phase: str) -> Path:
    cfg_name = Path(config_path).stem
    cfg_name = f"{cfg_name}_graph_profiler"
    work_dir = Path("./work_dirs") / cfg_name / phase / eval_type
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def prepare_graph_module_runtime(
    args,
    work_dir: Path,
    logger,
    device: str,
    dtype,
) -> Tuple[List[Any], fx.GraphModule, Any]:
    cfg = Config.fromfile(args.config)
    cfg.model.context_max_length = args.prefill_length + 1
    cfg.model.prefill_chunk_length = args.prefill_length
    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = work_dir / Path(args.config).name
    cfg.dump(config_file)

    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    model_cfg.num_logits_to_keep = 0
    model_cfg.use_cache = False

    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    model_state = LLMModelState.from_string(args.eval_type)
    xh_model.set_state(model_state)
    xh_model.to(device)
    xh_model.to(dtype)
    xh_model.eval()
    data_preprocesor = xh_model.get_data_preprocessor()
    is_decode = args.decode
    if is_decode:
        logger.info("Running in decode mode, skipping prefill data preparation.")
        xh_model.set_decode()
        data_batch = xh_model.get_decode_dummy_inputs()
    else:
        data_batch = xh_model.get_prefill_dummy_inputs()
    inputs = data_preprocesor(data_batch)
    inference_contexts = [
        torch.no_grad(),
        LLMInferenceContextManager(xh_model),
    ]

    return inference_contexts, xh_model, inputs, data_batch["past_seq_length"]


def main(args):
    eval_type = args.eval_type
    is_decode = args.decode
    phase = "decode" if is_decode else "prefill"
    work_dir = prepare_work_dir(args.config, args.eval_type, phase)
    log_file = str(work_dir / "graph_profiler.log")
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xhquant_init(log_file, debug=False)
    logger = get_xhquant_logger()
    inference_contexts, xh_model, inputs, context_length = prepare_graph_module_runtime(
        args, work_dir, logger, device, dtype
    )

    logger.info(f"------------ Running in {phase} mode, context_length: {context_length} ------------")
    logger.info(f"Prepared graph module runtime with device={device}, dtype={dtype}, context_length={context_length}.")
    with ContextManagers(inference_contexts):
        inference_model = xh_model._inference_model
        if isinstance(inference_model, FXInterpreter):
            graph_module = inference_model.module
        else:
            graph_module = inference_model

        if not isinstance(graph_module, fx.GraphModule):
            raise TypeError(
                f"Expected xh_model._inference_model to be a torch.fx.GraphModule, but got {type(graph_module).__name__}."
            )
        warmup_runner("graph_module", graph_module, args.warmup_runs, device, logger, *inputs)
        fx_interpreter = FXInterpreter(graph_module)
        warmup_runner(
            "FXInterpreter",
            fx_interpreter.run,
            args.warmup_runs,
            device,
            logger,
            *inputs,
        )
        profiler = GraphModuleNodeProfiler(graph_module, logger=logger, device=device)
        profiler.warmup(args.warmup_runs, *inputs)

        profile_contexts = [
            TimeProfiler(f"graph_module_profile_{eval_type}", logger),
            MemoryTracker(device=device, name=f"node_profile_{eval_type}", logger=logger),
        ]

        graph_module_total_latency_ms = measure_graph_module_total_latency_ms(graph_module, device, *inputs)
        fx_interpreter_total_latency_ms = measure_fx_interpreter_total_latency_ms(graph_module, device, *inputs)

        total_latency_csv = dump_total_latency_csv(
            work_dir,
            args.eval_type,
            graph_module_total_latency_ms,
            fx_interpreter_total_latency_ms,
        )
        latency_diff_ms = fx_interpreter_total_latency_ms - graph_module_total_latency_ms
        slowdown_ratio = fx_interpreter_total_latency_ms / graph_module_total_latency_ms
        logger.info(f"GraphModule end-to-end total latency: {graph_module_total_latency_ms:.3f} ms")
        logger.info(f"FXInterpreter end-to-end total latency: {fx_interpreter_total_latency_ms:.3f} ms")
        logger.info(f"FXInterpreter overhead vs graph_module: {latency_diff_ms:.3f} ms, slowdown={slowdown_ratio:.3f}x")
        logger.info(f"Saved total latency comparison to: {total_latency_csv}")

        with ContextManagers(profile_contexts):
            profiler.profile(args.profile_runs, *inputs)

        summary_csv, raw_csv, log_summary_csv = profiler.dump_csv(work_dir, eval_type, args.topk)
        profiler.log_summary(args.topk)
        logger.info(f"Saved node profile summary to: {summary_csv}")
        logger.info(f"Saved raw node profile records to: {raw_csv}")
        logger.info(f"Saved node profile log summary to: {log_summary_csv}")


if __name__ == "__main__":
    eval_types = LLMModelState.get_all_values()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config", type=str, default="configs_merak/xh2a/llm_models/qwen3/8b/qwen3_8b_xh2a_2k.py"
    )
    parser.add_argument("--warmup-runs", type=int, default=2, help="number of warmup runs before node profiling")
    parser.add_argument("--profile-runs", type=int, default=1, help="number of profiled runs to average per node")
    parser.add_argument(
        "--topk", type=int, default=20, help="number of hottest nodes to print in the log, <=0 prints all"
    )
    parser.add_argument("--prefill-length", type=int, default=2048, help="the sequence length for the prefill phase")
    parser.add_argument(
        "--decode",
        action="store_true",
        help="run the decode phase for evaluation, skipping the prefill phase",
    )
    parser.add_argument("--eval-type", type=str, default="quanted_disable", choices=eval_types)
    args = parser.parse_args()
    main(args)
