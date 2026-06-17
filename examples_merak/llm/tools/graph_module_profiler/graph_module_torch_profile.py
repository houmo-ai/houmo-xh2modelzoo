import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.fx as fx
from torch.profiler import ProfilerActivity, record_function
from torch.profiler import profile as torch_profile

from xhmodel_merak.xh_llm import (
    AutoLLMConfig,
    AutoLLMModel,
    LLMInferenceContextManager,
    LLMModelState,
)
from xhmodel_merak.xh_llm.types import ModelSwitcher
from xhquant.api import Config, FXInterpreter, get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers


def synchronize_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def unbind_args(args):
    new_args: list[Any] = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            if len(arg) == 0:
                new_args.append(arg)
            else:
                new_args.extend(arg)
        else:
            new_args.append(arg)
    return new_args


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
    if isinstance(runner, (fx.GraphModule, FXInterpreter)):
        args = unbind_args(args)
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
    return measure_runner_total_latency_ms(graph_module, device, *args, **kwargs)


def measure_fx_interpreter_total_latency_ms(
    graph_module: fx.GraphModule,
    device: str,
    *args,
    **kwargs,
) -> float:
    interpreter = FXInterpreter(graph_module)
    return measure_runner_total_latency_ms(interpreter.run, device, *args, **kwargs)


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


def build_node_profile_label(node: fx.Node) -> str:
    node_target = str(node.target)
    if node_target and node_target != node.name:
        return f"graph_module_node::{node.name}::{node_target}"
    return f"graph_module_node::{node.name}"


def build_node_profile_key(node: fx.Node, run_index: int, call_index: int) -> str:
    return f"{build_node_profile_label(node)}##run={run_index}##call={call_index}"


def resolve_graph_module(inference_model: Any) -> fx.GraphModule:
    if isinstance(inference_model, ModelSwitcher):
        inference_model = inference_model.activate_model
    if isinstance(inference_model, FXInterpreter):
        graph_module = inference_model.module
    else:
        graph_module = inference_model

    if not isinstance(graph_module, fx.GraphModule):
        raise TypeError(
            f"Expected xh_model._inference_model to be a torch.fx.GraphModule, but got {type(graph_module).__name__}."
        )

    return graph_module


class GraphModuleTorchProfiler(FXInterpreter):
    def __init__(self, graph_module: fx.GraphModule, logger, device: str):
        super().__init__(graph_module)
        self.logger = logger
        self.device = device
        self._use_device_time = device.startswith("cuda") and torch.cuda.is_available()
        self._primary_metric = "device_total_ms" if self._use_device_time else "cpu_total_ms"
        self._record_enabled = False
        self._current_run_index = -1
        self._current_call_index = 0
        self._node_index_by_name = {node.name: node_index for node_index, node in enumerate(graph_module.graph.nodes)}
        self.records: List[Dict[str, Any]] = []

    def run_node(self, node: fx.Node) -> Any:
        if not self._record_enabled:
            return FXInterpreter.run_node(self, node)

        call_index = self._current_call_index
        self._current_call_index += 1
        profile_key = build_node_profile_key(node, self._current_run_index, call_index)
        self.records.append(
            {
                "run_index": self._current_run_index,
                "call_index": call_index,
                "node_index": self._node_index_by_name.get(node.name, -1),
                "profile_key": profile_key,
                "profile_label": build_node_profile_label(node),
                "node_name": node.name,
                "node_op": node.op,
                "node_target": str(node.target),
                "module_type": self._get_module_type(node),
                "device_type": "",
                "device_index": "",
                "physical_device": "",
                "cpu_total_ms": 0.0,
                "cpu_self_ms": 0.0,
                "device_total_ms": 0.0,
                "device_self_ms": 0.0,
                "kernel_count": 0,
                "launch_count": 0,
                "total_ms": 0.0,
            }
        )

        with record_function(profile_key):
            return FXInterpreter.run_node(self, node)

    def warmup(self, warmup_runs: int, *args, **kwargs) -> None:
        if warmup_runs <= 0:
            return

        self.logger.info(f"Start warmup for torch.profiler interpreter: {warmup_runs} run(s).")
        self._record_enabled = False
        for warmup_index in range(warmup_runs):
            self.run(*args, **kwargs)
            synchronize_device(self.device)
            self.logger.info(f"Finished warmup run {warmup_index + 1}/{warmup_runs}.")

    def profile(self, profile_runs: int, *args, **kwargs) -> Any:
        if profile_runs <= 0:
            raise ValueError("profile_runs must be greater than 0.")

        self.records.clear()
        self.logger.info(f"Start torch.profiler node profiling: {profile_runs} run(s).")
        last_output = None
        for run_index in range(profile_runs):
            self._current_run_index = run_index
            self._current_call_index = 0
            self._record_enabled = True
            last_output = self.run(*args, **kwargs)
            synchronize_device(self.device)
            self.logger.info(f"Finished profile run {run_index + 1}/{profile_runs}.")
        self._record_enabled = False
        return last_output

    def attach_profiler_results(self, profiler_result) -> None:
        metrics_by_key = {
            record["profile_key"]: {
                "cpu_total_us": 0.0,
                "cpu_self_us": 0.0,
                "device_total_us": 0.0,
                "device_self_us": 0.0,
                "kernel_count": 0,
                "launch_count": 0,
                "device_types": set(),
                "device_indices": set(),
                "physical_devices": set(),
            }
            for record in self.records
        }

        event_owner_by_id: Dict[int, str] = {}
        events = list(profiler_result.events())

        for event in events:
            owner_profile_key = self._resolve_owner_profile_key_from_parent_chain(event, metrics_by_key)
            if not owner_profile_key:
                continue

            event_id = self._read_int_metric(event, "id")
            if event_id is not None and event_id >= 0 and event_id not in event_owner_by_id:
                event_owner_by_id[event_id] = owner_profile_key

        for event in events:
            profile_key = getattr(event, "name", "")
            metrics = metrics_by_key.get(profile_key)
            if metrics is not None:
                if self._is_cpu_event(event):
                    metrics["cpu_total_us"] += self._read_metric(event, "cpu_time_total")
                    metrics["cpu_self_us"] += self._read_metric(event, "self_cpu_time_total")
                else:
                    metrics["device_total_us"] += self._read_metric(event, "device_time_total", "cuda_time_total")
                    metrics["device_self_us"] += self._read_metric(
                        event,
                        "self_device_time_total",
                        "self_cuda_time_total",
                    )

                    device_type = self._format_device_type(event)
                    if device_type:
                        metrics["device_types"].add(device_type)

                    device_index = self._read_int_metric(event, "device_index")
                    if device_index is not None and device_index >= 0:
                        metrics["device_indices"].add(device_index)
                        physical_device = self._resolve_physical_device(device_index)
                        if physical_device:
                            metrics["physical_devices"].add(physical_device)

            owner_profile_key = self._resolve_owner_profile_key(event, metrics_by_key, event_owner_by_id)
            if not owner_profile_key:
                continue

            owner_metrics = metrics_by_key[owner_profile_key]
            if self._is_launch_event(event):
                owner_metrics["launch_count"] += 1
            if self._is_kernel_event(event, metrics_by_key):
                owner_metrics["kernel_count"] += 1

        missing_records = 0
        for record in self.records:
            metrics = metrics_by_key.get(record["profile_key"])
            if metrics is None:
                missing_records += 1
                continue

            record["cpu_total_ms"] = metrics["cpu_total_us"] / 1000.0
            record["cpu_self_ms"] = metrics["cpu_self_us"] / 1000.0
            record["device_total_ms"] = metrics["device_total_us"] / 1000.0
            record["device_self_ms"] = metrics["device_self_us"] / 1000.0
            record["kernel_count"] = metrics["kernel_count"]
            record["launch_count"] = metrics["launch_count"]
            record["device_type"] = self._join_str_values(metrics["device_types"])
            record["device_index"] = self._join_int_values(metrics["device_indices"])
            record["physical_device"] = self._join_str_values(metrics["physical_devices"])
            record["total_ms"] = record[self._primary_metric]

            if record["cpu_total_ms"] == 0.0 and record["device_total_ms"] == 0.0:
                missing_records += 1

        if missing_records > 0:
            self.logger.warning(f"torch.profiler returned no timing payload for {missing_records} node execution(s).")

    def build_summary(self) -> List[Dict[str, Any]]:
        summary_by_node: Dict[str, Dict[str, Any]] = {}
        for record in self.records:
            node_name = record["node_name"]
            if node_name not in summary_by_node:
                summary_by_node[node_name] = {
                    "node_index": record["node_index"],
                    "node_name": node_name,
                    "node_op": record["node_op"],
                    "node_target": record["node_target"],
                    "module_type": record["module_type"],
                    "device_type_values": set(),
                    "device_index_values": set(),
                    "physical_device_values": set(),
                    "call_count": 0,
                    "total_ms": 0.0,
                    "total_cpu_total_ms": 0.0,
                    "total_cpu_self_ms": 0.0,
                    "total_device_total_ms": 0.0,
                    "total_device_self_ms": 0.0,
                    "total_kernel_count": 0,
                    "total_launch_count": 0,
                    "primary_metric": self._primary_metric,
                }

            summary_item = summary_by_node[node_name]
            summary_item["call_count"] += 1
            summary_item["total_ms"] += record["total_ms"]
            summary_item["total_cpu_total_ms"] += record["cpu_total_ms"]
            summary_item["total_cpu_self_ms"] += record["cpu_self_ms"]
            summary_item["total_device_total_ms"] += record["device_total_ms"]
            summary_item["total_device_self_ms"] += record["device_self_ms"]
            summary_item["total_kernel_count"] += record["kernel_count"]
            summary_item["total_launch_count"] += record["launch_count"]
            summary_item["device_type_values"].update(self._split_csv_values(record["device_type"]))
            summary_item["device_index_values"].update(self._split_csv_int_values(record["device_index"]))
            summary_item["physical_device_values"].update(self._split_csv_values(record["physical_device"]))

        summary = list(summary_by_node.values())
        for item in summary:
            call_count = item["call_count"]
            item["avg_ms"] = item["total_ms"] / call_count
            item["avg_cpu_total_ms"] = item["total_cpu_total_ms"] / call_count
            item["avg_cpu_self_ms"] = item["total_cpu_self_ms"] / call_count
            item["avg_device_total_ms"] = item["total_device_total_ms"] / call_count
            item["avg_device_self_ms"] = item["total_device_self_ms"] / call_count
            item["avg_kernel_count"] = item["total_kernel_count"] / call_count
            item["avg_launch_count"] = item["total_launch_count"] / call_count
            item["device_type"] = self._join_str_values(item.pop("device_type_values"))
            item["device_indices"] = self._join_int_values(item.pop("device_index_values"))
            item["physical_devices"] = self._join_str_values(item.pop("physical_device_values"))

        summary.sort(key=lambda item: item["total_ms"], reverse=True)
        return summary

    def build_log_summary_rows(self, topk: int) -> List[Dict[str, Any]]:
        summary = self.build_summary()
        if topk <= 0:
            topk = len(summary)

        rows: List[Dict[str, Any]] = []
        for rank, item in enumerate(summary[:topk], start=1):
            rows.append(
                {
                    "rank": rank,
                    "node_index": item["node_index"],
                    "node_name": item["node_name"],
                    "node_op": item["node_op"],
                    "module_type": item["module_type"] or "-",
                    "node_target": item["node_target"],
                    "device_type": item["device_type"] or "-",
                    "device_indices": item["device_indices"],
                    "physical_devices": item["physical_devices"],
                    "call_count": item["call_count"],
                    "avg_kernel_count": item["avg_kernel_count"],
                    "total_kernel_count": item["total_kernel_count"],
                    "avg_launch_count": item["avg_launch_count"],
                    "total_launch_count": item["total_launch_count"],
                    "avg_ms": item["avg_ms"],
                    "total_ms": item["total_ms"],
                    "avg_cpu_total_ms": item["avg_cpu_total_ms"],
                    "avg_device_total_ms": item["avg_device_total_ms"],
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
                    "call_index",
                    "node_index",
                    "node_name",
                    "node_op",
                    "node_target",
                    "module_type",
                    "device_type",
                    "device_index",
                    "physical_device",
                    "profile_label",
                    "profile_key",
                    "cpu_total_ms",
                    "cpu_self_ms",
                    "device_total_ms",
                    "device_self_ms",
                    "kernel_count",
                    "launch_count",
                    "total_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(self.records)

        summary = self.build_summary()
        with summary_csv.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "node_index",
                    "node_name",
                    "node_op",
                    "node_target",
                    "module_type",
                    "device_type",
                    "device_indices",
                    "physical_devices",
                    "call_count",
                    "avg_kernel_count",
                    "total_kernel_count",
                    "avg_launch_count",
                    "total_launch_count",
                    "avg_ms",
                    "total_ms",
                    "avg_cpu_total_ms",
                    "total_cpu_total_ms",
                    "avg_cpu_self_ms",
                    "total_cpu_self_ms",
                    "avg_device_total_ms",
                    "total_device_total_ms",
                    "avg_device_self_ms",
                    "total_device_self_ms",
                    "primary_metric",
                ],
            )
            writer.writeheader()
            writer.writerows(summary)

        log_summary_rows = self.build_log_summary_rows(topk)
        with log_summary_csv.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "rank",
                    "node_index",
                    "node_name",
                    "node_op",
                    "module_type",
                    "node_target",
                    "device_type",
                    "device_indices",
                    "physical_devices",
                    "call_count",
                    "avg_kernel_count",
                    "total_kernel_count",
                    "avg_launch_count",
                    "total_launch_count",
                    "avg_ms",
                    "total_ms",
                    "avg_cpu_total_ms",
                    "avg_device_total_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(log_summary_rows)

        return summary_csv, raw_csv, log_summary_csv

    def log_summary(self, topk: int) -> None:
        summary = self.build_summary()
        total_node_time_ms = sum(item["total_ms"] for item in summary)
        self.logger.info(
            f"torch.profiler node summary: {len(summary)} unique node(s), "
            f"{len(self.records)} total node execution(s), primary metric={self._primary_metric}, "
            f"accumulated {total_node_time_ms:.3f} ms."
        )

        for item in self.build_log_summary_rows(topk):
            self.logger.info(
                f"[{item['rank']:03d}] idx={item['node_index']:<4} {item['node_name']:<40} "
                f"op={item['node_op']:<12} module={item['module_type']:<28} "
                f"avg={item['avg_ms']:.3f} ms total={item['total_ms']:.3f} ms "
                f"kernels={item['total_kernel_count']} launches={item['total_launch_count']} "
                f"cpu={item['avg_cpu_total_ms']:.3f} ms device={item['avg_device_total_ms']:.3f} ms "
                f"device_type={item['device_type'] or '-'} device_idx={item['device_indices'] or '-'} "
                f"physical={item['physical_devices'] or '-'} "
                f"target={item['node_target']}"
            )

    def _get_module_type(self, node: fx.Node) -> str:
        if node.op != "call_module":
            return ""
        try:
            return type(self.module.get_submodule(node.target)).__name__
        except AttributeError:
            return ""

    @staticmethod
    def _read_metric(event: Any, *names: str) -> float:
        for name in names:
            value = getattr(event, name, None)
            if value is not None:
                return float(value)
        return 0.0

    @staticmethod
    def _read_int_metric(event: Any, *names: str) -> Optional[int]:
        for name in names:
            value = getattr(event, name, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _is_cpu_event(event: Any) -> bool:
        return "CPU" in str(getattr(event, "device_type", "")).upper()

    @staticmethod
    def _format_device_type(event: Any) -> str:
        device_type = getattr(event, "device_type", None)
        if device_type is None:
            return ""
        return str(device_type).split(".")[-1].upper()

    def _resolve_physical_device(self, logical_device_index: int) -> str:
        if logical_device_index < 0 or not self._use_device_time:
            return ""

        visible_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible_cuda_devices:
            return str(logical_device_index)

        visible_device_tokens = [token.strip() for token in visible_cuda_devices.split(",") if token.strip()]
        if logical_device_index >= len(visible_device_tokens):
            return str(logical_device_index)

        return visible_device_tokens[logical_device_index]

    @staticmethod
    def _join_int_values(values: set[int]) -> str:
        return ",".join(str(value) for value in sorted(values))

    @staticmethod
    def _join_str_values(values: set[str]) -> str:
        return ",".join(sorted(value for value in values if value))

    @staticmethod
    def _split_csv_values(value: str) -> List[str]:
        if not value:
            return []
        return [item for item in value.split(",") if item]

    @staticmethod
    def _split_csv_int_values(value: str) -> List[int]:
        if not value:
            return []

        parsed_values: List[int] = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                parsed_values.append(int(item))
            except ValueError:
                continue
        return parsed_values

    @staticmethod
    def _resolve_owner_profile_key_from_parent_chain(event: Any, metrics_by_key: Dict[str, Dict[str, Any]]) -> str:
        profile_key = getattr(event, "name", "")
        if profile_key in metrics_by_key:
            return profile_key

        parent_event = getattr(event, "cpu_parent", None)
        while parent_event is not None:
            parent_name = getattr(parent_event, "name", "")
            if parent_name in metrics_by_key:
                return parent_name
            parent_event = getattr(parent_event, "cpu_parent", None)

        return ""

    def _resolve_owner_profile_key(
        self,
        event: Any,
        metrics_by_key: Dict[str, Dict[str, Any]],
        event_owner_by_id: Dict[int, str],
    ) -> str:
        owner_profile_key = self._resolve_owner_profile_key_from_parent_chain(event, metrics_by_key)
        if owner_profile_key:
            return owner_profile_key

        event_id = self._read_int_metric(event, "id")
        if event_id is not None and event_id >= 0:
            return event_owner_by_id.get(event_id, "")

        return ""

    @staticmethod
    def _is_launch_event(event: Any) -> bool:
        if not GraphModuleTorchProfiler._is_cpu_event(event):
            return False

        event_name = getattr(event, "name", "").lower()
        return any(
            token in event_name
            for token in (
                "cudalaunchkernel",
                "cudalaunchkernelexc",
                "cudalaunchcooperativekernel",
                "culaunchkernel",
                "graphlaunch",
            )
        )

    @staticmethod
    def _is_kernel_event(event: Any, metrics_by_key: Dict[str, Dict[str, Any]]) -> bool:
        if GraphModuleTorchProfiler._is_cpu_event(event):
            return False

        event_name = getattr(event, "name", "")
        if not event_name or event_name in metrics_by_key:
            return False

        lowered_name = event_name.lower()
        if "memcpy" in lowered_name or "memset" in lowered_name:
            return False

        return True


def build_profiler_activities(device: str) -> List[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return activities


def prepare_work_dir(config_path: str, eval_type: str) -> Path:
    cfg_name = Path(config_path).stem
    cfg_name = f"{cfg_name}_graph_profiler"
    work_dir = Path("./work_dirs") / cfg_name / eval_type / "torch_profiler"
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
    model_cfg.num_logits_to_keep = 1
    model_cfg.use_cache = True

    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    model_state = LLMModelState.from_string(args.eval_type)
    xh_model.set_state(model_state)
    xh_model.to(device)
    xh_model.to(dtype)
    xh_model.eval()
    data_batch = xh_model.get_prefill_dummy_inputs()
    inference_contexts = [torch.no_grad(), LLMInferenceContextManager(xh_model)]

    return inference_contexts, xh_model, data_batch


def main(args):
    eval_type = LLMModelState.from_string(args.eval_type)

    assert eval_type in [
        LLMModelState.FRONTED,
        LLMModelState.QUANTED_DISABLE,
        LLMModelState.QUANTED_ALIGNED,
        LLMModelState.QUANTED_FAST,
    ], (
        f"Unsupported eval_type: {args.eval_type}. Supported values are: fronted, quanted_disable, quanted_aligned, quanted_fast."
    )
    work_dir = prepare_work_dir(args.config, args.eval_type)
    log_file = str(work_dir / "graph_torch_profiler.log")
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xhquant_init(log_file, debug=False)
    os.environ["USE_TRITON_MATMUL"] = "1"
    logger = get_xhquant_logger()
    logger.info(f"torch.profiler activities: {build_profiler_activities(device)}")
    logger.info(
        "torch.profiler options: "
        f"record_shapes={args.record_shapes}, profile_memory={args.profile_memory}, with_stack={args.with_stack}"
    )

    inference_contexts, xh_model, data_batch = prepare_graph_module_runtime(args, work_dir, logger, device, dtype)
    if not args.decode:
        xh_model.set_prefill()
    else:
        xh_model.set_decode()

    with ContextManagers(inference_contexts):
        xh_model.to(device)
        data_preprocessor = xh_model.get_data_preprocessor()
        inputs = data_preprocessor(data_batch)

        inputs = unbind_args(inputs)
        graph_module = resolve_graph_module(xh_model._inference_model)
        warmup_runner("graph_module", graph_module, args.warmup_runs, device, logger, *inputs)

        fx_interpreter = FXInterpreter(graph_module)
        warmup_runner("FXInterpreter", fx_interpreter.run, args.warmup_runs, device, logger, *inputs)

        profiler = GraphModuleTorchProfiler(graph_module, logger=logger, device=device)
        profiler.warmup(args.warmup_runs, *inputs)

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
        args.record_shapes = True
        args.profile_memory = True
        args.with_stack = True
        tensorboard_log_dir = work_dir / "tensorboard"
        with torch_profile(
            activities=build_profiler_activities(device),
            record_shapes=args.record_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.with_stack,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(tensorboard_log_dir)),
        ) as profiler_result:
            profiler.profile(args.profile_runs, *inputs)

        profiler.attach_profiler_results(profiler_result)
        summary_csv, raw_csv, log_summary_csv = profiler.dump_csv(work_dir, args.eval_type, args.topk)
        profiler.log_summary(args.topk)
        logger.info(f"Saved torch.profiler node summary to: {summary_csv}")
        logger.info(f"Saved torch.profiler raw node records to: {raw_csv}")
        logger.info(f"Saved torch.profiler top-k node summary to: {log_summary_csv}")
        logger.info(f"Saved torch.profiler tensorboard traces to: {tensorboard_log_dir}")


if __name__ == "__main__":
    eval_types = LLMModelState.get_all_values()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config path. Legacy Qwen3.5 xh2a Python config defaults were removed; pass an explicit compatible config.",
    )
    parser.add_argument("--warmup-runs", type=int, default=2, help="number of warmup runs before profiling")
    parser.add_argument("--profile-runs", type=int, default=1, help="number of profiled runs")
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="number of hottest nodes to print in the log, <=0 prints all",
    )
    parser.add_argument(
        "--prefill-length",
        type=int,
        default=2048,
        help="the sequence length for the prefill phase",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="enable shape recording in torch.profiler",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="enable memory profiling in torch.profiler",
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="capture Python stack traces in torch.profiler",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="only run the prefill phase for evaluation, skipping the decode phase",
    )
    parser.add_argument("--eval-type", type=str, default="fronted", choices=eval_types)
    args = parser.parse_args()
    main(args)
