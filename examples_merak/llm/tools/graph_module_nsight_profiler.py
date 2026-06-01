# ruff: noqa: I001

import argparse
import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
import torch.fx as fx

from graph_module_profiler import (
    dump_total_latency_csv,
    prepare_graph_module_runtime,
    prepare_work_dir,
    synchronize_device,
    warmup_runner,
)
from xhmodel_merak.xh_llm import LLMModelState
from xhquant.api import FXInterpreter, get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers


def can_emit_nvtx() -> bool:
    return torch.cuda.is_available() and hasattr(torch.cuda, "nvtx")


@contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled and can_emit_nvtx():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
        return

    yield


@contextmanager
def cuda_profiler_capture(enabled: bool, device: str, logger, name: str):
    if not enabled:
        yield
        return

    if not torch.cuda.is_available() or not hasattr(torch.cuda, "cudart"):
        raise RuntimeError("CUDA profiler API capture requires torch.cuda.cudart() with CUDA available.")

    cudart = torch.cuda.cudart()
    synchronize_device(device)
    logger.info(f"Start CUDA profiler capture: {name}")
    cudart.cudaProfilerStart()
    try:
        yield
    finally:
        synchronize_device(device)
        cudart.cudaProfilerStop()
        logger.info(f"Stop CUDA profiler capture: {name}")


def measure_runner_total_latency_ms(
    runner: Callable[..., Any],
    device: str,
    range_name: str,
    emit_nvtx: bool,
    *args,
    **kwargs,
) -> float:
    synchronize_device(device)
    start_time = time.perf_counter()
    with nvtx_range(range_name, emit_nvtx):
        runner(*args, **kwargs)
    synchronize_device(device)

    return (time.perf_counter() - start_time) * 1000.0


def measure_graph_module_total_latency_ms(
    graph_module: fx.GraphModule,
    device: str,
    emit_nvtx: bool,
    *args,
    **kwargs,
) -> float:
    return measure_runner_total_latency_ms(
        graph_module,
        device,
        "graph_module::total_latency",
        emit_nvtx,
        *args,
        **kwargs,
    )


def measure_fx_interpreter_total_latency_ms(
    graph_module: fx.GraphModule,
    device: str,
    emit_nvtx: bool,
    *args,
    **kwargs,
) -> float:
    interpreter = FXInterpreter(graph_module)
    return measure_runner_total_latency_ms(
        interpreter.run,
        device,
        "fx_interpreter::total_latency",
        emit_nvtx,
        *args,
        **kwargs,
    )


def build_node_range_name(node: fx.Node) -> str:
    node_target = str(node.target)
    if node_target and node_target != node.name:
        return f"graph_module_node::{node.name}::{node_target}"
    return f"graph_module_node::{node.name}"


def should_profile_total_latency(profile_mode: str) -> bool:
    return profile_mode in {"full", "total_latency_only"}


def should_profile_nodes(profile_mode: str) -> bool:
    return profile_mode in {"full", "node_profile_only"}


def should_capture_total_latency_with_cuda_profiler(profile_mode: str) -> bool:
    return profile_mode == "total_latency_only"


def should_capture_nodes_with_cuda_profiler(profile_mode: str) -> bool:
    return should_profile_nodes(profile_mode)


def prepare_nsight_work_dir(config_path: str, eval_type: str, profile_mode: str) -> Path:
    work_dir = prepare_work_dir(config_path, eval_type) / "nsight" / profile_mode
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


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


def get_module_type(graph_module: fx.GraphModule, node: fx.Node) -> str:
    if node.op != "call_module":
        return ""
    try:
        return type(graph_module.get_submodule(node.target)).__name__
    except AttributeError:
        return ""


def dump_node_manifest(graph_module: fx.GraphModule, work_dir: Path) -> Path:
    manifest_csv = work_dir / "graph_module_node_manifest.csv"
    rows: List[Dict[str, Any]] = []
    for node_index, node in enumerate(graph_module.graph.nodes):
        rows.append(
            {
                "node_index": node_index,
                "node_name": node.name,
                "node_op": node.op,
                "node_target": str(node.target),
                "module_type": get_module_type(graph_module, node),
                "nvtx_range": build_node_range_name(node),
            }
        )

    with manifest_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "node_index",
                "node_name",
                "node_op",
                "node_target",
                "module_type",
                "nvtx_range",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest_csv


class GraphModuleNodeNvtxInterpreter(FXInterpreter):
    def __init__(self, graph_module: fx.GraphModule, emit_nvtx: bool):
        super().__init__(graph_module)
        self._emit_nvtx = emit_nvtx

    def run_node(self, node: fx.Node) -> Any:
        with nvtx_range(build_node_range_name(node), self._emit_nvtx):
            outs = FXInterpreter.run_node(self, node)
            synchronize_device("cuda" if torch.cuda.is_available() else "cpu")
            return outs


def run_node_marked_profile(
    interpreter: GraphModuleNodeNvtxInterpreter,
    profile_runs: int,
    device: str,
    logger,
    *args,
    **kwargs,
) -> None:
    if profile_runs <= 0:
        raise ValueError("profile_runs must be greater than 0.")

    logger.info(f"Start node-marked Nsight profile: {profile_runs} run(s).")
    for run_index in range(profile_runs):
        run_name = f"graph_module::node_profile::run_{run_index}"
        with nvtx_range(run_name, interpreter._emit_nvtx):
            interpreter.run(*args, **kwargs)
        synchronize_device(device)
        logger.info(f"Finished node-marked Nsight run {run_index + 1}/{profile_runs}.")


def main(args):
    eval_type = args.eval_type
    work_dir = prepare_nsight_work_dir(args.config, eval_type, args.profile_mode)
    log_file = str(work_dir / "graph_nsight_profiler.log")
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xhquant_init(log_file, debug=False)
    logger = get_xhquant_logger()

    if args.emit_nvtx:
        logger.info("NVTX ranges are enabled for Nsight profiling.")
    if args.cuda_profiler_api:
        logger.info("CUDA profiler API capture is enabled for Nsight Systems capture-range.")
    logger.info(f"Nsight profile mode: {args.profile_mode}")
    if args.cuda_profiler_api and args.profile_mode == "full":
        logger.info(
            "Full mode keeps total-latency measurements outside CUDA profiler capture and reserves Nsight capture "
            "for node-marked runs."
        )

    inference_contexts, xh_model, inputs = prepare_graph_module_runtime(args, work_dir, logger, device, dtype)

    with ContextManagers(inference_contexts):
        graph_module = resolve_graph_module(xh_model._inference_model)
        manifest_csv = dump_node_manifest(graph_module, work_dir)
        logger.info(f"Saved GraphModule node manifest to: {manifest_csv}")

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
        node_marked_interpreter = GraphModuleNodeNvtxInterpreter(graph_module, emit_nvtx=args.emit_nvtx)
        warmup_runner(
            "node_marked_fx_interpreter",
            node_marked_interpreter.run,
            args.warmup_runs,
            device,
            logger,
            *inputs,
        )

        if should_profile_total_latency(args.profile_mode):
            with cuda_profiler_capture(
                args.cuda_profiler_api and should_capture_total_latency_with_cuda_profiler(args.profile_mode),
                device,
                logger,
                "graph_module_total_latency_profile",
            ):
                graph_module_total_latency_ms = measure_graph_module_total_latency_ms(
                    graph_module, device, args.emit_nvtx, *inputs
                )
                fx_interpreter_total_latency_ms = measure_fx_interpreter_total_latency_ms(
                    graph_module, device, args.emit_nvtx, *inputs
                )

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
                logger.info(
                    f"FXInterpreter overhead vs graph_module: {latency_diff_ms:.3f} ms, slowdown={slowdown_ratio:.3f}x"
                )
                logger.info(f"Saved total latency comparison to: {total_latency_csv}")

        if should_profile_nodes(args.profile_mode):
            with cuda_profiler_capture(
                args.cuda_profiler_api and should_capture_nodes_with_cuda_profiler(args.profile_mode),
                device,
                logger,
                "graph_module_node_profile",
            ):
                run_node_marked_profile(
                    node_marked_interpreter,
                    args.profile_runs,
                    device,
                    logger,
                    *inputs,
                )


if __name__ == "__main__":
    eval_types = LLMModelState.get_all_values()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config", type=str, default="configs_merak/xh2a/llm_models/qwen3_legacy/8b/qwen3_8b_legacy_xh2a_2k.py"
    )
    parser.add_argument("--prefill-length", type=int, default=2048, help="the sequence length for the prefill phase")
    parser.add_argument(
        "--only-prefill",
        action="store_true",
        help="only run the prefill phase for evaluation, skipping the decode phase",
    )
    parser.add_argument("--warmup-runs", type=int, default=2, help="number of warmup runs before Nsight capture")
    parser.add_argument("--profile-runs", type=int, default=1, help="number of Nsight-captured runs")
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="reserved for launcher compatibility; unused by the Nsight-only runner",
    )
    parser.add_argument(
        "--profile-mode",
        type=str,
        default="full",
        choices=["full", "total_latency_only", "node_profile_only"],
        help="Nsight capture mode",
    )
    parser.add_argument("--emit-nvtx", action="store_true", help="emit NVTX ranges for NVIDIA Nsight profiling")
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help="wrap measured sections with cudaProfilerStart/Stop for Nsight Systems capture-range",
    )
    parser.add_argument("--eval-type", type=str, default="quanted_disable", choices=eval_types)
    args = parser.parse_args()
    main(args)
