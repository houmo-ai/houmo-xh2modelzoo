# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Run LIBERO rollout eval with Qwen visual + context graph + ActionHead HMONNX.

This is an experimental full-HMONNX evaluator. The exported Qwen/context graph is
static-shape, so this script intentionally supports batch_size=1 first and fails
fast when live LIBERO inputs do not match the exported HMONNX schemas.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VLA_JEPA_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _VLA_JEPA_ROOT.parents[2]
for _path in (_VLA_JEPA_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import json
import os
import time
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from eval.eval_action_head_hmonnx_libero import (
    ActionHeadHMONNXRuntime,
    as_jsonable,
    install_action_head_hmonnx,
    parse_task_ids,
)
from export.export_context_graph_wrapper import GRAPH_INPUT_NAMES, dense_deepstack
from common.inspect_policy import DEFAULT_MODEL, load_policy
from common.paths import (
    DEFAULT_STANDARD_ACTION_HEAD_HMONNX,
    DEFAULT_CONTEXT_GRAPH_HMONNX,
    DEFAULT_VISUAL_ENCODER_HMONNX,
    output_str,
    set_default_libero_config_path,
)

DEFAULT_ACTION_HMONNX = DEFAULT_STANDARD_ACTION_HEAD_HMONNX
DEFAULT_VISUAL_HMONNX = DEFAULT_VISUAL_ENCODER_HMONNX
DEFAULT_CONTEXT_HMONNX = DEFAULT_CONTEXT_GRAPH_HMONNX
DEFAULT_REPORT = output_str("eval", "full_hmonnx_libero_eval_report.json")


def log(message: str) -> None:
    print(f"[eval_full_hmonnx] {message}", flush=True)


def configure_hmonnx_progress(enabled: bool) -> None:
    if not enabled:
        return
    import xhquant.xhonnxruntime.config as hmonnx_config

    hmonnx_config.disable_progress = False
    hmonnx_config.verbose_progress = True
    log("enabled HMONNXInference node progress")


def tensor_summary(name: str, tensor: torch.Tensor) -> str:
    return f"{name}:shape={tuple(tensor.shape)},dtype={tensor.dtype},device={tensor.device}"


def value_summary(name: str, value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return tensor_summary(name, value)
    if isinstance(value, (tuple, list)):
        return f"{name}:" + ", ".join(value_summary(f"out{idx}", item) for idx, item in enumerate(value))
    return f"{name}:type={type(value).__name__}"


def _detach_mapping_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_mapping_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detach_mapping_to_cpu(item) for item in value)
    return value


class NamedHMONNXRuntime:
    def __init__(self, hmonnx_path: str, device: str, trace: bool = False, label: str | None = None) -> None:
        from xhquant.api import HMONNXInference

        self.hmonnx_path = hmonnx_path
        self.device = device
        self.trace = trace
        self.label = label or Path(hmonnx_path).name
        self.session = HMONNXInference(hmonnx_path)
        self.session.to(device)
        self.num_calls = 0
        self.input_schema = [
            {"name": item.name, "dtype": str(item.dtype), "shape": [int(dim) for dim in item.shape]}
            for item in self.session.inputs
        ]
        if self.trace:
            log(f"[hmonnx_trace] {self.label} session ready path={self.hmonnx_path} inputs={self.input_schema}")

    def __call__(self, input_by_name: dict[str, torch.Tensor]) -> Any:
        hmonnx_inputs = []
        for input_info in self.session.inputs:
            if input_info.name not in input_by_name:
                raise KeyError(f"Missing HMONNX input {input_info.name}; available={sorted(input_by_name)}")
            tensor = input_by_name[input_info.name].detach().to(device=self.device, dtype=input_info.dtype)
            expected_shape = tuple(int(dim) for dim in input_info.shape)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Input shape mismatch for {input_info.name}: got={tuple(tensor.shape)} expected={expected_shape}"
                )
            hmonnx_inputs.append(tensor)
        if self.trace:
            input_text = ", ".join(
                tensor_summary(input_info.name, tensor)
                for input_info, tensor in zip(self.session.inputs, hmonnx_inputs, strict=True)
            )
            log(f"[hmonnx_trace] {self.label} call={self.num_calls + 1} path={self.hmonnx_path} inputs=[{input_text}]")
        with torch.no_grad():
            output = self.session(*hmonnx_inputs)
        self.num_calls += 1
        if self.trace:
            log(f"[hmonnx_trace] {self.label} output={value_summary('output', output)}")
        return output

    def stats(self) -> dict[str, Any]:
        return {
            "hmonnx_path": self.hmonnx_path,
            "device": self.device,
            "trace": self.trace,
            "label": self.label,
            "num_calls": self.num_calls,
            "input_schema": self.input_schema,
        }


class MultiShapeHMONNXRuntime:
    SEQUENCE_INPUT_NAMES = {
        "inputs_embeds",
        "time_position_ids",
        "height_position_ids",
        "width_position_ids",
        "deepstack_visual_embed_0",
        "deepstack_visual_embed_1",
        "deepstack_visual_embed_2",
    }

    def __init__(self, hmonnx_paths: list[str], device: str, trace: bool = False, label: str = "context") -> None:
        if not hmonnx_paths:
            raise ValueError("At least one context HMONNX path is required")
        self.trace = trace
        self.label = label
        self.runtimes = [
            NamedHMONNXRuntime(path, device, trace=trace, label=f"{label}[{idx}]")
            for idx, path in enumerate(hmonnx_paths)
        ]
        self.num_calls = 0
        self.selected_counts = {runtime.hmonnx_path: 0 for runtime in self.runtimes}

    @staticmethod
    def _schema_by_name(runtime: NamedHMONNXRuntime) -> dict[str, tuple[int, ...]]:
        return {item["name"]: tuple(item["shape"]) for item in runtime.input_schema}

    @staticmethod
    def _pad_sequence_tensor(tensor: torch.Tensor, expected_shape: tuple[int, ...]) -> torch.Tensor:
        if tuple(tensor.shape) == expected_shape:
            return tensor
        if tensor.dim() == 1:
            seq_dim = 0
        elif tensor.dim() == 3:
            seq_dim = 1
        else:
            raise ValueError(f"Cannot pad non-sequence tensor shape={tuple(tensor.shape)} to {expected_shape}")

        got_shape = list(tensor.shape)
        target_shape = list(expected_shape)
        got_seq = got_shape[seq_dim]
        target_seq = target_shape[seq_dim]
        got_shape[seq_dim] = target_seq
        if got_seq > target_seq or got_shape != target_shape:
            raise ValueError(f"Cannot pad tensor shape={tuple(tensor.shape)} to {expected_shape}")

        pad_shape = list(tensor.shape)
        pad_shape[seq_dim] = target_seq - got_seq
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=seq_dim)

    def _prepare_for_runtime(
        self,
        runtime: NamedHMONNXRuntime,
        input_by_name: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        schema_by_name = self._schema_by_name(runtime)
        prepared = dict(input_by_name)
        for name, expected_shape in schema_by_name.items():
            if name not in input_by_name:
                raise KeyError(f"Missing HMONNX input {name}; available={sorted(input_by_name)}")
            tensor = input_by_name[name]
            if tuple(tensor.shape) == expected_shape:
                continue
            if name in self.SEQUENCE_INPUT_NAMES:
                prepared[name] = self._pad_sequence_tensor(tensor, expected_shape)
            else:
                raise ValueError(f"Input shape mismatch for {name}: got={tuple(tensor.shape)} expected={expected_shape}")

        if "current_input_length" in schema_by_name and "current_input_length" in prepared:
            target_seq = None
            for name in ("inputs_embeds", "time_position_ids"):
                expected_shape = schema_by_name.get(name)
                if expected_shape is not None:
                    target_seq = expected_shape[1] if len(expected_shape) == 3 else expected_shape[0]
                    break
            if target_seq is not None:
                current = prepared["current_input_length"]
                prepared["current_input_length"] = torch.tensor([target_seq], dtype=current.dtype, device=current.device)
        return prepared

    def __call__(self, input_by_name: dict[str, torch.Tensor]) -> Any:
        errors = []
        for runtime in self.runtimes:
            try:
                prepared = self._prepare_for_runtime(runtime, input_by_name)
            except Exception as exc:
                errors.append({"hmonnx_path": runtime.hmonnx_path, "error": str(exc)})
                continue

            self.num_calls += 1
            self.selected_counts[runtime.hmonnx_path] += 1
            if self.trace:
                original_shapes = {name: list(tensor.shape) for name, tensor in input_by_name.items()}
                prepared_shapes = {name: list(tensor.shape) for name, tensor in prepared.items()}
                log(
                    f"[hmonnx_trace] {self.label} selected path={runtime.hmonnx_path} "
                    f"original_shapes={original_shapes} prepared_shapes={prepared_shapes}"
                )
            return runtime(prepared)

        got = {name: list(tensor.shape) for name, tensor in input_by_name.items()}
        available = [
            {"hmonnx_path": runtime.hmonnx_path, "input_schema": runtime.input_schema}
            for runtime in self.runtimes
        ]
        raise ValueError(f"No context HMONNX graph matches live inputs: got={got} available={available} errors={errors}")

    def stats(self) -> dict[str, Any]:
        return {
            "num_calls": self.num_calls,
            "trace": self.trace,
            "label": self.label,
            "selected_counts": self.selected_counts,
            "runtimes": [runtime.stats() for runtime in self.runtimes],
        }


def parse_hmonnx_paths(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class QwenContextHMONNXRuntime:
    def __init__(
        self,
        policy: Any,
        visual_hmonnx: str,
        context_hmonnx: str,
        hmonnx_device: str,
        dtype: str,
        dump_context_sample: str | None = None,
        trace: bool = False,
    ) -> None:
        self.policy = policy
        self.trace = trace
        self.visual = NamedHMONNXRuntime(visual_hmonnx, hmonnx_device, trace=trace, label="visual")
        self.context = MultiShapeHMONNXRuntime(parse_hmonnx_paths(context_hmonnx), hmonnx_device, trace=trace, label="context")
        self.hmonnx_device = hmonnx_device
        self.dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        self.dump_context_sample = dump_context_sample
        self.num_calls = 0
        self.last_shapes: dict[str, Any] = {}
        if self.trace:
            log(
                "[hmonnx_trace] qwen_context runtime ready "
                f"visual={visual_hmonnx} context={context_hmonnx} device={hmonnx_device}"
            )

    def _visual_outputs(self, qwen_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        outputs = self.visual({"pixel_values": qwen_inputs["pixel_values"]})
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        outputs = tuple(outputs)
        if len(outputs) != 4:
            raise RuntimeError(f"Expected 4 visual HMONNX outputs, got {len(outputs)}")
        return outputs

    def _context_inputs(self, qwen_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        qwen_core = self.policy.model.qwen.model.model
        input_ids = qwen_inputs["input_ids"].to(next(qwen_core.parameters()).device)
        attention_mask = qwen_inputs["attention_mask"].to(input_ids.device)
        mm_token_type_ids = qwen_inputs["mm_token_type_ids"].to(input_ids.device)
        image_grid_thw = qwen_inputs["image_grid_thw"].to(input_ids.device)

        image_embeds, deep0, deep1, deep2 = self._visual_outputs(qwen_inputs)
        target_device = input_ids.device
        image_embeds = image_embeds.to(device=target_device, dtype=self.dtype)
        deep0 = deep0.to(device=target_device, dtype=self.dtype)
        deep1 = deep1.to(device=target_device, dtype=self.dtype)
        deep2 = deep2.to(device=target_device, dtype=self.dtype)

        inputs_embeds = qwen_core.get_input_embeddings()(input_ids).to(dtype=self.dtype)
        image_mask, _ = qwen_core.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0].bool()

        position_ids = qwen_core.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )
        if position_ids is None:
            raise RuntimeError("Qwen compute_3d_position_ids returned None for multimodal inputs")
        if position_ids.dim() == 3:
            position_ids = position_ids[:, 0, :]

        embodied_mask = input_ids == self.policy.model.embodied_action_token_id
        embodied_positions = embodied_mask.nonzero(as_tuple=False)[:, 1].view(input_ids.shape[0], -1).long()

        seq_len = inputs_embeds.shape[1]
        past_seq_length = torch.zeros(1, dtype=torch.int32, device=target_device)
        current_input_length = torch.tensor([seq_len], dtype=torch.int32, device=target_device)

        dense0 = dense_deepstack(inputs_embeds, visual_pos_masks, deep0)
        dense1 = dense_deepstack(inputs_embeds, visual_pos_masks, deep1)
        dense2 = dense_deepstack(inputs_embeds, visual_pos_masks, deep2)

        inputs = (
            inputs_embeds,
            position_ids[0].long(),
            position_ids[1].long(),
            position_ids[2].long(),
            past_seq_length,
            current_input_length,
            embodied_positions,
            dense0,
            dense1,
            dense2,
        )
        self.last_shapes = {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in zip(GRAPH_INPUT_NAMES, inputs, strict=True)
        }
        return inputs

    def _maybe_dump_context_sample(self, qwen_inputs: dict[str, torch.Tensor], inputs: tuple[torch.Tensor, ...]) -> None:
        if not self.dump_context_sample or self.num_calls != 1:
            return
        path = Path(self.dump_context_sample)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph_inputs = dict(zip(GRAPH_INPUT_NAMES, inputs, strict=True))
        inputs_embeds = graph_inputs["inputs_embeds"]
        embodied_positions = graph_inputs["embodied_positions"]
        conditioning_tokens = torch.zeros(
            inputs_embeds.shape[0],
            embodied_positions.shape[1],
            inputs_embeds.shape[-1],
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        torch.save(
            {
                "qwen_inputs": _detach_mapping_to_cpu(qwen_inputs),
                "context_graph_inputs": _detach_mapping_to_cpu(graph_inputs),
                "conditioning_tokens": conditioning_tokens.detach().cpu(),
            },
            path,
        )
        log(f"dumped live context sample: {path}")

    def last_hidden(self, qwen_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        self.num_calls += 1
        inputs = self._context_inputs(qwen_inputs)
        self._maybe_dump_context_sample(qwen_inputs, inputs)
        if self.trace:
            log(f"[hmonnx_trace] qwen_context call={self.num_calls} graph_inputs={self.last_shapes}")
        context_output = self.context(dict(zip(GRAPH_INPUT_NAMES, inputs, strict=True)))
        if isinstance(context_output, (tuple, list)):
            context_output = context_output[0]

        input_ids = qwen_inputs["input_ids"].to(next(self.policy.model.parameters()).device)
        embodied_positions = inputs[6].to(input_ids.device)
        batch_size = input_ids.shape[0]
        hidden_size = context_output.shape[-1]
        last_hidden = torch.zeros(
            batch_size,
            input_ids.shape[1],
            hidden_size,
            device=input_ids.device,
            dtype=context_output.dtype,
        )
        values = context_output.to(device=input_ids.device, dtype=last_hidden.dtype)
        for batch_idx in range(batch_size):
            last_hidden[batch_idx, embodied_positions[batch_idx]] = values[batch_idx]
        return last_hidden

    def stats(self) -> dict[str, Any]:
        return {
            "num_calls": self.num_calls,
            "hmonnx_device": self.hmonnx_device,
            "trace": self.trace,
            "dtype": str(self.dtype),
            "last_shapes": self.last_shapes,
            "visual": self.visual.stats(),
            "context": self.context.stats(),
        }


def install_qwen_context_hmonnx(policy: Any, runtime: QwenContextHMONNXRuntime) -> None:
    def qwen_last_decoder_hidden_hmonnx(self: Any, qwen_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return runtime.last_hidden(qwen_inputs)

    policy.model._qwen_last_decoder_hidden = MethodType(qwen_last_decoder_hidden_hmonnx, policy.model)


def build_policy_and_processors(args: argparse.Namespace):
    from lerobot.envs import make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies import make_pre_post_processors

    if args.batch_size != 1:
        raise ValueError("Full HMONNX context graph is static-shape; this script currently supports --batch-size 1 only")

    policy = load_policy(args.model, args.device, disable_world_model=True)
    policy.eval()

    log(f"loading Qwen visual HMONNX: {args.visual_hmonnx}")
    log(f"loading context graph HMONNX: {args.context_hmonnx}")
    qwen_context_runtime = QwenContextHMONNXRuntime(
        policy,
        args.visual_hmonnx,
        args.context_hmonnx,
        args.hmonnx_device,
        args.context_dtype,
        args.dump_context_sample,
        trace=args.trace_hmonnx,
    )
    install_qwen_context_hmonnx(policy, qwen_context_runtime)

    log(f"loading ActionHead HMONNX: {args.action_hmonnx}")
    action_runtime = ActionHeadHMONNXRuntime(args.action_hmonnx, args.hmonnx_device, trace=args.trace_hmonnx)
    install_action_head_hmonnx(policy.model.action_model, action_runtime)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.model,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_cfg = LiberoEnv(task=args.task, task_ids=parse_task_ids(args.task_ids))
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    return policy, env_cfg, env_preprocessor, env_postprocessor, preprocessor, postprocessor, qwen_context_runtime, action_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--action-hmonnx", default=DEFAULT_ACTION_HMONNX)
    parser.add_argument("--visual-hmonnx", default=DEFAULT_VISUAL_HMONNX)
    parser.add_argument("--context-hmonnx", default=DEFAULT_CONTEXT_HMONNX)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hmonnx-device", default="cuda:0")
    parser.add_argument("--context-dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    parser.add_argument("--task", default="libero_10")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-episodes-rendered", type=int, default=0)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--dump-context-sample", default=None)
    parser.add_argument("--trace-hmonnx", action="store_true")
    parser.add_argument("--trace-hmonnx-progress", action="store_true")
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    configure_hmonnx_progress(args.trace_hmonnx_progress)
    torch.manual_seed(args.seed)

    from lerobot.envs import close_envs, make_env
    from lerobot.scripts.lerobot_eval import eval_policy_all

    start = time.time()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        log(f"loading policy full_hmonnx: {args.model}")
        (
            policy,
            env_cfg,
            env_preprocessor,
            env_postprocessor,
            preprocessor,
            postprocessor,
            qwen_context_runtime,
            action_runtime,
        ) = build_policy_and_processors(args)

        log(f"creating env task={args.task} task_ids={args.task_ids} batch_size={args.batch_size}")
        envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=False)
        try:
            log("starting eval rollout")
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=args.n_episodes,
                max_episodes_rendered=args.max_episodes_rendered,
                videos_dir=report_path.parent / "videos" if args.max_episodes_rendered > 0 else None,
                start_seed=args.seed,
                max_parallel_tasks=1,
            )
        finally:
            close_envs(envs)

        report = {
            "status": "ok",
            "mode": "full_hmonnx",
            "model": args.model,
            "task": args.task,
            "task_ids": parse_task_ids(args.task_ids),
            "n_episodes_per_task": args.n_episodes,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": args.device,
            "hmonnx_device": args.hmonnx_device,
            "elapsed_s": time.time() - start,
            "runtime": {
                "qwen_context": qwen_context_runtime.stats(),
                "action": action_runtime.stats(),
            },
            "eval": as_jsonable(info),
        }
    except Exception as exc:  # keep shape/runtime failures inspectable
        report = {
            "status": "failed",
            "mode": "full_hmonnx",
            "model": args.model,
            "task": args.task,
            "task_ids": parse_task_ids(args.task_ids),
            "n_episodes_per_task": args.n_episodes,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": args.device,
            "hmonnx_device": args.hmonnx_device,
            "elapsed_s": time.time() - start,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        raise

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"wrote report: {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
