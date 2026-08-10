import copy
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.config import WorkflowConfig
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult


PI05_EXPORT_DTYPE = torch.float16


class PI05Workflow(BaseOtherModelWorkflow):
    SUPPORTED_COMPONENTS = {"vision", "gemma", "expert", "other"}

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise NotImplementedError(f"{type(self).__name__} does not support standalone quantization")
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        from xhquant.api import set_random_seed

        set_random_seed(self.seed)
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_model_dir = self._resolve_export_model_dir(quant_result)
        effective_overrides = {"export.model.hf_model": export_model_dir}
        if "expert_model" in workflow_config.export:
            effective_overrides["export.expert_model.hf_model"] = export_model_dir
        workflow_config = workflow_config.with_overrides(effective_overrides)
        workflow_config = WorkflowConfig(data=workflow_config.data, source=self.workflow_config.source)
        export_cfg = workflow_config.build_export_dict()
        target_device = str(export_cfg.get("target_device", "XH2a"))
        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))

        raw_components = export_cfg.get("components")
        if raw_components is None:
            components = ["vision", "gemma", "expert", "other"]
        elif isinstance(raw_components, str):
            components = [raw_components]
        elif isinstance(raw_components, Mapping):
            components = [str(name) for name, enabled in raw_components.items() if enabled is not False]
        else:
            components = [str(item) for item in raw_components]
        unsupported = [name for name in components if name not in self.SUPPORTED_COMPONENTS]
        if unsupported:
            raise ValueError(f"Unsupported PI05 component(s): {unsupported}")

        quant_types = export_cfg.get("quant_types") or {}
        if not isinstance(quant_types, Mapping):
            raise TypeError("PI05 export.quant_types must be a mapping when provided")

        config_dir = _require_pi05_config_dir(export_cfg)
        runtime_cfg = {
            "hf_model_dir": export_model_dir,
            "config_dir": config_dir,
            "device": _select_torch_device(device),
            "target_device": target_device,
            "variant": str(export_cfg.get("variant", workflow_config.name)),
            "prompt": str(export_cfg.get("prompt", "请解释Gemma模型的核心优势是什么？")),
        }
        runtime_cfg.update(_build_pi05_runtime_contract(export_cfg))

        meta: dict[str, Any] = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": str(Path(config_file).relative_to(work_dir)),
            "hf_model": export_model_dir,
            "target_device": target_device,
            "variant": runtime_cfg["variant"],
            "compact_prefix": {
                key: runtime_cfg[key]
                for key in (
                    "selected_image_indices",
                    "text_max_length",
                    "prefix_sequence_length",
                    "action_horizon",
                    "cache_length",
                )
            },
            "components": {},
        }

        policy = None
        if any(name in components for name in ("vision", "other")):
            from ._export_utils import load_pi05_policy, set_seed

            set_seed(int(export_cfg.get("seed", self.seed)))
            policy = load_pi05_policy(export_model_dir, device="cpu")

        if "vision" in components:
            result = export_pi05_vision(
                policy=policy,
                component_dir=work_dir / "Vision",
                target_device=target_device,
                quant_type=str(quant_types.get("vision", "w8a8h1_sefp")),
                component_cfg=export_cfg.get("vision") or {},
            )
            meta["components"]["vision"] = _component_meta(work_dir, result)

        if "other" in components:
            result = export_pi05_other(
                policy=policy,
                component_dir=work_dir / "Other",
                target_device=target_device,
                quant_type=str(quant_types.get("other", "w8a8h1_sefp")),
                component_cfg=export_cfg.get("other") or {},
            )
            meta["components"]["other"] = _component_meta(work_dir, result)

        del policy
        _empty_cuda_cache()

        if "gemma" in components:
            model_cfg = copy.deepcopy(export_cfg["model"])
            model_cfg["hf_model"] = export_model_dir
            _ensure_quant_config(model_cfg, target_device, str(quant_types.get("gemma", "w8a8h1_sefp")))
            result = export_pi05_llm_component(
                model_cfg=model_cfg,
                runtime_cfg=runtime_cfg,
                component_dir=work_dir / "Gemma2B",
                target_device=target_device,
                quant_type=str(quant_types.get("gemma", "w8a8h1_sefp")),
                component_name="pi05_gemma_2b",
                kind="gemma",
            )
            meta["components"]["gemma"] = _component_meta(work_dir, result)

        if "expert" in components:
            expert_cfg = copy.deepcopy(export_cfg.get("expert_model"))
            if not isinstance(expert_cfg, dict):
                raise ValueError("PI05 export.expert_model must be provided when component 'expert' is enabled")
            expert_cfg["hf_model"] = export_model_dir
            _ensure_quant_config(expert_cfg, target_device, str(quant_types.get("expert", "w8a8h1_sefp")))
            result = export_pi05_llm_component(
                model_cfg=expert_cfg,
                runtime_cfg=runtime_cfg,
                component_dir=work_dir / "GemmaExpert",
                target_device=target_device,
                quant_type=str(quant_types.get("expert", "w8a8h1_sefp")),
                component_name="pi05_gemma_expert_300m",
                kind="expert",
            )
            meta["components"]["expert"] = _component_meta(work_dir, result)

        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(json.dumps(_jsonable(meta), indent=4, ensure_ascii=False), encoding="utf-8")
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        from xhquant.core import CacheTensor

        work_dir = Path(export_result.work_dir)
        meta = json.loads((work_dir / "export_meta_info.json").read_text(encoding="utf-8"))
        torch_device = _select_torch_device(device)
        export_cfg = _load_dump_export_cfg(export_result, work_dir, meta, self.workflow_config)

        for name in ("vision",):
            comp = meta["components"].get(name)
            if not comp:
                continue
            comp_dir = work_dir / comp["component_dir"]
            hmonnx_file = comp_dir / comp["hmonnx_file"]
            shape = tuple(int(dim) for dim in comp["input_shape"])
            _run_hmonnx_golden(
                hmonnx_file,
                hmonnx_file.parent / "golden",
                torch_device,
                _prepare_hmonnx_inputs_for_runtime(
                    [torch.randn(shape, dtype=torch.float32, device=torch_device)],
                    hmonnx_file,
                    torch_device,
                    CacheTensor,
                ),
            )

        other = meta["components"].get("other")
        if other:
            comp_dir = work_dir / other["component_dir"]
            for item in other["graphs"]:
                hmonnx_file = comp_dir / item["hmonnx_file"]
                shape = tuple(int(dim) for dim in item["input_shape"])
                _run_hmonnx_golden(
                    hmonnx_file,
                    hmonnx_file.parent / "golden" / item["name"],
                    torch_device,
                    _prepare_hmonnx_inputs_for_runtime(
                        [torch.randn(shape, dtype=torch.float32, device=torch_device)],
                        hmonnx_file,
                        torch_device,
                        CacheTensor,
                    ),
                )

        for name in ("gemma", "expert"):
            comp = meta["components"].get(name)
            if not comp:
                continue
            comp_dir = work_dir / comp["component_dir"]
            comp_meta = json.loads((comp_dir / comp["meta_file"]).read_text(encoding="utf-8"))
            kind = "gemma" if name == "gemma" else "expert"
            model_cfg_key = "model" if kind == "gemma" else "expert_model"
            model_cfg = copy.deepcopy(export_cfg[model_cfg_key])
            model_cfg["hf_model"] = meta["hf_model"]
            prefill_inputs, decode_inputs = _build_pi05_llm_golden_inputs(
                model_cfg=model_cfg,
                config_dir=_require_pi05_config_dir(export_cfg),
                prompt=str(export_cfg.get("prompt", "请解释Gemma模型的核心优势是什么？")),
                kind=kind,
                device=torch_device,
                comp_meta=comp_meta,
                selected_image_indices=export_cfg["selected_image_indices"],
                text_max_length=int(export_cfg["text_max_length"]),
            )

            prefill_path = comp_dir / comp_meta["prefill_onnx_file"]
            _run_hmonnx_golden(
                prefill_path,
                prefill_path.parent / "golden",
                torch_device,
                _prepare_hmonnx_inputs_for_runtime(prefill_inputs, prefill_path, torch_device, CacheTensor),
            )

            decode_path = comp_dir / comp_meta["decode_onnx_file"]
            _run_hmonnx_golden(
                decode_path,
                decode_path.parent / "golden",
                torch_device,
                _prepare_hmonnx_inputs_for_runtime(decode_inputs, decode_path, torch_device, CacheTensor),
            )

        return str(work_dir)


def _build_pi05_runtime_contract(export_cfg: Mapping[str, Any]) -> dict[str, Any]:
    selected_image_indices = export_cfg.get("selected_image_indices")
    if (
        not isinstance(selected_image_indices, Sequence)
        or isinstance(selected_image_indices, (str, bytes))
        or not selected_image_indices
    ):
        raise ValueError("PI05 export.selected_image_indices must be a non-empty sequence")
    selected_image_indices = [int(index) for index in selected_image_indices]
    if len(set(selected_image_indices)) != len(selected_image_indices) or any(
        index < 0 for index in selected_image_indices
    ):
        raise ValueError("PI05 export.selected_image_indices must contain unique non-negative indices")

    text_max_length = int(export_cfg.get("text_max_length", 0))
    if text_max_length <= 0:
        raise ValueError("PI05 export.text_max_length must be positive")

    model_cfg = export_cfg.get("model")
    expert_cfg = export_cfg.get("expert_model")
    other_cfg = export_cfg.get("other")
    if not isinstance(model_cfg, Mapping) or not isinstance(expert_cfg, Mapping):
        raise ValueError("PI05 export requires model and expert_model mappings")
    model_wrap_cfg = model_cfg.get("wrap_cfg")
    expert_wrap_cfg = expert_cfg.get("wrap_cfg")
    if not isinstance(model_wrap_cfg, Mapping) or not isinstance(expert_wrap_cfg, Mapping):
        raise ValueError("PI05 model and expert_model require wrap_cfg mappings")

    prefix_sequence_length = int(model_wrap_cfg["input_sequence_length"])
    expected_prefix_length = len(selected_image_indices) * 256 + text_max_length
    if prefix_sequence_length != expected_prefix_length:
        raise ValueError(
            f"PI05 Gemma input_sequence_length is {prefix_sequence_length}, expected "
            f"{expected_prefix_length} for {len(selected_image_indices)} selected image(s)"
        )

    action_horizon = int(expert_wrap_cfg["input_sequence_length"])
    if action_horizon <= 0:
        raise ValueError("PI05 Expert input_sequence_length must be positive")
    if isinstance(other_cfg, Mapping) and int(other_cfg.get("sequence_length", action_horizon)) != action_horizon:
        raise ValueError("PI05 other.sequence_length must match the Expert action horizon")

    cache_length = int(model_wrap_cfg["max_sequence_length"])
    expert_cache_length = int(expert_wrap_cfg["max_sequence_length"])
    required_cache_length = prefix_sequence_length + action_horizon
    if cache_length < required_cache_length or expert_cache_length < required_cache_length:
        raise ValueError(f"PI05 cache capacity must be at least compact prefix + horizon ({required_cache_length})")
    if cache_length != expert_cache_length:
        raise ValueError("PI05 Gemma and Expert cache capacities must match")

    expected_input_names = {
        "model": ["inputs_embeds", "past_seq_length", "current_input_length", "attention_mask"],
        "expert_model": [
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "cond",
            "attention_mask",
        ],
    }
    for config_name, component_cfg in (("model", model_cfg), ("expert_model", expert_cfg)):
        component_export_cfg = component_cfg.get("export_cfg")
        input_names = component_export_cfg.get("input_names") if isinstance(component_export_cfg, Mapping) else None
        if input_names != expected_input_names[config_name]:
            raise ValueError(f"PI05 {config_name}.export_cfg.input_names must be {expected_input_names[config_name]}")

    return {
        "selected_image_indices": selected_image_indices,
        "text_max_length": text_max_length,
        "prefix_sequence_length": prefix_sequence_length,
        "action_horizon": action_horizon,
        "cache_length": cache_length,
    }


def _require_pi05_config_dir(export_cfg: Mapping[str, Any]) -> str:
    config_dir = export_cfg.get("config_dir")
    if not isinstance(config_dir, str) or not config_dir.strip():
        raise ValueError("PI05 export.config_dir must be supplied externally")
    return config_dir


def export_pi05_vision(
    *,
    policy: Any,
    component_dir: Path,
    target_device: str,
    quant_type: str,
    component_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    from xhquant.api import convert_onnx_to_hmonnx

    from ._export_utils import Siglip, export_onnx_and_simplify

    if policy is None:
        raise ValueError("PI05 policy must be loaded before exporting vision")
    component_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = component_dir / "onnx"
    hmonnx_dir = component_dir / "hmonnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)
    hmonnx_dir.mkdir(exist_ok=True, parents=True)

    input_shape = tuple(int(dim) for dim in component_cfg.get("input_shape", [1, 3, 224, 224]))
    input_features = torch.randn(input_shape, dtype=torch.float32)
    model = Siglip(policy.model.paligemma_with_expert.paligemma.model).float().cpu().eval()
    onnx_file = onnx_dir / "pi05_siglip.onnx"
    simplified_onnx = onnx_dir / "pi05_siglip_simplified.onnx"
    hmonnx_file = hmonnx_dir / f"vision_{target_device}_{quant_type}.onnx"
    export_onnx_and_simplify(
        model,
        input_features,
        onnx_file,
        simplified_onnx,
        ["pixel_values"],
        ["output"],
        {"pixel_values": list(input_shape)},
        opset_version=int(component_cfg.get("opset", 17)),
        verbose=bool(component_cfg.get("verbose", False)),
    )
    convert_onnx_to_hmonnx(
        simplified_onnx,
        (input_features,),
        out_hmonnx_file=hmonnx_file,
        device_type=_to_hmonnx_device_type(target_device),
        quant_config=_build_quant_config(target_device, quant_type),
    )
    return {
        "component_dir": str(component_dir),
        "quant_type": quant_type,
        "onnx_file": str(onnx_file.relative_to(component_dir)),
        "simplified_onnx_file": str(simplified_onnx.relative_to(component_dir)),
        "hmonnx_file": str(hmonnx_file.relative_to(component_dir)),
        "input_shape": list(input_shape),
    }


def export_pi05_other(
    *,
    policy: Any,
    component_dir: Path,
    target_device: str,
    quant_type: str,
    component_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    from xhquant.api import convert_onnx_to_hmonnx

    from ._export_utils import TimeMLPWrapper, export_onnx_and_simplify

    if policy is None:
        raise ValueError("PI05 policy must be loaded before exporting other components")
    component_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = component_dir / "onnx"
    hmonnx_dir = component_dir / "hmonnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    quant_config = _build_quant_config(target_device, quant_type)

    sequence_length = int(component_cfg.get("sequence_length", policy.config.chunk_size))
    action_dim = int(component_cfg.get("action_dim", policy.config.max_action_dim))
    hidden_size = int(component_cfg.get("hidden_size", 1024))

    graphs = []
    action_in = policy.model.action_in_proj.float().cpu().eval()
    action_in_shape = (1, sequence_length, action_dim)
    action_in_input = torch.randn(action_in_shape, dtype=torch.float32)
    action_in_onnx = export_onnx_and_simplify(
        action_in,
        action_in_input,
        onnx_dir / "pi05_action_in_proj.onnx",
        onnx_dir / "pi05_action_in_proj_simplified.onnx",
        ["action_in"],
        ["action_in_proj_out"],
        {"action_in": list(action_in_shape)},
        opset_version=int(component_cfg.get("opset", 17)),
        verbose=bool(component_cfg.get("verbose", False)),
    )
    action_in_hmonnx = hmonnx_dir / f"action_in_proj_{target_device}_{quant_type}.onnx"
    convert_onnx_to_hmonnx(
        action_in_onnx,
        (action_in_input,),
        out_hmonnx_file=action_in_hmonnx,
        device_type=_to_hmonnx_device_type(target_device),
        quant_config=quant_config,
    )
    graphs.append(
        _simple_graph_meta("action_in_proj", component_dir, action_in_onnx, action_in_hmonnx, action_in_shape)
    )

    action_out = policy.model.action_out_proj.float().cpu().eval()
    action_out_shape = (1, sequence_length, hidden_size)
    action_out_input = torch.randn(action_out_shape, dtype=torch.float32)
    action_out_onnx = export_onnx_and_simplify(
        action_out,
        action_out_input,
        onnx_dir / "pi05_action_out_proj.onnx",
        onnx_dir / "pi05_action_out_proj_simplified.onnx",
        ["action_out"],
        ["action_out_proj_out"],
        {"action_out": list(action_out_shape)},
        opset_version=int(component_cfg.get("opset", 17)),
        verbose=bool(component_cfg.get("verbose", False)),
    )
    action_out_hmonnx = hmonnx_dir / f"action_out_proj_{target_device}_{quant_type}.onnx"
    convert_onnx_to_hmonnx(
        action_out_onnx,
        (action_out_input,),
        out_hmonnx_file=action_out_hmonnx,
        device_type=_to_hmonnx_device_type(target_device),
        quant_config=quant_config,
    )
    graphs.append(
        _simple_graph_meta("action_out_proj", component_dir, action_out_onnx, action_out_hmonnx, action_out_shape)
    )

    time_mlp = TimeMLPWrapper(policy.model.time_mlp_in, policy.model.time_mlp_out).float().cpu().eval()
    time_mlp_shape = (1, hidden_size)
    time_mlp_input = torch.randn(time_mlp_shape, dtype=torch.float32)
    time_mlp_onnx = export_onnx_and_simplify(
        time_mlp,
        time_mlp_input,
        onnx_dir / "pi05_time_mlp.onnx",
        onnx_dir / "pi05_time_mlp_simplified.onnx",
        ["time_emb"],
        ["time_mlp_out"],
        {"time_emb": list(time_mlp_shape)},
        opset_version=int(component_cfg.get("opset", 17)),
        verbose=bool(component_cfg.get("verbose", False)),
    )
    time_mlp_hmonnx = hmonnx_dir / f"time_mlp_{target_device}_{quant_type}.onnx"
    convert_onnx_to_hmonnx(
        time_mlp_onnx,
        (time_mlp_input,),
        out_hmonnx_file=time_mlp_hmonnx,
        device_type=_to_hmonnx_device_type(target_device),
        quant_config=quant_config,
    )
    graphs.append(_simple_graph_meta("time_mlp", component_dir, time_mlp_onnx, time_mlp_hmonnx, time_mlp_shape))

    return {
        "component_dir": str(component_dir),
        "quant_type": quant_type,
        "graphs": graphs,
    }


def export_pi05_llm_component(
    *,
    model_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    component_dir: Path,
    target_device: str,
    quant_type: str,
    component_name: str,
    kind: str,
) -> dict[str, Any]:
    from xhmodel_merak.xh_other_model.builder import MODELS
    from xhmodel_merak.xh_other_model.eval_model_type import EvalModelType
    from xhquant.api import ConfigDict, PrecisionMode, get_root_logger, ptq_quantize

    component_dir.mkdir(parents=True, exist_ok=True)
    logger = get_root_logger()
    device = torch.device(str(runtime_cfg["device"]))
    exec_device = torch.device(str(runtime_cfg["device"]))
    dtype = PI05_EXPORT_DTYPE
    cfg_name = f"{component_name}_{target_device}_{quant_type}"

    xh_model = MODELS.build(ConfigDict(dict(model_cfg)))
    tokenizer = xh_model.get_tokenizer(str(runtime_cfg["config_dir"]))
    if kind == "gemma":
        policy = xh_model.get_hf_model(model="pi0.5").to(device=device, dtype=dtype).eval()
        calibration_contexts = _build_pi05_calibration_contexts(
            policy=policy,
            tokenizer=tokenizer,
            prefix_sequence_length=int(runtime_cfg["prefix_sequence_length"]),
            selected_image_indices=runtime_cfg["selected_image_indices"],
            text_max_length=int(runtime_cfg["text_max_length"]),
        )
        xh_model.init_wrap_model(policy.model.paligemma_with_expert.paligemma.model.language_model)
    elif kind == "expert":
        policy = xh_model.get_hf_model().to(device=device, dtype=dtype).eval()
        _configure_pi05_action_horizon(policy, int(runtime_cfg["action_horizon"]))
        calibration_contexts = _build_pi05_calibration_contexts(
            policy=policy,
            tokenizer=tokenizer,
            prefix_sequence_length=int(runtime_cfg["prefix_sequence_length"]),
            include_prefix_kv=True,
            selected_image_indices=runtime_cfg["selected_image_indices"],
            text_max_length=int(runtime_cfg["text_max_length"]),
        )
        policy.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        xh_model.init_wrap_model(policy.model.paligemma_with_expert.gemma_expert)
    else:
        raise ValueError(f"Unsupported PI05 LLM kind: {kind}")

    prefill_onnx_dir = component_dir / "prefill_onnx"
    decode_onnx_dir = component_dir / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)
    config_dir = Path(str(runtime_cfg["config_dir"]))
    _copy_hf_config_files(config_dir, component_dir / "hf_config")

    input_ids = _build_prompt_input_ids(tokenizer, str(runtime_cfg["prompt"]), device)
    valid_prefix_length = int(calibration_contexts[0]["valid_prefix_length"])
    graph_input_ids = input_ids
    current_input_length = valid_prefix_length
    past_seq_length = 0
    if kind == "expert":
        graph_input_ids = _build_decode_input_ids(kind, input_ids, int(runtime_cfg["action_horizon"]))
        current_input_length = int(runtime_cfg["action_horizon"])
        past_seq_length = valid_prefix_length
    data_batch = {
        "input_ids": graph_input_ids.to(device),
        "past_seq_length": [past_seq_length],
        "current_input_length": [current_input_length],
    }

    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
    xh_model.to(device)
    xh_model.to(dtype)
    xh_model.interactive_mode = True
    xh_model.convert_to_fronted_graph(data_batch)
    _empty_cuda_cache()
    xh_model.convert_to_quant_graph(target_device)

    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    xh_model.to(dtype)
    xh_model.to(device)
    calib_data = _flatten_inputs(xh_model.prepare_inputs(data_batch))
    if kind == "gemma":
        calibration_batches = _build_pi05_gemma_calibration_inputs(
            contexts=calibration_contexts,
            template_inputs=calib_data,
            cache_length=int(xh_model.cache_length),
            device=device,
        )
    else:
        calibration_batches = _build_pi05_expert_calibration_inputs(
            policy=policy,
            contexts=calibration_contexts,
            template_inputs=calib_data,
            sequence_length=int(xh_model.input_sequence_length),
            cache_length=int(xh_model.cache_length),
            device=device,
        )
    ptq_quantize(xh_model.quanted_model, calibration_batches, PrecisionMode.ALIGNED, [exec_device])

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(torch.float16)

    token_embedding_file = None
    if kind == "gemma":
        token_embedding_file = component_dir / "token_embedding.pt"
        torch.save(xh_model.token_embedding.state_dict(), str(token_embedding_file))

    prefill_input_sequence_length = int(xh_model.input_sequence_length)
    hidden_size = int(xh_model.past_key_caches[0].shape[-1]) * int(xh_model.past_key_caches[0].shape[1])
    if hasattr(xh_model, "token_embedding") and xh_model.token_embedding is not None:
        hidden_size = int(xh_model.token_embedding.embedding_dim)

    xh_model = xh_model.to("cpu")
    data_batch["input_ids"] = data_batch["input_ids"].to("cpu")
    prefill_onnx_file = _xhmodel_export_hmonnx(
        xh_model,
        data_batch,
        prefill_onnx_dir,
        f"{cfg_name}_prefill",
        logger,
    )
    xh_model.release_exported_model()

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(torch.float16)
    data_batch["input_ids"] = data_batch["input_ids"].to(device)
    decode_input_sequence_length = _decode_input_sequence_length(kind, prefill_input_sequence_length)
    xh_model.set_input_sequence_length(decode_input_sequence_length)
    past_seq_len = valid_prefix_length
    decode_input_ids = _build_decode_input_ids(kind, input_ids, decode_input_sequence_length)
    decode_batch = {
        "input_ids": decode_input_ids.to(device),
        "past_seq_length": [past_seq_len],
        "current_input_length": [decode_input_sequence_length],
    }
    xh_model = xh_model.to("cpu")
    decode_batch["input_ids"] = decode_batch["input_ids"].to("cpu")
    decode_onnx_file = _xhmodel_export_hmonnx(
        xh_model,
        decode_batch,
        decode_onnx_dir,
        f"{cfg_name}_decode",
        logger,
    )
    xh_model.release_exported_model()

    kv_cache_shape = list(xh_model.past_key_caches[0].shape)
    wrap_cfg_meta = _jsonable(xh_model.wrap_cfg.to_dict())
    wrap_cfg_meta["input_sequence_length"] = prefill_input_sequence_length
    meta_info: dict[str, Any] = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "wrap_cfg": wrap_cfg_meta,
        "hf_model": str(runtime_cfg["hf_model_dir"]),
        "hf_config": "hf_config",
        "target_device": target_device,
        "quant_type": quant_type,
        "prefill_onnx_file": str(Path(prefill_onnx_file).relative_to(component_dir)),
        "decode_onnx_file": str(Path(decode_onnx_file).relative_to(component_dir)),
        "use_cache": True,
        "kv_cache_shape": kv_cache_shape,
        "num_hidden_layers": len(xh_model.past_key_caches),
        "hidden_size": hidden_size,
        "prefill_input_sequence_length": prefill_input_sequence_length,
        "decode_input_sequence_length": decode_input_sequence_length,
        "prefill_input_token_length": past_seq_len,
        "prefill_attention_mask_shape": [1, 1, 1, int(xh_model.cache_length)],
        "decode_attention_mask_shape": [1, 1, 1, int(xh_model.cache_length)],
        "selected_image_indices": list(runtime_cfg["selected_image_indices"]),
        "text_max_length": int(runtime_cfg["text_max_length"]),
        "prefix_sequence_length": int(runtime_cfg["prefix_sequence_length"]),
        "valid_prefix_length": valid_prefix_length,
        "action_horizon": int(runtime_cfg["action_horizon"]),
        "cache_length": int(runtime_cfg["cache_length"]),
    }
    if token_embedding_file is not None:
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(component_dir))

    meta_file = component_dir / "export_meta_info.json"
    meta_file.write_text(json.dumps(_jsonable(meta_info), indent=4), encoding="utf-8")
    del policy
    del xh_model
    _empty_cuda_cache()
    return {
        "component_dir": str(component_dir),
        "meta_file": str(meta_file),
        "quant_type": quant_type,
    }


def _xhmodel_export_hmonnx(
    xh_model: Any,
    data_batch: dict[str, Any],
    output_dir: Path,
    prefix: str,
    logger: Any,
) -> str:
    logger.info("Start exporting PI05 HMONNX graph...")
    xh_model.to("cpu")
    _empty_cuda_cache()
    xh_model.convert_to_export_graph(data_batch)
    _empty_cuda_cache()
    from xhmodel_merak.xh_other_model.eval_model_type import EvalModelType

    xh_model.change_eval_type(EvalModelType.EXPORTED)
    xh_model.to("cpu")
    _empty_cuda_cache()
    return xh_model.to_export_onnx(data_batch, str(output_dir), prefix)[0]


def _load_dump_export_cfg(
    export_result: ExportResult,
    work_dir: Path,
    meta: Mapping[str, Any],
    fallback_config: WorkflowConfig,
) -> dict[str, Any]:
    config_file = Path(export_result.config_file) if export_result.config_file else None
    if config_file is not None and not config_file.is_file() and not config_file.is_absolute():
        config_file = work_dir / config_file
    if config_file is None or not config_file.is_file():
        config_file = work_dir / str(meta["config"])

    workflow_config = WorkflowConfig.from_file(str(config_file)) if config_file.is_file() else fallback_config
    export_cfg = workflow_config.build_export_dict()
    export_cfg["model"]["hf_model"] = meta["hf_model"]
    if "expert_model" in export_cfg and isinstance(export_cfg["expert_model"], dict):
        export_cfg["expert_model"]["hf_model"] = meta["hf_model"]
    return export_cfg


def _build_pi05_llm_golden_inputs(
    *,
    model_cfg: Mapping[str, Any],
    config_dir: str,
    prompt: str,
    kind: str,
    device: str,
    comp_meta: Mapping[str, Any],
    selected_image_indices: Sequence[int],
    text_max_length: int,
) -> tuple[list[Any], list[Any]]:
    from xhmodel_merak.xh_other_model.builder import MODELS
    from xhquant.api import ConfigDict

    xh_model = MODELS.build(ConfigDict(copy.deepcopy(dict(model_cfg))))
    tokenizer = xh_model.get_tokenizer(config_dir)
    torch_device = torch.device(device)
    input_ids = _build_prompt_input_ids(tokenizer, prompt, torch_device)

    if kind == "gemma":
        policy = xh_model.get_hf_model(model="pi0.5").to(device=torch_device, dtype=PI05_EXPORT_DTYPE).eval()
        contexts = _build_pi05_calibration_contexts(
            policy=policy,
            tokenizer=tokenizer,
            prefix_sequence_length=int(comp_meta["prefill_input_sequence_length"]),
            selected_image_indices=selected_image_indices,
            text_max_length=text_max_length,
        )
        xh_model.init_wrap_model(policy.model.paligemma_with_expert.paligemma.model.language_model)
    elif kind == "expert":
        policy = xh_model.get_hf_model().to(device=torch_device, dtype=PI05_EXPORT_DTYPE).eval()
        _configure_pi05_action_horizon(policy, int(comp_meta["prefill_input_sequence_length"]))
        prefix_sequence_length = len(selected_image_indices) * 256 + text_max_length
        contexts = _build_pi05_calibration_contexts(
            policy=policy,
            tokenizer=tokenizer,
            prefix_sequence_length=prefix_sequence_length,
            include_prefix_kv=True,
            selected_image_indices=selected_image_indices,
            text_max_length=text_max_length,
        )
        policy.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        xh_model.init_wrap_model(policy.model.paligemma_with_expert.gemma_expert)
    else:
        raise ValueError(f"Unsupported PI05 LLM kind: {kind}")

    xh_model.to(torch_device)
    xh_model.to(torch.float16)

    prefill_length = int(comp_meta["prefill_input_sequence_length"])
    xh_model.set_input_sequence_length(prefill_length)
    valid_prefix_length = int(contexts[0]["valid_prefix_length"])
    template_input_ids = input_ids
    current_input_length = valid_prefix_length
    past_seq_length = 0
    if kind == "expert":
        template_input_ids = _build_decode_input_ids(kind, input_ids, prefill_length)
        current_input_length = prefill_length
        past_seq_length = valid_prefix_length
    template_batch = {
        "input_ids": template_input_ids.to(torch_device),
        "past_seq_length": [past_seq_length],
        "current_input_length": [current_input_length],
    }
    template_inputs = _flatten_inputs(xh_model.prepare_inputs_for_graph(template_batch))

    if kind == "gemma":
        prefill_inputs = _build_pi05_gemma_calibration_inputs(
            contexts=contexts[:1],
            template_inputs=template_inputs,
            cache_length=int(xh_model.cache_length),
            device=torch_device,
        )[0]
    else:
        prefill_inputs = _build_pi05_expert_calibration_inputs(
            policy=policy,
            contexts=contexts[:1],
            template_inputs=template_inputs,
            sequence_length=prefill_length,
            cache_length=int(xh_model.cache_length),
            device=torch_device,
        )[0]

    decode_length = int(comp_meta["decode_input_sequence_length"])
    if kind == "expert" and decode_length == prefill_length:
        decode_inputs = prefill_inputs
    else:
        xh_model.set_input_sequence_length(decode_length)
        decode_input_ids = _build_decode_input_ids(kind, input_ids, decode_length)
        decode_batch = {
            "input_ids": decode_input_ids.to(torch_device),
            "past_seq_length": [valid_prefix_length],
            "current_input_length": [decode_length],
        }
        decode_inputs = _flatten_inputs(xh_model.prepare_inputs_for_graph(decode_batch))

    del policy
    del xh_model
    _empty_cuda_cache()
    return prefill_inputs, decode_inputs


def _build_pi05_calibration_contexts(
    *,
    policy: Any,
    tokenizer: Any,
    prefix_sequence_length: int,
    include_prefix_kv: bool = False,
    selected_image_indices: Sequence[int],
    text_max_length: int,
) -> list[dict[str, Any]]:
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    model_parameter = next(policy.parameters())
    model_device = model_parameter.device
    if model_parameter.dtype != PI05_EXPORT_DTYPE:
        raise TypeError(f"PI05 calibration policy must use {PI05_EXPORT_DTYPE}, got {model_parameter.dtype}")
    image_height, image_width = policy.config.image_resolution
    image_count = len(policy.config.image_features)
    token_length = int(policy.config.tokenizer_max_length)
    selected_indices = [int(image_index) for image_index in selected_image_indices]
    if not selected_indices or len(set(selected_indices)) != len(selected_indices):
        raise ValueError("PI05 selected_image_indices must be non-empty and unique")
    if any(image_index < 0 or image_index >= image_count for image_index in selected_indices):
        raise ValueError(f"PI05 selected_image_indices {selected_indices} exceed checkpoint image count {image_count}")
    if token_length != text_max_length:
        raise ValueError(f"PI05 checkpoint tokenizer_max_length is {token_length}, expected {text_max_length}")
    expected_prefix_length = len(selected_indices) * 256 + text_max_length
    if prefix_sequence_length != expected_prefix_length:
        raise ValueError(
            f"PI05 compact prefix length is {prefix_sequence_length}, expected {expected_prefix_length} "
            f"from {len(selected_indices)} image(s) and {text_max_length} language tokens"
        )
    contexts = []

    prompts = (
        ("pick up the green cube", (78, 103, 228, 136, 129, 42, 165, 0)),
        ("move the blue cup to the tray", (42, 170, 93, 199, 64, 155, 120, 255)),
        ("place the object on the table", (200, 90, 140, 33, 220, 100, 74, 128)),
    )
    for index, (task, state_values) in enumerate(prompts):
        generator = torch.Generator(device=model_device).manual_seed(42 + index)
        images = [
            torch.rand(
                (1, 3, image_height, image_width),
                generator=generator,
                dtype=PI05_EXPORT_DTYPE,
                device=model_device,
            )
            * 2.0
            - 1.0
            for _ in range(image_count)
        ]
        image_masks = [
            torch.full((1,), image_index in selected_indices, dtype=torch.bool, device=model_device)
            for image_index in range(image_count)
        ]
        state = " ".join(map(str, state_values))
        prompt = f"Task: {task}, State: {state};\nAction: "
        tokenized = tokenizer(
            [prompt],
            max_length=token_length,
            truncation=True,
            padding="max_length",
            padding_side="right",
            return_tensors="pt",
        )
        tokens = tokenized.input_ids.to(model_device)
        token_masks = tokenized.attention_mask.to(device=model_device, dtype=torch.bool)

        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
                images,
                image_masks,
                tokens,
                token_masks,
            )
        physical_prefix_length = int(prefix_embs.shape[1])
        image_token_count = physical_prefix_length - token_length
        if image_token_count != image_count * 256:
            raise ValueError(
                f"PI05 calibration produced {image_token_count} image tokens for {image_count} image(s); "
                "compact-prefix export requires 256 tokens per image"
            )
        compact_slices = [slice(image_index * 256, (image_index + 1) * 256) for image_index in selected_indices]
        compact_slices.append(slice(image_token_count, physical_prefix_length))
        prefix_embs = torch.cat([prefix_embs[:, item] for item in compact_slices], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks[:, item] for item in compact_slices], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks[:, item] for item in compact_slices], dim=1)

        if prefix_embs.shape[1] != prefix_sequence_length:
            raise ValueError(
                f"PI05 calibration prefix produced {prefix_embs.shape[1]} tokens from "
                f"{image_count} image(s) and {token_length} language tokens, expected {prefix_sequence_length}"
            )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_attention_mask = policy.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        valid_prefix_length = int(prefix_pad_masks.sum().item())
        if not bool(prefix_pad_masks[:, :valid_prefix_length].all()) or bool(
            prefix_pad_masks[:, valid_prefix_length:].any()
        ):
            raise ValueError("PI05 compact prefix must place every valid token before language padding")
        context = {
            "prefix_embs": prefix_embs.detach().cpu(),
            "prefix_pad_masks": prefix_pad_masks.detach().cpu(),
            "prefix_attention_mask": prefix_attention_mask.detach().cpu(),
            "prefix_position_ids": prefix_position_ids.detach().cpu(),
            "valid_prefix_length": valid_prefix_length,
        }
        if include_prefix_kv:
            with torch.no_grad():
                output = policy.model.paligemma_with_expert.paligemma.model.language_model(
                    inputs_embeds=prefix_embs,
                    attention_mask=prefix_attention_mask.to(dtype=prefix_embs.dtype),
                    position_ids=prefix_position_ids,
                    use_cache=True,
                )
            context["prefix_key_values"] = [
                (
                    key[..., :valid_prefix_length, :].contiguous(),
                    value[..., :valid_prefix_length, :].contiguous(),
                )
                for key, value in _extract_pi05_prefix_key_values(output.past_key_values)
            ]
        contexts.append(context)
    return contexts


def _extract_pi05_prefix_key_values(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past_key_values, "layers"):
        layers = past_key_values.layers
        return [(layer.keys.detach().cpu(), layer.values.detach().cpu()) for layer in layers]
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return [
            (key.detach().cpu(), value.detach().cpu())
            for key, value in zip(past_key_values.key_cache, past_key_values.value_cache, strict=True)
        ]
    return [(key.detach().cpu(), value.detach().cpu()) for key, value in past_key_values]


def _build_pi05_gemma_calibration_inputs(
    *,
    contexts: Sequence[Mapping[str, Any]],
    template_inputs: Sequence[Any],
    cache_length: int,
    device: torch.device,
) -> list[list[Any]]:
    calibration_batches = []
    for context in contexts:
        prefix_embs = context["prefix_embs"]
        valid_prefix_length = int(context["valid_prefix_length"])
        inputs = list(template_inputs)
        inputs[0] = prefix_embs.to(device=device, dtype=torch.float16)
        inputs[1] = torch.zeros(1, dtype=torch.int32, device=device)
        inputs[2] = torch.tensor([valid_prefix_length], dtype=torch.int32, device=device)
        inputs[3] = _build_pi05_key_attention_mask(valid_prefix_length, cache_length, device)
        calibration_batches.append(inputs)
    return calibration_batches


def _build_pi05_expert_calibration_inputs(
    *,
    policy: Any,
    contexts: Sequence[Mapping[str, Any]],
    template_inputs: Sequence[Any],
    sequence_length: int,
    cache_length: int,
    device: torch.device,
) -> list[list[Any]]:
    from xhquant.core import CacheTensor

    action_dim = int(policy.config.max_action_dim)
    action_parameter = next(policy.model.action_in_proj.parameters())
    model_device = action_parameter.device
    if action_parameter.dtype != PI05_EXPORT_DTYPE:
        raise TypeError(f"PI05 Expert calibration policy must use {PI05_EXPORT_DTYPE}, got {action_parameter.dtype}")
    calibration_batches = []

    timestep_values = (1.0, 0.5, 0.1)[: len(contexts)]
    for index, (context, timestep_value) in enumerate(zip(contexts, timestep_values, strict=True)):
        generator = torch.Generator(device=model_device).manual_seed(42 + index)
        noisy_actions = torch.randn(
            (1, sequence_length, action_dim),
            generator=generator,
            dtype=PI05_EXPORT_DTYPE,
            device=model_device,
        )
        timestep = torch.full((1,), timestep_value, dtype=PI05_EXPORT_DTYPE, device=model_device)
        with torch.no_grad():
            suffix_embs, suffix_pad_masks, suffix_att_masks, cond = policy.model.embed_suffix(noisy_actions, timestep)

        if suffix_embs.shape[1] != sequence_length:
            raise ValueError(
                f"PI05 expert calibration produced {suffix_embs.shape[1]} suffix tokens, expected {sequence_length}"
            )

        del suffix_pad_masks, suffix_att_masks
        valid_prefix_length = int(context["valid_prefix_length"])
        attention_mask = _build_pi05_key_attention_mask(
            valid_prefix_length + sequence_length,
            cache_length,
            device,
        )

        inputs = list(template_inputs)
        inputs[0] = suffix_embs.to(device=device, dtype=PI05_EXPORT_DTYPE)
        inputs[1] = torch.tensor([valid_prefix_length], dtype=torch.int32, device=device)
        inputs[2] = torch.tensor([sequence_length], dtype=torch.int32, device=device)
        inputs[3] = cond.to(device=device, dtype=PI05_EXPORT_DTYPE)
        inputs[4] = attention_mask
        prefix_key_values = context.get("prefix_key_values")
        if prefix_key_values is not None:
            cache_start = 5
            value_start = cache_start + len(prefix_key_values)
            for layer_index, (key, value) in enumerate(prefix_key_values):
                if key.shape[-2] != valid_prefix_length or value.shape[-2] != valid_prefix_length:
                    raise ValueError("PI05 compact prefix KV length does not match valid prefix length")
                if valid_prefix_length > cache_length:
                    raise ValueError("PI05 prefix KV length exceeds expert cache length")
                key_cache = torch.zeros(
                    (*key.shape[:-2], cache_length, key.shape[-1]),
                    dtype=torch.float16,
                    device=device,
                )
                value_cache = torch.zeros_like(key_cache)
                key_cache[..., :valid_prefix_length, :] = key.to(device=device, dtype=torch.float16)
                value_cache[..., :valid_prefix_length, :] = value.to(device=device, dtype=torch.float16)
                inputs[cache_start + layer_index] = CacheTensor(key_cache)
                inputs[value_start + layer_index] = CacheTensor(value_cache)
        calibration_batches.append(inputs)

    return calibration_batches


def _configure_pi05_action_horizon(policy: Any, action_horizon: int) -> None:
    policy.config.chunk_size = action_horizon
    policy.config.n_action_steps = action_horizon
    policy.model.config.chunk_size = action_horizon


def _build_pi05_key_attention_mask(
    valid_length: int,
    cache_length: int,
    device: torch.device,
) -> torch.Tensor:
    if valid_length <= 0 or valid_length > cache_length:
        raise ValueError(f"PI05 valid length {valid_length} exceeds cache capacity {cache_length}")
    mask_value = torch.finfo(torch.float16).min
    attention_mask = torch.full((1, 1, 1, cache_length), mask_value, dtype=torch.float16, device=device)
    attention_mask[..., :valid_length] = 0
    return attention_mask


def _prepare_hmonnx_inputs_for_runtime(
    inputs: Sequence[Any],
    hmonnx_file: Path,
    device: str,
    cache_tensor_cls: type,
) -> list[Any]:
    expected_dtypes = _hmonnx_input_dtypes(hmonnx_file)
    runtime_inputs: list[Any] = []
    for idx, value in enumerate(inputs):
        expected_dtype = expected_dtypes[idx] if idx < len(expected_dtypes) else None
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device)
            is_cache = False
        elif hasattr(value, "data") and isinstance(value.data, torch.Tensor):
            tensor = value.data.to(device=device)
            is_cache = True
        else:
            raise TypeError(f"Unsupported PI05 HMONNX input type at index {idx}: {type(value)}")
        if expected_dtype is not None:
            tensor = tensor.to(dtype=expected_dtype)
        if is_cache:
            runtime_inputs.append(cache_tensor_cls(tensor))
        else:
            runtime_inputs.append(tensor)
    return runtime_inputs


def _hmonnx_input_dtypes(hmonnx_file: Path) -> list[torch.dtype | None]:
    import onnx

    dtype_map = {
        1: torch.float32,
        6: torch.int32,
        7: torch.int64,
        9: torch.bool,
        10: torch.float16,
        16: torch.bfloat16,
    }
    model = onnx.load(hmonnx_file, load_external_data=False)
    return [dtype_map.get(input_value.type.tensor_type.elem_type) for input_value in model.graph.input]


def _run_hmonnx_golden(hmonnx_file: Path, golden_dir: Path, device: str, inputs: Sequence[Any]) -> None:
    import shutil

    from xhquant.api import HMONNXGoldenInference

    if golden_dir.exists():
        shutil.rmtree(golden_dir)
    golden_dir.mkdir(parents=True, exist_ok=True)
    session = HMONNXGoldenInference(str(hmonnx_file))
    session.to(device)
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    session.step = 0
    session(*inputs)


def _component_meta(work_dir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    component_dir = Path(result["component_dir"])
    meta: dict[str, Any] = {
        "component_dir": str(component_dir.relative_to(work_dir)),
        "quant_type": result["quant_type"],
    }
    if "meta_file" in result:
        meta["meta_file"] = str(Path(result["meta_file"]).relative_to(component_dir))
    for key in ("onnx_file", "simplified_onnx_file", "hmonnx_file", "input_shape", "graphs"):
        if key in result:
            meta[key] = _jsonable(result[key])
    return meta


def _simple_graph_meta(
    name: str,
    component_dir: Path,
    onnx_file: Path,
    hmonnx_file: Path,
    input_shape: tuple[int, ...],
):
    return {
        "name": name,
        "onnx_file": str(onnx_file.relative_to(component_dir)),
        "hmonnx_file": str(hmonnx_file.relative_to(component_dir)),
        "input_shape": list(input_shape),
    }


def _ensure_quant_config(model_cfg: dict[str, Any], target_device: str, quant_type: str) -> None:
    from xhquant.api import ConfigDict

    model_cfg["quant_config"] = ConfigDict(_build_quant_config(target_device, quant_type))


def _build_quant_config(target_device: str, quant_type: str):
    from xhquant.api import QuantScheme, create_quant_config

    quant_scheme = QuantScheme(target_device=_to_xh_device_type(target_device), quant_type=quant_type)
    return create_quant_config(quant_scheme)


def _to_xh_device_type(target_device: str):
    from xhquant.api import DeviceType

    if target_device != "XH2a":
        raise ValueError(f"PI05 workflow currently supports target_device='XH2a', got {target_device!r}")
    return DeviceType.XH2a


def _to_hmonnx_device_type(target_device: str) -> str:
    if target_device != "XH2a":
        raise ValueError(f"PI05 workflow currently supports target_device='XH2a', got {target_device!r}")
    return "XH2A"


def _select_torch_device(device: str) -> str:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return str(device)
    return "cpu"


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _copy_hf_config_files(config_dir: Path, dst_dir: Path) -> None:
    if not config_dir.is_dir():
        raise FileNotFoundError(f"PI05 tokenizer config_dir not found: {config_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    ):
        src = config_dir / filename
        if src.exists():
            shutil.copyfile(src, dst_dir / filename)


def _build_prompt_input_ids(tokenizer: Any, prompt: str, device: torch.device) -> torch.Tensor:
    text = f"""
    你是一个专业的AI助手，回答简洁、准确，使用中文。
    <start_of_turn>user
    {prompt}<end_of_turn>
    <start_of_turn>assistant
    """.strip()
    return tokenizer([text], return_tensors="pt").to(device).input_ids


def _decode_input_sequence_length(kind: str, prefill_input_sequence_length: int) -> int:
    if kind == "gemma":
        return 1
    return prefill_input_sequence_length


def _build_decode_input_ids(kind: str, input_ids: torch.Tensor, decode_input_sequence_length: int) -> torch.Tensor:
    if kind == "gemma":
        return input_ids[:, :1]
    return torch.full(
        (int(input_ids.shape[0]), decode_input_sequence_length),
        2,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )


def _flatten_inputs(inputs: Any) -> list[Any]:
    flattened = []
    for arg in inputs:
        if isinstance(arg, (list, tuple)):
            flattened.extend(arg)
        else:
            flattened.append(arg)
    return flattened


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value
