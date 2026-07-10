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

        runtime_cfg = {
            "hf_model_dir": export_model_dir,
            "config_dir": str(export_cfg.get("config_dir", "")),
            "device": _select_torch_device(device),
            "target_device": target_device,
            "variant": str(export_cfg.get("variant", workflow_config.name)),
            "prompt": str(export_cfg.get("prompt", "请解释Gemma模型的核心优势是什么？")),
        }

        meta: dict[str, Any] = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": str(Path(config_file).relative_to(work_dir)),
            "hf_model": export_model_dir,
            "target_device": target_device,
            "variant": runtime_cfg["variant"],
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
                config_dir=str(export_cfg.get("config_dir", "")),
                prompt=str(export_cfg.get("prompt", "请解释Gemma模型的核心优势是什么？")),
                kind=kind,
                device=torch_device,
                comp_meta=comp_meta,
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

    sequence_length = int(component_cfg.get("sequence_length", 50))
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
    graphs.append(_simple_graph_meta("action_in_proj", component_dir, action_in_onnx, action_in_hmonnx, action_in_shape))

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
    graphs.append(_simple_graph_meta("action_out_proj", component_dir, action_out_onnx, action_out_hmonnx, action_out_shape))

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
    from xhquant.api import ConfigDict, PrecisionMode, get_root_logger, ptq_quantize

    from xhmodel_merak.xh_other_model.builder import MODELS
    from xhmodel_merak.xh_other_model.eval_model_type import EvalModelType

    component_dir.mkdir(parents=True, exist_ok=True)
    logger = get_root_logger()
    device = torch.device(str(runtime_cfg["device"]))
    exec_device = torch.device(str(runtime_cfg["device"]))
    dtype = torch.float16 if kind == "gemma" else torch.float32
    cfg_name = f"{component_name}_{target_device}_{quant_type}"

    xh_model = MODELS.build(ConfigDict(dict(model_cfg)))
    if kind == "gemma":
        policy = xh_model.get_hf_model(model="pi0.5")
        xh_model.init_wrap_model(policy.model.paligemma_with_expert.paligemma.model.language_model)
    elif kind == "expert":
        policy = xh_model.get_hf_model()
        policy.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        xh_model.init_wrap_model(policy.model.paligemma_with_expert.gemma_expert)
    else:
        raise ValueError(f"Unsupported PI05 LLM kind: {kind}")

    tokenizer = xh_model.get_tokenizer(str(runtime_cfg["config_dir"]))
    prefill_onnx_dir = component_dir / "prefill_onnx"
    decode_onnx_dir = component_dir / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)
    config_dir = Path(str(runtime_cfg["config_dir"]))
    _copy_hf_config_files(config_dir, component_dir / "hf_config")

    input_ids = _build_prompt_input_ids(tokenizer, str(runtime_cfg["prompt"]), device)
    data_batch = {
        "input_ids": input_ids.to(device),
        "past_seq_length": [0],
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
    ptq_quantize(xh_model.quanted_model, [calib_data], PrecisionMode.ALIGNED, [exec_device])

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
    past_seq_len = int(input_ids.shape[-1])
    decode_input_ids = _build_decode_input_ids(kind, input_ids, decode_input_sequence_length)
    decode_batch = {
        "input_ids": decode_input_ids.to(device),
        "past_seq_length": [past_seq_len],
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
    meta_info: dict[str, Any] = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "wrap_cfg": _jsonable(xh_model.wrap_cfg.to_dict()),
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
        "prefill_attention_mask_shape": [1, 1, prefill_input_sequence_length, 1024],
        "decode_attention_mask_shape": [1, 1, decode_input_sequence_length, 1024],
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


def _xhmodel_export_hmonnx(xh_model: Any, data_batch: dict[str, Any], output_dir: Path, prefix: str, logger: Any) -> str:
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
) -> tuple[list[Any], list[Any]]:
    from xhquant.api import ConfigDict

    from xhmodel_merak.xh_other_model.builder import MODELS

    xh_model = MODELS.build(ConfigDict(copy.deepcopy(dict(model_cfg))))
    if kind == "gemma":
        policy = xh_model.get_hf_model(model="pi0.5")
        xh_model.init_wrap_model(policy.model.paligemma_with_expert.paligemma.model.language_model)
    elif kind == "expert":
        policy = xh_model.get_hf_model()
        policy.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        xh_model.init_wrap_model(policy.model.paligemma_with_expert.gemma_expert)
    else:
        raise ValueError(f"Unsupported PI05 LLM kind: {kind}")

    tokenizer = xh_model.get_tokenizer(config_dir)
    torch_device = torch.device(device)
    input_ids = _build_prompt_input_ids(tokenizer, prompt, torch_device)

    xh_model.to(torch_device)
    xh_model.to(torch.float16)

    prefill_length = int(comp_meta["prefill_input_sequence_length"])
    xh_model.set_input_sequence_length(prefill_length)
    prefill_batch = {
        "input_ids": input_ids.to(torch_device),
        "past_seq_length": [0],
    }
    prefill_inputs = _flatten_inputs(xh_model.prepare_inputs_for_graph(prefill_batch))

    decode_length = int(comp_meta["decode_input_sequence_length"])
    xh_model.set_input_sequence_length(decode_length)
    decode_input_ids = _build_decode_input_ids(kind, input_ids, decode_length)
    decode_batch = {
        "input_ids": decode_input_ids.to(torch_device),
        "past_seq_length": [int(input_ids.shape[-1])],
    }
    decode_inputs = _flatten_inputs(xh_model.prepare_inputs_for_graph(decode_batch))

    del policy
    del xh_model
    _empty_cuda_cache()
    return prefill_inputs, decode_inputs


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


def _simple_graph_meta(name: str, component_dir: Path, onnx_file: Path, hmonnx_file: Path, input_shape: tuple[int, ...]):
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
    pad_len = decode_input_sequence_length - int(input_ids.shape[-1])
    if pad_len < 0:
        raise ValueError(
            f"PI05 expert prompt length {input_ids.shape[-1]} exceeds decode input length {decode_input_sequence_length}"
        )
    padding = torch.full((1, pad_len), 2, dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat([input_ids, padding], dim=1)


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
