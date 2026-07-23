# Copyright 2025 HOUMO AI
#
# File: workflow.py
# Description:
#   CosyVoice3 export workflow (merak). Orchestrates quant/export/dump_golden
#   across all CosyVoice3 submodels:
#     - llm (Qwen2 0.5B, prefill + decode, PTQ)
#     - llm_decoder (lm-head projection)
#     - speech_tokenizer_v3 (mask-instrumented)
#     - campplus (speaker encoder frontend)
#     - flow_decoder (DiT decoder)
#     - hift (HiFT vocoder)
#     - spk_embed_affine_layer
#     - pre_lookahead_layer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult

from ._export_utils import (
    build_quant_config,
    convert_onnx_to_hmonnx,
    convert_speech_tokenizer_v3_onnx,
    fix_input_shape,
    hift_prepare_onnx,
    run_hmonnx_golden,
    simplify_model,
)


_DEFAULT_COMPONENTS = [
    "llm",
    "llm_decoder",
    "speech_tokenizer_v3",
    "campplus",
    "flow_decoder",
    "hift",
    "spk_embed_affine_layer",
    "pre_lookahead_layer",
]

# Default source-onnx filenames inside onnx_dir for each onnx-based component.
_DEFAULT_SOURCE_ONNX = {
    "llm_decoder": "llm_decoder.onnx",
    "speech_tokenizer_v3": "speech_tokenizer_v3.onnx",
    "campplus": "campplus.onnx",
    "flow_decoder": "flow_decoder_estimator_fp32.onnx",
    "hift": "hift.onnx",
    "spk_embed_affine_layer": "spk_embed_affine_layer.onnx",
    "pre_lookahead_layer": "pre_lookahead_layer.onnx",
}


def _select_torch_device(device: str) -> str:
    import torch

    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, torch.device):
        return str(device)
    return str(device)


def _empty_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _jsonable(obj: Any) -> Any:
    """Make nested objects json-serializable (Path/tensor/ConfigDict)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _normalize_components(components_cfg: Any) -> list[str]:
    if components_cfg is None:
        return list(_DEFAULT_COMPONENTS)
    if isinstance(components_cfg, Sequence) and not isinstance(components_cfg, (str, bytes)):
        return [str(item) for item in components_cfg]
    if isinstance(components_cfg, Mapping):
        return [str(name) for name, cfg in components_cfg.items() if cfg is not False and cfg is not None]
    raise TypeError("CosyVoice3 export.components must be a list or mapping")


def _resolve_onnx_dir(export_cfg: Mapping[str, Any], model_dir: str) -> Path:
    onnx_dir = export_cfg.get("onnx_dir")
    if onnx_dir:
        return Path(onnx_dir)
    # Prefer <model_dir>/onnx when present, otherwise fall back to <model_dir>
    # (the Fun-CosyVoice3 package ships onnx files directly in the root).
    sub = Path(model_dir) / "onnx"
    return sub if sub.exists() else Path(model_dir)


def _resolve_llm_checkpoint(export_cfg: Mapping[str, Any], model_dir: str) -> str:
    ckpt = export_cfg.get("llm_checkpoint")
    if ckpt:
        return str(ckpt)
    return str(Path(model_dir) / "llm.pt")


def _resolve_hf_model_dir(export_cfg: Mapping[str, Any], model_dir: str) -> str:
    model_cfg = export_cfg.get("model") or {}
    hf_model = model_cfg.get("hf_model")
    if hf_model:
        return str(hf_model)
    sub = Path(model_dir) / "CosyVoice-BlankEN"
    return str(sub if sub.exists() else Path(model_dir))


class CosyVoice3Workflow(BaseOtherModelWorkflow):
    """CosyVoice3 multi-submodel export workflow.

    The main export model is :class:`XHQwen2LegacyModel` (registered as
    ``XHCosyVoice3LLM``).  All other submodels are onnx-based and are exported
    by applying the transforms in :mod:`_export_utils` and converting to hmonnx.
    """

    SUPPORTED_COMPONENTS = set(_DEFAULT_COMPONENTS)

    def _validate_export_config(
        self,
        components: list[str],
        runtime_cfg: dict[str, Any],
        onnx_dir: Path,
        source_onnx_cfg: Mapping[str, Any],
    ) -> None:
        """Pre-flight checks before starting the export."""
        errors: list[str] = []
        onnx_dir = onnx_dir.resolve()
        llm_ckpt = Path(runtime_cfg.get("llm_checkpoint", "")).resolve()
        if "llm" in components and not llm_ckpt.is_file():
            errors.append(f"LLM checkpoint not found: {llm_ckpt}")
        hf_dir = Path(runtime_cfg.get("hf_model_dir", "")).resolve()
        if "llm" in components and not (hf_dir / "config.json").is_file():
            errors.append(f"HF model config.json not found in: {hf_dir}")
        model_dir = Path(runtime_cfg.get("model_dir", "")).resolve()
        if "llm" in components and not (model_dir / "flow.pt").is_file():
            errors.append(
                f"flow.pt not found in model_dir: {model_dir / 'flow.pt'} (required for input_embedding extraction)"
            )
        onnx_components = [c for c in components if c != "llm"]
        for name in onnx_components:
            src = source_onnx_cfg.get(name, _DEFAULT_SOURCE_ONNX.get(name))
            resolved = (onnx_dir / src).resolve() if src else None
            if resolved and not resolved.is_file():
                errors.append(f"Source ONNX for '{name}' not found: {resolved}")
            elif resolved:
                print(f"  [src] {name}: {resolved}")
        if "llm" in components:
            print(f"  [src] llm_checkpoint: {llm_ckpt}")
            print(f"  [src] hf_model_dir: {hf_dir}")
        if errors:
            raise FileNotFoundError("CosyVoice3 export pre-flight failed:\n" + "\n".join(f"  - {e}" for e in errors))

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise NotImplementedError(f"{type(self).__name__} does not support quantization")
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
        export_cfg = workflow_config.build_export_dict()
        target_device = str(export_cfg.get("target_device", "XH2a"))
        work_dir = Path(output_dir)
        existing_meta = work_dir / "export_meta_info.json"
        if existing_meta.exists():
            raise FileExistsError(
                f"Output directory already contains a previous export: {existing_meta}. "
                "Remove it manually or use --overwrite to force re-export."
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))

        components = _normalize_components(export_cfg.get("components"))
        unsupported = [name for name in components if name not in self.SUPPORTED_COMPONENTS]
        if unsupported:
            raise ValueError(f"Unsupported CosyVoice3 component(s): {unsupported}")

        torch_device = _select_torch_device(device)
        onnx_dir = _resolve_onnx_dir(export_cfg, export_model_dir)
        source_onnx_cfg = export_cfg.get("source_onnx") or {}
        quant_cfg = export_cfg.get("quant_types") or {}
        if not isinstance(quant_cfg, Mapping):
            raise TypeError("CosyVoice3 export.quant_types must be a mapping when provided")

        runtime_cfg = {
            "model_dir": export_model_dir,
            "hf_model_dir": _resolve_hf_model_dir(export_cfg, export_model_dir),
            "llm_checkpoint": _resolve_llm_checkpoint(export_cfg, export_model_dir),
            "onnx_dir": str(onnx_dir),
            "device": torch_device,
            "target_device": target_device,
            "config_file": str(config_file),
            "work_dir": str(work_dir),
            "model_name": str(export_cfg.get("model_name") or Path(export_model_dir).name),
        }

        self._validate_export_config(components, runtime_cfg, onnx_dir, source_onnx_cfg)

        meta: dict[str, Any] = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": str(Path(config_file).relative_to(work_dir)),
            "hf_model": export_model_dir,
            "target_device": target_device,
            "components": {},
        }

        if "llm" in components:
            llm_cfg = copy.deepcopy(export_cfg["model"])
            llm_cfg["hf_model"] = runtime_cfg["hf_model_dir"]
            quant_type = str(quant_cfg.get("llm", "w8a16_sefp"))
            component_dir = work_dir / "LLM"
            result = export_cosyvoice3_llm(
                model_cfg=llm_cfg,
                runtime_cfg=runtime_cfg,
                component_dir=component_dir,
                target_device=target_device,
                quant_type=quant_type,
                torch_device=torch_device,
                llm_checkpoint=runtime_cfg["llm_checkpoint"],
            )
            meta["components"]["llm"] = _component_meta(work_dir, result)

        # ---- onnx-based components: read pre-existing onnx → convert to hmonnx ----
        for name in ("llm_decoder", "spk_embed_affine_layer", "pre_lookahead_layer"):
            if name not in components:
                continue
            quant_type = str(quant_cfg.get(name, "w8a16h1_sefp"))
            src = source_onnx_cfg.get(name, _DEFAULT_SOURCE_ONNX[name])
            component_dir = work_dir / _component_dir_name(name)
            result = export_simple_onnx_component(
                name=name,
                source_onnx=str(onnx_dir / src),
                component_dir=component_dir,
                target_device=target_device,
                quant_type=quant_type,
                torch_device=torch_device,
            )
            meta["components"][name] = _component_meta(work_dir, result)

        # Back-fill llm_decoder_file into LLM meta_info.json (llm_decoder is exported after llm).
        # Store as relative path (relative to LLM component dir) for portability.
        if "llm" in meta["components"] and "llm_decoder" in meta["components"]:
            import os as _os

            _llm_comp_dir = work_dir / meta["components"]["llm"]["component_dir"]
            _dec = meta["components"]["llm_decoder"]
            if "meta_file" in _dec:
                _dec_meta = json.loads((work_dir / _dec["meta_file"]).read_text(encoding="utf-8"))
                _decoder_abs = work_dir / _dec["component_dir"] / _dec_meta["hmonnx"]
            else:
                _decoder_abs = work_dir / _dec["component_dir"] / _dec["hmonnx"]
            _decoder_file = _os.path.relpath(str(_decoder_abs), str(_llm_comp_dir))
            _meta_path = _llm_comp_dir / "meta_info.json"
            if _meta_path.exists():
                _meta = json.loads(_meta_path.read_text(encoding="utf-8"))
                _meta["llm_decoder_file"] = _decoder_file
                _meta_path.write_text(json.dumps(_jsonable(_meta), indent=4, ensure_ascii=False), encoding="utf-8")

        if "campplus" in components:
            quant_type = str(quant_cfg.get("campplus", "w8a16_sefp"))
            src = source_onnx_cfg.get("campplus", _DEFAULT_SOURCE_ONNX["campplus"])
            component_dir = work_dir / "Campplus"
            shapes = export_cfg.get("shapes") or {}
            fixed_dims = (shapes.get("campplus") if isinstance(shapes, Mapping) else None) or {
                "batch_size": 1,
                "sequence_length": 1000,
            }
            result = export_campplus(
                source_onnx=str(onnx_dir / src),
                component_dir=component_dir,
                target_device=target_device,
                quant_type=quant_type,
                torch_device=torch_device,
                fixed_dims=fixed_dims,
            )
            meta["components"]["campplus"] = _component_meta(work_dir, result)

        if "flow_decoder" in components:
            quant_type = str(quant_cfg.get("flow_decoder", "w8a16_sefp"))
            src = source_onnx_cfg.get("flow_decoder", _DEFAULT_SOURCE_ONNX["flow_decoder"])
            component_dir = work_dir / "FlowDecoder"
            shapes = export_cfg.get("shapes") or {}
            fixed_dims = (shapes.get("flow_decoder") if isinstance(shapes, Mapping) else None) or {"seq_len": 2048}
            result = export_flow_decoder(
                source_onnx=str(onnx_dir / src),
                component_dir=component_dir,
                target_device=target_device,
                quant_type=quant_type,
                torch_device=torch_device,
                fixed_dims=fixed_dims,
            )
            meta["components"]["flow_decoder"] = _component_meta(work_dir, result)

        if "hift" in components:
            quant_type = str(quant_cfg.get("hift", "w8a16_sefp"))
            src = source_onnx_cfg.get("hift", _DEFAULT_SOURCE_ONNX["hift"])
            component_dir = work_dir / "HiFT"
            result = export_hift(
                source_onnx=str(onnx_dir / src),
                component_dir=component_dir,
                target_device=target_device,
                quant_type=quant_type,
                torch_device=torch_device,
            )
            meta["components"]["hift"] = _component_meta(work_dir, result)

        if "speech_tokenizer_v3" in components:
            quant_type = str(quant_cfg.get("speech_tokenizer_v3", "w8a16_sefp"))
            src = source_onnx_cfg.get("speech_tokenizer_v3", _DEFAULT_SOURCE_ONNX["speech_tokenizer_v3"])
            component_dir = work_dir / "SpeechTokenizerV3"
            result = export_speech_tokenizer_v3(
                source_onnx=str(onnx_dir / src),
                component_dir=component_dir,
                target_device=target_device,
                quant_type=quant_type,
                torch_device=torch_device,
            )
            meta["components"]["speech_tokenizer_v3"] = _component_meta(work_dir, result)

        # Extract flow input_embedding (6561x80) from flow.pt for token2wav inference.
        flow_pt = Path(export_model_dir) / "flow.pt"
        if flow_pt.exists():
            import torch as _torch3

            _flow_state = _torch3.load(str(flow_pt), map_location="cpu", weights_only=True)
            if "input_embedding.weight" in _flow_state:
                _in_emb_file = work_dir / "input_embedding.pt"
                _torch3.save({"weight": _flow_state["input_embedding.weight"]}, str(_in_emb_file))
                meta["input_embedding_file"] = "input_embedding.pt"
            else:
                raise KeyError(f"flow.pt exists but missing required key 'input_embedding.weight': {flow_pt}")
        elif "llm" in components:
            raise FileNotFoundError(f"flow.pt is required for LLM inference but not found: {flow_pt}")

        # Fail fast: verify all inference-critical artifacts were produced.
        if "llm" in components:
            _missing: list[str] = []
            if "input_embedding_file" not in meta:
                _missing.append("input_embedding.pt (from flow.pt 'input_embedding.weight')")
            _llm_comp = meta["components"].get("llm", {})
            _llm_dir = work_dir / _llm_comp.get("component_dir", "LLM")
            if not (_llm_dir / "speech_embedding.pt").is_file():
                _missing.append(f"speech_embedding.pt (from llm.pt 'speech_embedding.weight') in {_llm_dir}")
            if _missing:
                raise FileNotFoundError(
                    "Export completed but inference-critical artifacts are missing:\n"
                    + "\n".join(f"  - {m}" for m in _missing)
                )

        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(json.dumps(_jsonable(meta), indent=4, ensure_ascii=False), encoding="utf-8")
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        import torch

        work_dir = Path(export_result.work_dir)
        meta = json.loads((work_dir / "export_meta_info.json").read_text(encoding="utf-8"))
        torch_device = _select_torch_device(device)
        golden_meta: dict[str, Any] = {
            "work_dir": str(work_dir),
            "device": torch_device,
            "input_messages": repr(input_messages),
            "components": {},
        }

        for name, comp in meta["components"].items():
            comp_dir = work_dir / comp["component_dir"]
            if "meta_file" in comp:
                comp_meta = json.loads((work_dir / comp["meta_file"]).read_text(encoding="utf-8"))
            else:
                comp_meta = comp

            if name == "llm":
                prefill_path = comp_dir / comp_meta["prefill_onnx_file"]
                decode_path = comp_dir / comp_meta["decode_onnx_file"]
                kv_shape = tuple(int(d) for d in comp_meta["kv_cache_shape"])
                layers = int(comp_meta["num_hidden_layers"])
                prefill_len = int(comp_meta["wrap_cfg"]["input_sequence_length"])
                from xhquant.core import CacheTensor

                prefill_inputs = _build_llm_hmonnx_inputs(
                    torch.randn((1, prefill_len, 896), dtype=torch.float16, device=torch_device),
                    past_seq_length=0,
                    current_input_length=prefill_len,
                    kv_cache_shape=kv_shape,
                    num_hidden_layers=layers,
                    torch_device=torch_device,
                    cache_tensor_cls=CacheTensor,
                )
                prefill_golden = prefill_path.parent / "golden"
                run_hmonnx_golden(str(prefill_path), str(prefill_golden), torch_device, prefill_inputs)

                decode_inputs = _build_llm_hmonnx_inputs(
                    torch.randn((1, 1, 896), dtype=torch.float16, device=torch_device),
                    past_seq_length=prefill_len,
                    current_input_length=1,
                    kv_cache_shape=kv_shape,
                    num_hidden_layers=layers,
                    torch_device=torch_device,
                    cache_tensor_cls=CacheTensor,
                )
                decode_golden = decode_path.parent / "golden"
                run_hmonnx_golden(str(decode_path), str(decode_golden), torch_device, decode_inputs)
                golden_meta["components"][name] = {
                    "prefill_golden_dir": str(prefill_golden.relative_to(work_dir)),
                    "decode_golden_dir": str(decode_golden.relative_to(work_dir)),
                }
                continue

            hmonnx_file = comp_dir / comp_meta["hmonnx"]
            golden_dir = hmonnx_file.parent / "golden"
            inputs = _build_component_golden_inputs(name, torch_device, comp_meta)
            run_hmonnx_golden(str(hmonnx_file), str(golden_dir), torch_device, inputs)
            golden_meta["components"][name] = {"golden_dir": str(golden_dir.relative_to(work_dir))}

        # Record golden paths in export_meta_info.json (no separate golden_meta_info.json
        # per MIGRATION_GUIDE.md: "不要新增 golden_meta_info.json 作为通用要求").
        for name, golden_info in golden_meta["components"].items():
            if name in meta["components"]:
                meta["components"][name]["golden"] = golden_info
        meta["golden_device"] = torch_device
        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(json.dumps(_jsonable(meta), indent=4, ensure_ascii=False), encoding="utf-8")
        return str(meta_file)


# ---------------------------------------------------------------------------
# component export functions
# ---------------------------------------------------------------------------


def _component_dir_name(name: str) -> str:
    return {
        "llm_decoder": "LLMDecoder",
        "spk_embed_affine_layer": "SpkEmbedAffineLayer",
        "pre_lookahead_layer": "PreLookaheadLayer",
    }.get(name, name)


def _component_meta(work_dir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    component_dir = Path(result["component_dir"])
    entry: dict[str, Any] = {
        "component_dir": str(component_dir.relative_to(work_dir)),
        "quant_type": result["quant_type"],
    }
    if "meta_file" in result:
        entry["meta_file"] = str(Path(result["meta_file"]).relative_to(work_dir))
    else:
        for k, v in result.items():
            if k not in ("component_dir", "quant_type", "meta_file"):
                entry[k] = _jsonable(v)
    return entry


def _write_component_meta(component_dir: Path, meta: dict[str, Any]) -> Path:
    component_dir.mkdir(parents=True, exist_ok=True)
    meta_file = component_dir / "meta.json"
    meta_file.write_text(json.dumps(_jsonable(meta), indent=4, ensure_ascii=False), encoding="utf-8")
    return meta_file


def _simple_dummy_inputs(name: str, torch_device: str) -> list:
    import torch

    if name == "llm_decoder":
        return [torch.randn(1, 896, device=torch_device)]
    if name == "spk_embed_affine_layer":
        return [torch.randn(1, 192, device=torch_device)]
    if name == "pre_lookahead_layer":
        return [torch.randn(1, 1024, 80, device=torch_device)]
    raise ValueError(f"unknown simple component: {name}")


def export_simple_onnx_component(
    *,
    name: str,
    source_onnx: str,
    component_dir: Path,
    target_device: str,
    quant_type: str,
    torch_device: str,
) -> dict[str, Any]:
    """Export a no-transform onnx component (llm_decoder / spk_embed / pre_lookahead)."""
    import shutil

    from ._export_utils import inspect_onnx

    component_dir.mkdir(parents=True, exist_ok=True)
    if not Path(source_onnx).exists():
        raise FileNotFoundError(f"source onnx for {name} not found: {source_onnx}")
    inputs = _simple_dummy_inputs(name, "cpu")
    onnx_dir = component_dir / "onnx"
    hmonnx_dir = component_dir / "hmonnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_file = onnx_dir / f"{name}.onnx"
    shutil.copyfile(source_onnx, str(onnx_file))
    inspect_onnx(str(onnx_file))
    hmonnx_file = hmonnx_dir / f"{name}_{target_device}_{quant_type}.onnx"
    convert_onnx_to_hmonnx(str(onnx_file), inputs, str(hmonnx_file), target_device, quant_type)
    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "name": name,
        "target_device": target_device,
        "quant_type": quant_type,
        "onnx": str(onnx_file.relative_to(component_dir)),
        "hmonnx": str(hmonnx_file.relative_to(component_dir)),
        "input_shapes": [list(t.shape) for t in inputs],
    }
    return {"component_dir": component_dir, "quant_type": quant_type, **meta}


def export_campplus(
    *,
    source_onnx: str,
    component_dir: Path,
    target_device: str,
    quant_type: str,
    torch_device: str,
    fixed_dims: dict[str, int],
) -> dict[str, Any]:
    import onnx
    import torch

    from ._export_utils import inspect_onnx

    component_dir.mkdir(parents=True, exist_ok=True)
    if not Path(source_onnx).exists():
        raise FileNotFoundError(f"campplus source onnx not found: {source_onnx}")
    model = onnx.load(source_onnx)
    inspect_onnx(source_onnx)
    model = fix_input_shape(model, fixed_dims)
    onnx_dir = component_dir / "onnx"
    hmonnx_dir = component_dir / "hmonnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_dir.mkdir(parents=True, exist_ok=True)
    simplified = simplify_model(model, str(onnx_dir / "campplus_simplify.onnx"))
    onnx_file = onnx_dir / "campplus.onnx"
    onnx.save(simplified, str(onnx_file))
    seq = fixed_dims.get("sequence_length", 1000)
    dummy_input = torch.randn(1, seq, 80)
    hmonnx_file = hmonnx_dir / f"campplus_{target_device}_{quant_type}.onnx"
    convert_onnx_to_hmonnx(str(onnx_file), [dummy_input], str(hmonnx_file), target_device, quant_type)
    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "target_device": target_device,
        "quant_type": quant_type,
        "onnx": str(onnx_file.relative_to(component_dir)),
        "hmonnx": str(hmonnx_file.relative_to(component_dir)),
        "fixed_dims": fixed_dims,
    }
    return {"component_dir": component_dir, "quant_type": quant_type, **meta}


def export_flow_decoder(
    *,
    source_onnx: str,
    component_dir: Path,
    target_device: str,
    quant_type: str,
    torch_device: str,
    fixed_dims: dict[str, int],
) -> dict[str, Any]:
    import onnx
    import torch

    from ._export_utils import inspect_onnx

    component_dir.mkdir(parents=True, exist_ok=True)
    if not Path(source_onnx).exists():
        raise FileNotFoundError(f"flow_decoder source onnx not found: {source_onnx}")
    model = onnx.load(source_onnx)
    inspect_onnx(source_onnx)
    model = fix_input_shape(model, fixed_dims)
    onnx_dir = component_dir / "onnx"
    hmonnx_dir = component_dir / "hmonnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_dir.mkdir(parents=True, exist_ok=True)
    simplified = simplify_model(model)
    onnx_file = onnx_dir / "flow_decoder.onnx"
    onnx.save(simplified, str(onnx_file))

    seq_len = fixed_dims.get("seq_len", 2048)
    out_channels = 80
    batch = 2
    x = torch.rand((batch, out_channels, seq_len), dtype=torch.float32)
    mask = torch.ones((batch, 1, seq_len), dtype=torch.float32)
    mu = torch.rand((batch, out_channels, seq_len), dtype=torch.float32)
    t = torch.rand((batch,), dtype=torch.float32)
    spks = torch.rand((batch, out_channels), dtype=torch.float32)
    cond = torch.rand((batch, out_channels, seq_len), dtype=torch.float32)
    dummy_inputs = (x, mask, mu, t, spks, cond)
    hmonnx_file = hmonnx_dir / f"flow_decoder_{target_device}_{quant_type}.onnx"
    convert_onnx_to_hmonnx(str(onnx_file), dummy_inputs, str(hmonnx_file), target_device, quant_type)
    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "target_device": target_device,
        "quant_type": quant_type,
        "onnx": str(onnx_file.relative_to(component_dir)),
        "hmonnx": str(hmonnx_file.relative_to(component_dir)),
        "fixed_dims": fixed_dims,
        "batch_size": batch,
        "seq_len": seq_len,
        "out_channels": out_channels,
    }
    return {"component_dir": component_dir, "quant_type": quant_type, **meta}


def export_hift(
    *,
    source_onnx: str,
    component_dir: Path,
    target_device: str,
    quant_type: str,
    torch_device: str,
) -> dict[str, Any]:
    import torch

    component_dir.mkdir(parents=True, exist_ok=True)
    if not Path(source_onnx).exists():
        raise FileNotFoundError(f"hift source onnx not found: {source_onnx}")
    onnx_dir = component_dir / "onnx"
    hmonnx_dir = component_dir / "hmonnx"
    final_onnx = hift_prepare_onnx(
        source_onnx,
        fixed_dims={"batch_size": 1, "seq_len": 1024},
        intermediate_dir=onnx_dir,
    )
    dummy_input = torch.randn(1, 80, 1024)
    hmonnx_file = hmonnx_dir / f"hift_{target_device}_{quant_type}.onnx"
    convert_onnx_to_hmonnx(
        final_onnx,
        [dummy_input],
        str(hmonnx_file),
        target_device,
        quant_type,
    )
    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "target_device": target_device,
        "quant_type": quant_type,
        "onnx": str(Path(final_onnx).relative_to(component_dir)),
        "hmonnx": str(hmonnx_file.relative_to(component_dir)),
    }
    return {"component_dir": component_dir, "quant_type": quant_type, **meta}


def _onnx_has_mask_inputs(onnx_path: str) -> bool:
    """Check if the source ONNX already has 'mask' and 'mask1' graph inputs."""
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    input_names = {inp.name for inp in model.graph.input}
    return "mask" in input_names and "mask1" in input_names


def export_speech_tokenizer_v3(
    *,
    source_onnx: str,
    component_dir: Path,
    target_device: str,
    quant_type: str,
    torch_device: str,
) -> dict[str, Any]:
    import shutil

    import torch

    component_dir.mkdir(parents=True, exist_ok=True)
    if not Path(source_onnx).exists():
        raise FileNotFoundError(f"speech_tokenizer_v3 source onnx not found: {source_onnx}")
    onnx_dir = component_dir / "onnx"
    hmonnx_dir = component_dir / "hmonnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_dir.mkdir(parents=True, exist_ok=True)
    final_onnx = str(onnx_dir / "speech_tokenizer_v3_3000_3.onnx")
    if _onnx_has_mask_inputs(source_onnx):
        print(f"Source ONNX already has mask/mask1 inputs, copy as-is: {source_onnx}")
        shutil.copyfile(source_onnx, final_onnx)
    else:
        print(f"Source ONNX lacks mask/mask1 inputs, running conversion: {source_onnx}")
        convert_speech_tokenizer_v3_onnx(source_onnx, final_onnx)

    dummy_input = torch.randn(1, 128, 3000)
    mask = torch.randn(1, 20, 750, 750)
    mask1 = torch.randn(1, 750, 1280)
    dummy_inputs = (dummy_input, mask, mask1)
    hmonnx_file = hmonnx_dir / f"speech_tokenizer_v3_{target_device}_{quant_type}.onnx"
    convert_onnx_to_hmonnx(final_onnx, dummy_inputs, str(hmonnx_file), target_device, quant_type)
    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "target_device": target_device,
        "quant_type": quant_type,
        "onnx": str(Path(final_onnx).relative_to(component_dir)),
        "hmonnx": str(hmonnx_file.relative_to(component_dir)),
        "input_shapes": [list(dummy_input.shape), list(mask.shape), list(mask1.shape)],
    }
    return {"component_dir": component_dir, "quant_type": quant_type, **meta}


def export_cosyvoice3_llm(
    *,
    model_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    component_dir: Path,
    target_device: str,
    quant_type: str,
    torch_device: str,
    llm_checkpoint: str,
) -> dict[str, Any]:
    """Export the CosyVoice3 Qwen2-0.5B LLM: PTQ + prefill/decode onnx.

    Loosely based on the legacy ``qwen2_xh2a_export_0.5B.py`` flow.
    """
    import torch

    from xhmodel_merak.xh_other_model.builder import MODELS
    from xhmodel_merak.xh_other_model.eval_model_type import EvalModelType
    from xhquant.api import ConfigDict, convert_fx_model_to_quanted_model, get_root_logger

    from .qwen_llm_model import XHQwen2LegacyModel

    component_dir.mkdir(parents=True, exist_ok=True)
    cfg_name = f"{runtime_cfg['model_name']}_llm_{target_device}_{quant_type}"
    logger = get_root_logger()
    dtype = torch.float16

    xh_model: XHQwen2LegacyModel = MODELS.build(ConfigDict(dict(model_cfg)))
    native_model = xh_model.get_hf_model("cpu")

    # Load CosyVoice3 llm.pt weights (strip "llm.model." prefix).
    if Path(llm_checkpoint).exists():
        xh_model.load_wraped_model_state_dict_prefix(native_model, llm_checkpoint)
    else:
        logger.warning(f"llm checkpoint not found, skip loading: {llm_checkpoint}")

    xh_model.init_wrap_model(native_model)
    native_model = None

    token_embedding_file = component_dir / "token_embedding.pt"
    torch.save(xh_model.token_embedding.state_dict(), str(token_embedding_file))

    # Extract speech_embedding (6761x896) from llm.pt for XHQwen2HMONNXModel inference.
    # llm_decoder_file is back-filled by export() after the llm_decoder component is exported.
    speech_embedding_file = component_dir / "speech_embedding.pt"
    if Path(llm_checkpoint).exists():
        _llm_state = torch.load(llm_checkpoint, map_location="cpu", weights_only=True)
        if "speech_embedding.weight" in _llm_state:
            _se_weight = _llm_state["speech_embedding.weight"]
            torch.save({"weight": _se_weight}, str(speech_embedding_file))
            # Pre-extract sos/task_id embeddings so runtime never writes to the product dir.
            torch.save(_se_weight[6561:6562].unsqueeze(0), str(component_dir / "sos_emb.pt"))
            torch.save(_se_weight[6563:6564].unsqueeze(0), str(component_dir / "task_id_emb.pt"))

    meta_info: dict[str, Any] = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model_name": cfg_name,
        "hf_model": runtime_cfg["hf_model_dir"],
        "target_device": target_device,
        "quant_type": quant_type,
        "wrap_cfg": xh_model.wrap_cfg.to_dict(),
        "token_embedding_file": token_embedding_file.name,
        "speech_embedding_file": speech_embedding_file.name if speech_embedding_file.exists() else None,
    }
    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info["use_cache"] = True
        meta_info["kv_cache_shape"] = xh_model.past_key_caches[0].shape
        meta_info["num_hidden_layers"] = len(xh_model.past_key_caches)

    # Build weight quantization config from quant_type (e.g., w8a16_sefp).
    # Then overlay input-quantization specs from model config (input dtype overrides
    # such as float16/int32) so that both weight quantization and input dtypes are respected.
    quant_config = ConfigDict(build_quant_config(target_device, quant_type))
    user_quant_config = model_cfg.get("quant_config")
    if user_quant_config:
        quant_config.update(copy.deepcopy(dict(user_quant_config)))

    xh_model.change_eval_type(EvalModelType.WRAPED)
    xh_model.to(torch_device).to(dtype)

    input_sequence_length = xh_model.wrap_cfg.input_sequence_length
    try:
        _calib_tokenizer = xh_model.get_tokenizer()
        _calib_text = _calib_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "你多大了？用中文回答。"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        _tokenized = _calib_tokenizer([_calib_text], return_tensors="pt")
        input_ids = _tokenized.input_ids
        # Truncate or pad to the wrap model's expected input_sequence_length.
        if input_ids.shape[-1] > input_sequence_length:
            input_ids = input_ids[:, :input_sequence_length]
        elif input_ids.shape[-1] < input_sequence_length:
            _pad = input_sequence_length - input_ids.shape[-1]
            _pad_id = getattr(_calib_tokenizer, "pad_token_id", None) or 0
            input_ids = torch.nn.functional.pad(input_ids, (0, _pad), value=_pad_id)
        input_ids = input_ids.to(device=torch_device, dtype=torch.long)
    except Exception:
        logger.warning("PTQ calibration tokenizer unavailable, falling back to zero inputs")
        input_ids = torch.zeros((1, input_sequence_length), dtype=torch.long, device=torch_device)
    data_batch = {"input_ids": input_ids, "past_seq_length": 0}
    inputs = xh_model.prepare_inputs_for_graph(data_batch)

    xh_model._quanted_model = convert_fx_model_to_quanted_model(
        xh_model._wrap_model,
        inputs,
        target_device,
        quant_config=quant_config,
    )
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(torch_device)
    xh_model.to(dtype)

    # Export prefill graph
    prefill_dir = component_dir / "Prefill"
    prefill_dir.mkdir(parents=True, exist_ok=True)
    xh_model.to("cpu")
    _empty_cuda_cache()
    xh_model.convert_to_export_graph(data_batch)
    xh_model.change_eval_type(EvalModelType.EXPORTED)
    xh_model.to("cpu")
    _empty_cuda_cache()
    prefill_onnx_file = xh_model.to_export_onnx(data_batch, str(prefill_dir), f"{cfg_name}_prefill")[0]
    meta_info["prefill_onnx_file"] = str(Path(prefill_onnx_file).relative_to(component_dir))
    xh_model.release_exported_model()

    # Export decode graph (input_sequence_length=1)
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(torch_device)
    xh_model.to(dtype)
    xh_model.set_input_sequence_length(1)
    prefill_length = input_ids.shape[-1]
    decode_data_batch = {
        "input_ids": torch.zeros((1, 1), dtype=torch.long, device=torch_device),
        "past_seq_length": prefill_length,
    }
    _empty_cuda_cache()
    xh_model.to("cpu")
    _empty_cuda_cache()
    xh_model.convert_to_export_graph(decode_data_batch)
    xh_model.change_eval_type(EvalModelType.EXPORTED)
    xh_model.to("cpu")
    _empty_cuda_cache()
    decode_dir = component_dir / "Decoder"
    decode_dir.mkdir(parents=True, exist_ok=True)
    decode_onnx_file = xh_model.to_export_onnx(decode_data_batch, str(decode_dir), f"{cfg_name}_decode")[0]
    meta_info["decode_onnx_file"] = str(Path(decode_onnx_file).relative_to(component_dir))
    xh_model.release_exported_model()

    # Write meta_info.json (XHQwen2HMONNXModel expects this filename).
    # This is the original metadata contract for LLM - do NOT add meta.json.
    (component_dir / "meta_info.json").write_text(
        json.dumps(_jsonable(meta_info), indent=4, ensure_ascii=False), encoding="utf-8"
    )
    # Return metadata inline (no separate meta_file) - will be stored in export_meta_info.json
    return {"component_dir": component_dir, "quant_type": quant_type, **meta_info}


# ---------------------------------------------------------------------------
# golden input builders
# ---------------------------------------------------------------------------


def _build_llm_hmonnx_inputs(
    inputs_embeds,
    *,
    past_seq_length: int,
    current_input_length: int,
    kv_cache_shape: tuple[int, ...],
    num_hidden_layers: int,
    torch_device: str,
    cache_tensor_cls,
    generate_steps: int | None = None,
) -> list:
    import torch

    past_seq_length_tensor = torch.tensor([past_seq_length], dtype=torch.int32, device=torch_device)
    current_input_length_tensor = torch.tensor([current_input_length], dtype=torch.int32, device=torch_device)
    past_key_caches = [
        cache_tensor_cls(torch.zeros(kv_cache_shape, dtype=torch.float16, device=torch_device))
        for _ in range(num_hidden_layers)
    ]
    past_value_caches = [
        cache_tensor_cls(torch.zeros(kv_cache_shape, dtype=torch.float16, device=torch_device))
        for _ in range(num_hidden_layers)
    ]
    inputs = [inputs_embeds, past_seq_length_tensor, current_input_length_tensor, *past_key_caches, *past_value_caches]
    if generate_steps is not None:
        inputs.append(torch.tensor([generate_steps], dtype=torch.int32, device=torch_device))
    return inputs


def _build_component_golden_inputs(name: str, torch_device: str, comp_meta: Mapping[str, Any]) -> list:
    import torch

    if name == "llm_decoder":
        return [torch.randn(1, 896, dtype=torch.float16, device=torch_device)]
    if name == "spk_embed_affine_layer":
        return [torch.randn(1, 192, dtype=torch.float16, device=torch_device)]
    if name == "pre_lookahead_layer":
        return [torch.randn(1, 1024, 80, dtype=torch.float16, device=torch_device)]
    if name == "campplus":
        seq = int(comp_meta.get("fixed_dims", {}).get("sequence_length", 1000))
        return [torch.randn(1, seq, 80, dtype=torch.float16, device=torch_device)]
    if name == "flow_decoder":
        seq = int(comp_meta.get("seq_len", 2048))
        b = int(comp_meta.get("batch_size", 2))
        c = int(comp_meta.get("out_channels", 80))
        return [
            torch.rand((b, c, seq), dtype=torch.float16, device=torch_device),
            torch.ones((b, 1, seq), dtype=torch.float16, device=torch_device),
            torch.rand((b, c, seq), dtype=torch.float16, device=torch_device),
            torch.rand((b,), dtype=torch.float16, device=torch_device),
            torch.rand((b, c), dtype=torch.float16, device=torch_device),
            torch.rand((b, c, seq), dtype=torch.float16, device=torch_device),
        ]
    if name == "hift":
        return [torch.randn(1, 80, 1024, dtype=torch.float16, device=torch_device)]
    if name == "speech_tokenizer_v3":
        return [
            torch.randn(1, 128, 3000, dtype=torch.float16, device=torch_device),
            torch.randn(1, 20, 750, 750, dtype=torch.float16, device=torch_device),
            torch.randn(1, 750, 1280, dtype=torch.float16, device=torch_device),
        ]
    raise ValueError(f"Unknown component for golden input construction: {name}")
