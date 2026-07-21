import json
import shutil
import tempfile
import time
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult

from .utils import activate_export_device


class VoxCPM2Workflow(BaseOtherModelWorkflow):
    SUPPORTED_COMPONENTS = {
        "lm",
        "locenc",
        "locdit",
        "audiovae_encoder",
        "audiovae_decoder_stream",
        "audiovae_decoder_full",
        "audiovae_decoder_stateful",
    }

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
        """Export components, then materialize the legacy-named release layout."""
        from .release_layout import build_release_directory

        resolved_device = str(activate_export_device(device))
        export_model_dir = self._resolve_export_model_dir(quant_result)
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_cfg = workflow_config.build_export_dict()
        target_device = str(export_cfg.get("target_device", "XH2a"))
        if target_device != "XH2a":
            raise ValueError(
                f"VoxCPM2 currently supports target_device=\'XH2a\', got {target_device!r}"
            )
        components_cfg = _normalize_components(export_cfg.get("components"))
        unsupported = [name for name in components_cfg if name not in self.SUPPORTED_COMPONENTS]
        if unsupported:
            raise ValueError(f"Unsupported VoxCPM2 component(s): {unsupported}")

        output_dir_path = Path(output_dir).expanduser().resolve()
        if output_dir_path.exists():
            raise FileExistsError(
                f"Export directory already exists: {output_dir_path}. "
                "Remove it first or use the example's --overwrite option."
            )
        output_dir_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".voxcpm2_export_staging_",
            dir=output_dir_path.parent,
        ) as staging_value:
            work_dir = Path(staging_value)
            config_file = workflow_config.dump(
                str(work_dir / f"{workflow_config.name}.yaml")
            )
            self._run_components(
                export_model_dir=export_model_dir,
                work_dir=work_dir,
                export_cfg=export_cfg,
                components_cfg=components_cfg,
                gen_golden=False,
                device=resolved_device,
            )
            self._write_meta(
                work_dir,
                config_file,
                export_model_dir,
                export_cfg,
                components_cfg,
                device=resolved_device,
            )
            release_dir = build_release_directory(work_dir, output_dir_path)

        manifest_file = release_dir / "export_meta_info.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        release_config = release_dir / manifest["config"]
        return ExportResult(
            work_dir=str(release_dir),
            config_file=str(release_config),
            meta=manifest,
        )

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        """Generate golden data by running the already exported HMONNX graphs."""
        from .release_layout import _copy_golden_step, _read_json, _write_json

        if export_result is None:
            raise ValueError("export_result can't be None")
        release_dir = Path(export_result.work_dir).expanduser().resolve()
        if not release_dir.is_dir():
            raise FileNotFoundError(f"Export directory does not exist: {release_dir}")

        resolved_device = str(activate_export_device(device))
        manifest_path = _find_release_manifest(release_dir)
        manifest = _read_json(manifest_path)
        components = manifest.get("components")
        if not isinstance(components, Mapping) or not components:
            raise ValueError(f"No release components found in {manifest_path}")
        if input_messages is not None:
            raise ValueError(
                "VoxCPM2 dump_golden does not accept synthetic input_messages; "
                "it builds deterministic inputs from each released graph contract."
            )

        generated_steps: dict[str, Path] = {}
        with tempfile.TemporaryDirectory(
            prefix=".voxcpm2_golden_staging_",
            dir=release_dir,
        ) as staging_value:
            staging_dir = Path(staging_value)
            for component_name, values in components.items():
                if not isinstance(values, Mapping):
                    raise TypeError(f"Invalid manifest entry for {component_name!r}")
                hmonnx_file = release_dir / str(
                    values.get("hmonnx_file") or values["with_act_onnx"]
                )
                if not hmonnx_file.is_file():
                    raise FileNotFoundError(
                        f"Missing released HMONNX for {component_name}: {hmonnx_file}"
                    )
                component_inputs = _build_release_golden_inputs(
                    hmonnx_file,
                    component_name,
                    resolved_device,
                    manifest,
                )
                golden_dir = staging_dir / component_name / "golden"
                _run_release_hmonnx_golden(
                    hmonnx_file,
                    golden_dir,
                    resolved_device,
                    component_inputs,
                )
                source_step = golden_dir / "step_0"
                if not source_step.is_dir():
                    raise FileNotFoundError(
                        f"HMONNX did not create step_0 for {component_name}: {source_step}"
                    )
                generated_steps[component_name] = source_step

            for component_name, values in components.items():
                component_dir = release_dir / str(
                    values.get("component_dir") or values["directory"]
                )
                hmonnx_file = release_dir / str(
                    values.get("hmonnx_file") or values["with_act_onnx"]
                )
                external_value = values.get("external_data")
                external_file = release_dir / str(external_value) if external_value else None
                step_dir = component_dir / "step_0"
                if step_dir.exists() or step_dir.is_symlink():
                    if step_dir.is_dir() and not step_dir.is_symlink():
                        shutil.rmtree(step_dir)
                    else:
                        step_dir.unlink()

                graph_stem = hmonnx_file.stem.removesuffix("_with_act")
                _copy_golden_step(generated_steps[component_name], step_dir, graph_stem)
                (step_dir / hmonnx_file.name).symlink_to(Path("..") / hmonnx_file.name)
                if external_file is not None:
                    (step_dir / external_file.name).symlink_to(Path("..") / external_file.name)

                relative_step = str(step_dir.relative_to(release_dir))
                if "golden_dir" in values:
                    values["golden_dir"] = relative_step
                else:
                    values["step_dir"] = relative_step

        manifest["export_device"] = resolved_device
        manifest["components"] = components
        manifest["golden"] = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "device": resolved_device,
            "input_messages": repr(input_messages),
        }
        _write_json(manifest_path, manifest)
        export_result.meta = manifest
        return str(manifest_path)

    def _run_components(
        self,
        *,
        export_model_dir: str,
        work_dir: Path,
        export_cfg: Mapping[str, Any],
        components_cfg: Mapping[str, Any],
        gen_golden: bool,
        device: str,
    ) -> None:
        quant_types = _mapping(export_cfg.get("quant_types"))
        dtype = str(export_cfg.get("dtype", "float16"))
        lm_prefill_length, lm_cache_length = _lm_lengths(export_cfg)

        if _component_enabled(components_cfg, "lm"):
            from . import export_lm

            cfg = _mapping(export_cfg.get("lm"))
            lm_contract = _lm_export_contract(cfg)
            export_lm.main(
                Namespace(
                    model=export_model_dir,
                    output_dir=str(work_dir),
                    prefill_length=lm_prefill_length,
                    cache_length=lm_cache_length,
                    device=device,
                    dtype=dtype,
                    frontend_type=lm_contract["frontend_type"],
                    use_cache=lm_contract["use_cache"],
                    cache_axis=lm_contract["cache_axis"],
                    batch_size=lm_contract["batch_size"],
                    input_names=lm_contract["input_names"],
                    output_names=lm_contract["output_names"],
                    quant_type=str(_quant_type(cfg, quant_types, "lm", "w8a8_sefp")),
                    gen_golden=gen_golden,
                    skip_verify=bool(cfg.get("skip_verify", True)),
                    verify_max_abs_tol=float(cfg.get("verify_max_abs_tol", 1.0)),
                    verify_mean_abs_tol=float(cfg.get("verify_mean_abs_tol", 0.1)),
                    verify_cosine_tol=float(cfg.get("verify_cosine_tol", 0.95)),
                    verify_fail_on_mismatch=bool(cfg.get("verify_fail_on_mismatch", False)),
                )
            )

        if _component_enabled(components_cfg, "locenc"):
            from . import export_locenc

            cfg = _mapping(export_cfg.get("locenc"))
            export_locenc.main(
                Namespace(
                    model=export_model_dir,
                    output_dir=str(work_dir),
                    cal_wav=cfg.get("cal_wav"),
                    num_cal_samples=int(cfg.get("num_cal_samples", 8)),
                    device=device,
                    dtype=dtype,
                    quant_type=str(_quant_type(cfg, quant_types, "locenc", "w8a8_sefp")),
                    gen_golden=gen_golden,
                    skip_verify=bool(cfg.get("skip_verify", True)),
                    verify_max_abs_tol=float(cfg.get("verify_max_abs_tol", 1.0)),
                    verify_mean_abs_tol=float(cfg.get("verify_mean_abs_tol", 0.1)),
                    verify_cosine_tol=float(cfg.get("verify_cosine_tol", 0.95)),
                    verify_fail_on_mismatch=bool(cfg.get("verify_fail_on_mismatch", False)),
                )
            )

        if _component_enabled(components_cfg, "locdit"):
            from . import export_locdit

            cfg = _mapping(export_cfg.get("locdit"))
            export_locdit.main(
                Namespace(
                    model=export_model_dir,
                    output_dir=str(work_dir),
                    device=device,
                    dtype=dtype,
                    quant_type=str(_quant_type(cfg, quant_types, "locdit", "w8a8_sefp")),
                    gen_golden=gen_golden,
                    skip_verify=bool(cfg.get("skip_verify", True)),
                    verify_max_abs_tol=float(cfg.get("verify_max_abs_tol", 1.0)),
                    verify_mean_abs_tol=float(cfg.get("verify_mean_abs_tol", 0.1)),
                    verify_cosine_tol=float(cfg.get("verify_cosine_tol", 0.95)),
                    verify_fail_on_mismatch=bool(cfg.get("verify_fail_on_mismatch", False)),
                )
            )

        if _component_enabled(components_cfg, "audiovae_encoder"):
            from . import export_audiovae_encoder

            cfg = _mapping(export_cfg.get("audiovae_encoder"))
            export_audiovae_encoder.main(
                Namespace(
                    model=export_model_dir,
                    output_dir=str(work_dir),
                    num_patches=int(cfg.get("num_patches", 128)),
                    audio=cfg.get("audio"),
                    device=device,
                    dtype=dtype,
                    quant_type=str(_quant_type(cfg, quant_types, "audiovae_encoder", "w16a16_sefp")),
                    gen_golden=gen_golden,
                    skip_verify=bool(cfg.get("skip_verify", True)),
                    verify_max_abs_tol=float(cfg.get("verify_max_abs_tol", 0.5)),
                    verify_mean_abs_tol=float(cfg.get("verify_mean_abs_tol", 0.05)),
                    verify_cosine_tol=float(cfg.get("verify_cosine_tol", 0.99)),
                    verify_fail_on_mismatch=bool(cfg.get("verify_fail_on_mismatch", False)),
                )
            )

        decoder_cfg = _mapping(export_cfg.get("audiovae_decoder"))
        if _component_enabled(components_cfg, "audiovae_decoder_stream"):
            self._run_audiovae_decoder(
                export_model_dir,
                work_dir,
                decoder_cfg,
                quant_types,
                "audiovae_decoder_stream",
                "stream_num_patches",
                3,
                gen_golden,
                device,
                dtype,
            )
        if _component_enabled(components_cfg, "audiovae_decoder_full"):
            self._run_audiovae_decoder(
                export_model_dir,
                work_dir,
                decoder_cfg,
                quant_types,
                "audiovae_decoder_full",
                "full_num_patches",
                128,
                gen_golden,
                device,
                dtype,
            )

        if _component_enabled(components_cfg, "audiovae_decoder_stateful"):
            from . import export_audiovae_decoder_streaming_stateful

            cfg = _mapping(export_cfg.get("audiovae_decoder_stateful"))
            export_audiovae_decoder_streaming_stateful.main(
                Namespace(
                    model=export_model_dir,
                    output_dir=str(work_dir),
                    num_patches=int(cfg.get("num_patches", 1)),
                    quant_type=str(_quant_type(cfg, quant_types, "audiovae_decoder_stateful", "w8a8_sefp")),
                    seed=int(cfg.get("seed", 0)),
                    verify_steps=int(cfg.get("verify_steps", 4)),
                    device=device,
                    dtype=dtype,
                    cpu=False,
                    skip_hmonnx=bool(cfg.get("skip_hmonnx", False)),
                    skip_verify=bool(cfg.get("skip_verify", True)),
                    gen_golden=gen_golden,
                )
            )

    @staticmethod
    def _run_audiovae_decoder(
        export_model_dir: str,
        work_dir: Path,
        cfg: Mapping[str, Any],
        quant_types: Mapping[str, Any],
        component_name: str,
        patches_key: str,
        default_patches: int,
        gen_golden: bool,
        device: str,
        dtype: str,
    ) -> None:
        from . import export_audiovae_decoder

        export_audiovae_decoder.main(
            Namespace(
                model=export_model_dir,
                output_dir=str(work_dir),
                num_patches=int(cfg.get(patches_key, default_patches)),
                device=device,
                dtype=dtype,
                quant_type=str(_quant_type(cfg, quant_types, component_name, "w8a8_sefp")),
                gen_golden=gen_golden,
                skip_verify=bool(cfg.get("skip_verify", True)),
                verify_max_abs_tol=float(cfg.get("verify_max_abs_tol", 0.5)),
                verify_mean_abs_tol=float(cfg.get("verify_mean_abs_tol", 0.05)),
                verify_cosine_tol=float(cfg.get("verify_cosine_tol", 0.99)),
                verify_fail_on_mismatch=bool(cfg.get("verify_fail_on_mismatch", False)),
            )
        )

    @staticmethod
    def _write_meta(
        work_dir: Path,
        config_file: str,
        model_dir: str,
        export_cfg: Mapping[str, Any],
        components_cfg: Mapping[str, Any],
        device: str | None = None,
    ) -> dict[str, Any]:
        prefill_length, context_length = _lm_lengths(export_cfg)
        component_meta = {
            "lm": "lm_export_meta_info.json",
            "locenc": "LocEnc/locenc_meta_info.json",
            "locdit": "LocDiT/locdit_meta_info.json",
        }
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": str(Path(config_file).relative_to(work_dir)),
            "hf_model": model_dir,
            "model_name": export_cfg.get("model_name", Path(model_dir).name),
            "target_device": export_cfg.get("target_device", "XH2a"),
            "dtype": str(export_cfg.get("dtype", "float16")),
            "export_device": device,
            "prefill_length": prefill_length,
            "context_length": context_length,
            "components": {},
        }
        for name, rel in component_meta.items():
            if _component_enabled(components_cfg, name):
                meta["components"][name] = {"meta_file": rel, "exists": (work_dir / rel).exists()}

        enc_cfg = _mapping(export_cfg.get("audiovae_encoder"))
        if _component_enabled(components_cfg, "audiovae_encoder"):
            n = int(enc_cfg.get("num_patches", 128))
            rel = f"AudioVAE_Encoder_np{n}/audiovae_encoder_meta_info.json"
            meta["components"]["audiovae_encoder"] = {"meta_file": rel, "exists": (work_dir / rel).exists()}

        dec_cfg = _mapping(export_cfg.get("audiovae_decoder"))
        for name, key, default in (
            ("audiovae_decoder_stream", "stream_num_patches", 3),
            ("audiovae_decoder_full", "full_num_patches", 128),
        ):
            if _component_enabled(components_cfg, name):
                n = int(dec_cfg.get(key, default))
                rel = f"AudioVAE_Decoder_np{n}/audiovae_decoder_np{n}_meta_info.json"
                meta["components"][name] = {"meta_file": rel, "exists": (work_dir / rel).exists()}

        st_cfg = _mapping(export_cfg.get("audiovae_decoder_stateful"))
        if _component_enabled(components_cfg, "audiovae_decoder_stateful"):
            n = int(st_cfg.get("num_patches", 1))
            rel = (
                f"AudioVAE_Decoder_StreamState_np{n}/"
                f"audiovae_decoder_streaming_stateful_np{n}_meta_info.json"
            )
            meta["components"]["audiovae_decoder_stateful"] = {
                "meta_file": rel,
                "exists": (work_dir / rel).exists(),
            }

        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(json.dumps(meta, indent=4, ensure_ascii=False), encoding="utf-8")
        return meta



def _lm_lengths(export_cfg: Mapping[str, Any]) -> tuple[int, int]:
    lm_cfg = _mapping(export_cfg.get("lm"))
    wrap_cfg = _mapping(lm_cfg.get("wrap_cfg"))
    try:
        prefill_length = int(wrap_cfg["input_sequence_length"])
        cache_length = int(wrap_cfg["max_sequence_length"])
    except KeyError as exc:
        raise ValueError(
            "VoxCPM2 requires export.lm.wrap_cfg.input_sequence_length and "
            "export.lm.wrap_cfg.max_sequence_length."
        ) from exc

    if cache_length <= 0 or prefill_length > cache_length:
        raise ValueError(
            "VoxCPM2 LM cache_length must be positive and no smaller than "
            "prefill_length. "
            f"Got prefill_length={prefill_length}, cache_length={cache_length}."
        )
    return prefill_length, cache_length


def _lm_export_contract(lm_cfg: Mapping[str, Any]) -> dict[str, Any]:
    wrap_cfg = _mapping(lm_cfg.get("wrap_cfg"))
    kv_cache = _mapping(wrap_cfg.get("kv_cache"))
    export_cfg = _mapping(lm_cfg.get("export_cfg"))
    contract = {
        "use_cache": bool(wrap_cfg.get("use_cache", True)),
        "cache_axis": int(kv_cache.get("cache_axis", 2)),
        "batch_size": int(wrap_cfg.get("batch_size", 1)),
        "frontend_type": str(lm_cfg.get("frontend_type", "TorchFX")),
        "input_names": list(
            export_cfg.get(
                "input_names",
                ["inputs_embeds", "past_seq_length", "current_input_length"],
            )
        ),
        "output_names": list(export_cfg.get("output_names", ["hidden"])),
    }
    if not contract["use_cache"]:
        raise ValueError("VoxCPM2 LM export requires wrap_cfg.use_cache=true")
    if contract["cache_axis"] != 2:
        raise ValueError("VoxCPM2 LM export currently requires kv_cache.cache_axis=2")
    if contract["batch_size"] != 1:
        raise ValueError("VoxCPM2 LM export currently supports batch_size=1 only")
    if len(contract["input_names"]) != 3:
        raise ValueError("VoxCPM2 LM export_cfg.input_names must contain exactly 3 names")
    if len(contract["output_names"]) != 1:
        raise ValueError("VoxCPM2 LM export_cfg.output_names must contain exactly 1 name")
    return contract

def _quant_type(
    component_cfg: Mapping[str, Any],
    quant_types: Mapping[str, Any],
    component_name: str,
    default: str,
) -> Any:
    return component_cfg.get("quant_type", quant_types.get(component_name, default))


def _mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping config, got {type(value).__name__}")
    return value


def _normalize_components(value: Any) -> Mapping[str, Any]:
    if value is None:
        raise ValueError(
            "VoxCPM2 requires export.components to be explicitly configured; "
            "list the components that should be exported."
        )
    if isinstance(value, Mapping):
        components = value
    elif isinstance(value, list):
        components = {str(name): {"enabled": True} for name in value}
    else:
        raise TypeError("VoxCPM2 export.components must be a mapping or a list")
    if not components or not any(_component_enabled(components, name) for name in components):
        raise ValueError("VoxCPM2 export.components must enable at least one component")
    return components


def _component_enabled(components_cfg: Mapping[str, Any], name: str) -> bool:
    value = components_cfg.get(name, False)
    if isinstance(value, Mapping):
        return bool(value.get("enabled", True))
    return bool(value)


def _find_release_manifest(release_dir: Path) -> Path:
    export_meta = release_dir / "export_meta_info.json"
    if export_meta.is_file():
        return export_meta
    expected = release_dir / f"{release_dir.name}_manifest.json"
    if expected.is_file():
        return expected
    matches = sorted(release_dir.glob("*_manifest.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one release manifest under {release_dir}, got {len(matches)}"
        )
    return matches[0]


def _build_release_golden_inputs(
    hmonnx_file: Path,
    component_name: str,
    device: str,
    manifest: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Build deterministic inputs from a released graph's static contract."""
    import onnx
    import torch
    from onnx import TensorProto
    from xhquant.core import CacheTensor

    model = onnx.load_model(str(hmonnx_file), load_external_data=False)
    initializer_names = {value.name for value in model.graph.initializer}
    initializer_names.update(value.values.name for value in model.graph.sparse_initializer)
    graph_inputs = [value for value in model.graph.input if value.name not in initializer_names]

    sequence_length = 1
    for value in graph_inputs:
        if value.name in {"input_1", "inputs_embeds"}:
            shape = _fixed_onnx_shape(value, hmonnx_file)
            if len(shape) > 1:
                sequence_length = shape[1]
            break

    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(1024)
    dtype_map = {
        TensorProto.FLOAT16: torch.float16,
        TensorProto.FLOAT: torch.float32,
        TensorProto.DOUBLE: torch.float64,
        TensorProto.BFLOAT16: torch.bfloat16,
        TensorProto.INT32: torch.int32,
        TensorProto.INT64: torch.int64,
        TensorProto.BOOL: torch.bool,
    }
    prefill_length = int((manifest or {}).get("prefill_length", 1))
    result: list[Any] = []
    for value in graph_inputs:
        tensor_type = value.type.tensor_type
        dtype = dtype_map.get(tensor_type.elem_type)
        if dtype is None:
            raise TypeError(
                f"Unsupported ONNX input dtype {tensor_type.elem_type} for "
                f"{component_name}.{value.name}"
            )
        shape = _fixed_onnx_shape(value, hmonnx_file)
        name = value.name
        if dtype == torch.bool:
            tensor = torch.zeros(shape, dtype=dtype, device=torch_device)
        elif dtype in (torch.int32, torch.int64):
            if name in {"valid_length", "past_seq_length"}:
                fill_value = prefill_length if component_name.endswith("_decode") else 0
            elif name in {"current_length", "current_input_length"}:
                fill_value = sequence_length
            elif name == "sr_idx":
                fill_value = 3
            else:
                fill_value = 0
            tensor = torch.full(shape, fill_value, dtype=dtype, device=torch_device)
        elif name.endswith("cache_input") or name.startswith("state_in_") or name == "dt":
            tensor = torch.zeros(shape, dtype=dtype, device=torch_device)
        elif name == "t":
            tensor = torch.full(shape, 0.5, dtype=dtype, device=torch_device)
        else:
            tensor = torch.randn(
                shape,
                dtype=dtype,
                device=torch_device,
                generator=generator,
            ) * 0.5

        if name.endswith("cache_input"):
            tensor = CacheTensor(tensor)
        result.append(tensor)
    return result


def _fixed_onnx_shape(value: Any, hmonnx_file: Path) -> tuple[int, ...]:
    shape: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            raise ValueError(
                "Golden generation requires fixed positive input shapes; "
                f"{hmonnx_file.name}:{value.name} has a dynamic dimension"
            )
        shape.append(int(dim.dim_value))
    return tuple(shape)


def _run_release_hmonnx_golden(
    hmonnx_file: Path,
    golden_dir: Path,
    device: str,
    inputs: list[Any],
) -> None:
    from xhquant.api import HMONNXGoldenInference

    golden_dir.mkdir(parents=True, exist_ok=True)
    session = HMONNXGoldenInference(str(hmonnx_file))
    session.to(device)
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    session.step = 0
    session(*inputs)
