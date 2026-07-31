import json
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult

from .hy_mt2_convert_config import HyMT2ConvertConfig
from .hy_mt2_converter import HyMT2ConverterXH2a


class HyMT2Workflow(BaseOtherModelWorkflow):
    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise NotImplementedError("Hy-MT2 does not support a separate quant stage; set quant: null")
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        model_dir = self._resolve_export_model_dir(quant_result)
        export_cfg = workflow_config.build_export_dict()
        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))

        convert_config = _build_convert_config(export_cfg)
        HyMT2ConverterXH2a(convert_config)._convert(model_dir, str(work_dir))

        legacy_meta_file = work_dir / "meta.json"
        legacy_meta = json.loads(legacy_meta_file.read_text(encoding="utf-8"))
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": str(Path(config_file).relative_to(work_dir)),
            "model_type": export_cfg["model"]["type"],
            "source_model_dir": model_dir,
            "target_device": str(export_cfg.get("target_device", "XH2a")),
            "components": ["prefill", "decode"],
            "legacy_meta": str(legacy_meta_file.relative_to(work_dir)),
            "hy_mt2": legacy_meta,
        }
        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(json.dumps(_jsonable(meta), ensure_ascii=False, indent=4), encoding="utf-8")
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        import torch
        from xhquant.api import CacheTensor

        work_dir = Path(export_result.work_dir)
        root_meta_file = work_dir / "export_meta_info.json"
        if not root_meta_file.is_file():
            raise FileNotFoundError(f"export_meta_info.json not found under {work_dir}")
        root_meta = json.loads(root_meta_file.read_text(encoding="utf-8"))
        meta = root_meta["hy_mt2"]
        torch_device = _select_torch_device(device)
        embedding = torch.load(work_dir / meta["token_embedding_file"], map_location="cpu", weights_only=True)["weight"]
        hidden_size = int(embedding.shape[1])
        prefill_len = int(meta["wrap_cfg"]["input_sequence_length"])
        kv_cache_shape = tuple(int(dim) for dim in meta["kv_cache"]["shape"])
        num_layers = int(meta["kv_cache"]["num_decoder_layers"])

        prefill_path = work_dir / meta["prefill_onnx"]
        prefill_inputs = _build_llm_hmonnx_inputs(
            torch.randn((1, prefill_len, hidden_size), dtype=torch.float16, device=torch_device),
            past_seq_length=0,
            current_input_length=prefill_len,
            kv_cache_shape=kv_cache_shape,
            num_hidden_layers=num_layers,
            torch_device=torch_device,
            cache_tensor_cls=CacheTensor,
        )
        prefill_golden = prefill_path.parent / "golden"
        _reset_golden_dir(prefill_golden)
        _run_hmonnx_golden(prefill_path, prefill_golden, torch_device, prefill_inputs)

        decode_path = work_dir / meta["decode_onnx"]
        decode_inputs = _build_llm_hmonnx_inputs(
            torch.randn((1, 1, hidden_size), dtype=torch.float16, device=torch_device),
            past_seq_length=prefill_len,
            current_input_length=1,
            kv_cache_shape=kv_cache_shape,
            num_hidden_layers=num_layers,
            torch_device=torch_device,
            cache_tensor_cls=CacheTensor,
        )
        decode_golden = decode_path.parent / "golden"
        _reset_golden_dir(decode_golden)
        _run_hmonnx_golden(decode_path, decode_golden, torch_device, decode_inputs)
        return str(work_dir)


def _build_convert_config(export_cfg: Mapping[str, Any]) -> HyMT2ConvertConfig:
    from xhquant.api import DeviceType, QuantScheme

    cfg = export_cfg.get("hy_mt2") or {}
    if not isinstance(cfg, Mapping):
        raise TypeError("export.hy_mt2 must be a mapping")
    target_device = str(export_cfg.get("target_device", "XH2a"))
    quant_type = str(cfg.get("quant_type", "w8a8h1_sefp"))
    return HyMT2ConvertConfig(
        batch_size=int(cfg.get("batch_size", 1)),
        context_length=int(cfg.get("context_length", 4096)),
        input_sequence_length=int(cfg.get("input_sequence_length", 256)),
        quant_scheme=QuantScheme(target_device=getattr(DeviceType, target_device), quant_type=quant_type),
        quant_weight=cfg.get("quant_weight"),
        mix_search=cfg.get("mix_search"),
        num_logits_to_keep=int(cfg.get("num_logits_to_keep", 1)),
    )


def _build_llm_hmonnx_inputs(
    inputs_embeds: Any,
    *,
    past_seq_length: int,
    current_input_length: int,
    kv_cache_shape: tuple[int, ...],
    num_hidden_layers: int,
    torch_device: str,
    cache_tensor_cls: type,
) -> list[Any]:
    import torch

    past_seq_length_tensor = torch.tensor([past_seq_length], dtype=torch.int32, device=torch_device)
    current_input_length_tensor = torch.tensor([current_input_length], dtype=torch.int32, device=torch_device)
    past_key_caches = [cache_tensor_cls(torch.zeros(kv_cache_shape, dtype=torch.float16, device=torch_device)) for _ in range(num_hidden_layers)]
    past_value_caches = [cache_tensor_cls(torch.zeros(kv_cache_shape, dtype=torch.float16, device=torch_device)) for _ in range(num_hidden_layers)]
    return [inputs_embeds, past_seq_length_tensor, current_input_length_tensor, *past_key_caches, *past_value_caches]


def _run_hmonnx_golden(hmonnx_file: Path, golden_dir: Path, device: str, inputs: Sequence[Any]) -> None:
    from xhquant.api import HMONNXGoldenInference

    session = HMONNXGoldenInference(str(hmonnx_file))
    session.to(device)
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    session.step = 0
    session(*inputs)


def _reset_golden_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _select_torch_device(device: str) -> str:
    import torch

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return str(device)
    return "cpu"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return value
