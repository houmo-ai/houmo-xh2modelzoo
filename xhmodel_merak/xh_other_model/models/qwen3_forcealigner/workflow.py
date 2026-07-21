import copy
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_other_model.workflows.base import (
    BaseOtherModelWorkflow,
)
from xhmodel_merak.xh_other_model.workflows.result import (
    ExportResult,
    QuantResult,
)

from ._export_utils import (
    _build_llm_hmonnx_inputs,
    _jsonable,
    _reset_golden_dir,
    _run_hmonnx_golden,
    _select_torch_device,
    export_qwen3_forcealigner_encoder,
    export_qwen3_forcealigner_prefill,
)


class Qwen3ForceAlignerWorkflow(BaseOtherModelWorkflow):
    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(
            config_overrides
        )
        if workflow_config.quant is not None:
            raise NotImplementedError(
                "Qwen3-ForceAligner does not support a separate quant stage"
            )
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        workflow_config = self.workflow_config.with_overrides(
            config_overrides
        )
        model_dir = self._resolve_export_model_dir(quant_result)
        export_cfg = workflow_config.build_export_dict()
        model_cfg = copy.deepcopy(export_cfg["model"])
        model_cfg["hf_model"] = model_dir

        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        target_device = str(export_cfg.get("target_device", "XH2a"))
        config_file = workflow_config.dump(
            str(work_dir / f"{workflow_config.name}.yaml")
        )

        audio_cfg = export_cfg.get("audio") or {}
        prefill_cfg = export_cfg.get("prefill") or {}
        if not isinstance(audio_cfg, Mapping):
            raise TypeError(
                "Qwen3-ForceAligner export.audio must be a mapping"
            )
        if not isinstance(prefill_cfg, Mapping):
            raise TypeError(
                "Qwen3-ForceAligner export.prefill must be a mapping"
            )

        max_audio_length = int(audio_cfg.get("max_audio_length", 3000))
        sequence_length = int(prefill_cfg.get("sequence_length", 411))
        encoder_quant_type = str(
            audio_cfg.get("quant_type", "w8a8_sefp")
        )
        prefill_quant_type = str(
            prefill_cfg.get("quant_type", "w8a8_sefp")
        )

        meta: dict[str, Any] = {
            "create_time": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(),
            ),
            "config": str(Path(config_file).relative_to(work_dir)),
            "hf_model": model_dir,
            "target_device": target_device,
            "components": ["encoder", "prefill"],
        }
        meta["encoder"] = export_qwen3_forcealigner_encoder(
            model_dir=model_dir,
            work_dir=work_dir,
            target_device=target_device,
            quant_type=encoder_quant_type,
            max_audio_length=max_audio_length,
        )
        meta.update(
            export_qwen3_forcealigner_prefill(
                model_dir=model_dir,
                work_dir=work_dir,
                target_device=target_device,
                model_cfg=model_cfg,
                device=device,
                quant_type=prefill_quant_type,
                sequence_length=sequence_length,
            )
        )

        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(
            json.dumps(_jsonable(meta), indent=4),
            encoding="utf-8",
        )
        return ExportResult(
            work_dir=str(work_dir),
            config_file=config_file,
            meta=meta,
        )

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        import torch
        from xhquant.core import CacheTensor

        work_dir = Path(export_result.work_dir)
        meta_file = work_dir / "export_meta_info.json"
        if not meta_file.is_file():
            raise FileNotFoundError(
                f"export_meta_info.json not found under {work_dir}"
            )
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        torch_device = _select_torch_device(device)

        encoder_path = work_dir / meta["encoder"]["hmonnx_file"]
        encoder_cfg = meta["encoder"]["model_cfg"]
        max_audio_length = int(
            encoder_cfg["fixed_max_audio_length"]
        )
        encoder_inputs = [
            torch.randn(
                1,
                int(encoder_cfg["num_mel_bins"]),
                max_audio_length,
                dtype=torch.float16,
                device=torch_device,
            ),
            torch.tensor(
                [max_audio_length],
                dtype=torch.int32,
                device=torch_device,
            ),
        ]
        encoder_golden = encoder_path.parent / "golden"
        _reset_golden_dir(encoder_golden)
        _run_hmonnx_golden(
            encoder_path,
            encoder_golden,
            torch_device,
            encoder_inputs,
        )

        embedding = torch.load(
            work_dir / meta["token_embedding_file"],
            map_location="cpu",
        )["weight"]
        prefill_length = int(meta["prefill_input_sequence_length"])
        prefill_path = work_dir / meta["prefill_onnx_file"]
        prefill_inputs = _build_llm_hmonnx_inputs(
            torch.randn(
                (1, prefill_length, int(embedding.shape[1])),
                dtype=torch.float16,
                device=torch_device,
            ),
            past_seq_length=0,
            current_input_length=prefill_length,
            kv_cache_shape=tuple(
                int(dim) for dim in meta["kv_cache_shape"]
            ),
            num_hidden_layers=int(meta["num_hidden_layers"]),
            torch_device=torch_device,
            cache_tensor_cls=CacheTensor,
        )
        prefill_golden = prefill_path.parent / "hmonnx" / "golden"
        _reset_golden_dir(prefill_golden)
        _run_hmonnx_golden(
            prefill_path,
            prefill_golden,
            torch_device,
            prefill_inputs,
        )

        return str(work_dir)
