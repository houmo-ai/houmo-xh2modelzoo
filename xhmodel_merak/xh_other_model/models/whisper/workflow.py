import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult

from ._export_utils import (
    Decoder,
    build_decoder_export_payload,
    build_decoder_golden_inputs,
    build_encoder_input_features,
    flatten_inputs,
    jsonable,
    run_hmonnx_golden,
    select_torch_device,
    simplify_onnx,
    to_xh_device_type,
)


def _validate_whisper_export_config(components_cfg: Mapping[str, Any], model_dir: str) -> None:
    from transformers import WhisperProcessor

    _prefill_cfg = dict(components_cfg.get("prefill") or {})
    _decoder_cfg = dict(components_cfg.get("decoder") or {})

    _ppt: list[int] = list(_prefill_cfg.get("prompt_token_ids") or [50258, 50259, 50359, 50363])
    _pcp: list[int] = list(_prefill_cfg.get("cache_position") or [0, 1, 2, 3])
    _ppl: int = int(_prefill_cfg.get("past_len") or 0)

    if len(_ppt) != len(_pcp):
        raise ValueError(
            f"prefill.prompt_token_ids (len={len(_ppt)}) and "
            f"prefill.cache_position (len={len(_pcp)}) must have the same length."
        )
    if _ppl != 0:
        raise ValueError(f"prefill.past_len must be 0, got {_ppl}.")

    _proc = WhisperProcessor.from_pretrained(model_dir)
    for _tid in _ppt:
        if not _proc.tokenizer.decode([_tid]).strip():
            raise ValueError(
                f"prefill.prompt_token_ids contains undecodable token {_tid}. "
                "Fix prompt_token_ids in the YAML export config."
            )

    _dpt: list[int] = list(_decoder_cfg.get("prompt_token_ids") or [2221])
    _dcp: list[int] = list(_decoder_cfg.get("cache_position") or [4])
    _dpl: int = int(_decoder_cfg.get("past_len") or 4)

    if len(_dpt) != len(_dcp):
        raise ValueError(
            f"decoder.prompt_token_ids (len={len(_dpt)}) and "
            f"decoder.cache_position (len={len(_dcp)}) must have the same length."
        )
    if _dpl != len(_ppt):
        raise ValueError(
            f"decoder.past_len ({_dpl}) must equal "
            f"len(prefill.prompt_token_ids) ({len(_ppt)}). "
            f"After a {len(_ppt)}-token prefill the decoder cache write "
            f"position starts at cache slot {len(_ppt)}."
        )


class WhisperWorkflow(BaseOtherModelWorkflow):
    SUPPORTED_COMPONENTS = {"encoder", "prefill", "decoder"}

    # ------------------------------------------------------------------ quant
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

    # ---------------------------------------------------------------- export
    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        from transformers import WhisperForConditionalGeneration

        from xhquant.api import get_root_logger

        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_model_dir = self._resolve_export_model_dir(quant_result)
        export_cfg = workflow_config.build_export_dict()

        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        target_device = str(export_cfg.get("target_device", "XH2a"))
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))

        components_cfg = export_cfg.get("components") or {}
        if not isinstance(components_cfg, Mapping):
            raise TypeError("Whisper export.components must be a mapping")
        unsupported = [name for name in components_cfg if name not in self.SUPPORTED_COMPONENTS]
        if unsupported:
            raise ValueError(
                f"Unsupported Whisper export component(s): {unsupported}. "
                f"Supported: {sorted(self.SUPPORTED_COMPONENTS)}"
            )

        _validate_whisper_export_config(components_cfg, export_model_dir)

        logger = get_root_logger()
        logger.info(f"Loading Whisper model from {export_model_dir}")
        model = WhisperForConditionalGeneration.from_pretrained(export_model_dir)
        model.eval()
        model.config.forced_decoder_ids = None
        model.config._attn_implementation = "eager"
        model.model.encoder.decoder_m = model.model.decoder

        decoder_layer0 = model.model.decoder.layers[0].self_attn
        model_meta_cfg = {
            "head_dim": int(decoder_layer0.head_dim),
            "num_heads": int(decoder_layer0.num_heads),
            "embed_dim": int(decoder_layer0.embed_dim),
            "max_source_positions": int(model.config.max_source_positions),
            "num_decode_layers": int(model.config.decoder_layers),
        }

        meta: dict[str, Any] = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": str(Path(config_file).relative_to(work_dir)),
            "hf_model": export_model_dir,
            "target_device": target_device,
            "model_cfg": model_meta_cfg,
            "components": [],
        }

        torch_device = select_torch_device(device)

        # ---- encoder -------------------------------------------------------
        encoder_cfg = components_cfg.get("encoder", {})
        if encoder_cfg is not False and (
            not isinstance(encoder_cfg, Mapping) or encoder_cfg.get("enabled", True)
        ):
            if not isinstance(encoder_cfg, Mapping):
                raise TypeError("Whisper export.components.encoder must be a mapping or false")
            meta["encoder"] = export_whisper_encoder(
                model=model,
                model_dir=export_model_dir,
                work_dir=work_dir,
                target_device=target_device,
                quant_type=str(encoder_cfg.get("quant_type", "w8a8_sefp")),
                audio_path=str(encoder_cfg.get("audio_path", "")),
                torch_device=torch_device,
                logger=logger,
            )
            meta["components"].append("encoder")
        else:
            raise ValueError("Whisper export requires the encoder component to be enabled")

        # ---- prefill --------------------------------------------------------
        prefill_cfg = components_cfg.get("prefill", {})
        if prefill_cfg is not False and (
            not isinstance(prefill_cfg, Mapping) or prefill_cfg.get("enabled", True)
        ):
            if not isinstance(prefill_cfg, Mapping):
                raise TypeError("Whisper export.components.prefill must be a mapping or false")
            meta["prefill"] = export_whisper_decoder_graph(
                model=model,
                model_dir=export_model_dir,
                work_dir=work_dir,
                target_device=target_device,
                quant_type=str(prefill_cfg.get("quant_type", "w8a8_sefp")),
                component_name="prefill",
                prompt_token_ids=list(prefill_cfg.get("prompt_token_ids", [50258, 50259, 50359, 50363])),
                cache_position=list(prefill_cfg.get("cache_position", [0, 1, 2, 3])),
                past_len=int(prefill_cfg.get("past_len", 0)),
                model_meta_cfg=model_meta_cfg,
                device=device,
                logger=logger,
            )
            meta["components"].append("prefill")

        # ---- decoder --------------------------------------------------------
        decoder_cfg = components_cfg.get("decoder", {})
        if decoder_cfg is not False and (
            not isinstance(decoder_cfg, Mapping) or decoder_cfg.get("enabled", True)
        ):
            if not isinstance(decoder_cfg, Mapping):
                raise TypeError("Whisper export.components.decoder must be a mapping or false")
            meta["decoder"] = export_whisper_decoder_graph(
                model=model,
                model_dir=export_model_dir,
                work_dir=work_dir,
                target_device=target_device,
                quant_type=str(decoder_cfg.get("quant_type", "w8a8_sefp")),
                component_name="decoder",
                prompt_token_ids=list(decoder_cfg.get("prompt_token_ids", [2221])),
                cache_position=list(decoder_cfg.get("cache_position", [4])),
                past_len=int(decoder_cfg.get("past_len", 4)),
                model_meta_cfg=model_meta_cfg,
                device=device,
                logger=logger,
            )
            meta["components"].append("decoder")

        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(json.dumps(jsonable(meta), indent=4, ensure_ascii=False), encoding="utf-8")
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

    # -------------------------------------------------------------- dump_golden
    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        import torch

        work_dir = Path(export_result.work_dir)
        meta_file = work_dir / "export_meta_info.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"export_meta_info.json not found under {work_dir}")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

        torch_device = select_torch_device(device)
        model_meta_cfg = meta["model_cfg"]
        num_heads = int(model_meta_cfg["num_heads"])
        head_dim = int(model_meta_cfg["head_dim"])
        embed_dim = int(model_meta_cfg["embed_dim"])
        max_source_positions = int(model_meta_cfg["max_source_positions"])
        num_decode_layers = int(model_meta_cfg["num_decode_layers"])

        import shutil

        # ---- encoder golden -------------------------------------------------
        if "encoder" in meta:
            encoder_path = work_dir / meta["encoder"]["hmonnx_file"]
            hf_model_dir = meta.get("hf_model", "")
            audio_path = meta["encoder"].get("audio_path", "")

            input_features = build_encoder_input_features(
                hf_model_dir, str(torch_device), audio_path,
            ).to(dtype=torch.float16)
            golden_dir = encoder_path.parent / "golden"
            if golden_dir.exists():
                shutil.rmtree(golden_dir)
            run_hmonnx_golden(encoder_path, golden_dir, torch_device, [input_features])

        # ---- prefill golden -------------------------------------------------
        if "prefill" in meta:
            from xhquant.core import CacheTensor

            prefill_path = work_dir / meta["prefill"]["hmonnx_file"]
            prefill_inputs = build_decoder_golden_inputs(
                prompt_token_ids=meta["prefill"]["prompt_token_ids"],
                cache_position=meta["prefill"]["cache_position"],
                past_len=meta["prefill"]["past_len"],
                num_heads=num_heads,
                head_dim=head_dim,
                embed_dim=embed_dim,
                max_source_positions=max_source_positions,
                num_decode_layers=num_decode_layers,
                torch_device=torch_device,
                cache_tensor_cls=CacheTensor,
            )
            prefill_golden_dir = prefill_path.parent / "golden"
            if prefill_golden_dir.exists():
                shutil.rmtree(prefill_golden_dir)
            run_hmonnx_golden(prefill_path, prefill_golden_dir, torch_device, prefill_inputs)

        # ---- decoder golden -------------------------------------------------
        if "decoder" in meta:
            from xhquant.core import CacheTensor

            decode_path = work_dir / meta["decoder"]["hmonnx_file"]
            decode_inputs = build_decoder_golden_inputs(
                prompt_token_ids=meta["decoder"]["prompt_token_ids"],
                cache_position=meta["decoder"]["cache_position"],
                past_len=meta["decoder"]["past_len"],
                num_heads=num_heads,
                head_dim=head_dim,
                embed_dim=embed_dim,
                max_source_positions=max_source_positions,
                num_decode_layers=num_decode_layers,
                torch_device=torch_device,
                cache_tensor_cls=CacheTensor,
            )
            decode_golden_dir = decode_path.parent / "golden"
            if decode_golden_dir.exists():
                shutil.rmtree(decode_golden_dir)
            run_hmonnx_golden(decode_path, decode_golden_dir, torch_device, decode_inputs)

        return str(work_dir)


def export_whisper_encoder(
    *,
    model,
    model_dir: str,
    work_dir: Path,
    target_device: str,
    quant_type: str,
    audio_path: str,
    torch_device: str,
    logger,
) -> dict[str, Any]:
    import onnx
    import torch

    from xhquant.api import (
        DeviceType,
        QuantScheme,
        convert_onnx_to_hmonnx,
        create_quant_config,
    )
    from xhquant.patch.core import RewriterContext

    from ._export_utils import GB

    device_type = to_xh_device_type(target_device, DeviceType)
    model_name = Path(os.path.normpath(model_dir)).name

    encoder_dir = work_dir / "encoder"
    encoder_dir.mkdir(exist_ok=True, parents=True)
    onnx_file = encoder_dir / f"{model_name}_encoder.onnx"
    hmonnx_file = encoder_dir / "hmonnx" / f"{model_name}_encoder_{target_device}_{quant_type}.onnx"

    input_features = build_encoder_input_features(model_dir, "cpu", audio_path)
    encoder = model.model.encoder
    num_mel_bins = int(encoder.conv1.in_channels)
    input_length = int(
        encoder.config.max_source_positions * encoder.conv1.stride[0] * encoder.conv2.stride[0]
    )

    encoder_output_names: list[str] = []
    num_decode_layers = int(model.config.decoder_layers)
    for i in range(num_decode_layers):
        encoder_output_names.append(f"key_state_{i}")
    for i in range(num_decode_layers):
        encoder_output_names.append(f"value_state_{i}")

    if not onnx_file.exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            with RewriterContext(None, backend="onnxruntime"):
                temp_onnx_file = str(Path(tmp_dir) / onnx_file.name)
                torch.onnx.export(
                    model.model.encoder,
                    input_features,
                    temp_onnx_file,
                    input_names=["input_features"],
                    output_names=encoder_output_names,
                )
                onnx_model = onnx.load(temp_onnx_file)
                onnx_model, _ = simplify_onnx(onnx_model)
        onnx.save(
            onnx_model,
            onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{onnx_file.stem}_external_data",
        )
    else:
        onnx_model = onnx.load(onnx_file)
    logger.info(f"Encoder ONNX saved: {onnx_file}, size: {onnx_model.ByteSize() / GB:.2f} GB")

    if not hmonnx_file.exists():
        quant_scheme = QuantScheme(target_device=device_type, quant_type=quant_type)
        convert_onnx_to_hmonnx(
            str(onnx_file),
            [input_features],
            device_type,
            hmonnx_file,
            quant_config=create_quant_config(quant_scheme),
            input_names=["input_features"],
            output_names=encoder_output_names,
        )

    return {
        "onnx_file": str(onnx_file.relative_to(work_dir)),
        "hmonnx_file": str(hmonnx_file.relative_to(work_dir)),
        "quant_type": quant_type,
        "num_mel_bins": num_mel_bins,
        "input_length": input_length,
        "audio_path": audio_path,
    }


def export_whisper_decoder_graph(
    *,
    model,
    model_dir: str,
    work_dir: Path,
    target_device: str,
    quant_type: str,
    component_name: str,
    prompt_token_ids: list[int],
    cache_position: list[int],
    past_len: int,
    model_meta_cfg: Mapping[str, Any],
    device: str,
    logger,
) -> dict[str, Any]:
    import torch

    from xhquant.api import (
        ConfigDict,
        DeviceType,
        PrecisionMode,
        QuantScheme,
        create_quant_config,
        ptq_quantize,
        to_frontend_graph,
        to_quant_graph,
    )
    from xhquant.api.ptq_export_hmonnx import convert_quanted_model_to_hmonnx
    from xhquant.core.datatype_mapping import TORCH_DTYPE_TO_FAKE_DTYPE
    from xhquant.patch.core import RewriterContext

    from ...builder import wrap_llm_model

    device_type = to_xh_device_type(target_device, DeviceType)
    model_name = Path(os.path.normpath(model_dir)).name

    component_dir = work_dir / component_name
    component_dir.mkdir(exist_ok=True, parents=True)
    hmonnx_file = (
        component_dir / "hmonnx" / f"{model_name}_{component_name}_{target_device}_{quant_type}.onnx"
    )

    num_heads = int(model_meta_cfg["num_heads"])
    head_dim = int(model_meta_cfg["head_dim"])
    embed_dim = int(model_meta_cfg["embed_dim"])
    max_source_positions = int(model_meta_cfg["max_source_positions"])
    num_decode_layers = int(model_meta_cfg["num_decode_layers"])

    payload = build_decoder_export_payload(
        prompt_token_ids=prompt_token_ids,
        cache_position=cache_position,
        past_len=past_len,
        num_heads=num_heads,
        head_dim=head_dim,
        embed_dim=embed_dim,
        max_source_positions=max_source_positions,
        num_decode_layers=num_decode_layers,
    )
    warp_inp = payload["warp_inp"]
    inputs_names = payload["inputs_names"]
    output_names = payload["output_names"]

    model_cus = Decoder(model.model, model.proj_out, config=model.config)

    if not hmonnx_file.exists():
        with RewriterContext(None, backend="onnxruntime"):
            warp_model_cus = wrap_llm_model(model_cus)
            warp_model_cus = warp_model_cus.half()
            fronted_graph_module = to_frontend_graph(warp_model_cus, "DynamoFX", warp_inp)

        quant_scheme = QuantScheme(target_device=device_type, quant_type=quant_type)
        quant_config = ConfigDict(create_quant_config(quant_scheme))
        if "inputs" not in quant_config:
            quant_config.inputs = ConfigDict()

        input_args = flatten_inputs(warp_inp)
        _input_names = fronted_graph_module.get_input_names()
        assert len(_input_names) == len(input_args), (
            f"input_names: {len(_input_names)}, input_args: {len(input_args)}"
        )
        for input_name, input_arg in zip(_input_names, input_args, strict=True):
            input_qconfig = ConfigDict(dict(quantizer=dict(qspec=dict())))
            if isinstance(input_arg, torch.Tensor):
                if input_arg.dtype in TORCH_DTYPE_TO_FAKE_DTYPE:
                    input_qconfig.quantizer.qspec.fake_dtype = (
                        TORCH_DTYPE_TO_FAKE_DTYPE[input_arg.dtype]
                    )
                else:
                    raise ValueError(f"Unsupported dtype: {input_arg.dtype}")
            quant_config.inputs[input_name] = input_qconfig

        quanted_graph_module = to_quant_graph(fronted_graph_module, DeviceType.XH2a.name, quant_config)
        execution_device = torch.device(device)
        ptq_quantize(quanted_graph_module, [input_args], PrecisionMode.ALIGNED, execution_device)
        convert_quanted_model_to_hmonnx(
            quanted_graph_module,
            warp_inp,
            hmonnx_file,
            inputs_names,
            output_names,
        )
    logger.info(f"{component_name} HMONNX saved: {hmonnx_file}")

    return {
        "hmonnx_file": str(hmonnx_file.relative_to(work_dir)),
        "quant_type": quant_type,
        "prompt_token_ids": list(prompt_token_ids),
        "cache_position": list(cache_position),
        "past_len": int(past_len),
    }
