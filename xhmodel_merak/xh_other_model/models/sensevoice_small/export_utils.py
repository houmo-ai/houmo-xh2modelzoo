# Copyright 2025 HOUMO AI
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


INPUT_NAMES = ["speech", "speech_lengths", "language", "textnorm"]
OUTPUT_NAMES = ["ctc_logits", "encoder_out_lens"]


class ScaledLayerNorm(torch.nn.Module):
    def __init__(self, original_layer_norm: torch.nn.LayerNorm, scale: float = 32.0):
        super().__init__()
        # Keep the legacy attribute name so ONNX scope names remain aligned.
        self.ln = original_layer_norm
        self.register_buffer(
            "inv_scale",
            torch.tensor(1.0 / scale, device=original_layer_norm.weight.device),
        )

    def forward(self, value: Any) -> Any:
        return self.ln(value * self.inv_scale)


def replace_layer_norm_with_scaled(model: Any, scale: float = 32.0) -> None:
    for name, module in model.named_children():
        if isinstance(module, torch.nn.LayerNorm):
            setattr(model, name, ScaledLayerNorm(module, scale))
        else:
            replace_layer_norm_with_scaled(module, scale)


def rebuild_for_onnx(
    model: Any,
    max_seq_len: int = 512,
    device: str = "cpu",
    dynamic: bool = True,
) -> Any:
    from funasr.utils.torch_function import sequence_mask

    model.device = device
    model.onnx_config = {"max_seq_len": int(max_seq_len), "dynamic": bool(dynamic)}
    model.make_pad_mask = sequence_mask(int(max_seq_len), flip=False)
    model.forward = types.MethodType(_export_forward, model)
    model.export_dummy_inputs = types.MethodType(_export_dummy_inputs, model)
    model.export_input_names = types.MethodType(_export_input_names, model)
    model.export_output_names = types.MethodType(_export_output_names, model)
    model.export_dynamic_axes = types.MethodType(_export_dynamic_axes, model)
    return model


def _export_forward(
    self: Any,
    speech: Any,
    speech_lengths: Any,
    language: Any,
    textnorm: Any,
    **kwargs: Any,
) -> tuple[Any, Any]:
    import torch

    language_query = self.embed(language.to(speech.device)).unsqueeze(1)
    textnorm_query = self.embed(textnorm.to(speech.device)).unsqueeze(1)
    speech = torch.cat((textnorm_query, speech), dim=1)
    speech_lengths = speech_lengths + 1

    event_emo_query = self.embed(torch.tensor([[1, 2]], dtype=torch.long, device=speech.device)).repeat(
        speech.size(0), 1, 1
    )
    input_query = torch.cat((language_query, event_emo_query), dim=1)
    speech = torch.cat((input_query, speech), dim=1)
    speech_lengths = speech_lengths + 3

    encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]
    return self.ctc.ctc_lo(encoder_out), encoder_out_lens


def _export_dummy_inputs(self: Any) -> tuple[Any, ...]:
    import torch

    if not self.onnx_config["dynamic"]:
        length = int(self.onnx_config["max_seq_len"])
        return (
            torch.randn(1, length, 560, dtype=torch.float32),
            torch.tensor([length], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([15], dtype=torch.int32),
        )
    return (
        torch.randn(2, 30, 560, dtype=torch.float32),
        torch.tensor([6, 30], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([15, 15], dtype=torch.int32),
    )


def _export_input_names(self: Any) -> list[str]:
    return list(INPUT_NAMES)


def _export_output_names(self: Any) -> list[str]:
    return list(OUTPUT_NAMES)


def _export_dynamic_axes(self: Any) -> dict[str, dict[int, str]] | None:
    if not self.onnx_config["dynamic"]:
        return None
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {0: "batch_size"},
        "language": {0: "batch_size"},
        "textnorm": {0: "batch_size"},
        "ctc_logits": {0: "batch_size", 1: "logits_length"},
        "encoder_out_lens": {0: "batch_size"},
    }


def export_fp32_onnx(
    *,
    model_dir: str,
    work_dir: Path,
    device: str,
    max_seq_len: int,
    opset: int,
    static: bool,
    simplify: bool,
    layer_norm_scale: float,
    verbose: bool,
    seed: int,
) -> dict[str, Any]:
    import torch
    from funasr import AutoModel

    onnx_dir = work_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_file = onnx_dir / "model.onnx"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, _ = AutoModel.build_model(model=model_dir, device=device, trust_remote_code=False)
    model.eval()
    replace_layer_norm_with_scaled(model, scale=float(layer_norm_scale))
    rebuilt_model = rebuild_for_onnx(
        model,
        max_seq_len=int(max_seq_len),
        device=str(device),
        dynamic=not static,
    )
    dummy_inputs = tuple(value.to(device) for value in rebuilt_model.export_dummy_inputs())
    torch.onnx.export(
        rebuilt_model,
        dummy_inputs,
        str(onnx_file),
        verbose=bool(verbose),
        opset_version=int(opset),
        input_names=rebuilt_model.export_input_names(),
        output_names=rebuilt_model.export_output_names(),
        dynamic_axes=rebuilt_model.export_dynamic_axes(),
    )

    simplified = False
    if static and simplify:
        import onnx
        from onnxsim import simplify as simplify_onnx

        model_proto = onnx.load(str(onnx_file))
        simplified_model, checked = simplify_onnx(model_proto)
        if not checked:
            raise RuntimeError("onnxsim validation failed for the simplified SenseVoice model")
        onnx.save(simplified_model, str(onnx_file))
        simplified = True

    input_names, output_names = onnx_io_names(onnx_file)
    input_shapes = {name: list(shape) for name, shape in static_input_shapes(onnx_file).items()}
    meta = {
        "model_dir": model_dir,
        "device": device,
        "max_seq_len": int(max_seq_len),
        "opset": int(opset),
        "static": bool(static),
        "simplified": simplified,
        "layer_norm_scale": float(layer_norm_scale),
        "onnx": str(onnx_file),
        "input_names": input_names,
        "output_names": output_names,
        "input_shapes": input_shapes,
    }
    (onnx_dir / "export_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def onnx_io_names(onnx_path: Path) -> tuple[list[str], list[str]]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = [value.name for value in model.graph.input if value.name not in initializer_names]
    outputs = [value.name for value in model.graph.output]
    return inputs, outputs


def static_input_shapes(onnx_path: Path) -> dict[str, tuple[int, ...]]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    shapes: dict[str, tuple[int, ...]] = {}
    for value in model.graph.input:
        shape: list[int] = []
        for dim in value.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                shape = []
                break
            shape.append(int(dim.dim_value))
        if shape:
            shapes[value.name] = tuple(shape)
    return shapes


def _align_calibration_tensor(name: str, value: Any, target_shapes: Mapping[str, tuple[int, ...]]) -> Any:
    import torch
    import torch.nn.functional as functional

    tensor = torch.as_tensor(value)
    tensor = tensor.to(torch.float32 if name == "speech" else torch.int32)
    target_shape = target_shapes.get(name)
    if name == "speech" and target_shape and tensor.ndim == 3 and len(target_shape) == 3:
        current_length = tensor.shape[1]
        target_length = target_shape[1]
        if current_length < target_length:
            tensor = functional.pad(tensor, (0, 0, 0, target_length - current_length))
        elif current_length > target_length:
            tensor = tensor[:, :target_length, :]
    if name == "speech_lengths":
        speech_shape = target_shapes.get("speech")
        if speech_shape:
            tensor = torch.clamp(tensor, max=int(speech_shape[1]))
    return tensor


def _calibration_from_file(
    calib_path: Path,
    input_names: Sequence[str],
    target_shapes: Mapping[str, tuple[int, ...]],
) -> list[Any]:
    import torch

    payload = torch.load(calib_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Calibration file must contain a mapping: {calib_path}")
    missing = [name for name in input_names if name not in payload]
    if missing:
        raise KeyError(f"Calibration file is missing {missing}; available fields: {sorted(payload)}")
    return [_align_calibration_tensor(name, payload[name], target_shapes) for name in input_names]


def _calibration_from_hf_dataset(
    *,
    model_dir: str,
    calibration_cfg: Mapping[str, Any],
    input_names: Sequence[str],
    target_shapes: Mapping[str, tuple[int, ...]],
) -> tuple[list[Any], int]:
    import torch
    import torch.nn.functional as functional
    from tqdm import tqdm

    from . import runtime

    sample_limit = int(calibration_cfg.get("samples", 128))
    samples = runtime.load_hf_dataset(
        dataset=str(calibration_cfg.get("hf_dataset", "openslr/librispeech_asr")),
        config=str(calibration_cfg.get("hf_config", "clean")),
        split=str(calibration_cfg.get("hf_split", "validation")),
        limit=sample_limit,
        streaming=bool(calibration_cfg.get("hf_streaming", True)),
        audio_field=str(calibration_cfg.get("hf_audio_field", "audio")),
        text_field="",
    )
    if not samples:
        raise ValueError("No calibration samples were loaded from the Hugging Face dataset")

    frontend = runtime.build_frontend(Path(model_dir))
    all_inputs: dict[str, list[Any]] = {name: [] for name in input_names}
    for sample in tqdm(samples, desc="SenseVoice calibration"):
        try:
            waveform = runtime.load_audio_any(sample, target_sr=int(frontend.cfg.fs))
            feat, feat_len = runtime.extract_features(frontend, waveform)
            inputs = runtime.make_inputs_for_sample(feat, feat_len, sample.language, sample.textnorm)
            for name in input_names:
                all_inputs[name].append(_align_calibration_tensor(name, inputs[name], target_shapes))
        except Exception as exc:
            print(f"Warning: failed to prepare calibration sample {sample.audio_id}: {exc}")

    valid_samples = len(all_inputs[input_names[0]])
    if valid_samples == 0:
        raise ValueError("All Hugging Face calibration samples failed during audio preprocessing")

    batched_inputs: list[Any] = []
    for name in input_names:
        tensors = all_inputs[name]
        if name == "speech":
            max_length = max(tensor.shape[1] for tensor in tensors)
            tensors = [
                functional.pad(tensor, (0, 0, 0, max_length - tensor.shape[1]))
                if tensor.shape[1] < max_length
                else tensor
                for tensor in tensors
            ]
        batched_inputs.append(torch.cat(tensors, dim=0))
    return batched_inputs, valid_samples


def _configure_calibration_metric(quant_config: Any, calib_metric: str) -> None:
    for key in ("w_cfg", "i_cfg"):
        if key in quant_config and "quantizer" in quant_config[key]:
            quant_config[key]["quantizer"]["calib_metric"] = calib_metric


def _convert_multi_sample(
    *,
    onnx_path: Path,
    output_path: Path,
    input_list: Sequence[Any],
    input_names: Sequence[str],
    output_names: Sequence[str],
    device_type: Any,
    quant_config: Any,
    execution_device: str,
) -> None:
    import torch

    from xhquant.api.ptq_export_hmonnx import (
        _convert_model_to_quanted_model,
        convert_quanted_model_to_hmonnx,
    )
    from xhquant.common.types import FrontendType, PrecisionMode
    from xhquant.quantization import ptq_quantize

    first_sample = [value[0:1] if isinstance(value, torch.Tensor) else value for value in input_list]
    quanted_graph_module = _convert_model_to_quanted_model(
        str(onnx_path),
        FrontendType.ONNX,
        first_sample,
        device_type,
        quant_config,
        use_ptq=False,
        input_names=list(input_names),
    )
    batch_size = int(input_list[0].shape[0])
    calibration_data = [
        [value[index : index + 1].cpu() if isinstance(value, torch.Tensor) else value for value in input_list]
        for index in range(batch_size)
    ]
    target_device = torch.device(execution_device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        target_device = torch.device("cpu")
    ptq_quantize(quanted_graph_module, calibration_data, PrecisionMode.ALIGNED, target_device)
    convert_quanted_model_to_hmonnx(
        quanted_graph_module,
        first_sample,
        str(output_path),
        list(input_names),
        list(output_names),
    )


def export_hmonnx(
    *,
    onnx_path: Path,
    model_dir: str,
    work_dir: Path,
    target_device: str,
    quant_type: str,
    calib_metric: str,
    force_fp32_ops: Sequence[str],
    calibration_cfg: Mapping[str, Any],
    execution_device: str,
    debug: bool,
) -> dict[str, Any]:
    from xhquant.api import (
        DeviceType,
        QuantScheme,
        convert_onnx_to_hmonnx,
        create_quant_config,
        get_root_logger,
        xhquant_init,
    )

    if target_device != "XH2a":
        raise ValueError(f"SenseVoiceSmall currently supports target_device='XH2a', got {target_device!r}")
    if calib_metric not in {"minmax", "mse", "kl"}:
        raise ValueError(f"Unsupported calibration metric: {calib_metric!r}")
    if isinstance(force_fp32_ops, (str, bytes)):
        raise TypeError("export.hmonnx.force_fp32_ops must be a list of operator type names")

    input_names, output_names = onnx_io_names(onnx_path)
    target_shapes = static_input_shapes(onnx_path)
    if "speech" not in target_shapes:
        raise ValueError("SenseVoice HMONNX export requires a static ONNX speech input")

    xhquant_init(str(work_dir / "convert.log"), debug=bool(debug))
    device_type = DeviceType.XH2a
    quant_scheme = QuantScheme(target_device=device_type, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    for op_type in force_fp32_ops:
        quant_config.ops_cfg[str(op_type)] = {"force_fp32": True}
    _configure_calibration_metric(quant_config, calib_metric)

    calibration_source = str(calibration_cfg.get("source", "hf_dataset"))
    if calibration_source == "file":
        value = calibration_cfg.get("file")
        if not value:
            raise ValueError("export.hmonnx.calibration.file is required when source='file'")
        calib_file = Path(str(value)).expanduser().resolve()
        input_list = _calibration_from_file(calib_file, input_names, target_shapes)
        valid_samples = int(input_list[0].shape[0])
    elif calibration_source == "hf_dataset":
        if int(calibration_cfg.get("samples", 128)) <= 0:
            raise ValueError("export.hmonnx.calibration.samples must be greater than zero")
        input_list, valid_samples = _calibration_from_hf_dataset(
            model_dir=model_dir,
            calibration_cfg=calibration_cfg,
            input_names=input_names,
            target_shapes=target_shapes,
        )
    else:
        raise ValueError(f"Unsupported calibration source {calibration_source!r}; expected 'hf_dataset' or 'file'")

    hmonnx_dir = work_dir / "hmonnx"
    hmonnx_dir.mkdir(parents=True, exist_ok=True)
    output_path = hmonnx_dir / f"sensevoice_small_{target_device}_{quant_type}.onnx"
    batch_size = int(input_list[0].shape[0]) if hasattr(input_list[0], "shape") else 1
    if batch_size > 1:
        _convert_multi_sample(
            onnx_path=onnx_path,
            output_path=output_path,
            input_list=input_list,
            input_names=input_names,
            output_names=output_names,
            device_type=device_type,
            quant_config=quant_config,
            execution_device=execution_device,
        )
    else:
        convert_onnx_to_hmonnx(
            str(onnx_path),
            list(input_list),
            device_type,
            str(output_path),
            quant_config=quant_config,
            input_names=input_names,
            output_names=output_names,
        )

    get_root_logger().info(f"SenseVoice HMONNX saved: {output_path}")
    return {
        "hmonnx": str(output_path),
        "quant_type": quant_type,
        "calib_metric": calib_metric,
        "force_fp32_ops": list(force_fp32_ops),
        "calibration_source": calibration_source,
        "calibration_samples": valid_samples,
        "calibration": dict(calibration_cfg),
        "input_names": input_names,
        "output_names": output_names,
    }


__all__ = [
    "INPUT_NAMES",
    "OUTPUT_NAMES",
    "export_fp32_onnx",
    "export_hmonnx",
    "onnx_io_names",
    "rebuild_for_onnx",
    "replace_layer_norm_with_scaled",
    "static_input_shapes",
]
