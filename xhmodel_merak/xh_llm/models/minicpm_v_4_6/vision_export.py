"""ONNX/HMONNX export for MiniCPM-V-4.6 token-capacity Vision graphs."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from .vision import (
    build_token_capacity_vision_export_model,
    crop_token_capacity_image_embeds,
    prepare_token_capacity_vision_inputs,
    validate_token_capacity,
)


BASE_INPUT_NAMES = ("pixel_values", "position_ids", "pre_attention_bias")


def vision_input_names(downsample_mode: str) -> tuple[str, ...]:
    if downsample_mode == "16x":
        return (*BASE_INPUT_NAMES, "post_attention_bias")
    if downsample_mode == "4x":
        return BASE_INPUT_NAMES
    raise ValueError(f"Unsupported downsample mode: {downsample_mode}")


def _device_type(target_device: str, device_type_cls):
    normalized = target_device.lower()
    for member in device_type_cls:
        if member.name.lower() == normalized or str(member.value).lower() == normalized:
            return member
    raise ValueError(f"Unsupported target device: {target_device}")


def _cosine_similarity(lhs, rhs) -> float:
    import torch

    return torch.nn.functional.cosine_similarity(
        lhs.float().reshape(-1),
        rhs.float().reshape(-1),
        dim=0,
    ).item()


def _export_onnx(model, model_inputs, output_path: Path, names: tuple[str, ...]) -> None:
    import onnx
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.float().eval().cpu()
    cpu_inputs = tuple(
        value.cpu() if value.dtype in (torch.int32, torch.int64) else value.float().cpu() for value in model_inputs
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / output_path.name
        torch.onnx.export(
            model,
            cpu_inputs,
            str(temp_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=list(names),
            output_names=["image_embeds"],
            dynamo=False,
        )
        onnx_model = onnx.load(str(temp_path), load_external_data=True)

    external_name = f"{output_path.stem}_external_data"
    onnx.save(
        onnx_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_name,
        size_threshold=1024,
        convert_attribute=True,
    )


def _validate_onnx(onnx_path: Path, model_inputs, reference, names: tuple[str, ...]) -> tuple[float, float]:
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {name: value.cpu().numpy() for name, value in zip(names, model_inputs, strict=True)}
    (output,) = session.run(None, feed)
    actual = torch.from_numpy(output)
    max_abs = (reference.float().cpu() - actual.float()).abs().max().item()
    return max_abs, _cosine_similarity(reference.cpu(), actual)


def _convert_hmonnx(
    onnx_path: Path,
    hmonnx_path: Path,
    model_inputs,
    names: tuple[str, ...],
    target_device: str,
    quant_type: str,
) -> None:
    from xhquant.api import (
        ConfigDict,
        DeviceType,
        QuantScheme,
        convert_onnx_to_hmonnx,
        create_quant_config,
    )

    device_type = _device_type(target_device, DeviceType)
    quant_scheme = QuantScheme(target_device=device_type, quant_type=quant_type)
    quant_config = ConfigDict(create_quant_config(quant_scheme))

    def float_input_qspec() -> dict[str, Any]:
        return {"quantizer": {"qspec": {"fake_dtype": "float16"}}}

    quant_inputs = {
        "pixel_values": float_input_qspec(),
        "pre_attention_bias": float_input_qspec(),
    }
    if "post_attention_bias" in names:
        quant_inputs["post_attention_bias"] = float_input_qspec()
    quant_config.inputs = ConfigDict(quant_inputs)

    hmonnx_path.parent.mkdir(parents=True, exist_ok=True)
    convert_onnx_to_hmonnx(
        str(onnx_path),
        [value.cpu() for value in model_inputs],
        device_type,
        str(hmonnx_path),
        quant_config=quant_config,
        input_names=list(names),
        output_names=["image_embeds"],
    )


def _validate_hmonnx(
    hmonnx_path: Path,
    model_inputs,
    native_reference,
    *,
    target_size: tuple[int, int],
    token_capacity: int,
    downsample_mode: str,
    device: str,
) -> tuple[float, float]:
    import torch

    from xhquant.api import HMONNXInference

    torch_device = torch.device(device)
    session = HMONNXInference(str(hmonnx_path))
    session.exec_device = torch_device
    session.to(torch_device)
    runtime_inputs = tuple(
        value.to(torch_device)
        if value.dtype in (torch.int32, torch.int64)
        else value.to(device=torch_device, dtype=torch.float16)
        for value in model_inputs
    )
    output = session(*runtime_inputs)
    output = output[0] if isinstance(output, (tuple, list)) else output
    actual = (
        crop_token_capacity_image_embeds(
            output,
            target_size,
            token_capacity,
            downsample_mode,
        )
        .float()
        .cpu()
    )
    max_abs = (native_reference.float().cpu() - actual).abs().max().item()
    return max_abs, _cosine_similarity(native_reference.cpu(), actual)


def export_vision_profile(
    *,
    model_dir: str,
    work_dir: Path,
    target_device: str,
    quant_type: str,
    token_capacity: int,
    downsample_mode: str,
    validation_target: tuple[int, int],
    device: str,
    keep_onnx: bool = False,
) -> dict[str, Any]:
    """Export and validate one static token-capacity Vision profile."""

    import torch
    from transformers import AutoModelForImageTextToText

    from xhquant.api import get_root_logger

    validate_token_capacity(validation_target, token_capacity, downsample_mode)
    logger = get_root_logger()
    profile_dir = work_dir / f"vision_{downsample_mode}"
    profile_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = f"MiniCPM-V-4.6_Vision_tokens{token_capacity}_{downsample_mode}_{target_device}_{quant_type}"
    onnx_path = profile_dir / "onnx" / f"{artifact_stem}.onnx"
    hmonnx_path = profile_dir / "hmonnx" / f"{artifact_stem}.onnx"

    full_model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    positions_per_side = int(full_model.model.vision_tower.embeddings.num_patches_per_side)
    wrapper = build_token_capacity_vision_export_model(
        full_model,
        token_capacity,
        downsample_mode,
    )

    height, width = validation_target
    pixel_values = torch.randn(
        1,
        3,
        14,
        height * width * 14,
        dtype=torch.float32,
    )
    target_sizes = torch.tensor([validation_target], dtype=torch.int32)
    prepared_inputs = prepare_token_capacity_vision_inputs(
        pixel_values,
        validation_target,
        token_capacity,
        downsample_mode,
        positions_per_side,
    )
    names = vision_input_names(downsample_mode)
    model_inputs = prepared_inputs if downsample_mode == "16x" else prepared_inputs[:3]

    torch_device = torch.device(device)
    full_model.to(torch_device)
    wrapper.to(torch_device)
    with torch.inference_mode():
        capacity_reference = wrapper(*(value.to(torch_device) for value in model_inputs)).cpu()
        cropped_reference = crop_token_capacity_image_embeds(
            capacity_reference,
            validation_target,
            token_capacity,
            downsample_mode,
        )
        native_features = full_model.get_image_features(
            pixel_values.to(torch_device),
            target_sizes.to(torch_device),
            downsample_mode=downsample_mode,
        ).pooler_output
        native_features = torch.cat(native_features, dim=0).unsqueeze(0).cpu()

    native_max_abs = (cropped_reference.float() - native_features.float()).abs().max().item()
    native_cosine = _cosine_similarity(cropped_reference, native_features)
    if native_cosine < 0.999999:
        raise RuntimeError(f"Token-capacity wrapper/native mismatch: cosine={native_cosine}, max_abs={native_max_abs}")
    logger.info(
        "MiniCPM Vision %s wrapper/native: max_abs=%.8e cosine=%.10f",
        downsample_mode,
        native_max_abs,
        native_cosine,
    )

    del full_model
    _export_onnx(wrapper, model_inputs, onnx_path, names)
    onnx_max_abs, onnx_cosine = _validate_onnx(
        onnx_path,
        model_inputs,
        capacity_reference,
        names,
    )
    _convert_hmonnx(
        onnx_path,
        hmonnx_path,
        model_inputs,
        names,
        target_device,
        quant_type,
    )
    hmonnx_max_abs, hmonnx_cosine = _validate_hmonnx(
        hmonnx_path,
        model_inputs,
        native_features,
        target_size=validation_target,
        token_capacity=token_capacity,
        downsample_mode=downsample_mode,
        device=device,
    )

    result = {
        "downsample_mode": downsample_mode,
        "token_capacity": token_capacity,
        "positions_per_side": positions_per_side,
        "quant_type": quant_type,
        "hmonnx": str(hmonnx_path.relative_to(work_dir)),
        "input_names": list(names),
        "input_shapes": {name: list(value.shape) for name, value in zip(names, model_inputs, strict=True)},
        "output_shape": list(capacity_reference.shape),
        "validation_target_size": list(validation_target),
        "validation": {
            "native_wrapper_max_abs_error": native_max_abs,
            "native_wrapper_cosine": native_cosine,
            "onnx_max_abs_error": onnx_max_abs,
            "onnx_cosine": onnx_cosine,
            "hmonnx_native_max_abs_error": hmonnx_max_abs,
            "hmonnx_native_cosine": hmonnx_cosine,
        },
    }
    if keep_onnx:
        result["onnx"] = str(onnx_path.relative_to(work_dir))
    else:
        shutil.rmtree(onnx_path.parent)
    return result


__all__ = ["export_vision_profile", "vision_input_names"]
