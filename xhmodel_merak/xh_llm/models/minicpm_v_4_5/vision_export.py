"""ONNX/HMONNX export for the MiniCPM-V-4.5 static-capacity Vision graph."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .vision import (
    DEFAULT_GROUP_CAPACITY,
    build_resampler_pos_embed_cache,
    build_resampler_temporal_pos_embed_cache,
    build_video_group_vision_export_model,
    build_vision_export_model,
    prepare_video_group_inputs,
    prepare_vision_inputs,
    validate_patch_capacity,
)


VISION_INPUT_NAMES = (
    "pixel_values",
    "position_ids",
    "attention_bias",
    "resampler_pos_embed",
    "resampler_key_bias",
)

VIDEO_GROUP_INPUT_NAMES = (
    "pixel_values",
    "position_ids",
    "attention_bias",
    "resampler_pos_embed",
    "resampler_temporal_pos_embed",
    "resampler_key_bias",
)


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


def _export_onnx(
    model,
    model_inputs,
    output_path: Path,
    input_names: tuple[str, ...],
) -> None:
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
            input_names=list(input_names),
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


def _validate_onnx(
    onnx_path: Path,
    model_inputs,
    reference,
    input_names: tuple[str, ...],
) -> tuple[float, float]:
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {name: value.cpu().numpy() for name, value in zip(input_names, model_inputs, strict=True)}
    (output,) = session.run(None, feed)
    actual = torch.from_numpy(output)
    max_abs = (reference.float().cpu() - actual.float()).abs().max().item()
    return max_abs, _cosine_similarity(reference.cpu(), actual)


def _convert_hmonnx(
    onnx_path: Path,
    hmonnx_path: Path,
    model_inputs,
    input_names: tuple[str, ...],
    output_name: str,
    target_device: str,
    quant_scheme_config: Mapping[str, Any],
) -> None:
    import torch

    from xhquant.api import (
        ConfigDict,
        DeviceType,
        QuantScheme,
        convert_onnx_to_hmonnx,
        create_quant_config,
    )

    device_type = _device_type(target_device, DeviceType)
    quant_scheme = _build_quant_scheme(QuantScheme, device_type, quant_scheme_config)
    quant_config = ConfigDict(create_quant_config(quant_scheme))

    def float_input_qspec() -> dict[str, Any]:
        return {"quantizer": {"qspec": {"fake_dtype": "float16"}}}

    quant_inputs = {
        name: float_input_qspec()
        for name, value in zip(input_names, model_inputs, strict=True)
        if value.dtype in (torch.float16, torch.float32, torch.float64)
    }
    quant_config.inputs = ConfigDict(quant_inputs)

    hmonnx_path.parent.mkdir(parents=True, exist_ok=True)
    convert_onnx_to_hmonnx(
        str(onnx_path),
        [value.cpu() for value in model_inputs],
        device_type,
        str(hmonnx_path),
        quant_config=quant_config,
        input_names=list(input_names),
        output_names=[output_name],
    )


def _build_quant_scheme(quant_scheme_cls, device_type, quant_scheme_config: Mapping[str, Any]):
    """Build a converter QuantScheme without dropping advanced overrides."""
    if not isinstance(quant_scheme_config, Mapping):
        raise TypeError("vision quant_scheme must be a mapping")
    config = dict(quant_scheme_config)
    quant_type = config.get("quant_type")
    if not quant_type:
        raise ValueError("vision quant_scheme must define quant_type")
    allowed = {
        "nodes",
        "ops",
        "input_ppc_config",
        "input_enable_fp32",
        "output_enable_fp32",
        "w_scheme",
        "act_scheme",
    }
    kwargs = {key: config[key] for key in allowed if key in config}
    return quant_scheme_cls(
        target_device=device_type,
        quant_type=str(quant_type),
        **kwargs,
    )


def _validate_hmonnx(
    hmonnx_path: Path,
    model_inputs,
    native_reference,
    *,
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
    actual = output.float().cpu()
    max_abs = (native_reference.float().cpu() - actual).abs().max().item()
    return max_abs, _cosine_similarity(native_reference.cpu(), actual)


def export_vision_profile(
    *,
    model_dir: str,
    work_dir: Path,
    target_device: str,
    quant_scheme: Mapping[str, Any],
    patch_capacity: int,
    validation_target: tuple[int, int],
    device: str,
    keep_onnx: bool = False,
    artifact_stem: str | None = None,
    profile_dir: Path | None = None,
) -> dict[str, Any]:
    """Export and validate one static-capacity Vision profile.

    ``artifact_stem`` defaults to the legacy per-profile name; pass the release
    prefix (``hmquant_<prefix>_visual``) for convention-compliant artifacts.
    ``profile_dir`` defaults to ``work_dir/vision``.
    """

    import torch
    from transformers import AutoModel

    from xhquant.api import get_root_logger

    validate_patch_capacity(validation_target, patch_capacity)
    logger = get_root_logger()
    if profile_dir is None:
        profile_dir = work_dir / "vision"
    profile_dir.mkdir(parents=True, exist_ok=True)
    quant_type = str(quant_scheme["quant_type"])
    if artifact_stem is None:
        artifact_stem = f"MiniCPM-V-4.5_Vision_patches{patch_capacity}_{target_device}_{quant_type}"
    onnx_path = profile_dir / "onnx" / f"{artifact_stem}.onnx"
    hmonnx_path = profile_dir / f"{artifact_stem}.onnx"

    full_model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    positions_per_side = int(full_model.vpm.embeddings.num_patches_per_side)
    embed_dim = int(full_model.resampler.embed_dim)
    pos_embed_cache = full_model.resampler.pos_embed
    computed_cache = build_resampler_pos_embed_cache(
        embed_dim,
        positions_per_side,
        device=pos_embed_cache.device,
    )
    if tuple(computed_cache.shape) != tuple(pos_embed_cache.shape) or not torch.allclose(
        computed_cache,
        pos_embed_cache,
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Deterministic resampler pos-embed cache does not match the checkpoint cache: "
            f"{tuple(computed_cache.shape)} vs {tuple(pos_embed_cache.shape)}"
        )
    wrapper = build_vision_export_model(full_model, patch_capacity)

    height, width = validation_target
    token_count = height * width
    pixel_slice = torch.randn(
        3,
        14,
        token_count * 14,
        dtype=torch.float32,
    )
    prepared_inputs = prepare_vision_inputs(
        pixel_slice,
        validation_target,
        patch_capacity,
        positions_per_side,
        pos_embed_cache,
        dtype=torch.float32,
    )

    torch_device = torch.device(device)
    full_model.to(torch_device)
    wrapper.to(torch_device)
    with torch.inference_mode():
        capacity_reference = wrapper(*(value.to(torch_device) for value in prepared_inputs)).cpu()
        native_features = native_vision_features(
            full_model,
            pixel_slice.to(torch_device),
            validation_target,
            torch_device,
        )

    native_max_abs = (capacity_reference.float() - native_features.float()).abs().max().item()
    native_cosine = _cosine_similarity(capacity_reference, native_features)
    if native_cosine < 0.999999:
        raise RuntimeError(f"Static-capacity wrapper/native mismatch: cosine={native_cosine}, max_abs={native_max_abs}")
    logger.info(
        "MiniCPM Vision wrapper/native: max_abs=%.8e cosine=%.10f",
        native_max_abs,
        native_cosine,
    )

    del full_model
    _export_onnx(wrapper, prepared_inputs, onnx_path, VISION_INPUT_NAMES)
    onnx_max_abs, onnx_cosine = _validate_onnx(
        onnx_path,
        prepared_inputs,
        capacity_reference,
        VISION_INPUT_NAMES,
    )
    _convert_hmonnx(
        onnx_path,
        hmonnx_path,
        prepared_inputs,
        VISION_INPUT_NAMES,
        "image_embeds",
        target_device,
        quant_scheme,
    )
    hmonnx_max_abs, hmonnx_cosine = _validate_hmonnx(
        hmonnx_path,
        prepared_inputs,
        native_features,
        device=device,
    )
    # 发布规范：正式 HMONNX 文件名应为 <stem>_with_act.onnx。
    # converter 可能额外写出同内容的 <stem>_with_act.onnx 副本——统一为
    # _with_act 名并删除多余副本（external_data 引用名不变，不手动改 external）。
    hmonnx_path = _normalize_hmonnx_name(hmonnx_path)

    result = {
        "patch_capacity": patch_capacity,
        "positions_per_side": positions_per_side,
        "embed_dim": embed_dim,
        "quant_type": quant_type,
        "quant_scheme": dict(quant_scheme),
        "hmonnx": str(hmonnx_path.relative_to(work_dir)),
        "input_names": list(VISION_INPUT_NAMES),
        "input_shapes": {
            name: list(value.shape) for name, value in zip(VISION_INPUT_NAMES, prepared_inputs, strict=True)
        },
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


def _normalize_hmonnx_name(hmonnx_path: Path) -> Path:
    """Rename the HMONNX artifact to ``<stem>_with_act.onnx`` (release naming).

    The converter writes to ``hmonnx_path`` and may additionally emit a
    same-content ``<stem>_with_act.onnx`` copy.  The release convention expects
    the formal file to carry the ``_with_act`` suffix; external_data keeps its
    original name and is never renamed manually.
    """
    import os

    if hmonnx_path.name.endswith("_with_act.onnx"):
        return hmonnx_path
    with_act = hmonnx_path.with_name(hmonnx_path.stem + "_with_act.onnx")
    if with_act.exists():
        with_act.unlink()
    os.replace(hmonnx_path, with_act)
    return with_act


def native_vision_features(
    full_model,
    pixel_slice,
    target_size: tuple[int, int],
    device,
) -> Any:
    """Compute the native per-slice SigLIP2 + Resampler reference output."""
    import torch

    height, width = target_size
    token_count = height * width
    pixel_values = pixel_slice.unsqueeze(0).to(device)
    patch_attn_mask = torch.zeros(
        (1, 1, token_count),
        dtype=torch.bool,
        device=device,
    )
    patch_attn_mask[0, 0, :] = True
    tgt_sizes = torch.tensor([[height, width]], dtype=torch.int32, device=device)
    hidden_states = full_model.vpm(
        pixel_values,
        patch_attention_mask=patch_attn_mask,
        tgt_sizes=tgt_sizes,
    ).last_hidden_state
    return full_model.resampler(hidden_states, tgt_sizes=tgt_sizes).cpu()


def export_video_group_profile(
    *,
    model_dir: str,
    work_dir: Path,
    target_device: str,
    quant_scheme: Mapping[str, Any],
    patch_capacity: int,
    group_capacity: int = DEFAULT_GROUP_CAPACITY,
    validation_target: tuple[int, int] = (32, 32),
    validation_frames: int = 3,
    device: str,
    keep_onnx: bool = False,
    artifact_stem: str | None = None,
    profile_dir: Path | None = None,
) -> dict[str, Any]:
    """Export and validate the temporal-group Vision profile (video path).

    ``artifact_stem`` defaults to the legacy per-profile name; pass the release
    prefix (``hmquant_<prefix>_visual``) for convention-compliant artifacts.
    ``profile_dir`` defaults to ``work_dir/vision_video``.
    """

    import torch
    from transformers import AutoModel

    from xhquant.api import get_root_logger

    validate_patch_capacity(validation_target, patch_capacity)
    if not (1 <= validation_frames <= group_capacity):
        raise ValueError(f"validation_frames must be in [1, group_capacity={group_capacity}]")
    logger = get_root_logger()
    if profile_dir is None:
        profile_dir = work_dir / "vision_video"
    profile_dir.mkdir(parents=True, exist_ok=True)
    quant_type = str(quant_scheme["quant_type"])
    if artifact_stem is None:
        artifact_stem = (
            f"MiniCPM-V-4.5_VisionVideo_groups{group_capacity}_patches{patch_capacity}_{target_device}_{quant_type}"
        )
    onnx_path = profile_dir / "onnx" / f"{artifact_stem}.onnx"
    hmonnx_path = profile_dir / f"{artifact_stem}.onnx"

    full_model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    positions_per_side = int(full_model.vpm.embeddings.num_patches_per_side)
    embed_dim = int(full_model.resampler.embed_dim)
    pos_embed_cache = full_model.resampler.pos_embed
    computed_cache = build_resampler_pos_embed_cache(
        embed_dim,
        positions_per_side,
        device=pos_embed_cache.device,
    )
    if tuple(computed_cache.shape) != tuple(pos_embed_cache.shape) or not torch.allclose(
        computed_cache,
        pos_embed_cache,
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Deterministic resampler pos-embed cache does not match the checkpoint cache: "
            f"{tuple(computed_cache.shape)} vs {tuple(pos_embed_cache.shape)}"
        )
    temporal_cache = full_model.resampler.temporal_pos_embed
    computed_temporal_cache = build_resampler_temporal_pos_embed_cache(
        embed_dim,
        int(temporal_cache.shape[0]),
        device=temporal_cache.device,
    )
    if tuple(computed_temporal_cache.shape) != tuple(temporal_cache.shape) or not torch.allclose(
        computed_temporal_cache,
        temporal_cache,
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Deterministic resampler temporal-pos-embed cache does not match the checkpoint "
            f"cache: {tuple(computed_temporal_cache.shape)} vs {tuple(temporal_cache.shape)}"
        )
    wrapper = build_video_group_vision_export_model(
        full_model,
        patch_capacity,
        group_capacity,
    )

    height, width = validation_target
    token_count = height * width
    temporal_ids = list(range(validation_frames))
    pixel_slices = [torch.randn(3, 14, token_count * 14, dtype=torch.float32) for _ in range(validation_frames)]
    tgt_sizes = [validation_target] * validation_frames
    prepared_inputs = prepare_video_group_inputs(
        pixel_slices,
        tgt_sizes,
        temporal_ids,
        patch_capacity,
        group_capacity,
        positions_per_side,
        pos_embed_cache,
        temporal_cache,
        dtype=torch.float32,
    )

    torch_device = torch.device(device)
    full_model.to(torch_device)
    wrapper.to(torch_device)
    with torch.inference_mode():
        capacity_reference = wrapper(*(value.to(torch_device) for value in prepared_inputs)).cpu()
        native_features = native_video_group_features(
            full_model,
            pixel_slices,
            tgt_sizes,
            temporal_ids,
            torch_device,
        )

    native_max_abs = (capacity_reference.float() - native_features.float()).abs().max().item()
    native_cosine = _cosine_similarity(capacity_reference, native_features)
    if native_cosine < 0.999999:
        raise RuntimeError(f"Video-group wrapper/native mismatch: cosine={native_cosine}, max_abs={native_max_abs}")
    logger.info(
        "MiniCPM VisionVideo wrapper/native: max_abs=%.8e cosine=%.10f",
        native_max_abs,
        native_cosine,
    )

    del full_model
    _export_onnx(wrapper, prepared_inputs, onnx_path, VIDEO_GROUP_INPUT_NAMES)
    onnx_max_abs, onnx_cosine = _validate_onnx(
        onnx_path,
        prepared_inputs,
        capacity_reference,
        VIDEO_GROUP_INPUT_NAMES,
    )
    _convert_hmonnx(
        onnx_path,
        hmonnx_path,
        prepared_inputs,
        VIDEO_GROUP_INPUT_NAMES,
        "image_embeds",
        target_device,
        quant_scheme,
    )
    hmonnx_max_abs, hmonnx_cosine = _validate_hmonnx(
        hmonnx_path,
        prepared_inputs,
        native_features,
        device=device,
    )
    # 发布规范：正式 HMONNX 文件名应为 <stem>_with_act.onnx。
    hmonnx_path = _normalize_hmonnx_name(hmonnx_path)

    result = {
        "patch_capacity": patch_capacity,
        "group_capacity": group_capacity,
        "positions_per_side": positions_per_side,
        "embed_dim": embed_dim,
        "quant_type": quant_type,
        "quant_scheme": dict(quant_scheme),
        "hmonnx": str(hmonnx_path.relative_to(work_dir)),
        "input_names": list(VIDEO_GROUP_INPUT_NAMES),
        "input_shapes": {
            name: list(value.shape) for name, value in zip(VIDEO_GROUP_INPUT_NAMES, prepared_inputs, strict=True)
        },
        "output_shape": list(capacity_reference.shape),
        "validation_target_size": list(validation_target),
        "validation_frames": validation_frames,
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


def native_video_group_features(
    full_model,
    pixel_slices: list,
    tgt_sizes: list[tuple[int, int]],
    temporal_ids: list[int],
    device,
) -> Any:
    """Native batched vpm + merged Resampler reference for one temporal group."""
    import torch

    frame_count = len(pixel_slices)
    token_counts = [h * w for h, w in tgt_sizes]
    max_tokens = max(token_counts)
    pixel_values = pixel_slices[0].new_zeros((frame_count, 3, 14, max_tokens * 14))
    for index, (pixel_slice, token_count) in enumerate(zip(pixel_slices, token_counts, strict=True)):
        pixel_values[index, :, :, : token_count * 14] = pixel_slice[:, :, : token_count * 14]
    patch_attn_mask = torch.zeros(
        (frame_count, 1, max_tokens),
        dtype=torch.bool,
        device=device,
    )
    for index, token_count in enumerate(token_counts):
        patch_attn_mask[index, 0, :token_count] = True
    tgt = torch.tensor(
        [[h, w] for h, w in tgt_sizes],
        dtype=torch.int32,
        device=device,
    )
    hidden_states = full_model.vpm(
        pixel_values.to(device),
        patch_attention_mask=patch_attn_mask,
        tgt_sizes=tgt,
    ).last_hidden_state
    return full_model.resampler(
        hidden_states,
        tgt_sizes=tgt,
        temporal_ids=[temporal_ids],
    ).cpu()


__all__ = [
    "VISION_INPUT_NAMES",
    "VIDEO_GROUP_INPUT_NAMES",
    "export_vision_profile",
    "export_video_group_profile",
    "native_vision_features",
    "native_video_group_features",
]
