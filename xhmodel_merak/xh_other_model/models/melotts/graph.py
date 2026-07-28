from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper, shape_inference
from onnxsim import simplify

from .modeling import (
    EncoderDurationStatic,
    FlowMaskedDecoderStatic,
    load_official_model,
    prepare_decoder_inputs,
)


def _static_shape(
    value_info: onnx.ValueInfoProto,
) -> tuple[int, ...] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dimensions = tensor_type.shape.dim
    if any(not dimension.HasField("dim_value") for dimension in dimensions):
        return None
    return tuple(int(dimension.dim_value) for dimension in dimensions)


def replace_static_constant_of_shape(model: onnx.ModelProto) -> int:
    """Replace statically-known ConstantOfShape nodes unsupported by xhquant."""

    inferred = shape_inference.infer_shapes(model)
    model.CopyFrom(inferred)
    value_info = {
        item.name: item for item in (list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info))
    }
    replacements: dict[int, onnx.NodeProto] = {}
    for index, node in enumerate(model.graph.node):
        if node.op_type != "ConstantOfShape":
            continue
        shape = _static_shape(value_info[node.output[0]])
        if shape is None:
            continue
        value = np.asarray(0, dtype=np.float32)
        for attribute in node.attribute:
            if attribute.name == "value":
                value = numpy_helper.to_array(attribute.t)
                break
        tensor = numpy_helper.from_array(np.full(shape, value.reshape(-1)[0], dtype=value.dtype))
        replacements[index] = helper.make_node(
            "Constant",
            inputs=[],
            outputs=list(node.output),
            name=node.name,
            value=tensor,
        )
    if replacements:
        nodes = [replacements.get(index, node) for index, node in enumerate(model.graph.node)]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    return len(replacements)


def materialize_axes_constants(model: onnx.ModelProto) -> int:
    """Turn Unsqueeze/Squeeze axes Constant nodes into initializers.

    ONNX Runtime accepts either representation. xhquant 0.4's parser requires
    the second input to be a graphsurgeon Constant, which an initializer
    guarantees.
    """

    producer = {output: node for node in model.graph.node for output in node.output}
    target_names = {
        node.input[1] for node in model.graph.node if node.op_type in {"Unsqueeze", "Squeeze"} and len(node.input) > 1
    }
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    remove_ids: set[int] = set()
    count = 0
    for name in sorted(target_names):
        if name in initializer_names:
            continue
        node = producer.get(name)
        if node is None or node.op_type != "Constant":
            continue
        attribute = next((item for item in node.attribute if item.name == "value"), None)
        if attribute is None:
            raise ValueError(f"axes Constant {node.name} has no tensor value")
        tensor = onnx.TensorProto()
        tensor.CopyFrom(attribute.t)
        tensor.name = name
        model.graph.initializer.append(tensor)
        initializer_names.add(name)
        remove_ids.add(id(node))
        count += 1
    if remove_ids:
        nodes = [node for node in model.graph.node if id(node) not in remove_ids]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    return count


def _add_metadata(
    model: onnx.ModelProto,
    *,
    graph_role: str,
    lmax: int,
    tmax: int,
    hop_length: int,
) -> None:
    metadata = {
        "model_type": "melo-vits-v2-static-mask-aware",
        "graph_role": graph_role,
        "batch_size": "1",
        "text_max_length": str(lmax),
        "acoustic_max_length": str(tmax),
        "hop_length": str(hop_length),
        "sample_rate": "44100",
        "padding_contract": "right-padding; mask is contiguous ones then zeros",
        "duration_contract": "CPU: ceil(exp(logw)*x_mask*length_scale)",
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = value


def _fix_output_shapes(
    model: onnx.ModelProto,
    *,
    role: str,
    lmax: int,
    tmax: int,
    hop_length: int,
) -> None:
    if role == "encoder_duration":
        expected = {
            "m_p": [1, 192, lmax],
            "logs_p": [1, 192, lmax],
            "logw": [1, 1, lmax],
            "x_mask": [1, 1, lmax],
            "g": [1, 256, 1],
        }
    elif role == "inverse_flow_masked_generator":
        expected = {"y": [1, 1, tmax * hop_length]}
    else:
        raise ValueError(f"unknown graph role: {role}")
    for value in model.graph.output:
        if value.name not in expected:
            raise RuntimeError(f"unexpected {role} output: {value.name}")
        dimensions = value.type.tensor_type.shape
        del dimensions.dim[:]
        for size in expected[value.name]:
            dimensions.dim.add().dim_value = int(size)


def _finalize(
    path: Path,
    *,
    role: str,
    lmax: int,
    tmax: int,
    hop_length: int,
) -> dict[str, int | bool]:
    model = onnx.load(path)
    axes_materialized = materialize_axes_constants(model)
    model, check = simplify(model, check_n=3)
    if not check:
        raise RuntimeError(f"onnxsim equivalence check failed: {path}")
    replaced = replace_static_constant_of_shape(model)
    axes_materialized += materialize_axes_constants(model)
    _add_metadata(
        model,
        graph_role=role,
        lmax=lmax,
        tmax=tmax,
        hop_length=hop_length,
    )
    _fix_output_shapes(
        model,
        role=role,
        lmax=lmax,
        tmax=tmax,
        hop_length=hop_length,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return {
        "onnxsim_checked": True,
        "axes_materialized": axes_materialized,
        "constant_of_shape_replaced": replaced,
    }


def _run_ort(
    session: ort.InferenceSession,
    feed: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    types = {value.name: value.type for value in session.get_inputs()}
    arrays = {}
    for name, tensor in feed.items():
        if types[name] == "tensor(int32)":
            tensor = tensor.to(torch.int32)
        elif types[name] == "tensor(int64)":
            tensor = tensor.to(torch.int64)
        else:
            tensor = tensor.to(torch.float32)
        arrays[name] = tensor.detach().cpu().numpy()
    return [torch.from_numpy(value) for value in session.run(None, arrays)]


def _max_abs(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return float(torch.max(torch.abs(reference.float() - actual.float())))


def _dynamic_encoder(
    model: torch.nn.Module,
    x: torch.Tensor,
    tones: torch.Tensor,
    sid: torch.Tensor,
    lang_id: int,
) -> tuple[torch.Tensor, ...]:
    length = x.shape[1]
    x_lengths = torch.tensor([length], dtype=torch.long)
    language = (torch.arange(length).remainder(2).view(1, -1) * int(lang_id)).long()
    g = model.emb_g(sid).unsqueeze(-1)
    hidden, m_p, logs_p, x_mask = model.enc_p(
        x,
        x_lengths,
        tones,
        language,
        torch.zeros(1, 1024, length),
        torch.zeros(1, 768, length),
        g=g,
    )
    logw = model.dp(hidden, x_mask, g=g)
    return m_p, logs_p, logw, x_mask, g


def _validate(
    *,
    model: torch.nn.Module,
    static_encoder: EncoderDurationStatic,
    static_decoder: FlowMaskedDecoderStatic,
    encoder_path: Path,
    decoder_path: Path,
    reference_path: Path,
    lmax: int,
    tmax: int,
    lang_id: int,
    speaker_id: int,
    lengths: tuple[int, ...],
    atol: float,
) -> dict[str, object]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    encoder_session = ort.InferenceSession(str(encoder_path), options, providers=["CPUExecutionProvider"])
    decoder_session = ort.InferenceSession(str(decoder_path), options, providers=["CPUExecutionProvider"])
    reference_session = ort.InferenceSession(str(reference_path), options, providers=["CPUExecutionProvider"])
    cases = []
    for actual_length in lengths:
        generator = torch.Generator().manual_seed(1000 + actual_length)
        valid_x = torch.randint(
            0,
            model.n_vocab,
            (1, actual_length),
            generator=generator,
        ).long()
        valid_tones = torch.randint(
            0,
            model.enc_p.tone_emb.num_embeddings,
            (1, actual_length),
            generator=generator,
        ).long()
        sid = torch.tensor([speaker_id]).long()
        x = torch.zeros(1, lmax).long()
        tones = torch.zeros(1, lmax).long()
        x[:, :actual_length] = valid_x
        tones[:, :actual_length] = valid_tones
        x_lengths = torch.tensor([actual_length]).long()
        dynamic = _dynamic_encoder(model, valid_x, valid_tones, sid, lang_id)
        static = static_encoder(x, x_lengths, tones, sid)
        encoder_error = max(
            _max_abs(reference, padded[..., : reference.shape[-1]])
            for reference, padded in zip(dynamic, static, strict=True)
        )
        if encoder_error > atol:
            raise AssertionError(f"L={actual_length} padded encoder error {encoder_error}")
        dynamic_inputs = prepare_decoder_inputs(*dynamic, acoustic_max_length=tmax, noise_scale=0)
        static_inputs = prepare_decoder_inputs(*static, acoustic_max_length=tmax, noise_scale=0)
        y_length = dynamic_inputs.y_length
        z_dynamic = model.flow(
            dynamic_inputs.z_p[:, :, :y_length],
            torch.ones(1, 1, y_length),
            g=dynamic_inputs.g,
            reverse=True,
        )
        pytorch_y = model.dec(z_dynamic, g=dynamic_inputs.g)
        padded_y = static_decoder(
            static_inputs.z_p,
            static_inputs.y_mask,
            static_inputs.g,
        )[:, :, : y_length * 512]
        padded_wave_error = _max_abs(pytorch_y, padded_y)
        if padded_wave_error > atol:
            raise AssertionError(f"L={actual_length} padded decoder error {padded_wave_error}")
        original_y = _run_ort(
            reference_session,
            {
                "x": valid_x,
                "x_lengths": torch.tensor([actual_length]),
                "tones": valid_tones,
                "sid": sid,
                "noise_scale": torch.tensor([0.0]),
                "length_scale": torch.tensor([1.0]),
                "noise_scale_w": torch.tensor([0.8]),
            },
        )[0]
        original_error = _max_abs(pytorch_y, original_y)
        onnx_features = _run_ort(
            encoder_session,
            {
                "x": x,
                "x_lengths": x_lengths,
                "tones": tones,
                "sid": sid,
            },
        )
        onnx_inputs = prepare_decoder_inputs(*onnx_features, acoustic_max_length=tmax, noise_scale=0)
        onnx_y = _run_ort(
            decoder_session,
            {
                "z_p": onnx_inputs.z_p,
                "y_mask": onnx_inputs.y_mask,
                "g": onnx_inputs.g,
            },
        )[0][:, :, : onnx_inputs.y_length * 512]
        onnx_error = _max_abs(padded_y, onnx_y)
        if max(original_error, onnx_error) > atol:
            raise AssertionError(f"L={actual_length} ONNX error exceeds {atol}")
        cases.append(
            {
                "text_length": actual_length,
                "acoustic_length": y_length,
                "samples": y_length * 512,
                "padded_encoder_max_abs": encoder_error,
                "padded_waveform_max_abs": padded_wave_error,
                "release_onnx_vs_pytorch_max_abs": original_error,
                "static_onnx_vs_pytorch_max_abs": onnx_error,
            }
        )
    return {
        "atol": atol,
        "cases": cases,
        "maximum_error": max(
            max(
                case["padded_encoder_max_abs"],
                case["padded_waveform_max_abs"],
                case["release_onnx_vs_pytorch_max_abs"],
                case["static_onnx_vs_pytorch_max_abs"],
            )
            for case in cases
        ),
    }


def export_static_graphs(
    *,
    melo_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    release_onnx: str | Path,
    output_dir: str | Path,
    lmax: int = 32,
    tmax: int = 64,
    lang_id: int = 3,
    speaker_id: int = 1,
    opset: int = 17,
    validation_lengths: tuple[int, ...] = (3, 5, 9),
) -> tuple[Path, Path, dict[str, object]]:
    if lmax <= 0 or tmax <= 0:
        raise ValueError("lmax/tmax must be positive")
    if max(validation_lengths) > lmax:
        raise ValueError("validation length exceeds lmax")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    model, config = load_official_model(melo_root, config_path, checkpoint_path)
    encoder = EncoderDurationStatic(model, text_max_length=lmax, lang_id=lang_id).eval()
    decoder = FlowMaskedDecoderStatic(model).eval()
    x = torch.zeros(1, lmax, dtype=torch.int32)
    x_lengths = torch.tensor([lmax], dtype=torch.int32)
    tones = torch.zeros(1, lmax, dtype=torch.int32)
    sid = torch.tensor([speaker_id], dtype=torch.int32)
    z_p = torch.zeros(1, model.inter_channels, tmax)
    y_mask = torch.ones(1, 1, tmax)
    g = model.emb_g(sid.long()).unsqueeze(-1)
    encoder_path = output / f"melotts_encoder_dp_b1_l{lmax}.onnx"
    decoder_path = output / f"melotts_flow_masked_decoder_b1_t{tmax}.onnx"
    torch.onnx.export(
        encoder,
        (x, x_lengths, tones, sid),
        encoder_path,
        input_names=["x", "x_lengths", "tones", "sid"],
        output_names=["m_p", "logs_p", "logw", "x_mask", "g"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    encoder_rewrites = _finalize(
        encoder_path,
        role="encoder_duration",
        lmax=lmax,
        tmax=tmax,
        hop_length=int(config["data"]["hop_length"]),
    )
    torch.onnx.export(
        decoder,
        (z_p, y_mask, g),
        decoder_path,
        input_names=["z_p", "y_mask", "g"],
        output_names=["y"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    decoder_rewrites = _finalize(
        decoder_path,
        role="inverse_flow_masked_generator",
        lmax=lmax,
        tmax=tmax,
        hop_length=int(config["data"]["hop_length"]),
    )
    validation = _validate(
        model=model,
        static_encoder=encoder,
        static_decoder=decoder,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        reference_path=Path(release_onnx),
        lmax=lmax,
        tmax=tmax,
        lang_id=lang_id,
        speaker_id=speaker_id,
        lengths=validation_lengths,
        atol=2e-5,
    )
    validation["static_shape_rewrites"] = {
        "encoder": encoder_rewrites,
        "decoder": decoder_rewrites,
    }
    validation["sample_rate"] = int(config["data"]["sampling_rate"])
    validation["hop_length"] = int(config["data"]["hop_length"])
    return encoder_path, decoder_path, validation


__all__ = [
    "export_static_graphs",
    "replace_static_constant_of_shape",
]
