from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import onnx
import torch

from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config


_CAMPPLUS_INPUT = "input"  # (batch, seq, 80) fbank features
_CAMPPLUS_OUTPUT = "output"  # (batch, 192) speaker embedding


def _fix_input_shape(model: onnx.ModelProto, fixed_dims: dict[str, int]) -> onnx.ModelProto:
    """Replace symbolic input dims with fixed values declared in ``fixed_dims``."""
    for input_node in model.graph.input:
        dims: list[Any] = []
        for d in input_node.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                dims.append(d.dim_param)
        new_dims: list[int] = []
        for dim in dims:
            if dim in fixed_dims:
                new_dims.append(int(fixed_dims[dim]))
            elif isinstance(dim, int):
                new_dims.append(dim)
            else:
                raise ValueError(f"Unhandled dynamic dim: {dim}")
        input_node.type.tensor_type.shape.dim.clear()
        for val in new_dims:
            input_node.type.tensor_type.shape.dim.add(dim_value=val)
    return model


def _convert_to_hmonnx(
    onnx_file: Path,
    dummy_inputs: list[torch.Tensor],
    input_names: list[str],
    output_names: list[str],
    output_file: Path,
    target_device: str,
    quant_type: str,
) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if target_device != "XH2a":
        raise RuntimeError(f"Unsupported speaker component target device: {target_device}")
    quant_config = create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type))
    convert_onnx_to_hmonnx(
        str(onnx_file),
        [value.cpu() for value in dummy_inputs],
        DeviceType.XH2a,
        str(output_file),
        quant_config=quant_config,
        input_names=input_names,
        output_names=output_names,
    )
    return output_file


def _export_onnx_artifact(
    source_onnx: Path,
    out_onnx: Path,
    fixed_dims: dict[str, int],
    preprocess: Any = None,
) -> Path:
    """Fix dynamic input shapes of a source onnx into a static-shape onnx.

    campplus has no dynamic Expand; fixing the batch/sequence dims is sufficient.
    speech_tokenizer additionally folds its dynamic Expand via ``preprocess``.
    """
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source_onnx))
    model = _fix_input_shape(model, fixed_dims)
    if preprocess is not None:
        model = preprocess(model)
    onnx.save(model, str(out_onnx))
    return out_onnx


def export_minicpm_o_4_5_campplus(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, Any],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
) -> dict[str, Any]:
    """Export the token2wav campplus speaker-encoder onnx as an HMONNX graph.

    The official runtime runs campplus via onnxruntime (CPU); exporting it as an
    HMONNX graph keeps the whole speaker-embedding path on the target accelerator.
    Input: (1, sequence_length, 80) fbank features; Output: (1, 192) speaker
    embedding. Mirrors the CosyVoice3 campplus export.
    """
    del device
    source = Path(model_dir) / "assets" / "token2wav" / "campplus.onnx"
    if not source.exists():
        raise FileNotFoundError(f"campplus source onnx not found: {source}")
    seq_capacity = int(component_cfg.get("sequence_length", 1000))
    component_dir = work_dir / "Speaker"
    onnx_file = _export_onnx_artifact(
        source,
        component_dir / "onnx" / "campplus.onnx",
        {"batch_size": 1, "sequence_length": seq_capacity},
    )
    dummy_input = torch.randn(1, seq_capacity, 80)
    quant_type = str(component_cfg.get("quant_type", "w8a16_sefp"))
    hmonnx_file = (
        component_dir / "hmonnx" / f"{(model_name or 'minicpm_o_4_5')}_campplus_{target_device}_{quant_type}.onnx"
    )
    _convert_to_hmonnx(
        onnx_file,
        [dummy_input],
        [_CAMPPLUS_INPUT],
        [_CAMPPLUS_OUTPUT],
        hmonnx_file,
        target_device,
        quant_type,
    )
    return {
        "quant_type": quant_type,
        "graphs": {"campplus": str(hmonnx_file.relative_to(work_dir))},
        "onnx": str(onnx_file.relative_to(work_dir)),
        "sequence_length": seq_capacity,
        "input_dim": 80,
        "output_dim": 192,
    }


__all__ = ["export_minicpm_o_4_5_campplus"]


# ---------------------------------------------------------------------------
# speech_tokenizer_v2_25hz export
# ---------------------------------------------------------------------------
# The upstream graph derives an Expand output shape from ``feats_length`` via a
# dynamic chain:
#   T' = ReduceMax(Add(Cast(Div(Add(Sub(Add(Cast(Div(Sub(feats_length, 3), 2)), 1), 3), 2), 2)), 1))
#   Expand(data=Unsqueeze(Range(0, T', 1)), shape=Where(Equal([1,T'], -1), ..., [1,T']))
# With feats_length fixed to the exported capacity, T' is a compile-time
# constant and the whole Expand becomes a constant tensor. We evaluate that
# chain in numpy and replace the Expand node with a Constant initializer so the
# hmonnx frontend (which requires constant Expand shapes) accepts the graph.
#
# This mirrors the CosyVoice3 speech_tokenizer_v3 conversion which also fixes
# feats_length to a constant and folds the dynamic shape chain.


def _feats_length_to_tokens(feats_length: int) -> int:
    """Compute the token count from the mel frame capacity (T' = 750 @ 3000)."""
    x = (feats_length - 3) // 2
    x = int(x) + 1
    x = x - 3 + 2
    x = (x // 2) + 1
    return int(x)


def _fold_speech_tokenizer_expand(model: onnx.ModelProto, feats_length: int) -> onnx.ModelProto:
    """Replace the dynamic Expand in the speech_tokenizer graph with a constant.

    The Expand (and its Unsqueeze/Range/Where producers) only depends on the
    feats_length-derived token count. With feats_length fixed, it is evaluated
    and the whole Expand node is replaced by a Constant tensor (its output
    becomes the indices value). The remaining graph then has no dynamic-shape
    Expand and can be exported through the hmonnx frontend.
    """
    import numpy as np
    import onnx_graphsurgeon as gs

    token_count = _feats_length_to_tokens(feats_length)
    indices = np.arange(0, token_count, 1, dtype=np.int64).reshape(1, -1)

    g = gs.import_onnx(model)
    # Fold the Expand output into a constant (its T' producer chain then becomes
    # orphaned). The feats_length graph input is retained: the T' chain keeps
    # consuming it, and the hmonnx export receives both inputs (feats +
    # feats_length), matching the upstream graph contract.
    expand_nodes = [n for n in g.nodes if n.op == "Expand"]
    for node in expand_nodes:
        # Replace the output tensor with a constant; drop the Expand node.
        out = node.outputs[0]
        g.outputs = [o for o in g.outputs if o.name != out.name]
        const_tensor = gs.Constant(
            name=out.name,
            values=indices,
        )
        # Rebind consumers of the old output to the constant.
        for other in g.nodes:
            for idx, inp in enumerate(other.inputs):
                if inp is out or (hasattr(inp, "name") and inp.name == out.name):
                    other.inputs[idx] = const_tensor
        g.nodes.remove(node)
    g.cleanup().toposort()
    return gs.export_onnx(g)


def export_minicpm_o_4_5_speech_tokenizer(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, Any],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
) -> dict[str, Any]:
    """Export the token2wav speech_tokenizer_v2_25hz onnx as an HMONNX graph.

    The official runtime loads it via s3tokenizer (onnx2torch, host PyTorch).
    This export fixes feats_length to the capacity, folds the dynamic Expand to
    a constant, and converts the graph to HMONNX so the mel->token quantization
    runs on the target accelerator. Input: (1, 128, feats_length) mel;
    Output: indices (quantized tokens).
    """
    del device
    source = Path(model_dir) / "assets" / "token2wav" / "speech_tokenizer_v2_25hz.onnx"
    if not source.exists():
        raise FileNotFoundError(f"speech_tokenizer source onnx not found: {source}")
    n_mels = int(component_cfg.get("n_mels", 128))
    feats_length = int(component_cfg.get("feats_length", 3000))
    component_dir = work_dir / "SpeechTokenizer"
    onnx_file = _export_onnx_artifact(
        source,
        component_dir / "onnx" / "speech_tokenizer_v2_25hz.onnx",
        {"T": feats_length},
        preprocess=lambda model: _fold_speech_tokenizer_expand(model, feats_length),
    )
    dummy_feats = torch.randn(1, n_mels, feats_length)
    dummy_length = torch.tensor([feats_length], dtype=torch.int32)
    quant_type = str(component_cfg.get("quant_type", "w8a16_sefp"))
    hmonnx_file = (
        component_dir
        / "hmonnx"
        / f"{(model_name or 'minicpm_o_4_5')}_speech_tokenizer_v2_25hz_{target_device}_{quant_type}.onnx"
    )
    _convert_to_hmonnx(
        onnx_file,
        [dummy_feats, dummy_length],
        ["feats", "feats_length"],
        ["indices"],
        hmonnx_file,
        target_device,
        quant_type,
    )
    return {
        "quant_type": quant_type,
        "graphs": {"speech_tokenizer": str(hmonnx_file.relative_to(work_dir))},
        "onnx": str(onnx_file.relative_to(work_dir)),
        "n_mels": n_mels,
        "feats_length": feats_length,
        "input_names": ["feats", "feats_length"],
        "output_names": ["indices"],
    }
