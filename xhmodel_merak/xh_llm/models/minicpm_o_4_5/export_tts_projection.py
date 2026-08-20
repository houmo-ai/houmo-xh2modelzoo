from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import onnx
import torch
from torch import Tensor, nn

from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config

from .export_token2wav import (
    _exportable_normalize,
    constantize_dynamic_expand_shapes,
    remove_identity_expands,
    remove_static_broadcast_expands,
    staticize_known_shape_nodes,
)


def _convert_small_module(
    module: nn.Module,
    inputs: Sequence[Tensor],
    names: tuple[Sequence[str], Sequence[str]],
    output_file: Path,
    target_device: str,
    quant_type: str,
    prepare: Callable[[onnx.ModelProto], onnx.ModelProto] | None = None,
    source_dir: Path | None = None,
) -> Path:
    """ONNX-direct export for a small projection/head module.

    Mirrors ``export_token2wav._convert``: torch.onnx.export -> shape
    staticization -> convert_onnx_to_hmonnx. Small MLP/Linear modules have no
    stateful or streaming contract, so the fixed-shape ONNX path is sufficient
    and avoids the TorchFX wrap chain used for the LLM backbone graphs. The
    float source (``*_source.onnx``) is kept in ``source_dir`` (when given) so
    the float graph and the HMONNX artifact live in separate directories.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if source_dir is None:
        source_file = output_file.with_name(f"{output_file.stem}_source.onnx")
    else:
        source_dir.mkdir(parents=True, exist_ok=True)
        source_file = source_dir / f"{output_file.stem}_source.onnx"
    with _exportable_normalize():
        torch.onnx.export(
            module,
            tuple(inputs),
            str(source_file),
            input_names=list(names[0]),
            output_names=list(names[1]),
            opset_version=17,
            do_constant_folding=True,
        )
    folded_model = staticize_known_shape_nodes(onnx.load(str(source_file)))
    folded_model = constantize_dynamic_expand_shapes(folded_model)
    folded_model = remove_identity_expands(folded_model)
    folded_model = remove_static_broadcast_expands(folded_model)
    if prepare is not None:
        folded_model = prepare(folded_model)
    onnx.save(folded_model, str(source_file))
    onnx.checker.check_model(str(source_file))
    if target_device != "XH2a":
        raise RuntimeError(f"Unsupported TTS projection target device: {target_device}")
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    convert_onnx_to_hmonnx(
        str(source_file),
        [value.cpu() for value in inputs],
        DeviceType.XH2a,
        str(output_file),
        quant_config=create_quant_config(quant_scheme),
        input_names=list(names[0]),
        output_names=list(names[1]),
    )
    return output_file


class _ProjectorSemantic(nn.Module):
    """``tts.projector_semantic`` wrapper with a fixed sequence capacity.

    The official projector is a per-token MLP (4096 -> 768 -> 768, relu in
    between); the sequence axis is independent so a fixed-capacity graph can be
    called for arbitrary real lengths and cropped on the Host.
    """

    def __init__(self, projector: nn.Module, seq_capacity: int) -> None:
        super().__init__()
        self.projector = projector
        self.seq_capacity = int(seq_capacity)

    def forward(self, hidden_states: Tensor) -> Tensor:
        padded = torch.nn.functional.pad(
            hidden_states,
            (0, 0, 0, self.seq_capacity - hidden_states.shape[1]),
        )
        return self.projector(padded)


class _HeadCode(nn.Module):
    """``tts.head_code[0]`` wrapper (num_vq == 1) with fixed sequence capacity."""

    def __init__(self, head_code: nn.Module, seq_capacity: int) -> None:
        super().__init__()
        self.head_code = head_code
        self.seq_capacity = int(seq_capacity)

    def forward(self, hidden_states: Tensor) -> Tensor:
        padded = torch.nn.functional.pad(
            hidden_states,
            (0, 0, 0, self.seq_capacity - hidden_states.shape[1]),
        )
        return self.head_code(padded)


def export_minicpm_o_4_5_tts_projection(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, Any],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
) -> dict[str, Any]:
    """Export the TTS host-side projection/head modules as HMONNX small graphs.

    - ``projector_semantic``: LLM hidden (4096) -> TTS hidden (768), applied per
      text chunk before emb_text merge.
    - ``head_code``: TTS hidden (768) -> audio code logits (6562), applied per
      decode step (num_vq == 1).

    Both are fixed-shape small modules; ONNX-direct export mirrors the audio
    main graph path.
    """
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    host = AutoModel.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    ).eval()
    tts = host.tts
    if tts.num_vq != 1:
        raise RuntimeError(f"TTS head_code export requires num_vq=1, got {tts.num_vq}")
    seq_capacity = int(component_cfg["wrap_cfg"]["input_sequence_length"])
    quant_type = str(component_cfg["quant_type"])
    model_name = model_name or "minicpm_o_4_5"
    tts_others_dir = work_dir / "TTS" / "Others"
    onnx_dir = tts_others_dir / "onnx"
    hmonnx_dir = tts_others_dir / "hmonnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_dir.mkdir(parents=True, exist_ok=True)

    projector = _ProjectorSemantic(tts.projector_semantic, seq_capacity).to(device)
    projector_input = torch.randn((1, seq_capacity, 4096), dtype=torch.float16, device=device)
    projector_graph = _convert_small_module(
        projector,
        [projector_input],
        (["hidden_states"], ["projected"]),
        hmonnx_dir / f"{model_name}_tts_projector_semantic_{target_device}_{quant_type}.onnx",
        target_device,
        quant_type,
        source_dir=onnx_dir,
    )

    head_code = _HeadCode(tts.head_code[0], seq_capacity).to(device)
    head_input = torch.randn((1, seq_capacity, 768), dtype=torch.float16, device=device)
    head_graph = _convert_small_module(
        head_code,
        [head_input],
        (["hidden_states"], ["logits"]),
        hmonnx_dir / f"{model_name}_tts_head_code_{target_device}_{quant_type}.onnx",
        target_device,
        quant_type,
        source_dir=onnx_dir,
    )
    return {
        "quant_type": quant_type,
        "graphs": {
            "projector_semantic": str(projector_graph.relative_to(work_dir)),
            "head_code": str(head_graph.relative_to(work_dir)),
        },
        "projection_seq_capacity": seq_capacity,
        "num_vq": tts.num_vq,
    }


__all__ = ["export_minicpm_o_4_5_tts_projection"]
