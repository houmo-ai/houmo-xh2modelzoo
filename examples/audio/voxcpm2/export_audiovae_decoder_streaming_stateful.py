"""Export VoxCPM2 AudioVAE decoder as a true stateful streaming graph.

Unlike ``export_audiovae_decoder.py``, this graph does not decode an overlap
window and crop the newest patch. It exposes every causal-conv state as graph
inputs and outputs:

    audio, state_out_0, ... = f(z_new, sr_idx, state_in_0, ...)

The host should initialize all states to zero at the beginning of an utterance,
then feed the returned states back into the next step. With ``--num_patches 1``
the input ``z_new`` is exactly one VoxCPM2 generated audio patch:
``[1, latent_dim, patch_size]``.
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import onnx
import onnxsim
import torch
import torch.nn as nn
from onnx import TensorProto

from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    HMONNXInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
)
from xhquant.patch.core import RewriterContext

try:
    from voxcpm import VoxCPM2Model
except ImportError:
    from voxcpm.model.voxcpm2 import VoxCPM2Model

from utils import remove_weight_norm_recursively, write_json_file


GB = int(2**30)
_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 0.1)

_ONNX_DTYPE_TO_NAME = {
    TensorProto.FLOAT: "float32",
    TensorProto.FLOAT16: "float16",
    TensorProto.DOUBLE: "float64",
    TensorProto.BFLOAT16: "bfloat16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
}

_NAME_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}


@dataclass(frozen=True)
class StateSpec:
    name: str
    kind: str
    channels: int
    length: int
    module_class: str

    @property
    def shape(self) -> list[int]:
        return [1, int(self.channels), int(self.length)]


def _first_output(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _as_tuple(outputs):
    if isinstance(outputs, tuple):
        return outputs
    if isinstance(outputs, list):
        return tuple(outputs)
    return (outputs,)


def _read_onnx_input_dtype(onnx_path: str, input_name: str) -> str | None:
    try:
        model = onnx.load(onnx_path)
    except Exception:
        return None
    for graph_input in model.graph.input:
        if graph_input.name != input_name:
            continue
        elem_type = graph_input.type.tensor_type.elem_type
        return _ONNX_DTYPE_TO_NAME.get(elem_type)
    return None


def _cast_to_graph_dtype(tensor: torch.Tensor, dtype_name: str | None) -> torch.Tensor:
    if dtype_name is None:
        return tensor
    target = _NAME_TO_TORCH_DTYPE.get(dtype_name)
    if target is None or tensor.dtype == target:
        return tensor
    return tensor.to(dtype=target)


def _is_causal_conv(module: nn.Module) -> bool:
    return module.__class__.__name__ == "CausalConv1d"


def _is_causal_tconv(module: nn.Module) -> bool:
    return module.__class__.__name__ == "CausalTransposeConv1d"


def _causal_conv_pad(module: nn.Module) -> int:
    padding = int(getattr(module, "_CausalConv1d__padding"))
    output_padding = int(getattr(module, "_CausalConv1d__output_padding"))
    return padding * 2 - output_padding


def _causal_tconv_ctx_trim(module: nn.Module) -> tuple[int, int]:
    padding = int(getattr(module, "_CausalTransposeConv1d__padding"))
    output_padding = int(getattr(module, "_CausalTransposeConv1d__output_padding"))
    trim = padding * 2 - output_padding
    ctx = (int(module.kernel_size[0]) - 1) // int(module.stride[0])
    return ctx, trim


def _has_stateful_child(module: nn.Module) -> bool:
    return any(_is_causal_conv(m) or _is_causal_tconv(m) for m in module.modules())


class StatefulAudioVAEDecoderExportWrapper(nn.Module):
    """AudioVAE decoder with explicit causal-conv states.

    The wrapper mirrors ``StreamingVAEDecoder`` from VoxCPM2, but the state is
    passed through graph inputs/outputs instead of hidden Python buffers.
    """

    def __init__(self, vae_decoder: nn.Module):
        super().__init__()
        self.decoder = vae_decoder
        self.has_sr_cond = vae_decoder.sr_bin_boundaries is not None
        self.state_specs = self._collect_state_specs()

    def _collect_state_specs(self) -> list[StateSpec]:
        specs: list[StateSpec] = []

        def visit(module: nn.Module, prefix: str) -> None:
            if _is_causal_conv(module):
                pad = _causal_conv_pad(module)
                if pad > 0:
                    specs.append(
                        StateSpec(
                            name=f"{prefix}_state",
                            kind="causal_conv",
                            channels=int(module.in_channels),
                            length=pad,
                            module_class=module.__class__.__name__,
                        )
                    )
                return
            if _is_causal_tconv(module):
                ctx, _ = _causal_tconv_ctx_trim(module)
                if ctx > 0:
                    specs.append(
                        StateSpec(
                            name=f"{prefix}_state",
                            kind="causal_transpose_conv",
                            channels=int(module.in_channels),
                            length=ctx,
                            module_class=module.__class__.__name__,
                        )
                    )
                return
            if module.__class__.__name__ in {"CausalResidualUnit", "CausalDecoderBlock"}:
                visit(module.block, f"{prefix}_block")
                return
            if isinstance(module, nn.Sequential):
                for idx, child in enumerate(module):
                    visit(child, f"{prefix}_{idx}")
                return
            if _has_stateful_child(module):
                for name, child in module.named_children():
                    visit(child, f"{prefix}_{name}")

        if self.has_sr_cond:
            for idx, (layer, sr_layer) in enumerate(zip(self.decoder.model, self.decoder.sr_cond_model)):
                if sr_layer is not None:
                    visit(sr_layer, f"srcond_{idx}")
                visit(layer, f"layer_{idx}")
        else:
            visit(self.decoder.model, "model")
        return specs

    def initial_states(self, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        return [
            torch.zeros(spec.shape, device=device, dtype=dtype)
            for spec in self.state_specs
        ]

    def _forward_causal_conv(
        self,
        module: nn.Module,
        x: torch.Tensor,
        state_iter: Iterable[torch.Tensor],
        new_states: list[torch.Tensor],
    ) -> torch.Tensor:
        pad = _causal_conv_pad(module)
        if pad <= 0:
            return nn.Conv1d.forward(module, x)
        state = next(state_iter)
        x_pad = torch.cat([state, x], dim=-1)
        new_states.append(x_pad[..., -pad:].detach())
        return nn.Conv1d.forward(module, x_pad)

    def _forward_causal_tconv(
        self,
        module: nn.Module,
        x: torch.Tensor,
        state_iter: Iterable[torch.Tensor],
        new_states: list[torch.Tensor],
    ) -> torch.Tensor:
        ctx, trim = _causal_tconv_ctx_trim(module)
        if ctx <= 0:
            out = nn.ConvTranspose1d.forward(module, x)
            return out[..., :-trim] if trim > 0 else out
        state = next(state_iter)
        x_full = torch.cat([state, x], dim=-1)
        new_states.append(x[..., -ctx:].detach())
        out = nn.ConvTranspose1d.forward(module, x_full)
        left = ctx * int(module.stride[0])
        return out[..., left:-trim] if trim > 0 else out[..., left:]

    def _forward_module(
        self,
        module: nn.Module,
        x: torch.Tensor,
        state_iter: Iterable[torch.Tensor],
        new_states: list[torch.Tensor],
    ) -> torch.Tensor:
        if _is_causal_conv(module):
            return self._forward_causal_conv(module, x, state_iter, new_states)
        if _is_causal_tconv(module):
            return self._forward_causal_tconv(module, x, state_iter, new_states)
        if module.__class__.__name__ in {"CausalResidualUnit"}:
            y = self._forward_module(module.block, x, state_iter, new_states)
            return x + y
        if module.__class__.__name__ in {"CausalDecoderBlock"}:
            return self._forward_module(module.block, x, state_iter, new_states)
        if isinstance(module, nn.Sequential):
            for child in module:
                x = self._forward_module(child, x, state_iter, new_states)
            return x
        if _has_stateful_child(module):
            # This path mainly protects SampleRateConditionLayer when
            # cond_out_layer=True. The current VoxCPM2 config uses Identity.
            raise RuntimeError(
                f"Unsupported stateful nested module: {module.__class__.__name__}. "
                "Add an explicit forward rule before exporting."
            )
        return module(x)

    def _apply_sr_cond(self, sr_layer: nn.Module, x: torch.Tensor, sr_idx: torch.Tensor) -> torch.Tensor:
        if sr_layer.cond_type in {"scale_bias", "scale_bias_init"}:
            x = x * sr_layer.scale_embed(sr_idx).unsqueeze(-1) + sr_layer.bias_embed(sr_idx).unsqueeze(-1)
        elif sr_layer.cond_type == "add":
            x = x + sr_layer.cond_embed(sr_idx).unsqueeze(-1)
        elif sr_layer.cond_type == "concat":
            cond = sr_layer.cond_embed(sr_idx).unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.cat([x, cond], dim=1)
        else:
            raise ValueError(f"Invalid cond_type: {sr_layer.cond_type}")
        if not isinstance(sr_layer.out_layer, nn.Identity):
            if _has_stateful_child(sr_layer.out_layer):
                raise RuntimeError("SampleRateConditionLayer.out_layer with causal state is not supported yet.")
            x = sr_layer.out_layer(x)
        return x

    def forward(self, z: torch.Tensor, sr_idx: torch.Tensor, *states: torch.Tensor):
        if len(states) != len(self.state_specs):
            raise RuntimeError(f"expected {len(self.state_specs)} states, got {len(states)}")
        state_iter = iter(states)
        new_states: list[torch.Tensor] = []

        x = z
        if self.has_sr_cond:
            for layer, sr_layer in zip(self.decoder.model, self.decoder.sr_cond_model):
                if sr_layer is not None:
                    x = self._apply_sr_cond(sr_layer, x, sr_idx)
                x = self._forward_module(layer, x, state_iter, new_states)
        else:
            x = self._forward_module(self.decoder.model, x, state_iter, new_states)

        if len(new_states) != len(self.state_specs):
            raise RuntimeError(f"produced {len(new_states)} states, expected {len(self.state_specs)}")
        return (x, *new_states)


def _compute_sr_idx(voxcpm2: VoxCPM2Model, sr_value: int) -> int:
    boundaries = voxcpm2.audio_vae.decoder.sr_bin_boundaries
    sr_tensor = torch.tensor([sr_value], dtype=torch.int32)
    return int(torch.bucketize(sr_tensor, boundaries.to(sr_tensor.device))[0].item())


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().float().reshape(-1).cpu()
    b = b.detach().float().reshape(-1).cpu()
    if a.numel() != b.numel():
        raise RuntimeError(f"shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}")
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "cosine": float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()),
    }


@torch.inference_mode()
def _verify_wrapper_vs_native_streaming(
    wrapper: StatefulAudioVAEDecoderExportWrapper,
    audio_vae,
    z_seq: torch.Tensor,
    sr_idx: torch.Tensor,
    states: list[torch.Tensor],
    logger,
) -> dict:
    wrapper_states = [s.clone() for s in states]
    wrapper_chunks = []
    native_chunks = []
    with audio_vae.streaming_decode() as native_dec:
        for step in range(z_seq.shape[0]):
            z_step = z_seq[step]
            wrapper_out = wrapper(z_step, sr_idx, *wrapper_states)
            wrapper_chunks.append(wrapper_out[0])
            wrapper_states = list(wrapper_out[1:])
            native_chunks.append(native_dec.decode_chunk(z_step.to(torch.float32)))

    wrapper_audio = torch.cat(wrapper_chunks, dim=-1)
    native_audio = torch.cat(native_chunks, dim=-1)
    metrics = _tensor_metrics(wrapper_audio, native_audio)
    logger.info(
        "Parity(wrapper vs native streaming): max_abs=%.8f mean_abs=%.8f cosine=%.8f",
        metrics["max_abs"],
        metrics["mean_abs"],
        metrics["cosine"],
    )
    return metrics


def _verify_hmonnx_step(
    hmonnx_file: Path,
    wrapper: StatefulAudioVAEDecoderExportWrapper,
    z_cal: torch.Tensor,
    sr_cal: torch.Tensor,
    states: list[torch.Tensor],
    graph_dtypes: dict[str, str | None],
    device: torch.device,
    logger,
) -> dict:
    session = HMONNXInference(str(hmonnx_file))
    session.to(str(device))

    z_in = _cast_to_graph_dtype(z_cal, graph_dtypes.get("z"))
    sr_in = _cast_to_graph_dtype(sr_cal, graph_dtypes.get("sr_idx"))
    state_inputs = [
        _cast_to_graph_dtype(state, graph_dtypes.get(f"state_in_{idx}"))
        for idx, state in enumerate(states)
    ]
    input_dict = {"z": z_in, "sr_idx": sr_in}
    input_dict.update({f"state_in_{idx}": state for idx, state in enumerate(state_inputs)})
    h_outputs = _as_tuple(session.run(input_dict))

    wrapper_dtype = next(wrapper.parameters()).dtype
    ref_outputs = wrapper(
        z_in.to(device=device, dtype=wrapper_dtype),
        sr_cal,
        *[s.to(device=device, dtype=wrapper_dtype) for s in state_inputs],
    )

    audio_metrics = _tensor_metrics(h_outputs[0], ref_outputs[0])
    state_mean_abs = []
    for got, ref in zip(h_outputs[1:], ref_outputs[1:]):
        state_mean_abs.append(_tensor_metrics(got, ref)["mean_abs"])
    metrics = {
        **audio_metrics,
        "state_mean_abs_max": float(max(state_mean_abs) if state_mean_abs else 0.0),
    }
    logger.info(
        "Parity(HMONNX step vs wrapper): max_abs=%.6f mean_abs=%.6f cosine=%.6f state_mean_abs_max=%.6f",
        metrics["max_abs"],
        metrics["mean_abs"],
        metrics["cosine"],
        metrics["state_mean_abs_max"],
    )
    return metrics


def main(args):
    model_path = str(Path(args.model).expanduser().resolve())
    model_name = Path(model_path).name
    target_device = "XH2a"
    script_dir = Path(__file__).resolve().parent
    work_root = script_dir / "work_dirs"
    work_dir = (
        work_root
        / f"{model_name}_{target_device}"
        / f"AudioVAE_Decoder_StreamState_np{args.num_patches}"
    )
    work_dir.mkdir(exist_ok=True, parents=True)
    logger = get_root_logger()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float32

    logger.info("Loading VoxCPM2 model from %s", model_path)
    voxcpm2 = VoxCPM2Model.from_local(model_path, optimize=False, training=False)
    voxcpm2.audio_vae.to(device=device, dtype=dtype)
    voxcpm2.audio_vae.eval()

    logger.info("Removing weight_norm from AudioVAE decoder")
    remove_weight_norm_recursively(voxcpm2.audio_vae.decoder)

    wrapper = StatefulAudioVAEDecoderExportWrapper(voxcpm2.audio_vae.decoder).to(device=device, dtype=dtype).eval()
    state_specs = wrapper.state_specs
    logger.info("Collected %d streaming states", len(state_specs))
    for idx, spec in enumerate(state_specs):
        logger.info("  state_%02d %-22s shape=%s", idx, spec.kind, spec.shape)

    patch_size = int(voxcpm2.patch_size)
    latent_dim = int(voxcpm2.audio_vae.latent_dim)
    decoder_rates = list(voxcpm2.audio_vae.decoder_rates)
    upscale = int(math.prod(decoder_rates))
    num_patches = int(args.num_patches)
    T = num_patches * patch_size
    L_out = T * upscale
    out_sample_rate = int(voxcpm2.audio_vae.out_sample_rate)
    sr_idx_value = _compute_sr_idx(voxcpm2, out_sample_rate)

    torch.manual_seed(args.seed)
    z_cal = torch.randn((1, latent_dim, T), device=device, dtype=dtype) * 0.5
    sr_cal = torch.tensor([sr_idx_value], device=device, dtype=torch.int32)
    states_cal = wrapper.initial_states(device=device, dtype=dtype)

    with torch.no_grad():
        ref_outputs = wrapper(z_cal, sr_cal, *states_cal)
    audio_ref = ref_outputs[0]
    assert tuple(audio_ref.shape) == (1, 1, L_out), (
        f"Expected audio shape (1, 1, {L_out}), got {tuple(audio_ref.shape)}"
    )
    logger.info("Stateful decoder forward OK, audio.shape=%s", tuple(audio_ref.shape))

    stream_metrics = _verify_wrapper_vs_native_streaming(
        wrapper=wrapper,
        audio_vae=voxcpm2.audio_vae,
        z_seq=torch.randn((args.verify_steps, 1, latent_dim, T), device=device, dtype=dtype) * 0.5,
        sr_idx=sr_cal,
        states=states_cal,
        logger=logger,
    )

    onnx_file = work_dir / "voxcpm2_audiovae_decoder_streaming_stateful.onnx"
    input_names = ["z", "sr_idx"] + [f"state_in_{idx}" for idx in range(len(state_specs))]
    output_names = ["audio"] + [f"state_out_{idx}" for idx in range(len(state_specs))]
    export_inputs = (z_cal, sr_cal, *states_cal)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_onnx = str(Path(tmp_dir) / onnx_file.name)
        with RewriterContext(None, backend="onnxruntime"):
            torch.onnx.export(
                wrapper,
                export_inputs,
                tmp_onnx,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
            )
            logger.info("ONNX exported")
            onnx_model = onnx.load(tmp_onnx)
            size = onnx_model.ByteSize()
            logger.info("ONNX size: %.3f GB", size / GB)
            skipped_optimizers = [
                "fuse_pad_into_conv",
                "fuse_consecutive_slices",
                "eliminate_common_subexpression",
                "fuse_qkv",
            ]
            if size <= _LARGE_MODEL_SIZE_THRESHOLD:
                simplified, ok = onnxsim.simplify(onnx_model, skipped_optimizers=skipped_optimizers)
            else:
                from xhquant.utils.onnxsim_large_model import simplify_large_onnx

                simplified, ok = simplify_large_onnx(onnx_model, skipped_optimizers=skipped_optimizers)
            if ok:
                onnx_model = simplified

    onnx.save(
        onnx_model,
        onnx_file,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{onnx_file.stem}_external_data",
    )
    logger.info("ONNX saved: %s", onnx_file)

    hmonnx_file = (
        work_dir
        / "hmonnx"
        / f"voxcpm2_audiovae_decoder_streaming_stateful_np{num_patches}_xh2a_{args.quant_type}.onnx"
    )
    hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    graph_dtypes: dict[str, str | None] = {}
    hmonnx_metrics = None

    if not args.skip_hmonnx:
        quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
        quant_config = create_quant_config(quant_scheme)
        logger.info("Converting to HMONNX")
        convert_onnx_to_hmonnx(
            str(onnx_file),
            list(export_inputs),
            DeviceType.XH2a,
            hmonnx_file,
            quant_config=quant_config,
            input_names=input_names,
            output_names=output_names,
        )
        logger.info("HMONNX saved: %s", hmonnx_file)

        graph_dtypes = {name: _read_onnx_input_dtype(str(hmonnx_file), name) for name in input_names}
        logger.info("HMONNX graph input dtypes: %s", graph_dtypes)

        if not args.skip_verify:
            hmonnx_metrics = _verify_hmonnx_step(
                hmonnx_file=hmonnx_file,
                wrapper=wrapper,
                z_cal=z_cal,
                sr_cal=sr_cal,
                states=states_cal,
                graph_dtypes=graph_dtypes,
                device=device,
                logger=logger,
            )

        if args.gen_golden:
            golden_dir = work_dir / "hmonnx" / "golden"
            if golden_dir.exists():
                shutil.rmtree(golden_dir)
            session = HMONNXGoldenInference(hmonnx_file)
            session.to(device)
            session.save_golden = True
            session.golden_dir = str(golden_dir)
            session.step = 0
            golden_inputs = [
                _cast_to_graph_dtype(tensor, graph_dtypes.get(name))
                for name, tensor in zip(input_names, export_inputs)
            ]
            session(*golden_inputs)
            logger.info("golden saved: %s", golden_dir)

    meta = dict(
        create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        model_name=model_name,
        target_device=target_device,
        module="AudioVAE-Decoder-Streaming-Stateful",
        onnx_file=str(onnx_file.relative_to(work_dir.parent)),
        hmonnx_file=str(hmonnx_file.relative_to(work_dir.parent)) if hmonnx_file.exists() else None,
        quant_type=args.quant_type,
        input_dtype="float32",
        graph_input_dtype=graph_dtypes,
        input_names=input_names,
        output_names=output_names,
        input_shapes={
            "z": [1, latent_dim, T],
            "sr_idx": [1],
            **{f"state_in_{idx}": spec.shape for idx, spec in enumerate(state_specs)},
        },
        output_shapes={
            "audio": [1, 1, L_out],
            **{f"state_out_{idx}": spec.shape for idx, spec in enumerate(state_specs)},
        },
        state_specs=[spec.__dict__ | {"shape": spec.shape} for spec in state_specs],
        num_patches=num_patches,
        patch_size=patch_size,
        latent_dim=latent_dim,
        decoder_rates=decoder_rates,
        upscale=upscale,
        out_sample_rate=out_sample_rate,
        precomputed_sr_idx=sr_idx_value,
        sr_bin_boundaries=[int(x) for x in voxcpm2.audio_vae.decoder.sr_bin_boundaries.tolist()],
        parity=dict(
            wrapper_vs_native_streaming=stream_metrics,
            hmonnx_step_vs_wrapper=hmonnx_metrics,
        ),
    )
    meta_file = work_dir / f"audiovae_decoder_streaming_stateful_np{num_patches}_meta_info.json"
    write_json_file(meta_file, meta)
    logger.info("Meta saved: %s", meta_file)
    logger.info("Stateful streaming AudioVAE decoder export done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_patches", type=int, default=1)
    parser.add_argument("--quant_type", type=str, default="w8a8_sefp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify_steps", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip_hmonnx", action="store_true")
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--gen_golden", action="store_true")
    args = parser.parse_args()
    main(args)
