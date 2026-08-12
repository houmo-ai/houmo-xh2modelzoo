from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput, Transformer2DModelOutput

from xhquant.api import HMONNXGoldenInference, PrecisionMode

from .qwen3_vl_preprocess import (
    LingBotQwen3VLDataPreprocess,
    LingBotQwen3VLProcessor,
)
from .text_encoder import TEXT_OUTPUT_SEMANTICS, prepare_language_graph_inputs
from .transformer_wrapper import LingBotVideoTransformerExportWrapper


def resolve_runtime_artifact(export_dir: Path, name: str) -> Path:
    export_dir = Path(export_dir).resolve()
    meta_file = export_dir / "export_meta_info.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"Missing LingBot export metadata: {meta_file}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    runtime = meta.get("runtime")
    if not isinstance(runtime, dict) or name not in runtime:
        raise ValueError(f"LingBot export metadata has no runtime.{name}; re-export a self-contained artifact.")
    relative_path = Path(runtime[name])
    if relative_path.is_absolute():
        raise ValueError(f"runtime.{name} must be relative to the export directory")
    artifact = (export_dir / relative_path).resolve()
    if artifact != export_dir and export_dir not in artifact.parents:
        raise ValueError(f"runtime.{name} resolves outside the export directory")
    if not artifact.exists():
        raise FileNotFoundError(f"Missing runtime artifact: {artifact}")
    return artifact


class _LazyHMONNXRuntime:
    def __init__(self, hmonnx_file: Path, device: torch.device, precision_mode: Optional[PrecisionMode] = None):
        self.hmonnx_file = Path(hmonnx_file)
        if not self.hmonnx_file.is_file():
            raise FileNotFoundError(f"Missing HMONNX file: {self.hmonnx_file}")
        self.device = torch.device(device)
        self.precision_mode = precision_mode
        self._runtime: Optional[HMONNXGoldenInference] = None

    def to(self, device: torch.device) -> "_LazyHMONNXRuntime":
        self.device = torch.device(device)
        if self._runtime is not None:
            self._runtime.exec_device = self.device
        return self

    def __call__(self, *inputs: torch.Tensor):
        if self._runtime is None:
            self._runtime = HMONNXGoldenInference(str(self.hmonnx_file), exec_device=self.device)
            if self.precision_mode is not None:
                self._runtime.set_precision_mode(self.precision_mode)
        self._runtime.exec_device = self.device
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self._runtime(*inputs)


def load_token_embedding(checkpoint: Path) -> nn.Embedding:
    """Rebuild the FP16 nn.Embedding used by Merak LLM HMONNX inference."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "weight" not in state:
        raise ValueError(f"Token embedding checkpoint must be an nn.Embedding state dict: {checkpoint}")
    weight = state["weight"]
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise ValueError("Token embedding must contain floating-point weights. Re-export the text encoder.")
    if weight.ndim != 2:
        raise ValueError(f"Token embedding weight must be rank 2, got shape {tuple(weight.shape)}.")
    embedding = nn.Embedding(*weight.shape, dtype=torch.float16)
    embedding.load_state_dict(state)
    return embedding


class LingBotQwen3VLTextEncoderInference(nn.Module):
    def __init__(
        self,
        *,
        export_root: Path,
        config_file: Path,
        device: torch.device,
    ):
        super().__init__()
        export_root = Path(export_root)
        meta = json.loads((export_root / "meta.json").read_text(encoding="utf-8"))
        output_semantics = meta.get("output_semantics")
        if output_semantics != TEXT_OUTPUT_SEMANTICS:
            raise ValueError(
                "Incompatible LingBot text encoder artifact: expected output_semantics="
                f"{TEXT_OUTPUT_SEMANTICS!r}, got {output_semantics!r}. Re-export the "
                "artifact with the current Merak workflow."
            )
        config = json.loads(Path(config_file).read_text(encoding="utf-8"))
        self._device = torch.device(device)
        self._dtype_marker = nn.Parameter(
            torch.empty(0, device=self._device, dtype=torch.float16),
            requires_grad=False,
        )
        self.language_runtime = _LazyHMONNXRuntime(export_root / meta["language_hmonnx"], self._device)
        self.visual_runtime = _LazyHMONNXRuntime(export_root / meta["visual"]["hmonnx"], self._device)
        self.sequence_length = int(meta["sequence_length"])
        self.embedding = load_token_embedding(export_root / meta["quant_embedding"])
        self.config = SimpleNamespace(**config)
        self.processor_config = {
            "image_token_id": int(config["image_token_id"]),
            "video_token_id": int(config["video_token_id"]),
            "vision_start_token_id": int(config["vision_start_token_id"]),
            "spatial_merge_size": int(config["vision_config"]["spatial_merge_size"]),
        }

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype_marker.dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._device = torch.device(device)
            self.language_runtime.to(self._device)
            self.visual_runtime.to(self._device)
            self.embedding.to(self._device)
        marker_dtype = dtype if dtype is not None else self._dtype_marker.dtype
        self._dtype_marker.data = self._dtype_marker.data.to(device=self._device, dtype=marker_dtype)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        hm_pixel_values: Optional[list[torch.Tensor]] = None,
        output_hidden_states: bool = True,
        **kwargs,
    ) -> Any:
        del pixel_values, pixel_values_videos, video_grid_thw, kwargs
        if input_ids.shape[0] != 1:
            raise ValueError("LingBot HMONNX text encoder supports batch_size=1 only.")
        true_length = int(attention_mask.sum().item() if attention_mask is not None else input_ids.shape[1])
        if true_length > self.sequence_length:
            raise ValueError(f"Prompt has {true_length} tokens, export limit is {self.sequence_length}.")

        image_embeds = None
        deepstack_image_embeds = None
        if hm_pixel_values:
            visual_outputs = []
            for pixels in hm_pixel_values:
                output = self.visual_runtime(_prepare_hmonnx_tensor(pixels, self._device, torch.float16))
                output = output if isinstance(output, tuple) else (output,)
                visual_outputs.append(output)
            image_embeds = torch.cat([output[0] for output in visual_outputs], dim=0).squeeze(0)
            deepstack_image_embeds = [
                torch.cat([output[index] for output in visual_outputs], dim=0).squeeze(0) for index in range(1, 4)
            ]

        preprocessor = LingBotQwen3VLDataPreprocess(
            token_embedding=self.embedding,
            input_sequence_length=self.sequence_length,
            image_token_id=self.processor_config["image_token_id"],
            video_token_id=self.processor_config["video_token_id"],
            vision_start_token_id=self.processor_config["vision_start_token_id"],
            spatial_merge_size=self.processor_config["spatial_merge_size"],
        ).to(self._device, torch.float16)
        data = {
            "input_ids": input_ids,
            "past_seq_length": 0,
            "image_grid_thw": image_grid_thw,
        }
        if image_embeds is not None:
            data["image_embeds"] = image_embeds
            data["deepstack_image_embeds"] = deepstack_image_embeds
        graph_inputs = prepare_language_graph_inputs(list(preprocessor(data)))
        graph_inputs = [value.to(torch.int32) if value.dtype == torch.int64 else value for value in graph_inputs]
        graph_inputs = [value.to(self._device) for value in graph_inputs]
        prompt_embeds = self.language_runtime(*graph_inputs)
        if isinstance(prompt_embeds, tuple):
            prompt_embeds = prompt_embeds[0]
        result = SimpleNamespace(last_hidden_state=prompt_embeds)
        result.hidden_states = (prompt_embeds,) if output_hidden_states else None
        return result


class LingBotVideoTransformerInference(nn.Module):
    def __init__(self, export_root: Path, config_file: Path, device: torch.device):
        super().__init__()
        from lingbot_video.transformer_lingbot_video import LingBotVideoRotaryEmbedding

        export_root = Path(export_root)
        meta = json.loads((export_root.parent / "export_meta_info.json").read_text(encoding="utf-8"))["transformer"]
        if meta.get("attention_backend") != "xh2a_flash_attention":
            raise ValueError(
                "Incompatible LingBot transformer artifact: full-duration video "
                "requires the xh2a_flash_attention export. Re-export this profile "
                "with the current Merak workflow."
            )
        transformer_config = json.loads(Path(config_file).read_text(encoding="utf-8"))
        self._device = torch.device(device)
        self._dtype_marker = nn.Parameter(
            torch.empty(0, device=self._device, dtype=torch.float16),
            requires_grad=False,
        )
        self.runtime = _LazyHMONNXRuntime(export_root / meta["hmonnx_file"], self._device, PrecisionMode.ALIGNED)
        self.config = SimpleNamespace(**transformer_config)
        self.config.patch_size = tuple(self.config.patch_size)
        self.rope = LingBotVideoRotaryEmbedding(
            tuple(self.config.axes_dims),
            tuple(self.config.axes_lens),
            float(self.config.rope_theta),
        )
        self.timestep_values = torch.tensor(meta["timestep_values"], dtype=torch.float32)
        self.text_sequence_length = int(meta["text_sequence_length"])

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype_marker.dtype

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._device = torch.device(device)
            self.runtime.to(self._device)
        marker_dtype = dtype if dtype is not None else self._dtype_marker.dtype
        self._dtype_marker.data = self._dtype_marker.data.to(device=self._device, dtype=marker_dtype)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Any:
        del kwargs
        if hidden_states.shape[0] != 1:
            raise ValueError("LingBot HMONNX transformer supports batch_size=1 only.")
        valid_text_length = int(
            encoder_attention_mask.sum().item()
            if encoder_attention_mask is not None
            else encoder_hidden_states.shape[1]
        )
        if valid_text_length > self.text_sequence_length:
            raise ValueError(f"Text length {valid_text_length} exceeds export limit {self.text_sequence_length}.")
        encoder_hidden_states = encoder_hidden_states[:, :valid_text_length]
        padding = self.text_sequence_length - valid_text_length
        if padding:
            encoder_hidden_states = torch.cat(
                (
                    encoder_hidden_states,
                    torch.zeros(
                        1,
                        padding,
                        encoder_hidden_states.shape[-1],
                        device=encoder_hidden_states.device,
                        dtype=encoder_hidden_states.dtype,
                    ),
                ),
                dim=1,
            )
        hidden_states = hidden_states.to(self._device, dtype=torch.float16)
        encoder_hidden_states = encoder_hidden_states.to(self._device, dtype=torch.float16)
        rotary_cos, rotary_sin, current_input_length, valid_token_mask = (
            LingBotVideoTransformerExportWrapper.build_rotary_inputs(
                self,
                hidden_states,
                valid_text_length=valid_text_length,
                padded_text_length=self.text_sequence_length,
            )
        )
        timestep_index = (
            torch.argmin(torch.abs(self.timestep_values - timestep.detach().float().cpu()[0]))
            .reshape(1)
            .to(self._device, dtype=torch.int32)
        )
        sample = self.runtime(
            hidden_states,
            encoder_hidden_states,
            timestep_index,
            rotary_cos,
            rotary_sin,
            current_input_length,
            valid_token_mask,
        )
        if isinstance(sample, tuple):
            sample = sample[0]
        if not return_dict:
            return (sample,)
        return Transformer2DModelOutput(sample=sample)


class LingBotWanVAEInference(nn.Module):
    def __init__(self, export_root: Path, config_file: Path, device: torch.device):
        super().__init__()
        export_root = Path(export_root)
        root_meta = json.loads((export_root.parent / "export_meta_info.json").read_text(encoding="utf-8"))["vae"]
        vae_config = json.loads(Path(config_file).read_text(encoding="utf-8"))
        self._device = torch.device(device)
        self._dtype_marker = nn.Parameter(
            torch.empty(0, device=self._device, dtype=torch.float16),
            requires_grad=False,
        )
        self.encoder_runtime = _LazyHMONNXRuntime(export_root / root_meta["encoder"]["hmonnx_file"], self._device)
        decoder_meta = root_meta["decoder"]
        if decoder_meta.get("type") != "stateful_temporal_cache":
            raise ValueError(
                "Incompatible LingBot VAE decoder artifact: full-duration video "
                "requires the stateful_temporal_cache export. Re-export this profile."
            )
        self.decoder_meta = decoder_meta
        self.decoder_first_runtime = _LazyHMONNXRuntime(
            export_root / decoder_meta["first"]["hmonnx_file"], self._device
        )
        self.decoder_next_runtime = None
        if decoder_meta.get("next") is not None:
            self.decoder_next_runtime = _LazyHMONNXRuntime(
                export_root / decoder_meta["next"]["hmonnx_file"], self._device
            )
        self.config = SimpleNamespace(**vae_config)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype_marker.dtype

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._device = torch.device(device)
            self.encoder_runtime.to(self._device)
            self.decoder_first_runtime.to(self._device)
            if self.decoder_next_runtime is not None:
                self.decoder_next_runtime.to(self._device)
        marker_dtype = dtype if dtype is not None else self._dtype_marker.dtype
        self._dtype_marker.data = self._dtype_marker.data.to(device=self._device, dtype=marker_dtype)
        return self

    def enable_tiling(self, *args, **kwargs):
        del args, kwargs
        return self

    def disable_tiling(self):
        return self

    def encode(self, sample: torch.Tensor, return_dict: bool = True, **kwargs):
        del kwargs
        parameters = self.encoder_runtime(_prepare_hmonnx_tensor(sample, self._device, torch.float16))
        if isinstance(parameters, tuple):
            parameters = parameters[0]
        distribution = DiagonalGaussianDistribution(parameters)
        if not return_dict:
            return (distribution,)
        return AutoencoderKLOutput(latent_dist=distribution)

    def decode(self, latents: torch.Tensor, return_dict: bool = True, **kwargs):
        del kwargs
        latent_frames = int(latents.shape[2])
        if latent_frames <= 0:
            raise ValueError("LingBot stateful VAE decoder requires at least one latent frame.")
        prepared = _prepare_hmonnx_tensor(latents, self._device, torch.float16)
        first_outputs = self.decoder_first_runtime(prepared[:, :, :1])
        first_outputs = first_outputs if isinstance(first_outputs, tuple) else (first_outputs,)
        expected_outputs = int(self.decoder_meta["cache_count"]) + 1
        if len(first_outputs) != expected_outputs:
            raise RuntimeError(
                f"Wan first-frame graph returned {len(first_outputs)} outputs; expected {expected_outputs}."
            )
        samples = [first_outputs[0]]
        caches = first_outputs[1:]
        for frame_index in range(1, latent_frames):
            if self.decoder_next_runtime is None:
                raise RuntimeError("Wan next-frame graph is missing for a video artifact.")
            next_outputs = self.decoder_next_runtime(prepared[:, :, frame_index : frame_index + 1], *caches)
            next_outputs = next_outputs if isinstance(next_outputs, tuple) else (next_outputs,)
            if len(next_outputs) != expected_outputs:
                raise RuntimeError(
                    f"Wan next-frame graph returned {len(next_outputs)} outputs; expected {expected_outputs}."
                )
            samples.append(next_outputs[0])
            caches = next_outputs[1:]
        sample = torch.cat(samples, dim=2)
        first_sample_frames = int(self.decoder_meta["first_sample_frames"])
        next_sample_frames = int(self.decoder_meta["next_sample_frames"])
        expected_sample_frames = first_sample_frames + next_sample_frames * (latent_frames - 1)
        if sample.shape[2] != expected_sample_frames:
            raise RuntimeError(
                "Wan stateful decoder output frame mismatch: "
                f"expected={expected_sample_frames}, output={sample.shape[2]}."
            )
        if not return_dict:
            return (sample,)
        return DecoderOutput(sample=sample)


def _prepare_hmonnx_tensor(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalize dtype, device, and layout at an HMONNX graph boundary."""
    return tensor.to(device=device, dtype=dtype).contiguous()


def build_hmonnx_pipeline_components(
    *, export_dir: Path, device: torch.device
) -> tuple[nn.Module, nn.Module, nn.Module, LingBotQwen3VLProcessor]:
    export_dir = Path(export_dir).resolve()
    text_encoder = LingBotQwen3VLTextEncoderInference(
        export_root=export_dir / "text_encoder",
        config_file=resolve_runtime_artifact(export_dir, "text_encoder_config"),
        device=device,
    )
    transformer = LingBotVideoTransformerInference(
        export_root=export_dir / "transformer",
        config_file=resolve_runtime_artifact(export_dir, "transformer_config"),
        device=device,
    )
    vae = LingBotWanVAEInference(
        export_root=export_dir / "vae",
        config_file=resolve_runtime_artifact(export_dir, "vae_config"),
        device=device,
    )
    processor = LingBotQwen3VLProcessor.from_pretrained(resolve_runtime_artifact(export_dir, "processor"))
    return transformer, vae, text_encoder, processor


def build_hmonnx_pipeline_components_from_dirs(
    *,
    text_visual_export_dir: Path,
    transformer_export_dir: Path,
    vae_export_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, nn.Module, LingBotQwen3VLProcessor]:
    text_visual_export_dir = Path(text_visual_export_dir).resolve()
    transformer_export_dir = Path(transformer_export_dir).resolve()
    vae_export_dir = Path(vae_export_dir).resolve()
    text_encoder = LingBotQwen3VLTextEncoderInference(
        export_root=text_visual_export_dir / "text_encoder",
        config_file=resolve_runtime_artifact(text_visual_export_dir, "text_encoder_config"),
        device=device,
    )
    transformer = LingBotVideoTransformerInference(
        export_root=transformer_export_dir / "transformer",
        config_file=resolve_runtime_artifact(transformer_export_dir, "transformer_config"),
        device=device,
    )
    vae = LingBotWanVAEInference(
        export_root=vae_export_dir / "vae",
        config_file=resolve_runtime_artifact(vae_export_dir, "vae_config"),
        device=device,
    )
    processor = LingBotQwen3VLProcessor.from_pretrained(resolve_runtime_artifact(text_visual_export_dir, "processor"))
    return transformer, vae, text_encoder, processor


def configure_pipeline_token_length(pipeline: Any, text_encoder: LingBotQwen3VLTextEncoderInference) -> None:
    pipeline.token_length = int(text_encoder.sequence_length)
