# Copyright 2025 HOUMO AI
#
# File: flux2_klein_converter.py
# Description:
#   FLUX.2-klein-4B converter implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from xhquant.api import (
    ConfigDict,
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_dynamo_model_to_quanted_model,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)
from xhquant.quantization.xh2a.builder import register_none_quanted_module


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_vendored_diffusers() -> None:
    vendored_diffusers = _repo_root() / "data" / "difs" / "diffusers-main" / "src"
    vendored_str = str(vendored_diffusers)
    if vendored_str not in sys.path:
        sys.path.insert(0, vendored_str)


_ensure_vendored_diffusers()

from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel  # noqa: E402

from .text_encoder_hmonnx import (  # noqa: E402
    DEFAULT_TEXT_ENCODER_OUT_LAYERS,
    Flux2KleinTextEncoderExportWrapper,
    build_qwen3_prompt_embeds,
    build_qwen3_text_inputs,
)
from .transformer_wrapper import Flux2TransformerExportWrapper, register_transformer_wrap_modules  # noqa: E402
from .vae_wrapper import Flux2VAEDecoderWrapper, Flux2VAEEncoderWrapper, register_vae_wrap_modules  # noqa: E402

torch.fx.wrap("len")


FLUX_EXPORT_COMPONENTS = ("text_encoder", "transformer", "vae", "vae_encoder")
FLUX_DEFAULT_EXPORT_COMPONENTS = ("text_encoder", "transformer", "vae")


@dataclass
class Flux2KleinConvertConfig:
    quant_scheme: QuantScheme = field(default_factory=QuantScheme)
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    width: int = 1024
    height: int = 1024
    prompt: str = "A cat holding a sign that says hello world"
    seed: int = 0
    batch_size: int = 1
    max_sequence_length: int = 512
    input_sequence_length: int = 256
    text_encoder_out_layers: Tuple[int, ...] = DEFAULT_TEXT_ENCODER_OUT_LAYERS
    export_components: Tuple[str, ...] = FLUX_DEFAULT_EXPORT_COMPONENTS
    torch_dtype: torch.dtype = torch.float16
    image_edit: bool = False


class Flux2KleinConverter:
    def __init__(self, pretrained_model_path: str, convert_config: Flux2KleinConvertConfig):
        self.pretrained_model_path = Path(pretrained_model_path)
        self.config = convert_config
        self.logger = get_root_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.export_components = self._normalize_components(self.config.export_components)

    @staticmethod
    def _normalize_components(components: Sequence[str]) -> Tuple[str, ...]:
        if not components:
            raise ValueError("export_components cannot be empty")

        normalized = []
        for component in components:
            name = component.strip().lower()
            if name == "vae_decoder":
                name = "vae"
            if name == "vae_encode":
                name = "vae_encoder"
            if name not in FLUX_EXPORT_COMPONENTS:
                raise ValueError(
                    f"Unsupported export component: {component}. Expected one of {FLUX_EXPORT_COMPONENTS}."
                )
            if name not in normalized:
                normalized.append(name)
        return tuple(normalized)

    @staticmethod
    def _prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = prompt_embeds.shape
        out_ids = []
        for _ in range(batch_size):
            coords = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(seq_len))
            out_ids.append(coords)
        return torch.stack(out_ids)

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = latents.shape
        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)
        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_ids

    @staticmethod
    def _prepare_image_ids(image_latents: torch.Tensor, scale: int = 10) -> torch.Tensor:
        batch_size, _, height, width = image_latents.shape
        image_ids = torch.cartesian_prod(torch.arange(scale, scale + 1), torch.arange(height), torch.arange(width), torch.arange(1))
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return image_ids

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = latents.shape
        return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

    def _load_text_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.pretrained_model_path / "tokenizer", trust_remote_code=True)

    def _load_text_encoder(self):
        return AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_path / "text_encoder",
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=True,
        ).eval()

    def _load_transformer(self):
        register_none_quanted_module(torch.nn.RMSNorm)
        transformer = Flux2Transformer2DModel.from_pretrained(
            self.pretrained_model_path / "transformer",
            torch_dtype=self.config.torch_dtype,
            low_cpu_mem_usage=False,
        )
        return transformer.eval()

    def _load_vae(self):
        return AutoencoderKLFlux2.from_pretrained(
            self.pretrained_model_path / "vae",
            torch_dtype=self.config.torch_dtype,
            low_cpu_mem_usage=False,
        ).eval()

    def _build_prompt_embeds(
        self,
        text_encoder,
        tokenizer,
        prompt: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        prompt_embeds, model_inputs = build_qwen3_prompt_embeds(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            max_sequence_length=self.config.max_sequence_length,
            hidden_states_layers=self.config.text_encoder_out_layers,
            dtype=self.config.torch_dtype,
            device=self.device,
        )
        text_ids = self._prepare_text_ids(prompt_embeds).to(self.device)
        return prompt_embeds, text_ids, model_inputs

    def _build_transformer_inputs(
        self,
        transformer: Flux2Transformer2DModel,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        generator = torch.Generator(device=self.device.type).manual_seed(self.config.seed)
        vae_scale_factor = 8
        height = 2 * (int(self.config.height) // (vae_scale_factor * 2))
        width = 2 * (int(self.config.width) // (vae_scale_factor * 2))
        shape = (
            self.config.batch_size,
            transformer.config.in_channels,
            height // 2,
            width // 2,
        )
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.config.torch_dtype)
        latent_ids = self._prepare_latent_ids(latents).to(self.device)
        packed_latents = self._pack_latents(latents)

        if self.config.image_edit:
            image_latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.config.torch_dtype)
            image_latent_ids = self._prepare_image_ids(image_latents).to(self.device)
            packed_image_latents = self._pack_latents(image_latents)
            packed_latents = torch.cat([packed_latents, packed_image_latents], dim=1)
            latent_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

        timestep = torch.tensor([1.0], device=self.device, dtype=torch.float32).expand(self.config.batch_size)
        guidance = torch.full(
            [self.config.batch_size],
            self.config.guidance_scale,
            device=self.device,
            dtype=torch.float32,
        )

        concat_rotary_cos, concat_rotary_sin = Flux2TransformerExportWrapper.build_rotary_inputs(
            transformer, latent_ids, text_ids
        )

        return (
            packed_latents,
            prompt_embeds,
            timestep,
            concat_rotary_cos,
            concat_rotary_sin,
            guidance,
            latent_ids,
        )

    def _validate_transformer_wrapper_output(
        self,
        transformer_wrapper: Flux2TransformerExportWrapper,
        transformer_inputs: Sequence[torch.Tensor],
        ref_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        hidden_states, prompt_embeds, timestep, concat_rotary_cos, concat_rotary_sin, guidance, latent_ids = transformer_inputs
        timestep_index = torch.zeros_like(timestep, dtype=torch.int32)
        wrapper_inputs = (
            hidden_states,
            prompt_embeds,
            timestep_index,
            concat_rotary_cos,
            concat_rotary_sin,
            guidance.half(),
        )

        with torch.no_grad():
            wrapped_output = transformer_wrapper(*wrapper_inputs)

        ref_float = ref_output.float()
        wrapped_float = wrapped_output.float()
        ref_nan = torch.isnan(ref_float)
        wrapped_nan = torch.isnan(wrapped_float)
        nan_mismatch = torch.logical_xor(ref_nan, wrapped_nan).sum().item()
        finite_mask = torch.isfinite(ref_float) & torch.isfinite(wrapped_float)
        if finite_mask.any():
            diff = (ref_float - wrapped_float).abs()
            max_abs = diff.max().item()
            denom = ref_float.abs().clamp_min(1e-6)
            max_rel = (diff / denom).max().item()
        else:
            max_abs = 0.0 if nan_mismatch == 0 else float("nan")
            max_rel = 0.0 if nan_mismatch == 0 else float("nan")
        self.logger.info(
            "Flux2 transformer wrapper consistency: max_abs=%.6g, max_rel=%.6g, ref_nan=%d, wrapped_nan=%d, nan_mismatch=%d",
            max_abs,
            max_rel,
            ref_nan.sum().item(),
            wrapped_nan.sum().item(),
            nan_mismatch,
        )
        # if not torch.allclose(ref_float, wrapped_float, rtol=1e-2, atol=1e-2, equal_nan=True):
        #     raise RuntimeError(
        #         f"Flux2 transformer wrapper output mismatch: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}, "
        #         f"nan_mismatch={nan_mismatch}"
        #     )

        return wrapper_inputs

    def _export_text_encoder(self, output_dir: Path) -> Dict[str, Any]:
        from .text_encoder_wrapper import wrap_text_encoder_model

        output_dir.mkdir(exist_ok=True, parents=True)
        tokenizer = self._load_text_tokenizer()
        text_encoder = self._load_text_encoder().to(self.device)
        token_embedding = text_encoder.get_input_embeddings()
        token_embedding_file = output_dir / "token_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file))
        wrap_cfg = ConfigDict(
            dict(
                max_sequence_length=self.config.max_sequence_length,
                input_sequence_length=self.config.max_sequence_length,
                use_cache=False,
                num_logits_to_keep=0,
            )
        )
        wrapped_text_encoder = wrap_text_encoder_model(text_encoder, wrap_cfg)
        wrapper = Flux2KleinTextEncoderExportWrapper(
            text_encoder=wrapped_text_encoder,
            hidden_states_layers=self.config.text_encoder_out_layers,
            output_dtype=self.config.torch_dtype,
            max_sequence_length=self.config.max_sequence_length,
            batch_size=self.config.batch_size,
        ).to(self.device)
        wrapper.eval()

        sample_inputs = build_qwen3_text_inputs(tokenizer, self.config.prompt, self.config.max_sequence_length)
        input_ids = sample_inputs["input_ids"].to(device=self.device, dtype=torch.int32)
        attention_mask = sample_inputs["attention_mask"].to(device=self.device, dtype=torch.int32)
        inputs_embeds = token_embedding(input_ids).to(device=self.device, dtype=self.config.torch_dtype)

        prefix = (
            f"flux2_klein_text_encoder-"
            f"{self.config.quant_scheme.target_device}-{self.config.quant_scheme.quant_type}-"
            f"{self.config.batch_size}x{self.config.max_sequence_length}"
        )
        golden_dir = output_dir / "golden" / prefix
        export_meta = self._export_component(
            model=wrapper,
            inputs=[inputs_embeds, attention_mask],
            input_names=["inputs_embeds", "attention_mask"],
            output_names=["prompt_embeds"],
            onnx_file=output_dir / "hmonnx" / f"{prefix}.onnx",
            golden_dir=golden_dir,
            trace_backend="fx",
        )

        meta_info = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "model_name": self.pretrained_model_path.name,
            "source_model_dir": str(self.pretrained_model_path),
            "tokenizer_dir": str(self.pretrained_model_path / "tokenizer"),
            "text_encoder_dir": str(self.pretrained_model_path / "text_encoder"),
            "token_embedding_file": str(token_embedding_file.relative_to(output_dir)),
            "device": str(self.config.quant_scheme.target_device),
            "quant_scheme": self.config.quant_scheme.to_dict(),
            "max_sequence_length": self.config.max_sequence_length,
            "text_encoder_out_layers": list(self.config.text_encoder_out_layers),
            "prompt_format": {
                "add_generation_prompt": True,
                "enable_thinking": False,
                "messages": [{"role": "user", "content": self.config.prompt}],
            },
            "hmonnx_file": str(Path(export_meta["onnx"]).relative_to(output_dir)),
            "golden_dir": str(Path(export_meta["golden_dir"]).relative_to(output_dir)),
            "input_names": export_meta["input_names"],
            "output_names": export_meta["output_names"],
            "sample_input_shapes": {
                "inputs_embeds": list(inputs_embeds.shape),
                "attention_mask": list(attention_mask.shape),
            },
            "sample_token_input_shapes": {
                "input_ids": list(input_ids.shape),
                "attention_mask": list(attention_mask.shape),
            },
        }
        json.dump(meta_info, open(output_dir / "meta.json", "w"), indent=4)
        return meta_info

    def _export_component(
        self,
        model: nn.Module,
        inputs: Sequence[torch.Tensor],
        input_names: Sequence[str],
        output_names: Sequence[str],
        onnx_file: Path,
        golden_dir: Path,
        trace_backend: str = "fx",
    ) -> Dict[str, Any]:
        quant_config = ConfigDict(create_quant_config(self.config.quant_scheme))
        if trace_backend == "dynamo":
            quant_graph_model = convert_dynamo_model_to_quanted_model(
                model,
                list(inputs),
                self.config.quant_scheme.target_device,
                quant_config,
            )
        else:
            quant_graph_model = convert_fx_model_to_quanted_model(
                model,
                list(inputs),
                self.config.quant_scheme.target_device,
                quant_config,
            )
        convert_quanted_model_to_hmonnx(
            quant_graph_model,
            list(inputs),
            str(onnx_file),
            list(input_names),
            list(output_names),
        )

        hm_session = HMONNXGoldenInference(str(onnx_file))
        hm_session.save_golden = True
        hm_session.exec_device = self.device
        golden_dir.mkdir(exist_ok=True, parents=True)
        hm_session.golden_dir = str(golden_dir)
        with torch.no_grad():
            hm_session.forward(*list(inputs))

        return {
            "onnx": str(onnx_file),
            "golden_dir": str(golden_dir),
            "input_names": list(input_names),
            "output_names": list(output_names),
        }

    def _convert(self, work_dir: str) -> None:
        work_dir_path = Path(work_dir)
        work_dir_path.mkdir(exist_ok=True, parents=True)

        export_text_encoder = "text_encoder" in self.export_components
        export_transformer = "transformer" in self.export_components
        export_vae = "vae" in self.export_components
        export_vae_encoder = "vae_encoder" in self.export_components
        need_prompt_embeds = export_transformer or export_vae

        tokenizer = self._load_text_tokenizer() if need_prompt_embeds else None
        text_encoder = self._load_text_encoder().to(self.device) if need_prompt_embeds else None
        transformer = self._load_transformer().to(self.device) if (export_transformer or export_vae) else None
        vae = self._load_vae().to(self.device) if (export_vae or export_vae_encoder) else None

        prompt_embeds = text_ids = text_model_inputs = None
        if need_prompt_embeds:
            assert tokenizer is not None and text_encoder is not None
            prompt_embeds, text_ids, text_model_inputs = self._build_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=self.config.prompt,
            )

        text_encoder_meta = None
        if export_text_encoder:
            text_encoder_meta = self._export_text_encoder(work_dir_path / "text_encoder")

        transformer_inputs = None
        transformer_meta = None
        if export_transformer or export_vae:
            assert transformer is not None and prompt_embeds is not None and text_ids is not None
            transformer_inputs = self._build_transformer_inputs(transformer, prompt_embeds, text_ids)

        vae_registered = False

        if export_transformer:
            assert transformer is not None and transformer_inputs is not None
            hidden_states, prompt_embeds, timestep, _, _, guidance, latent_ids = transformer_inputs
            with torch.no_grad():
                transformer_origin_output = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    img_ids=latent_ids,
                    txt_ids=text_ids,
                    guidance=None,
                    return_dict=False,
                )[0]

            register_transformer_wrap_modules(transformer)
            
            transformer_wrapper = Flux2TransformerExportWrapper(transformer).to(self.device).eval()
            transformer_prefix = (
                f"flux2_klein_transformer-{self.config.quant_scheme.target_device}-{self.config.quant_scheme.quant_type}"
            )

            transformer_export_inputs = self._validate_transformer_wrapper_output(
                transformer_wrapper=transformer_wrapper,
                transformer_inputs=transformer_inputs,
                ref_output=transformer_origin_output,
            )

            transformer_meta = self._export_component(
                model=transformer_wrapper,
                inputs=transformer_export_inputs,
                input_names=[
                    "hidden_states",
                    "encoder_hidden_states",
                    "timestep",
                    "concat_rotary_cos",
                    "concat_rotary_sin",
                    "guidance",
                ],
                output_names=["sample"],
                onnx_file=work_dir_path / "hmonnx" / f"{transformer_prefix}.onnx",
                golden_dir=work_dir_path / "golden" / transformer_prefix,
                trace_backend="fx",
            )

        vae_meta = None
        vae_inputs = None
        vae_encoder_meta = None
        vae_encoder_inputs = None
        if export_vae_encoder:
            with torch.no_grad():
                vae_encoder_origin_output = vae.encode(
                    torch.randn(
                        (self.config.batch_size, 3, self.config.height, self.config.width),
                        device=self.device,
                        dtype=self.config.torch_dtype,
                    )
                )
            assert vae is not None
            register_vae_wrap_modules(vae)
            vae_registered = True
            vae_encoder_wrapper = Flux2VAEEncoderWrapper(vae).to(self.device).eval()

            with torch.no_grad():
                vae_encoder_warper_output = vae_encoder_wrapper(
                    torch.randn(
                        (self.config.batch_size, 3, self.config.height, self.config.width),
                        device=self.device,
                        dtype=self.config.torch_dtype,
                    )
                )

            vae_encoder_prefix = f"flux2_klein_vae_encoder-{self.config.quant_scheme.target_device}-{self.config.quant_scheme.quant_type}"
            draft_image = torch.randn(
                (self.config.batch_size, 3, self.config.height, self.config.width),
                device=self.device,
                dtype=self.config.torch_dtype,
            )
            vae_encoder_inputs = [draft_image]
            vae_encoder_meta = self._export_component(
                model=vae_encoder_wrapper,
                inputs=vae_encoder_inputs,
                input_names=["image"],
                output_names=["image_latents"],
                onnx_file=work_dir_path / "hmonnx" / f"{vae_encoder_prefix}.onnx",
                golden_dir=work_dir_path / "golden" / vae_encoder_prefix,
            )

        if export_vae:
            assert vae is not None and transformer_inputs is not None
            if not vae_registered:
                register_vae_wrap_modules(vae)
                vae_registered = True
            vae_wrapper = Flux2VAEDecoderWrapper(vae).to(self.device).eval()
            vae_prefix = f"flux2_klein_vae-{self.config.quant_scheme.target_device}-{self.config.quant_scheme.quant_type}"
            latent_ids = transformer_inputs[6]
            decode_latents = vae_wrapper.preprocess_decode_latents(transformer_inputs[0], latent_ids)
            vae_inputs = [decode_latents]
            vae_meta = self._export_component(
                model=vae_wrapper,
                inputs=vae_inputs,
                input_names=["latents"],
                output_names=["image"],
                onnx_file=work_dir_path / "hmonnx" / f"{vae_prefix}.onnx",
                golden_dir=work_dir_path / "golden" / vae_prefix,
            )

        meta_info: Dict[str, Any] = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "model_name": self.pretrained_model_path.name,
            "device": str(self.config.quant_scheme.target_device),
            "quant_scheme": self.config.quant_scheme.to_dict(),
            "export_components": list(self.export_components),
            "width": self.config.width,
            "height": self.config.height,
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
            "prompt": self.config.prompt,
            "text_encoder_out_layers": list(self.config.text_encoder_out_layers),
            "max_sequence_length": self.config.max_sequence_length,
            "input_sequence_length": self.config.input_sequence_length,
            "image_edit": self.config.image_edit,
        }

        if text_encoder_meta is not None:
            meta_info["text_encoder_meta"] = str(Path("text_encoder") / "meta.json")

        if transformer_meta is not None:
            meta_info["transformer_hmonnx"] = str(Path(transformer_meta["onnx"]).relative_to(work_dir_path))
            meta_info["transformer_golden_dir"] = str(Path(transformer_meta["golden_dir"]).relative_to(work_dir_path))

        if vae_meta is not None:
            meta_info["vae_hmonnx"] = str(Path(vae_meta["onnx"]).relative_to(work_dir_path))
            meta_info["vae_golden_dir"] = str(Path(vae_meta["golden_dir"]).relative_to(work_dir_path))

        if vae_encoder_meta is not None:
            meta_info["vae_encoder_hmonnx"] = str(Path(vae_encoder_meta["onnx"]).relative_to(work_dir_path))
            meta_info["vae_encoder_golden_dir"] = str(Path(vae_encoder_meta["golden_dir"]).relative_to(work_dir_path))

        sample_input_shapes: Dict[str, Any] = {}
        if transformer_inputs is not None:
            sample_input_shapes["transformer"] = [list(t.shape) for t in transformer_inputs[:6]]
            sample_input_shapes["latent_ids"] = list(transformer_inputs[6].shape)
        if vae_inputs is not None:
            sample_input_shapes["vae"] = [list(t.shape) for t in vae_inputs]
        if vae_encoder_inputs is not None:
            sample_input_shapes["vae_encoder"] = [list(t.shape) for t in vae_encoder_inputs]
        if sample_input_shapes:
            meta_info["sample_input_shapes"] = sample_input_shapes

        if text_model_inputs is not None:
            meta_info["sample_text_input_shapes"] = {key: list(value.shape) for key, value in text_model_inputs.items()}

        json.dump(meta_info, open(work_dir_path / "meta.json", "w"), indent=4)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, convert_config: Flux2KleinConvertConfig, work_dir: str):
        converter = cls(pretrained_model_path, convert_config)
        converter._convert(work_dir)