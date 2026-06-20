import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import torch
from torch import nn
from transformers import AutoModelForImageTextToText
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration, Gemma4RMSNorm
from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph
import xhquant.nn.modules as xhnn

from ...base_vision_model import BaseVisionModel
from ...llm_data_processor import BaseVisualProcessor
from .gemma4_series_processor import XHGemma4Processor
from .xh_gemma4_series_config import Gemma4SeriesAudioModelMeta, XHGemma4SeriesAudioConfig


def _onnx_scalar_constant_map(model: onnx.ModelProto) -> dict[str, float]:
    constants: dict[str, float] = {}
    for initializer in model.graph.initializer:
        array = onnx.numpy_helper.to_array(initializer)
        if array.size == 1:
            constants[initializer.name] = float(np.asarray(array).reshape(-1)[0])

    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            array = onnx.numpy_helper.to_array(attr.t)
            if array.size == 1:
                constants[node.output[0]] = float(np.asarray(array).reshape(-1)[0])
            break
    return constants


def _is_noop_clip(node: onnx.NodeProto, constants: dict[str, float]) -> bool:
    if node.op_type != "Clip" or len(node.input) < 3:
        return False
    min_value = constants.get(node.input[1])
    max_value = constants.get(node.input[2])
    return bool(min_value is not None and max_value is not None and np.isneginf(min_value) and np.isposinf(max_value))


def _remove_noop_clip_nodes(onnx_file: str | Path) -> int:
    """Remove ONNX Clip nodes whose bounds are exactly ``[-inf, +inf]``.

    xhquant export may preserve HF no-op clamps as explicit Clip nodes.  Only
    the mathematically identity case is rewritten; finite activation/quant clips
    are intentionally retained.
    """
    onnx_path = Path(onnx_file)
    model = onnx.load(str(onnx_path), load_external_data=True)
    constants = _onnx_scalar_constant_map(model)
    replacements: dict[str, str] = {}
    kept_nodes = []
    removed = 0
    for node in model.graph.node:
        if _is_noop_clip(node, constants):
            replacements[node.output[0]] = node.input[0]
            removed += 1
        else:
            kept_nodes.append(node)

    if not removed:
        return 0

    for node in kept_nodes:
        for idx, value in enumerate(node.input):
            while value in replacements:
                value = replacements[value]
            node.input[idx] = value
    for output in model.graph.output:
        while output.name in replacements:
            output.name = replacements[output.name]

    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    onnx.save(model, str(onnx_path))
    return removed


def _replace_rmsnorm(module: nn.Module) -> int:
    # Gemma4 audio tower contains ~108 Gemma4RMSNorm modules; in HF, each one
    # decomposes into Pow + ReduceMean + Add + Sqrt + Div + Mul + Mul during
    # ONNX export. Under w8a8h1_sefp the intermediate Pow/ReduceMean lose
    # precision on the squared activations, which propagates as a ~scale-1
    # bias through every encoder layer and ultimately makes the audio embeds
    # disconnected from the input. Replace with the fused xhnn.RMSNorm so the
    # whole reduction is one quantization-aware op (visual tower already does
    # this). Audio uses with_scale=True everywhere, so the simple variant is
    # enough.
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, Gemma4RMSNorm):
            if not getattr(child, "with_scale", True):
                # with_scale=False means no weight param; hidden_size must be
                # inferred from the parent module (e.g. Gemma4MultimodalEmbedder
                # stores it as multimodal_hidden_size).
                hidden_size = getattr(module, "multimodal_hidden_size", None)
                if hidden_size is None:
                    # Cannot determine hidden_size; leave untouched for the
                    # _Gemma4RMSNorm wrapper to handle during wrap_llm_model.
                    replaced += _replace_rmsnorm(child)
                    continue
                fused = xhnn.RMSNorm(hidden_size, eps=child.eps)
                setattr(module, name, fused)
                replaced += 1
                continue
            hidden_size = child.weight.shape[0]
            device = child.weight.device
            dtype = child.weight.dtype
            fused = xhnn.RMSNorm(hidden_size, eps=child.eps)
            fused.weight.data.copy_(child.weight.data)
            fused = fused.to(device=device, dtype=dtype)
            setattr(module, name, fused)
            replaced += 1
        else:
            replaced += _replace_rmsnorm(child)
    return replaced


class _Gemma4AudioExportBridge(nn.Module):
    def __init__(self, hf_model: Gemma4ForConditionalGeneration):
        super().__init__()
        self.audio_tower = hf_model.model.audio_tower
        self.embed_audio = hf_model.model.embed_audio

    def forward(self, input_features, input_features_mask, audio_attention_mask):
        audio_outputs = self.audio_tower(
            input_features=input_features,
            attention_mask=input_features_mask,
            audio_attention_mask=audio_attention_mask,
        )
        output_mask = None
        if hasattr(audio_outputs, "last_hidden_state"):
            hidden_states = audio_outputs.last_hidden_state
            output_mask = getattr(audio_outputs, "output_mask", None)
        elif isinstance(audio_outputs, (tuple, list)):
            hidden_states = audio_outputs[0]
            if len(audio_outputs) > 1:
                output_mask = audio_outputs[1]
        else:
            hidden_states = audio_outputs

        audio_embeds = self.embed_audio(inputs_embeds=hidden_states)
        if isinstance(audio_embeds, (tuple, list)):
            audio_embeds = audio_embeds[0]
        if output_mask is None:
            return audio_embeds
        return audio_embeds, output_mask


class _Gemma4AudioProcessor(BaseVisualProcessor):
    def __init__(self, audio_config: XHGemma4SeriesAudioConfig):
        self.audio_config = audio_config

    def forward(self, data: dict) -> list[torch.Tensor]:
        assert isinstance(data, dict), "Input data should be a dictionary."
        input_features = data.get("input_features")
        input_features_mask = data.get("input_features_mask")
        audio_attention_mask = data.get("audio_attention_mask")
        assert input_features is not None and input_features_mask is not None, (
            "Gemma4 audio export requires both `input_features` and `input_features_mask`."
        )
        input_features_mask = input_features_mask.to(torch.float16)
        if audio_attention_mask is None:
            audio_attention_mask = XHGemma4Processor.build_audio_attention_mask(
                input_features_mask,
                chunk_size=self.audio_config.attention_chunk_size,
                context_left=self.audio_config.attention_context_left,
                context_right=self.audio_config.attention_context_right,
                dtype=torch.float16,
            )
        return (input_features, input_features_mask, audio_attention_mask)


class XHGemma4SeriesAudioModel(BaseVisionModel):
    HF_MODEL_CLS = Gemma4ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = Gemma4SeriesAudioModelMeta
    CONFIG_CLS = XHGemma4SeriesAudioConfig

    def __init__(self, config: XHGemma4SeriesAudioConfig):
        super().__init__(config)
        self.config = cast(XHGemma4SeriesAudioConfig, self.config)
        self._onnx_artifact_dir: Path | None = None

    def init_wrap_model(self, hf_model: Any = None) -> Any:
        from ._audio_model_impl import register_wrap_modules

        register_wrap_modules(hf_model)
        if hf_model is not None and hasattr(hf_model, "model") and hasattr(hf_model.model, "audio_tower"):
            replaced = _replace_rmsnorm(hf_model.model.audio_tower)
            replaced += _replace_rmsnorm(hf_model.model.embed_audio)
            if replaced:
                get_xhquant_logger().info(
                    f"Replaced {replaced} Gemma4RMSNorm modules in audio tower with fused xhnn.RMSNorm"
                )
            hf_model = _Gemma4AudioExportBridge(hf_model)
        return super().init_wrap_model(hf_model)

    def get_tf_processor(self):
        processor = XHGemma4Processor.from_pretrained(self.hf_model_dir)
        processor.config.sampling_rate = self.config.sampling_rate
        processor.config.audio_feature_length = self.config.input_feature_length
        processor.config.audio_attention_chunk_size = self.config.attention_chunk_size
        processor.config.audio_attention_context_left = self.config.attention_context_left
        processor.config.audio_attention_context_right = self.config.attention_context_right
        return processor

    def _get_data_preprocessor(self) -> BaseVisualProcessor:
        return _Gemma4AudioProcessor(self.config)

    def get_dummy_inputs(self) -> Any:
        processor = self.get_tf_processor()
        rng = np.random.default_rng(42)
        duration_samples = self.config.sampling_rate
        dummy_audio = rng.standard_normal(duration_samples).astype(np.float32) * 0.1
        model_inputs = processor(
            text="<|audio|>Transcribe this audio.",
            audio=dummy_audio,
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
        )
        return {
            "input_features": model_inputs["input_features"],
            "input_features_mask": model_inputs["input_features_mask"],
            "audio_attention_mask": model_inputs["audio_attention_mask"],
        }

    def _to_fronted(self, wrap_model):
        logger = get_xhquant_logger()
        dummy_inputs = self.get_dummy_inputs()
        input_names = list(dummy_inputs.keys())
        dummy_values = tuple(
            value.float().cpu() if value.is_floating_point() else value.cpu()
            for value in dummy_inputs.values()
        )

        work_dir = self.config.work_dir
        tmp_dir_ctx = None
        if not work_dir:
            tmp_dir_ctx = tempfile.TemporaryDirectory()
            work_dir = tmp_dir_ctx.name

        try:
            onnx_file = str(Path(work_dir) / "onnx" / "gemma4_audio.onnx")
            Path(onnx_file).parent.mkdir(parents=True, exist_ok=True)
            if not Path(onnx_file).exists():
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                    torch.onnx.export(
                        wrap_model.float().cpu(),
                        dummy_values,
                        tmp_onnx_file,
                        export_params=True,
                        opset_version=18,
                        do_constant_folding=True,
                        input_names=input_names,
                        output_names=["audio_embeds", "audio_embeds_mask"],
                        verbose=False,
                    )
                    onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)
                    from xhquant.utils.onnx_simplify import onnx_simplify

                    onnx_model_simp, check = onnx_simplify(onnx_model)
                    if check:
                        onnx_model = onnx_model_simp
                    onnx.save(
                        onnx_model,
                        onnx_file,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=f"{Path(onnx_file).stem}_external_data",
                    )
                self._wrap_model.to(self.device, self.dtype)
            else:
                logger.info(f"from cached onnx: {onnx_file}")

            self._onnx_artifact_dir = Path(onnx_file).parent
            onnx_model = onnx.load(onnx_file)
            return to_frontend_graph(onnx_model, FrontendType.ONNX, list(dummy_values))
        finally:
            if tmp_dir_ctx is not None:
                tmp_dir_ctx.cleanup()

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": ["input_features", "input_features_mask", "audio_attention_mask"],
            "output_names": ["audio_embeds", "audio_embeds_mask"],
        }

    def export_hmonnx(self, output_dir: str) -> Gemma4SeriesAudioModelMeta:
        meta_info = self.create_export_metadata(output_dir)
        exported_hmonnx_file = super()._export_hmonnx(output_dir)
        removed_noop_clips = _remove_noop_clip_nodes(exported_hmonnx_file)
        if removed_noop_clips:
            get_xhquant_logger().info(
                f"Removed {removed_noop_clips} no-op Clip(-inf, inf) nodes from Gemma4 audio HMONNX"
            )
        meta_info.hmonnx = str(exported_hmonnx_file)
        # The raw torch-exported audio ONNX is only an intermediate for xhquant
        # frontend conversion.  It still contains decomposed RMSNorm
        # (ReduceMean/Pow/Sqrt) nodes, while the deployable HMONNX above is
        # fused.  Do not copy or advertise the raw graph in the exported package.
        source_onnx_dir = self._onnx_artifact_dir or Path(self.config.work_dir) / "onnx"
        target_onnx_dir = Path(output_dir) / "onnx"
        removed_dirs: list[str] = []
        for onnx_dir in (source_onnx_dir, target_onnx_dir):
            if onnx_dir.exists():
                resolved = str(onnx_dir.resolve())
                if resolved in removed_dirs:
                    continue
                shutil.rmtree(onnx_dir, ignore_errors=True)
                removed_dirs.append(resolved)
        meta_info.onnx = None
        return meta_info

    def create_export_metadata(self, output_dir: str) -> Gemma4SeriesAudioModelMeta:
        meta_info = cast(Gemma4SeriesAudioModelMeta, self.get_export_metadata_cls()())
        meta_info.sampling_rate = self.config.sampling_rate
        meta_info.feature_size = self.config.feature_size
        meta_info.input_feature_length = self.config.input_feature_length
        meta_info.attention_chunk_size = self.config.attention_chunk_size
        meta_info.attention_context_left = self.config.attention_context_left
        meta_info.attention_context_right = self.config.attention_context_right
        return meta_info


# Compatibility aliases for existing Gemma4 code paths.
XHGemma4AudioModel = XHGemma4SeriesAudioModel


__all__ = [
    "XHGemma4AudioModel",
    "XHGemma4SeriesAudioModel",
]
