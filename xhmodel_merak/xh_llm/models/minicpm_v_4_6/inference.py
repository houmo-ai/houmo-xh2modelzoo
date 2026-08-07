"""Root-metadata-driven MiniCPM-V-4.6 HMONNX inference."""

from __future__ import annotations

import gc
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel
from xhmodel_merak.xh_llm.hmonnx.vision_llm_hmonnx_model import (
    VisonLLMHMONNXModel,
)
from xhmodel_merak.xh_llm.infer_mixin import LLMInferenceContextManager
from xhmodel_merak.xh_llm.models.qwen3_5.data_preprocess import (
    Qwen3_5_DataPreprocess,
)
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
    Qwen3_5HMONNXKVCacheMixin,
    XHQwen3_5_HMONNXModel,
)
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import (
    Qwen3_5_ModelMeta,
)

from .model import XHMiniCPMV46Model
from .vision import (
    crop_token_capacity_image_embeds,
    image_token_count,
    prepare_token_capacity_vision_inputs,
    split_packed_pixel_values,
    validate_token_capacity,
)


class MiniCPMV46TextHMONNXModel(XHQwen3_5_HMONNXModel):
    """Qwen3.5 cache runtime without Qwen's native Vision session."""

    LLM_MODEL_CLS = XHMiniCPMV46Model

    def __init__(self, meta_info, **kwargs):
        VisonLLMHMONNXModel.__init__(self, meta_info, **kwargs)
        self.visual_meta = None
        self.visual = None
        self._kvcache_mixin = Qwen3_5HMONNXKVCacheMixin(self.kvcache_config)
        self._kvcache_mixin.split_conv_cache = bool(getattr(meta_info.model_config, "split_conv_cache", False))
        self._sync_page_attention_mode_to_kvcache()

    def _set_device(self, device):
        return VisonLLMHMONNXModel._set_device(self, device)

    def _set_dtype(self, dtype):
        self._dtype = dtype
        self.embed_tokens.to(dtype=dtype)
        for model in self._models.values():
            model.to(dtype=dtype)
        return self

    def _get_data_preprocessor(self) -> Qwen3_5_DataPreprocess:
        return Qwen3_5_DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.get_input_sequence_length(),
            image_size_w=448,
            image_size_h=448,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            patch_size=14,
            image_token_id=self.meta_info.model_config.image_token_id,
            video_token_id=self.meta_info.model_config.video_token_id,
            vision_start_token_id=self.meta_info.model_config.vision_start_token_id,
            vision_end_token_id=self.meta_info.model_config.vision_end_token_id,
            spatial_merge_size=2,
            enable_page_attention=self.enable_page_attention,
        )

    def _set_enable_golden(self, enable: bool) -> None:
        # HMONNXBaseModel's property setter has already propagated this to
        # Prefill and Decode before calling this subclass hook. Avoid the
        # Qwen3.5 hook because MiniCPM has no Qwen Vision model.
        del enable


@dataclass
class PreparedMultimodalInputs:
    input_ids: torch.Tensor
    image_embeds: torch.Tensor | None
    processor: Any


class MiniCPMV46HMONNXRuntime:
    """Load all runtime artifacts through the root export metadata."""

    def __init__(
        self,
        export_meta: str | Path,
        *,
        device: str = "cuda:0",
        enable_golden: bool = False,
    ):
        self.meta_file = Path(export_meta).resolve()
        if not self.meta_file.is_file():
            raise FileNotFoundError(f"MiniCPM export metadata not found: {self.meta_file}")
        self.work_dir = self.meta_file.parent
        self.meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
        if self.meta.get("model_type") != "MiniCPM-V-4.6":
            raise ValueError(f"Not a MiniCPM-V-4.6 export: {self.meta_file}")
        self.device = torch.device(device)
        self.enable_golden = bool(enable_golden)
        self.text: MiniCPMV46TextHMONNXModel | None = None
        if "llm" in self.meta:
            child_meta_file = self._resolve(self.meta["llm"]["metadata"])
            child_meta_dict = json.loads(child_meta_file.read_text(encoding="utf-8"))
            child_meta_dict["_meta_path_"] = str(child_meta_file)
            llm_meta = Qwen3_5_ModelMeta.from_dict(child_meta_dict)
            self.text = MiniCPMV46TextHMONNXModel(
                llm_meta,
                enable_golden=enable_golden,
                device_map=[self.device],
            )
            self.text.to(self.device)
            # The legacy HMONNXModel constructor records ``enable_golden``
            # without invoking its setter. Re-apply it at the aggregate
            # model level so Prefill and Decode sessions enable dumping.
            self.text.enable_golden = self.enable_golden
        self._vision_sessions: dict[str, Any] = {}
        self._vision_golden_steps: dict[str, int] = {}
        self._processor = None
        if self.enable_golden and self.text is not None:
            self._configure_text_golden_dirs()

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.work_dir / path).resolve()

    def get_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            processor_source = self.meta["llm"]["hf_config"] if "llm" in self.meta else self.meta["hf_model"]
            processor_dir = self._resolve(processor_source)
            self._processor = AutoProcessor.from_pretrained(str(processor_dir))
        return self._processor

    def _configure_text_golden_dirs(self) -> None:
        assert self.text is not None
        _set_session_golden_dir(
            self.text.prefill_model,
            self._graph_dir(self.meta["llm"]["prefill_hmonnx"]),
        )
        _set_session_golden_dir(
            self.text.decode_model,
            self._graph_dir(self.meta["llm"]["decode_hmonnx"]),
        )

    def _graph_dir(self, graph: str | Path) -> Path:
        return self._resolve(graph).parent

    def _get_vision_session(self, downsample_mode: str):
        from xhquant.api import HMONNXGoldenInference, HMONNXInference

        if downsample_mode in self._vision_sessions:
            return self._vision_sessions[downsample_mode]
        try:
            profile = self.meta["vision"][downsample_mode]
        except KeyError as exc:
            raise ValueError(f"Vision profile {downsample_mode!r} was not exported") from exc
        graph_path = str(self._resolve(profile["hmonnx"]))
        if self.enable_golden:
            session = HMONNXGoldenInference(graph_path)
            session.save_golden = True
            profile_dir = self._graph_dir(profile["hmonnx"])
            profile_dir.mkdir(parents=True, exist_ok=True)
            session.golden_dir = str(profile_dir)
            session.step = 0
            session.to(self.device)
            session.exec_device = self.device
            session.initialize()
        else:
            session = HMONNXInference(graph_path)
            session.exec_device = self.device
            session.to(self.device)
        self._vision_sessions[downsample_mode] = session
        return session

    def prepare_multimodal(
        self,
        input_messages: Any,
        *,
        downsample_mode: str = "4x",
        max_slice_nums: int = 36,
        video_max_slice_nums: int = 1,
        video_max_num_frames: int = 4,
    ) -> PreparedMultimodalInputs:
        processor = self.get_processor()
        messages = _build_messages(input_messages)
        model_inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "images_kwargs": {
                    "downsample_mode": downsample_mode,
                    "max_slice_nums": max_slice_nums,
                },
                "videos_kwargs": {
                    "downsample_mode": downsample_mode,
                    "max_slice_nums": video_max_slice_nums,
                    "max_num_frames": video_max_num_frames,
                },
            },
        )
        input_ids = model_inputs["input_ids"]
        try:
            image_features = self._extract_features(
                model_inputs.get("pixel_values"),
                model_inputs.get("target_sizes"),
                downsample_mode,
            )
            video_features = self._extract_features(
                model_inputs.get("pixel_values_videos"),
                model_inputs.get("target_sizes_videos"),
                downsample_mode,
            )
        finally:
            self._release_vision_session(downsample_mode)
        unified_ids, visual_features = _combine_multimodal_features(
            input_ids,
            image_features,
            video_features,
            image_token_id=processor.image_token_id,
            video_token_id=processor.video_token_id,
        )
        return PreparedMultimodalInputs(
            input_ids=unified_ids,
            image_embeds=visual_features,
            processor=processor,
        )

    def _release_vision_session(self, downsample_mode: str) -> None:
        session = self._vision_sessions.pop(downsample_mode, None)
        if session is None:
            return
        del session
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _extract_features(
        self,
        pixel_values: torch.Tensor | None,
        target_sizes: torch.Tensor | None,
        downsample_mode: str,
    ) -> torch.Tensor | None:
        if pixel_values is None or target_sizes is None:
            return None
        profile = self.meta["vision"][downsample_mode]
        token_capacity = int(profile["token_capacity"])
        positions_per_side = int(profile["positions_per_side"])
        expected_tokens = image_token_count(target_sizes, downsample_mode)
        pixel_slices = split_packed_pixel_values(pixel_values, target_sizes)
        session = self._get_vision_session(downsample_mode)
        features = []
        for pixel_slice, target_tensor in zip(
            pixel_slices,
            target_sizes,
            strict=True,
        ):
            target_size = tuple(int(value) for value in target_tensor.tolist())
            validate_token_capacity(target_size, token_capacity, downsample_mode)
            prepared = prepare_token_capacity_vision_inputs(
                pixel_slice.to(device=self.device, dtype=torch.float16),
                target_size,
                token_capacity,
                downsample_mode,
                positions_per_side,
            )
            if self.enable_golden:
                step = self._vision_golden_steps.get(downsample_mode, 0)
                self._vision_golden_steps[downsample_mode] = step + 1
                session.step = step
            output = session(*(prepared if downsample_mode == "16x" else prepared[:3]))
            output = output[0] if isinstance(output, (tuple, list)) else output
            output = crop_token_capacity_image_embeds(
                output,
                target_size,
                token_capacity,
                downsample_mode,
            )
            features.append(output.squeeze(0))
        merged = torch.cat(features, dim=0).to(self.device, torch.float16)
        if merged.shape[0] != expected_tokens:
            raise RuntimeError(f"Vision token mismatch: expected {expected_tokens}, got {merged.shape[0]}")
        return merged

    def generate_prepared(
        self,
        prepared: PreparedMultimodalInputs,
        *,
        max_new_tokens: int = 128,
    ) -> str:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.text is None:
            raise ValueError("The LLM component was not exported")
        prompt_length = int(prepared.input_ids.shape[-1])
        context_length = int(self.meta["llm"]["context_max_length"])
        if prompt_length + max_new_tokens > context_length:
            raise ValueError(
                f"Prompt plus output needs {prompt_length + max_new_tokens} tokens, "
                f"exceeding context length {context_length}"
            )

        generated: list[int] = []
        eos_ids = _eos_ids(prepared.processor.tokenizer)
        with LLMInferenceContextManager(self.text):
            self.text.set_prefill()
            processor = self.text.get_data_preprocessor()
            prefill_length = int(self.meta["llm"]["prefill_chunk_length"])
            feature_offset = 0
            logits = None
            for offset in range(0, prompt_length, prefill_length):
                chunk_ids = prepared.input_ids[:, offset : offset + prefill_length].to(self.device)
                chunk_feature_count = int((chunk_ids == int(prepared.processor.image_token_id)).sum().item())
                chunk_features = None
                if chunk_feature_count:
                    if prepared.image_embeds is None:
                        raise RuntimeError("Visual placeholder has no embedding")
                    chunk_features = prepared.image_embeds[feature_offset : feature_offset + chunk_feature_count]
                    feature_offset += chunk_feature_count
                model_args = processor(
                    {
                        "input_ids": chunk_ids,
                        "past_seq_length": offset,
                        "image_embeds": chunk_features,
                        # MiniCPM's proven Qwen3.5 adapter uses sequential LLM
                        # positions after scattering visual embeddings.
                        "image_grid_thw": None,
                    }
                )
                model_args = _match_position_ranks(
                    model_args,
                    self.text.prefill_model.hmonnx_session,
                )
                logits = self.text.forward(*model_args)[0]
            if logits is None:
                raise ValueError("Prompt must contain at least one token")
            if prepared.image_embeds is not None and feature_offset != prepared.image_embeds.shape[0]:
                raise RuntimeError(
                    "Not all visual embeddings were consumed during chunked Prefill: "
                    f"{feature_offset}/{prepared.image_embeds.shape[0]}"
                )
            token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            generated.append(token)

            self.text.set_decode()
            past_seq_length = prompt_length
            while len(generated) < max_new_tokens and token not in eos_ids:
                decode_args = self.text.get_data_preprocessor()(
                    {
                        "input_ids": torch.tensor(
                            [[token]],
                            dtype=torch.long,
                            device=self.device,
                        ),
                        "past_seq_length": past_seq_length,
                        "image_grid_thw": None,
                    }
                )
                decode_args = _match_position_ranks(
                    decode_args,
                    self.text.decode_model.hmonnx_session,
                )
                logits = self.text.forward(*decode_args)[0]
                token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                generated.append(token)
                past_seq_length += 1

        return prepared.processor.tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def generate(
        self,
        input_messages: Any,
        *,
        downsample_mode: str = "4x",
        max_slice_nums: int = 36,
        video_max_slice_nums: int = 1,
        video_max_num_frames: int = 4,
        max_new_tokens: int = 128,
    ) -> str:
        prepared = self.prepare_multimodal(
            input_messages,
            downsample_mode=downsample_mode,
            max_slice_nums=max_slice_nums,
            video_max_slice_nums=video_max_slice_nums,
            video_max_num_frames=video_max_num_frames,
        )
        return self.generate_prepared(prepared, max_new_tokens=max_new_tokens)


def dump_minicpm_v46_golden(
    *,
    work_dir: Path,
    device: str,
    input_messages: Any = None,
) -> str:
    """Generate Vision 4x/16x and text Prefill/Decode golden data."""

    from PIL import Image

    root_meta_file = work_dir / "export_meta_info.json"
    root_meta = json.loads(root_meta_file.read_text(encoding="utf-8"))
    vision_modes = list((root_meta.get("vision") or {}).keys())
    if input_messages is None:
        input_messages = {"text": "请描述这张图片。"}
        if vision_modes:
            input_messages["image"] = Image.new(
                "RGB",
                (448, 448),
                color=(114, 114, 114),
            )
    elif vision_modes and isinstance(input_messages, str):
        input_messages = {
            "image": Image.new("RGB", (448, 448), color=(114, 114, 114)),
            "text": input_messages,
        }
    elif vision_modes and isinstance(input_messages, Mapping):
        input_messages = dict(input_messages)
        has_media = any(key in input_messages for key in ("image", "images", "video", "videos", "messages"))
        if not has_media:
            input_messages["image"] = Image.new(
                "RGB",
                (448, 448),
                color=(114, 114, 114),
            )
    graph_dirs = [
        (work_dir / profile["hmonnx"]).resolve().parent for profile in (root_meta.get("vision") or {}).values()
    ]
    if "llm" in root_meta:
        graph_dirs.extend(
            [(work_dir / root_meta["llm"][field]).resolve().parent for field in ("prefill_hmonnx", "decode_hmonnx")]
        )
    # xhquant's legacy symlink helper requires step_N to be an immediate child
    # of the graph directory. Clean every prior step plus obsolete layouts.
    legacy_golden_root = work_dir / "golden"
    if legacy_golden_root.exists():
        shutil.rmtree(legacy_golden_root)
    for graph_dir in graph_dirs:
        legacy_graph_golden = graph_dir / "golden"
        if legacy_graph_golden.exists():
            shutil.rmtree(legacy_graph_golden)
        for step_dir in graph_dir.glob("step_*"):
            if step_dir.is_dir():
                shutil.rmtree(step_dir)
    runtime = MiniCPMV46HMONNXRuntime(
        root_meta_file,
        device=device,
        enable_golden=True,
    )
    prepared_by_mode = {
        mode: runtime.prepare_multimodal(
            input_messages,
            downsample_mode=mode,
            max_slice_nums=1,
        )
        for mode in vision_modes
    }
    if runtime.text is not None:
        if prepared_by_mode:
            preferred_mode = "4x" if "4x" in prepared_by_mode else vision_modes[0]
            llm_input = prepared_by_mode[preferred_mode]
        else:
            llm_input = runtime.prepare_multimodal(input_messages)
        runtime.generate_prepared(llm_input, max_new_tokens=2)
    expected = [(work_dir / root_meta["vision"][mode]["hmonnx"]).resolve().parent / "step_0" for mode in vision_modes]
    if runtime.text is not None:
        expected.extend(
            [
                (work_dir / root_meta["llm"][field]).resolve().parent / "step_0"
                for field in ("prefill_hmonnx", "decode_hmonnx")
            ]
        )
    missing = [str(path) for path in expected if not path.is_dir()]
    if missing:
        raise RuntimeError("Golden generation did not create: " + ", ".join(missing))
    return str(work_dir)


def _build_messages(input_messages: Any) -> list[dict[str, Any]]:
    if isinstance(input_messages, list):
        return input_messages
    if isinstance(input_messages, str):
        return [{"role": "user", "content": input_messages}]
    if not isinstance(input_messages, Mapping):
        raise TypeError("input_messages must be a message list, string, or mapping")
    if "messages" in input_messages:
        messages = input_messages["messages"]
        if not isinstance(messages, list):
            raise TypeError("input_messages['messages'] must be a list")
        return messages

    content = []
    for media_type, singular, plural in (
        ("image", "image", "images"),
        ("video", "video", "videos"),
    ):
        media = input_messages.get(plural, input_messages.get(singular))
        if media is None:
            continue
        values = media if isinstance(media, Sequence) and not isinstance(media, (str, bytes)) else [media]
        for value in values:
            key = "path" if isinstance(value, (str, Path)) else media_type
            content.append({"type": media_type, key: str(value) if key == "path" else value})
    text = input_messages.get("text", input_messages.get("prompt", ""))
    if text:
        content.append({"type": "text", "text": str(text)})
    if not content:
        raise ValueError("input_messages mapping must contain text, image, or video")
    return [{"role": "user", "content": content}]


def _combine_multimodal_features(
    input_ids: torch.Tensor,
    image_features: torch.Tensor | None,
    video_features: torch.Tensor | None,
    *,
    image_token_id: int,
    video_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    image_count = int((input_ids == image_token_id).sum().item())
    video_count = int((input_ids == video_token_id).sum().item())
    if image_count == 0 and video_count == 0:
        return input_ids, None
    if image_count and (image_features is None or image_features.shape[0] != image_count):
        actual = 0 if image_features is None else image_features.shape[0]
        raise ValueError(f"Image tokens/features mismatch: {image_count} vs {actual}")
    if video_features is None and video_count:
        raise ValueError(f"Video tokens/features mismatch: {video_count} vs 0")
    if video_features is not None and video_features.shape[0] != video_count:
        raise ValueError(f"Video tokens/features mismatch: {video_count} vs {video_features.shape[0]}")

    template = image_features if image_count else video_features
    assert template is not None
    ordered = template.new_empty((image_count + video_count, template.shape[-1]))
    visual_ids = input_ids[(input_ids == image_token_id) | (input_ids == video_token_id)].to(template.device)
    if image_count:
        ordered[visual_ids == image_token_id] = image_features
    if video_count:
        assert video_features is not None
        ordered[visual_ids == video_token_id] = video_features
    unified = input_ids.clone()
    unified[unified == video_token_id] = image_token_id
    return unified, ordered


def _set_session_golden_dir(model: HMONNXModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = model.hmonnx_session
    session.golden_dir = str(output_dir)
    if hasattr(session, "save_golden_dir"):
        session.save_golden_dir = str(output_dir)


def _match_position_ranks(model_args, session):
    """Accept both legacy rank-2 and current rank-1 position-id ABIs."""

    if hasattr(session, "initialize") and not getattr(session, "_initialized", True):
        session.initialize()
    args = list(model_args)
    for index, candidates in (
        (1, ("time_position_ids",)),
        (2, ("hight_position_ids", "height_position_ids")),
        (3, ("width_position_ids",)),
    ):
        input_meta = None
        for name in candidates:
            try:
                input_meta = session.get_input(name)
                break
            except (KeyError, ValueError):
                continue
        if input_meta is None:
            continue
        value = args[index]
        while value.dim() < len(input_meta.shape):
            value = value.unsqueeze(0)
        args[index] = value
    return tuple(args)


def _eos_ids(tokenizer) -> set[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(value) for value in eos}


__all__ = [
    "MiniCPMV46HMONNXRuntime",
    "MiniCPMV46TextHMONNXModel",
    "dump_minicpm_v46_golden",
]
