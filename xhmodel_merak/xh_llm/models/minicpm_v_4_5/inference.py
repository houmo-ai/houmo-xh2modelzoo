"""Root-metadata-driven MiniCPM-V-4.5 HMONNX inference."""

from __future__ import annotations

import gc
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel
from xhmodel_merak.xh_llm.hmonnx.text_llm_hmonnx_model import TextLLMHMONNXModel
from xhmodel_merak.xh_llm.infer_mixin import LLMInferenceContextManager
from xhmodel_merak.xh_llm.types import LLMModelMeta

from .model import XHMiniCPMV45Model
from .vision import (
    build_resampler_pos_embed_cache,
    build_resampler_temporal_pos_embed_cache,
    prepare_video_group_inputs,
    prepare_vision_inputs,
    validate_patch_capacity,
)


# Official MiniCPM-V-4.5 video packing constants (README encode_video).
_MAX_NUM_FRAMES = 180
_MAX_NUM_PACKING = 6
_TIME_SCALE = 0.1


def encode_video(
    video_path: str,
    choose_fps: float = 3,
    force_packing: int | None = None,
) -> tuple[list, list]:
    """Official frame sampling + temporal grouping of the MiniCPM-V-4.5 demo.

    Returns ``(frames, frame_ts_id_group)`` where ``frames`` is a list of PIL
    images and ``frame_ts_id_group`` is the list of temporal-id groups (each
    group packs up to 6 consecutive frames for the 3D-Resampler).
    """
    import math

    import numpy as np
    from decord import VideoReader, cpu
    from scipy.spatial import cKDTree

    def uniform_sample(length: int, count: int) -> list[int]:
        gap = length / count
        return [int(index * gap + gap / 2) for index in range(count)]

    def map_to_nearest_scale(values, scale):
        tree = cKDTree(np.asarray(scale)[:, None])
        _, indices = tree.query(np.asarray(values)[:, None])
        return np.asarray(scale)[indices]

    def group_array(values, size):
        return [values[i : i + size] for i in range(0, len(values), size)]

    reader = VideoReader(video_path, ctx=cpu(0))
    fps = reader.get_avg_fps()
    video_duration = len(reader) / fps

    if choose_fps * int(video_duration) <= _MAX_NUM_FRAMES:
        packing_nums = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(_MAX_NUM_FRAMES, video_duration))
    else:
        packing_nums = math.ceil(video_duration * choose_fps / _MAX_NUM_FRAMES)
        if packing_nums <= _MAX_NUM_PACKING:
            choose_frames = round(video_duration * choose_fps)
        else:
            choose_frames = round(_MAX_NUM_FRAMES * _MAX_NUM_PACKING)
            packing_nums = _MAX_NUM_PACKING
    if force_packing is not None:
        packing_nums = min(int(force_packing), _MAX_NUM_PACKING)

    frame_idx = np.array(uniform_sample(len(reader), choose_frames))
    frames = reader.get_batch(frame_idx).asnumpy()
    frame_idx_ts = frame_idx / fps
    scale = np.arange(0, video_duration, _TIME_SCALE)
    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / _TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)
    assert len(frames) == len(frame_ts_id)

    from PIL import Image

    frames = [Image.fromarray(value.astype("uint8")).convert("RGB") for value in frames]
    return frames, group_array(frame_ts_id, packing_nums)


class MiniCPMV45TextHMONNXModel(TextLLMHMONNXModel):
    """Qwen3 cache runtime without Qwen's native Vision session."""

    LLM_MODEL_CLS = XHMiniCPMV45Model


@dataclass
class PreparedMultimodalInputs:
    input_ids: torch.Tensor
    image_bound: torch.Tensor | None
    image_embeds: torch.Tensor | None
    processor: Any


class MiniCPMV45HMONNXRuntime:
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
        raw_meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
        self.meta = self._normalize_meta(raw_meta)
        if self.meta.get("model_type") != "MiniCPM-V-4.5":
            raise ValueError(f"Not a MiniCPM-V-4.5 export: {self.meta_file}")
        self.device = torch.device(device)
        self.enable_golden = bool(enable_golden)
        self.text: MiniCPMV45TextHMONNXModel | None = None
        if "llm" in self.meta:
            child_meta_file = self._resolve(self.meta["llm"]["metadata"])
            child_meta_dict = json.loads(child_meta_file.read_text(encoding="utf-8"))
            child_meta_dict["_meta_path_"] = str(child_meta_file)
            llm_meta = LLMModelMeta.from_dict(child_meta_dict)
            # The LLM artifact's hf_config only carries config.json; MiniCPM-V-4.5
            # config/tokenizer/processor are remote code, so load them from the
            # original model directory.
            llm_meta.hf_config = str(self._resolve(self.meta["hf_model"]))
            self.text = MiniCPMV45TextHMONNXModel(
                llm_meta,
                enable_golden=enable_golden,
                device_map=[self.device],
            )
            self.text.to(self.device)
            # The legacy HMONNXModel constructor records ``enable_golden``
            # without invoking its setter. Re-apply it at the aggregate
            # model level so Prefill and Decode sessions enable dumping.
            self.text.enable_golden = self.enable_golden
        self._vision_session: Any = None
        self._vision_golden_step = 0
        self._video_group_session: Any = None
        self._video_group_golden_step = 0
        self._processor = None
        if self.enable_golden and self.text is not None:
            self._configure_text_golden_dirs()

    @staticmethod
    def _normalize_meta(raw_meta: dict[str, Any]) -> dict[str, Any]:
        """Normalize golden_meta_info.json (VLLMModelMeta) to the runtime view.

        The aligned export writes a ``golden_meta_info.json`` whose top level
        is an LLMModelMeta dict plus the ``visual_config`` /
        ``visual_video_config`` nests (qwen2_vl convention).  The runtime
        internally expects the legacy flattened keys (``vision`` /
        ``vision_video`` / ``llm`` / ``hf_model``); this maps between the two.
        The legacy ``export_meta_info.json`` layout is still accepted.
        """
        if "llm" in raw_meta:
            return raw_meta
        model_config = raw_meta.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("MiniCPM golden_meta_info.json must contain model_config")
        hf_model = model_config.get("hf_model")
        if not hf_model:
            raise ValueError("MiniCPM golden_meta_info.json model_config.hf_model is required")
        normalized: dict[str, Any] = {
            "model_type": raw_meta.get("model_type", "MiniCPM-V-4.5"),
            "hf_model": hf_model,
            "llm": {
                "metadata": "golden_meta_info.json",
                "prefill_hmonnx": raw_meta["prefill_hmonnx"],
                "decode_hmonnx": raw_meta["decode_hmonnx"],
                "quant_embedding": raw_meta["quant_embedding"],
                "hf_config": raw_meta["hf_config"],
                "context_max_length": int(model_config.get("context_max_length", 8192)),
                "prefill_chunk_length": int(model_config.get("prefill_chunk_length", 256)),
            },
        }
        if "visual_config" in raw_meta:
            normalized["vision"] = raw_meta["visual_config"]
        if "visual_video_config" in raw_meta:
            normalized["vision_video"] = raw_meta["visual_video_config"]
        return normalized

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.work_dir / path).resolve()

    def get_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            # MiniCPM-V-4.5's processor is remote code; load it from the
            # original model directory (the LLM artifact's hf_config only
            # carries config.json and lacks the remote-code files).
            processor_dir = self._resolve(self.meta["hf_model"])
            self._processor = AutoProcessor.from_pretrained(
                str(processor_dir),
                trust_remote_code=True,
            )
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

    def _get_vision_session(self):
        from xhquant.api import HMONNXGoldenInference, HMONNXInference

        if self._vision_session is not None:
            return self._vision_session
        profile = self.meta["vision"]
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
        self._vision_session = session
        return session

    def _release_vision_session(self) -> None:
        session = self._vision_session
        if session is None:
            return
        self._vision_session = None
        del session
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_video_group_session(self):
        from xhquant.api import HMONNXGoldenInference, HMONNXInference

        if self._video_group_session is not None:
            return self._video_group_session
        profile = self.meta["vision_video"]
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
        self._video_group_session = session
        return session

    def _release_video_group_session(self) -> None:
        session = self._video_group_session
        if session is None:
            return
        self._video_group_session = None
        del session
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_multimodal(
        self,
        input_messages: Any,
        *,
        max_slice_nums: int = 9,
        video_fps: float = 3,
        video_packing: int | None = None,
    ) -> PreparedMultimodalInputs:
        processor = self.get_processor()
        messages, images, temporal_ids = _build_messages(
            input_messages,
            video_fps=video_fps,
            video_packing=video_packing,
        )
        prompts = [
            processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                # Match the official chat() default: non-thinking mode.
                enable_thinking=False,
            )
        ]
        if temporal_ids is not None:
            # Official video usage: no image ids, one slice per frame.
            model_inputs = processor(
                prompts,
                [images],
                max_slice_nums=1,
                use_image_id=False,
                temporal_ids=[temporal_ids],
                return_tensors="pt",
            )
        else:
            model_inputs = processor(
                prompts,
                [images],
                max_slice_nums=max_slice_nums,
                return_tensors="pt",
            )
        input_ids = model_inputs["input_ids"]
        pixel_slices = model_inputs["pixel_values"][0]
        if pixel_slices:
            tgt_sizes = model_inputs["tgt_sizes"][0]
            image_bound = model_inputs["image_bound"][0]
        else:
            tgt_sizes = None
            image_bound = None
        try:
            if temporal_ids is not None:
                image_embeds = self._extract_video_group_features(
                    pixel_slices,
                    tgt_sizes,
                    temporal_ids,
                )
            else:
                image_embeds = self._extract_features(pixel_slices, tgt_sizes)
        finally:
            self._release_vision_session()
            self._release_video_group_session()
        return PreparedMultimodalInputs(
            input_ids=input_ids,
            image_bound=image_bound,
            image_embeds=image_embeds,
            processor=processor,
        )

    def _extract_features(
        self,
        pixel_slices: Sequence[torch.Tensor],
        tgt_sizes: torch.Tensor,
    ) -> torch.Tensor | None:
        if not pixel_slices:
            return None
        profile = self.meta["vision"]
        patch_capacity = int(profile["patch_capacity"])
        positions_per_side = int(profile["positions_per_side"])
        embed_dim = int(profile["embed_dim"])
        pos_embed_cache = build_resampler_pos_embed_cache(
            embed_dim,
            positions_per_side,
            device=self.device,
        )
        session = self._get_vision_session()
        features = []
        for pixel_slice, tgt in zip(
            pixel_slices,
            tgt_sizes.tolist(),
            strict=True,
        ):
            target_size = (int(tgt[0]), int(tgt[1]))
            validate_patch_capacity(target_size, patch_capacity)
            prepared = prepare_vision_inputs(
                pixel_slice.to(device=self.device, dtype=torch.float16),
                target_size,
                patch_capacity,
                positions_per_side,
                pos_embed_cache,
                dtype=torch.float16,
            )
            if self.enable_golden:
                session.step = self._vision_golden_step
                self._vision_golden_step += 1
            output = session(*prepared)
            output = output[0] if isinstance(output, (tuple, list)) else output
            features.append(output.squeeze(0))
        merged = torch.cat(features, dim=0).to(self.device, torch.float16)
        expected_tokens = len(pixel_slices) * 64
        if merged.shape[0] != expected_tokens:
            raise RuntimeError(
                f"Vision feature count mismatch: expected {expected_tokens} "
                f"(64 per slice x {len(pixel_slices)} slices), got {merged.shape[0]}"
            )
        return merged

    def _extract_video_group_features(
        self,
        pixel_slices: Sequence[torch.Tensor],
        tgt_sizes: torch.Tensor,
        temporal_groups: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Extract one 64-token feature vector per temporal group."""
        if not pixel_slices:
            return None
        profile = self.meta["vision_video"]
        patch_capacity = int(profile["patch_capacity"])
        group_capacity = int(profile["group_capacity"])
        positions_per_side = int(profile["positions_per_side"])
        embed_dim = int(profile["embed_dim"])
        max_temporal_id = max(
            (int(value) for group in temporal_groups for value in group),
            default=0,
        )
        pos_embed_cache = build_resampler_pos_embed_cache(
            embed_dim,
            positions_per_side,
            device=self.device,
        )
        temporal_cache = build_resampler_temporal_pos_embed_cache(
            embed_dim,
            max(max_temporal_id + 1, 1),
            device=self.device,
        )
        session = self._get_video_group_session()
        features = []
        slice_offset = 0
        for group_index, group in enumerate(temporal_groups):
            frame_count = len(group)
            if frame_count > group_capacity:
                raise ValueError(
                    f"Temporal group {group_index} has {frame_count} frames, exceeding group_capacity {group_capacity}"
                )
            if slice_offset + frame_count > len(pixel_slices):
                raise RuntimeError(
                    f"Temporal group {group_index} needs {frame_count} slices, "
                    f"only {len(pixel_slices) - slice_offset} remaining"
                )
            group_slices = list(pixel_slices[slice_offset : slice_offset + frame_count])
            group_tgts = [(int(h), int(w)) for h, w in tgt_sizes[slice_offset : slice_offset + frame_count].tolist()]
            slice_offset += frame_count
            prepared = prepare_video_group_inputs(
                group_slices,
                group_tgts,
                [int(value) for value in group],
                patch_capacity,
                group_capacity,
                positions_per_side,
                pos_embed_cache,
                temporal_cache,
                dtype=torch.float16,
            )
            prepared = tuple(
                value.to(device=self.device)
                if value.dtype in (torch.int32, torch.int64)
                else value.to(device=self.device, dtype=torch.float16)
                for value in prepared
            )
            if self.enable_golden:
                session.step = self._video_group_golden_step
                self._video_group_golden_step += 1
            output = session(*prepared)
            output = output[0] if isinstance(output, (tuple, list)) else output
            features.append(output.squeeze(0))
        if slice_offset != len(pixel_slices):
            raise RuntimeError(f"Video groups consumed {slice_offset} slices, got {len(pixel_slices)}")
        merged = torch.cat(features, dim=0).to(self.device, torch.float16)
        expected_tokens = len(temporal_groups) * 64
        if merged.shape[0] != expected_tokens:
            raise RuntimeError(
                f"Video-group feature count mismatch: expected {expected_tokens} "
                f"(64 per group x {len(temporal_groups)} groups), got {merged.shape[0]}"
            )
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

        embeds = self.text.embed_tokens(prepared.input_ids.to(self.device))
        if prepared.image_embeds is not None:
            embeds = _scatter_image_embeds(
                embeds,
                prepared.image_bound,
                prepared.image_embeds,
            )

        generated: list[int] = []
        eos_ids = _eos_ids(prepared.processor.tokenizer)
        with LLMInferenceContextManager(self.text):
            self.text.set_prefill()
            prefill_length = int(self.meta["llm"]["prefill_chunk_length"])
            steps = (prompt_length + prefill_length - 1) // prefill_length
            padding_len = steps * prefill_length - prompt_length
            if padding_len > 0:
                padding_embeds = self.text.embed_tokens(
                    torch.zeros((1, padding_len), dtype=torch.long, device=self.device)
                )
                embeds = torch.cat([embeds, padding_embeds], dim=1)
            logits = None
            for step in range(steps):
                start = step * prefill_length
                chunk = embeds[:, start : start + prefill_length, :]
                current_length = min((step + 1) * prefill_length, prompt_length) - start
                logits = self.text.forward(
                    chunk,
                    torch.tensor([start], dtype=torch.int32, device=self.device),
                    torch.tensor([current_length], dtype=torch.int32, device=self.device),
                    self.text.past_key_caches,
                    self.text.past_value_caches,
                )
            if logits is None:
                raise ValueError("Prompt must contain at least one token")
            token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            generated.append(token)

            self.text.set_decode()
            past_seq_length = prompt_length
            while len(generated) < max_new_tokens and token not in eos_ids:
                decode_embeds = self.text.embed_tokens(torch.tensor([[token]], dtype=torch.long, device=self.device))
                logits = self.text.forward(
                    decode_embeds,
                    torch.tensor([past_seq_length], dtype=torch.int32, device=self.device),
                    torch.tensor([1], dtype=torch.int32, device=self.device),
                    self.text.past_key_caches,
                    self.text.past_value_caches,
                )
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
        max_slice_nums: int = 9,
        video_fps: float = 3,
        video_packing: int | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        prepared = self.prepare_multimodal(
            input_messages,
            max_slice_nums=max_slice_nums,
            video_fps=video_fps,
            video_packing=video_packing,
        )
        return self.generate_prepared(prepared, max_new_tokens=max_new_tokens)


def dump_minicpm_v45_golden(
    *,
    work_dir: Path,
    device: str,
    input_messages: Any = None,
) -> str:
    """Generate Vision and text Prefill/Decode golden data."""

    from PIL import Image

    root_meta_file = work_dir / "golden_meta_info.json"
    if not root_meta_file.is_file():
        raise FileNotFoundError(f"MiniCPM golden metadata not found: {root_meta_file}")
    raw_root_meta = json.loads(root_meta_file.read_text(encoding="utf-8"))
    root_meta = MiniCPMV45HMONNXRuntime._normalize_meta(raw_root_meta)
    has_vision = bool(root_meta.get("vision"))
    has_video = bool(root_meta.get("vision_video"))
    if input_messages is None:
        input_messages = {"text": "请描述这张图片。"}
        if has_vision:
            input_messages["image"] = Image.new(
                "RGB",
                (448, 448),
                color=(114, 114, 114),
            )
    elif has_vision and isinstance(input_messages, str):
        input_messages = {
            "image": Image.new("RGB", (448, 448), color=(114, 114, 114)),
            "text": input_messages,
        }
    elif has_vision and isinstance(input_messages, Mapping):
        input_messages = dict(input_messages)
        has_media = any(key in input_messages for key in ("image", "images", "video", "videos", "messages"))
        if not has_media:
            input_messages["image"] = Image.new(
                "RGB",
                (448, 448),
                color=(114, 114, 114),
            )

    graph_dirs = [(work_dir / root_meta["vision"]["hmonnx"]).resolve().parent if has_vision else Path()]
    if has_video:
        graph_dirs.append((work_dir / root_meta["vision_video"]["hmonnx"]).resolve().parent)
    graph_dirs = [path for path in graph_dirs if path != Path()]
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

    runtime = MiniCPMV45HMONNXRuntime(
        root_meta_file,
        device=device,
        enable_golden=True,
    )
    prepared = runtime.prepare_multimodal(input_messages)
    if runtime.text is not None:
        runtime.generate_prepared(prepared, max_new_tokens=2)
    if has_video:
        _dump_video_group_golden(runtime)

    expected = []
    if has_vision:
        expected.append((work_dir / root_meta["vision"]["hmonnx"]).resolve().parent / "step_0")
    if has_video:
        expected.append((work_dir / root_meta["vision_video"]["hmonnx"]).resolve().parent / "step_0")
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
    # 发布规范注意点 8：每个 step_0 内软链图文件（onnx + external_data），
    # 便于编译器按相对路径找到权重。
    _link_graph_files_into_steps(work_dir, root_meta)
    return str(work_dir)


def _link_graph_files_into_steps(work_dir: Path, root_meta: dict[str, Any]) -> None:
    """Symlink each graph's onnx/external_data into its step_* directories."""

    def graph_files(graph_rel: str) -> list[Path]:
        graph_path = (work_dir / graph_rel).resolve()
        if not graph_path.is_file():
            return []
        # HMONNX may rename the graph to ``*_with_act.onnx`` while preserving
        # the original external-data location.  Accept both stems instead of
        # guessing only from the final graph filename.
        stems = {graph_path.stem}
        if graph_path.stem.endswith("_with_act"):
            stems.add(graph_path.stem[: -len("_with_act")])
        candidates = [graph_path]
        candidates.extend(graph_path.parent / f"{stem}_external_data" for stem in stems)
        return [path for path in candidates if path.is_file()]

    graph_refs: list[str] = []
    for key in ("vision", "vision_video"):
        profile = root_meta.get(key)
        if profile:
            graph_refs.append(profile["hmonnx"])
    llm = root_meta.get("llm")
    if llm:
        graph_refs.extend([llm["prefill_hmonnx"], llm["decode_hmonnx"]])

    for graph_rel in graph_refs:
        graph_path = (work_dir / graph_rel).resolve()
        files = graph_files(graph_rel)
        if not files:
            continue
        for step_dir in graph_path.parent.glob("step_*"):
            if not step_dir.is_dir():
                continue
            for source in files:
                link = step_dir / source.name
                if link.is_symlink():
                    try:
                        link.resolve(strict=True)
                        continue
                    except (OSError, RuntimeError):
                        link.unlink()
                elif link.exists():
                    if not link.is_file():
                        raise RuntimeError(f"Golden link target path is not a file: {link}")
                    continue
                target = Path(os.path.relpath(source, step_dir))
                try:
                    link.symlink_to(target)
                except OSError as exc:
                    raise RuntimeError(f"Failed to link {link} -> {target}: {exc}") from exc
                try:
                    link.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise RuntimeError(f"Created dangling Golden link: {link} -> {target}") from exc


def _dump_video_group_golden(runtime: MiniCPMV45HMONNXRuntime) -> None:
    """Generate Golden for the temporal-group Vision graph with a synthetic
    3-frame group (temporal ids 0, 1, 2)."""
    import torch
    from PIL import Image

    profile = runtime.meta["vision_video"]
    patch_capacity = int(profile["patch_capacity"])
    group_capacity = int(profile["group_capacity"])
    positions_per_side = int(profile["positions_per_side"])
    embed_dim = int(profile["embed_dim"])

    gray = Image.new("RGB", (448, 448), color=(114, 114, 114))
    processor = runtime.get_processor()
    proc = processor(
        ["(<image>./</image>)\n(<image>./</image>)\n(<image>./</image>)"],
        [[gray, gray, gray]],
        max_slice_nums=1,
        use_image_id=False,
        return_tensors="pt",
    )
    pixel_slices = proc["pixel_values"][0]
    tgt_sizes = proc["tgt_sizes"][0]
    group_tgts = [(int(h), int(w)) for h, w in tgt_sizes.tolist()]
    pos_embed_cache = build_resampler_pos_embed_cache(
        embed_dim,
        positions_per_side,
        device=runtime.device,
    )
    temporal_cache = build_resampler_temporal_pos_embed_cache(
        embed_dim,
        3,
        device=runtime.device,
    )
    prepared = prepare_video_group_inputs(
        list(pixel_slices),
        group_tgts,
        [0, 1, 2],
        patch_capacity,
        group_capacity,
        positions_per_side,
        pos_embed_cache,
        temporal_cache,
        dtype=torch.float16,
    )
    prepared = tuple(
        value.to(device=runtime.device)
        if value.dtype in (torch.int32, torch.int64)
        else value.to(device=runtime.device, dtype=torch.float16)
        for value in prepared
    )
    session = runtime._get_video_group_session()
    session.step = 0
    output = session(*prepared)
    output = output[0] if isinstance(output, (tuple, list)) else output
    if output.shape[0] != 1 or output.shape[1] != 64:
        raise RuntimeError(f"Unexpected video-group golden output shape: {tuple(output.shape)}")
    runtime._release_video_group_session()


def _build_messages(
    input_messages: Any,
    *,
    video_fps: float = 3,
    video_packing: int | None = None,
) -> tuple[list[dict[str, Any]], list[Any], list[list[int]] | None]:
    """Build chat messages with ``(<image>./</image>)`` placeholders, the
    ordered media list, and the video temporal-id groups (None for images),
    mirroring the official MiniCPM-V-4.5 chat() flow."""
    from PIL import Image

    if isinstance(input_messages, list):
        raise TypeError("MiniCPM-V-4.5 runtime expects a mapping, not a message list")
    if not isinstance(input_messages, Mapping):
        raise TypeError("input_messages must be a mapping")
    if "messages" in input_messages:
        raise TypeError("MiniCPM-V-4.5 runtime expects images/videos/text keys, not 'messages'")

    content: list[Any] = []
    media: list[Any] = []
    for media_type in ("image", "images"):
        media_value = input_messages.get(media_type)
        if media_value is None:
            continue
        values = (
            media_value
            if isinstance(media_value, Sequence) and not isinstance(media_value, (str, bytes))
            else [media_value]
        )
        for value in values:
            if isinstance(value, (str, Path)):
                image = Image.open(value).convert("RGB")
            else:
                image = value
            content.append(image)
            media.append(image)

    temporal_ids: list[list[int]] | None = None
    for video_type in ("video", "videos"):
        video_value = input_messages.get(video_type)
        if video_value is None:
            continue
        if isinstance(video_value, Sequence) and not isinstance(video_value, (str, bytes)) and not video_value:
            continue
        values = (
            video_value
            if isinstance(video_value, Sequence) and not isinstance(video_value, (str, bytes))
            else [video_value]
        )
        if temporal_ids is None:
            # Mixed calls: every static image is its own single-frame group
            # with temporal id -1 (zero temporal embedding, like the native
            # Resampler), followed by the video groups in content order.
            temporal_ids = [[-1] for _ in media]
        for value in values:
            if not isinstance(value, (str, Path)):
                raise TypeError(f"{video_type} entries must be file paths")
            frames, groups = encode_video(
                str(value),
                choose_fps=video_fps,
                force_packing=video_packing,
            )
            content.extend(frames)
            media.extend(frames)
            temporal_ids.extend(group.tolist() for group in groups)

    text = input_messages.get("text", input_messages.get("prompt", ""))
    if text:
        content.append(str(text))
    if not content:
        raise ValueError("input_messages mapping must contain text, image, images, or video")

    placeholder_items: list[str] = []
    for item in content:
        if isinstance(item, Image.Image):
            placeholder_items.append("(<image>./</image>)")
        else:
            placeholder_items.append(str(item))
    return (
        [{"role": "user", "content": "\n".join(placeholder_items)}],
        media,
        temporal_ids,
    )


def _scatter_image_embeds(
    embeds: torch.Tensor,
    image_bound: torch.Tensor,
    image_embeds: torch.Tensor,
) -> torch.Tensor:
    """Replace image placeholder embeddings with the vision features."""
    ranges = torch.cat(
        [torch.arange(int(start), int(end), device=embeds.device) for start, end in image_bound.tolist()]
    )
    feature_count = image_embeds.shape[0]
    if ranges.shape[0] != feature_count:
        raise RuntimeError(
            f"Vision token mismatch: {ranges.shape[0]} bound positions vs {feature_count} image features"
        )
    scattered = embeds.clone()
    scattered[0, ranges] = image_embeds.to(dtype=embeds.dtype)
    return scattered


def _set_session_golden_dir(model: HMONNXModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = model.hmonnx_session
    session.golden_dir = str(output_dir)
    if hasattr(session, "save_golden_dir"):
        session.save_golden_dir = str(output_dir)


def _eos_ids(tokenizer) -> set[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(value) for value in eos}


__all__ = [
    "MiniCPMV45HMONNXRuntime",
    "MiniCPMV45TextHMONNXModel",
    "dump_minicpm_v45_golden",
]
