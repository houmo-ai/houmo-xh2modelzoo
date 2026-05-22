from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageOps
from torch import Tensor, nn
from xhquant.api import HMONNXGoldenInference, HMONNXInference

from ..builder import MODELS
from ..llm_onnx_model import LLMONNXModel
from .utils import build_inputs, build_messages, get_rope_index, scatter_image_embeds

# LLMONNXGraphModel was a planned extension; alias to LLMONNXModel for compatibility
LLMONNXGraphModel = LLMONNXModel


class RepetitionPenaltyLogitsProcessor:
    """Apply repetition penalty to logits (same as Qwen2_5_VLONNXModel)."""

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        return scores.scatter(1, input_ids, score)


def decode_next_token(tokenizer, logits: torch.Tensor, do_sample: bool = False):
    if do_sample:
        probs = nn.functional.softmax(logits[:, -1, :].float(), dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        next_tokens = next_tokens.unsqueeze(0)
    else:
        next_tokens = torch.argmax(logits, dim=-1)
    next_token_str = tokenizer.batch_decode(next_tokens, skip_special_tokens=True)
    return next_tokens, next_token_str


@MODELS.register_module()
class GlmOcrONNXModel(LLMONNXGraphModel):
    """GLM-OCR inference model backed by HMONNX runtime.

    Extends :class:`LLMONNXModel` to leverage ``HMONNXInference`` for models
    exported with custom xh2a ops (e.g. ``ai.houmo.xh2a:RMSNorm``) that
    standard ONNX Runtime cannot handle.

    Config example::

        model = dict(
            type="GlmOcrONNXModel",
            image_feature=dict(onnx="...visual_1.onnx"),
            prefill=dict(onnx="...prefill.onnx", input_sequence_length=256),
            decode=dict(onnx="...decode.onnx"),
            kv_cache=dict(num_hidden_layers=16, shape=[1, 8, 2048, 128]),
            cache_len=2048,
            eos_token_id=[151329],
        )
    """

    def __init__(
        self,
        image_feature,
        prefill,
        decode,
        kv_cache,
        cache_len: int = 2048,
        image_size_w: int = 1024,
        image_size_h: int = 1024,
        presence_penalty: float = 0.0,
        image_token_id: int = 59280,
        video_start_token_id: int = 151343,
        video_end_token_id: int = 151344,
        eos_token_id: Optional[list[int]] = None,
        pad_token_id: int = 59246,
    ):
        super().__init__(prefill, decode, kv_cache)
        self.image_feature_config = image_feature
        self.image_feature_session = None
        self._vision_runtime = "hmonnx"

        self.image_size_w = image_size_w
        self.image_size_h = image_size_h
        self.cache_len = cache_len

        self.image_token_id = int(image_token_id)
        self.video_start_token_id = int(video_start_token_id)
        self.video_end_token_id = int(video_end_token_id)
        self.eos_token_id = eos_token_id or [151329]
        self.pad_token_id = int(pad_token_id)

        self.spatial_merge_size = 2

        self.presence_penalty = float(presence_penalty)
        effective_penalty = max(presence_penalty, 1e-6) if presence_penalty > 0 else 1.0
        self.repetition_penalty_logits_processor = RepetitionPenaltyLogitsProcessor(effective_penalty)

        self.rope_deltas = None
        self.input_sequence_length = 0

    # ------------------------------------------------------------------ #
    #  Vision session management                                          #
    # ------------------------------------------------------------------ #

    def _get_image_feature_onnx_path(self):
        if isinstance(self.image_feature_config, dict):
            return self.image_feature_config["onnx"]
        return self.image_feature_config.onnx

    def init_image_feature(self):
        self._vision_runtime = "hmonnx"
        hmonnx_session = HMONNXGoldenInference(self._get_image_feature_onnx_path())
        hmonnx_session.exec_device = self._exec_device
        hmonnx_session.to(self.device)
        try:
            hmonnx_session.initialize()
            self.image_feature_session = hmonnx_session
        except Exception as e:
            if "Unsupported ops" not in str(e):
                raise
            providers = ["CPUExecutionProvider"]
            if str(self._exec_device).startswith("cuda"):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.image_feature_session = ort.InferenceSession(self._get_image_feature_onnx_path(), providers=providers)
            self._vision_runtime = "ort"

    def save_image_feature_golden(self, output_dir):
        if self._vision_runtime == "hmonnx":
            self.image_feature_session.save_golden = True
            self.image_feature_session.golden_dir = output_dir
            self.image_feature_session.save_golden_dir = output_dir

    def save_prefill_golden(self, output_dir):
        super().save_prefill_golden(output_dir)
        self.prefill_session.save_golden_dir = output_dir

    def save_decode_golden(self, output_dir):
        super().save_decode_golden(output_dir)
        self.decode_session.save_golden_dir = output_dir

    def release_image_feature(self):
        self.image_feature_session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def extract_image_features(self, pixel_values: Tensor, image_grid_thw: Tensor) -> Tensor:
        """Run the vision ONNX / HMONNX model to get image embeddings."""
        if self.image_feature_session is None:
            raise RuntimeError("image_feature_session is not initialized")

        if self._vision_runtime == "ort":
            ort_input_names = {inp.name for inp in self.image_feature_session.get_inputs()}
            ort_inputs = {}
            if "pixel_values" in ort_input_names:
                ort_inputs["pixel_values"] = pixel_values.float().detach().cpu().numpy()
            if "image_grid_thw" in ort_input_names:
                ort_inputs["image_grid_thw"] = image_grid_thw.long().detach().cpu().numpy()
            if "grid_thw" in ort_input_names:
                ort_inputs["grid_thw"] = image_grid_thw.long().detach().cpu().numpy()

            if len(ort_inputs) == 0:
                raise RuntimeError(f"Unsupported ORT vision inputs: {sorted(ort_input_names)}")

            outputs = self.image_feature_session.run(
                None,
                ort_inputs,
            )
            return torch.from_numpy(outputs[0]).to(pixel_values.device, dtype=torch.float16)

        try:
            out = self.image_feature_session(pixel_values.half(), image_grid_thw.long())
        except Exception:
            # Some HMONNX-converted vision models bake grid_thw as a static constant.
            out = self.image_feature_session(pixel_values.half())
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    # ------------------------------------------------------------------ #
    #  RoPE index                                                         #
    # ------------------------------------------------------------------ #

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_start_token_id=self.video_start_token_id,
            video_end_token_id=self.video_end_token_id,
        )

    # ------------------------------------------------------------------ #
    #  Input preparation                                                  #
    # ------------------------------------------------------------------ #

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        device = self._exec_device
        input_ids = data["input_ids"].to(device)
        seq_length = input_ids.shape[1]

        assert self.token_embedding is not None, "Token embedding is not available."
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
        assert seq_length <= self.input_sequence_length, (
            f"Input sequence too long: max={self.input_sequence_length}, got={seq_length}"
        )

        attention_mask = data.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones((1, seq_length), dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)

        # Pad to input_sequence_length
        if self.input_sequence_length > seq_length:
            pad_len = self.input_sequence_length - seq_length
            padding_ids = torch.zeros((1, pad_len), dtype=torch.long, device=device).fill_(self.pad_token_id)
            input_ids = torch.cat([input_ids, padding_ids], dim=-1)
            padding_mask = torch.zeros((1, pad_len), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, padding_mask], dim=-1)

        inputs_embeds = self.token_embedding.to(device)(input_ids)

        # Scatter image embeddings into token embeddings
        n_image_tokens = int(torch.sum(input_ids == self.image_token_id).item())
        if n_image_tokens > 0 and data.get("image_embeds", None) is not None:
            image_embeds = data["image_embeds"].to(device)
            inputs_embeds = scatter_image_embeds(
                input_ids=input_ids,
                token_embeds=inputs_embeds,
                image_embeds=image_embeds,
                image_token_id=self.image_token_id,
            )
        elif data.get("image_embeds", None) is not None:
            image_embeds = data["image_embeds"].to(device)
            expected_tokens = int(image_embeds.shape[0])
            unique_ids, counts = torch.unique(input_ids, return_counts=True)
            matched = unique_ids[counts == expected_tokens]
            if matched.numel() == 1:
                detected_image_token_id = int(matched[0].item())
                self.image_token_id = detected_image_token_id
                inputs_embeds = scatter_image_embeds(
                    input_ids=input_ids,
                    token_embeds=inputs_embeds,
                    image_embeds=image_embeds,
                    image_token_id=self.image_token_id,
                )

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."

        if past_seq_length == 0:
            # Prefill: compute full rope index
            image_grid_thw = data.get("image_grid_thw", None)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)
            position_ids, rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
            # Fix rope_deltas for right-padding: get_rope_index computes
            # delta = max_position + 1 - total_len, where total_len includes
            # padding.  But the decode loop passes the *original* (unpadded)
            # seq_length as past_seq_length, so the delta must be relative to
            # the original length. Add back the padding offset.
            pad_len = self.input_sequence_length - seq_length
            if pad_len > 0:
                rope_deltas = rope_deltas + pad_len
            self.rope_deltas = rope_deltas
        else:
            # Decode: use cached rope_deltas
            assert self.rope_deltas is not None, f"rope_deltas is None but past_seq_length={past_seq_length}"
            batch_size, embed_seq_len, _ = inputs_embeds.shape
            delta = past_seq_length + self.rope_deltas
            position_ids = torch.arange(embed_seq_len, device=device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        past_key_caches = []
        past_value_caches = []
        for i in range(self.num_hidden_layers):
            past_key_caches.append(getattr(self, f"past_k_cache_{i}"))
            past_value_caches.append(getattr(self, f"past_v_cache_{i}"))

        return (
            inputs_embeds.to(device),
            position_ids.to(device=device, dtype=torch.float16),
            torch.tensor([past_seq_length], dtype=torch.int32, device=device),
            torch.tensor([seq_length], dtype=torch.int32, device=device),
            past_key_caches,
            past_value_caches,
        )

    # ------------------------------------------------------------------ #
    #  Prefill / Decode                                                   #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def prefill(self, data: Union[dict, tuple, list], save_golden: bool = False):
        input_ids = data["input_ids"]
        input_seq_len = input_ids.shape[-1]
        steps = (input_seq_len + self.prefill_input_sequence_length - 1) // self.prefill_input_sequence_length
        self.input_sequence_length = self.prefill_input_sequence_length * steps
        inputs = self.prepare_inputs(data)

        (
            inputs_embeds,
            position_ids,
            past_seq_length,
            _,
            past_key_caches,
            past_value_caches,
        ) = inputs

        golden_dir = getattr(self.prefill_session, "golden_dir", None) if save_golden else None
        for i in range(steps):
            if golden_dir:
                step_golden_dir = Path(golden_dir) / f"prefill_step_{i}"
                step_golden_dir.mkdir(exist_ok=True, parents=True)
                self.save_prefill_golden(step_golden_dir)

            start = i * self.prefill_input_sequence_length
            end = (i + 1) * self.prefill_input_sequence_length
            current_input_length = min(end, input_seq_len) - start
            output = self.prefill_session(
                inputs_embeds[:, start:end, :],
                position_ids[:, :, start:end],
                past_seq_length,
                torch.tensor([current_input_length], dtype=torch.int32, device=inputs_embeds.device),
                *past_key_caches,
                *past_value_caches,
            )
            past_seq_length += current_input_length

        if golden_dir:
            self.save_prefill_golden(golden_dir)
        return output

    @torch.no_grad()
    def decode(self, data: Union[dict, tuple, list]):
        self.input_sequence_length = 1
        inputs = self.prepare_inputs(data)
        (
            inputs_embeds,
            position_ids,
            past_seq_length,
            current_seq_length,
            past_key_caches,
            past_value_caches,
        ) = inputs
        return self.decode_session(
            inputs_embeds,
            position_ids,
            past_seq_length,
            current_seq_length,
            *past_key_caches,
            *past_value_caches,
        )

    # ------------------------------------------------------------------ #
    #  High-level chat interface                                          #
    # ------------------------------------------------------------------ #

    def release_all_sessions(self):
        """Explicitly release all cached HMONNX sessions and free GPU memory."""
        self.release_image_feature()
        self.release_prefill_session()
        self.release_decode_session()

    @torch.no_grad()
    def chat(
        self,
        prompt: str,
        image_path: str,
        processor,
        logger=None,
        use_fast: bool = False,
        do_sample: bool = False,
        max_new_tokens: int = 1024,
        keep_sessions: bool = False,
        image_feature_fn=None,
    ) -> str:
        from tqdm import tqdm

        runtime_device = self._exec_device
        if not isinstance(runtime_device, torch.device):
            runtime_device = torch.device(runtime_device)
        self.set_exec_device(runtime_device)
        self.to(runtime_device)

        stop_token_ids = set(self.eos_token_id)
        tokenizer_eos = getattr(processor.tokenizer, "eos_token_id", None)
        tokenizer_pad = getattr(processor.tokenizer, "pad_token_id", None)
        if tokenizer_eos is not None:
            stop_token_ids.add(int(tokenizer_eos))
        if tokenizer_pad is not None:
            stop_token_ids.add(int(tokenizer_pad))
        for token in ["<|user|>", "<|assistant|>", "<|observation|>", "<eop>"]:
            token_id = processor.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and int(token_id) >= 0:
                stop_token_ids.add(int(token_id))

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        target_w = int(self.image_size_w)
        target_h = int(self.image_size_h)
        if (orig_w, orig_h) != (target_w, target_h):
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=(114, 114, 114))

        messages = build_messages(image, prompt)
        inputs = build_inputs(processor, messages, device=runtime_device)
        inputs.pop("mm_token_type_ids", None)

        input_ids = inputs["input_ids"].to(runtime_device)
        pixel_values = inputs["pixel_values"].to(runtime_device)
        image_grid_thw = inputs["image_grid_thw"].to(runtime_device)

        # --- Vision ---
        if image_feature_fn is not None:
            image_features = image_feature_fn(pixel_values, image_grid_thw)
        elif self.image_feature_session is None:
            self.init_image_feature()
            self.to(runtime_device)
            self.set_exec_device(runtime_device)
            if use_fast and self._vision_runtime == "hmonnx":
                self.image_feature_session.initialize()
                self.image_feature_session._session.to_fast_mode()
            image_features = self.extract_image_features(pixel_values, image_grid_thw)
        else:
            image_features = self.extract_image_features(pixel_values, image_grid_thw)
        if image_feature_fn is None and not keep_sessions:
            self.release_image_feature()

        # --- Prefill ---
        decoder_ids: list = []

        data_prefill = {
            "input_ids": input_ids,
            "image_embeds": image_features,
            "past_seq_length": 0,
            "image_grid_thw": image_grid_thw,
        }

        if self.prefill_session is None:
            self.init_prefill()
            self.to(runtime_device)
            self.set_exec_device(runtime_device)
            if use_fast:
                self.prefill_session.initialize()
                self.prefill_session._session.to_fast_mode()
        prefill_logits = self.prefill(data_prefill, save_golden=False)
        prefill_logits = prefill_logits.to(input_ids.device)

        if self.presence_penalty > 0:
            prefill_logits = self.repetition_penalty_logits_processor(
                input_ids, prefill_logits[:, -1, :].float()
            ).unsqueeze(1)

        next_token_id, next_token_text = decode_next_token(
            processor.tokenizer, prefill_logits, do_sample=do_sample,
        )
        next_token_id = next_token_id.to(input_ids.device)
        all_input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        decoder_ids.append(next_token_id)
        if logger:
            logger.info(f"Prefill next token: {next_token_id} {next_token_text}")
        if not keep_sessions:
            self.release_prefill_session()

        # --- Decode ---
        if self.decode_session is None:
            self.init_decode()
            self.to(runtime_device)
            self.set_exec_device(runtime_device)
            if use_fast:
                self.decode_session.initialize()
                self.decode_session._session.to_fast_mode()

        current_length = input_ids.shape[-1]
        decode_limit = min(current_length + max_new_tokens, self.cache_len)
        if current_length + max_new_tokens > self.cache_len and logger:
            logger.warning(
                f"max_new_tokens({max_new_tokens}) + prompt({current_length}) = "
                f"{current_length + max_new_tokens} > cache_len({self.cache_len}), "
                f"output will be truncated to {self.cache_len - current_length} tokens"
            )
        for decoder_index in tqdm(range(current_length, decode_limit), desc="Decoder"):

            data_decode = {
                "input_ids": next_token_id,
                "past_seq_length": decoder_index,
            }
            decode_logits = self.decode(data_decode)
            decode_logits = decode_logits.to(all_input_ids.device)

            if self.presence_penalty > 0:
                decode_logits = self.repetition_penalty_logits_processor(
                    all_input_ids, decode_logits[:, -1, :].float()
                ).unsqueeze(1)

            next_token_id, next_token_text = decode_next_token(
                processor.tokenizer, decode_logits, do_sample=do_sample,
            )
            next_token_id = next_token_id.to(all_input_ids.device)
            all_input_ids = torch.cat([all_input_ids, next_token_id], dim=-1)
            decoder_ids.append(next_token_id)
            if logger:
                logger.info(f"Decode next token: {next_token_id} {next_token_text}")

            if int(next_token_id.item()) in stop_token_ids:
                break

            out = processor.decode(torch.cat(decoder_ids, dim=-1).view(-1)).strip()
            if logger:
                logger.info(f"Output: {out}")

        decoder_ids_tensor = torch.cat(decoder_ids, dim=-1).view(-1)
        out = processor.decode(decoder_ids_tensor).strip()
        if logger:
            logger.info(f"Output: {out}")
        if not keep_sessions:
            self.release_decode_session()
        return out
