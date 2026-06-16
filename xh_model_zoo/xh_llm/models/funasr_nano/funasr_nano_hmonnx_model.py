"""HMONNX inference wrapper for FunASR-Nano.

The wrapper is intentionally runtime-only: export scripts can emit the audio
encoder, audio adaptor, optional CTC branch and Qwen3 decoder as independent
HMONNX files, then this class stitches them back into the FunASR-Nano pipeline.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from xhquant.api import CacheTensor, ConfigDict, HMONNXInference
from funasr.models.fun_asr_nano.tools.utils import forced_align

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional dependency is validated at runtime
    AutoTokenizer = None

from ..builder import MODELS
from ..device_dtype_mixin import DeviceDtypeMixin


TensorLikeAudio = Union[str, np.ndarray, torch.Tensor]


def _as_config_dict(value: Any) -> ConfigDict:
    if isinstance(value, ConfigDict):
        return value
    if isinstance(value, dict):
        return ConfigDict(value)
    return ConfigDict(dict(value or {}))


def _resolve_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base / path


def _sequence_mask(lengths: torch.Tensor, maxlen: int) -> torch.Tensor:
    row = torch.arange(0, maxlen, 1, device=lengths.device)
    return (row < lengths.unsqueeze(-1)).to(torch.float32)[:, None, :]


def _attention_additive_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    invalid = valid_mask[:, None, :, :].eq(0)
    return torch.zeros_like(invalid, dtype=torch.float32).masked_fill(invalid, -32752.0)


def _downsample_lengths(lengths: torch.Tensor, rate: int) -> torch.Tensor:
    return ((lengths - 1) // int(rate) + 1).to(torch.int32)


def _downsample_time(time_length: int, rate: int) -> int:
    return (int(time_length) - 1) // int(rate) + 1


@MODELS.register_module()
class FunASRNanoHMONNXModel(DeviceDtypeMixin, nn.Module):
    """FunASR-Nano HMONNX pipeline.

    Expected ``export_meta_info.json`` fields are intentionally compatible with
    the existing Qwen3-ASR/CosyVoice style metadata::

        {
          "hf_config": "ConfigFiles",
          "token_embedding_file": "token_embedding.pt",
          "encoder_hmonnx_file": "Encoder/hmonnx/encoder.onnx",
          "audio_adaptor_hmonnx_file": "Adaptor/hmonnx/adaptor.onnx",
          "ctc_decoder_hmonnx_file": "CTCDecoder/hmonnx/ctc_decoder.onnx",
          "ctc_hmonnx_file": "CTC/hmonnx/ctc.onnx",
          "prefill_onnx_file": "Prefill/prefill.onnx",
          "decode_onnx_file": "Decoder/decode.onnx",
          "num_hidden_layers": 28,
          "kv_cache_shape": [1, 8, 2048, 128]
        }
    """

    def __init__(self, model_dir: str, meta_file: str = "export_meta_info.json"):
        super().__init__()
        self.model_dir = Path(model_dir).expanduser().resolve()
        meta_path = self.model_dir / meta_file
        if not meta_path.exists():
            alt_meta_path = self.model_dir / "meta_info.json"
            if alt_meta_path.exists():
                meta_path = alt_meta_path
            else:
                raise FileNotFoundError(f"metadata file not found: {meta_path}")

        self.meta_info = _as_config_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        self._exec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16

        self.encoder_session = self._make_session(
            "encoder_hmonnx_file", "encoder_onnx_file", "encoder_onnx", required=True
        )
        self.adaptor_session = self._make_session(
            "audio_adaptor_hmonnx_file", "adaptor_hmonnx_file", "audio_adaptor_onnx_file"
        )
        # ``ctc_onnx_file`` exported by examples/audio/funasr_nano_xh2a/export_audio_modules_onnx.py
        # is the full CTC branch: ctc_decoder + ctc.log_softmax. Keep this as
        # the primary runtime path. The separated decoder/head sessions are
        # only for metadata produced by other exporters.
        self.ctc_branch_session = self._make_session("ctc_branch_hmonnx_file", "ctc_hmonnx_file", "ctc_onnx_file")
        self.ctc_decoder_session = None if self.ctc_branch_session is not None else self._make_session(
            "ctc_decoder_hmonnx_file", "ctc_decoder_onnx_file"
        )
        self.ctc_head_session = self._make_session("ctc_head_hmonnx_file", "ctc_head_onnx_file")
        self.prefill_session = self._make_session(
            "prefill_hmonnx_file", "prefill_onnx_file", "prefill_onnx", required=True
        )
        self.decode_session = self._make_session(
            "decode_hmonnx_file", "decode_onnx_file", "decode_onnx", required=True
        )

        self.tokenizer = self._load_tokenizer()
        self.embed_tokens = self._load_embedding_layer()

        kv_shape = self.meta_info.get("kv_cache_shape", self.meta_info.get("kv_shape"))
        if kv_shape is None:
            raise KeyError("metadata must contain kv_cache_shape")
        self.kv_cache_shape = tuple(int(x) for x in kv_shape)
        self.num_hidden_layers = int(self.meta_info.get("num_hidden_layers", 0))
        if self.num_hidden_layers <= 0:
            self.num_hidden_layers = int(self.meta_info.get("model_cfg", {}).get("num_decode_layers", 0))
        if self.num_hidden_layers <= 0:
            raise KeyError("metadata must contain num_hidden_layers or model_cfg.num_decode_layers")

        self.cache_axis = int(self.meta_info.get("cache_axis", 2))
        self.max_prefill_length = int(
            self.meta_info.get(
                "prefill_input_sequence_length",
                self.meta_info.get("wrap_cfg", {}).get("input_sequence_length", 0),
            )
            or 0
        )
        self.max_frames = int(self.meta_info.get("max_frames", 0) or 0)
        self.pad_token_id = int(self.meta_info.get("pad_token_id", 0))
        self.use_low_frame_rate = bool(self.meta_info.get("use_low_frame_rate", True))
        self.blank_id = int(self.meta_info.get("blank_id", -1))

        self.ctc_tokenizer = None
        ctc_tokenizer_path = _resolve_path(self.model_dir, self.meta_info.get("ctc_tokenizer_file"))
        if ctc_tokenizer_path is not None and ctc_tokenizer_path.exists():
            self.ctc_tokenizer = self._load_ctc_tokenizer(str(ctc_tokenizer_path))
        elif self.meta_info.get("ctc_tokenizer") is not None:
            self.ctc_tokenizer = self._load_ctc_tokenizer(None)
        elif self.meta_info.get("ctc_tokenizer_source") == "funasr_model":
            self.ctc_tokenizer = self._load_ctc_tokenizer_from_funasr_model()

        self._to_sessions(self._exec_device)

    @property
    def device(self) -> torch.device:
        return self._exec_device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def to(self, *args, **kwargs):  # noqa: D401 - align with nn.Module.to
        module = super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        if args:
            for arg in args:
                if isinstance(arg, (str, torch.device)):
                    device = arg
                elif isinstance(arg, torch.dtype):
                    dtype = arg
        if device is not None:
            self._exec_device = torch.device(device)
            self._to_sessions(self._exec_device)
        if dtype is not None:
            self._dtype = dtype
        self.embed_tokens.to(device=self._exec_device, dtype=self._dtype)
        return module

    def _make_session(self, *keys: str, required: bool = False) -> Optional[HMONNXInference]:
        path = None
        for key in keys:
            path = _resolve_path(self.model_dir, self.meta_info.get(key))
            if path is not None:
                break
        if path is None:
            if required:
                raise KeyError(f"missing required session path, tried keys: {keys}")
            return None
        if not path.exists():
            if required:
                raise FileNotFoundError(str(path))
            return None
        return HMONNXInference(str(path))

    def _to_sessions(self, device: torch.device) -> None:
        for session in (
            self.encoder_session,
            self.adaptor_session,
            self.ctc_branch_session,
            self.ctc_decoder_session,
            self.ctc_head_session,
            self.prefill_session,
            self.decode_session,
        ):
            if session is not None:
                session.to(str(device))

    def _load_tokenizer(self):
        if AutoTokenizer is None:
            raise ImportError("transformers is required to load the FunASR-Nano tokenizer")
        cfg_dir = _resolve_path(self.model_dir, self.meta_info.get("hf_config"))
        if cfg_dir is None:
            cfg_dir = _resolve_path(self.model_dir, self.meta_info.get("hf_model"))
        if cfg_dir is None:
            cfg_dir = self.model_dir
        return AutoTokenizer.from_pretrained(str(cfg_dir), trust_remote_code=True, use_fast=True)

    def _load_embedding_layer(self) -> nn.Embedding:
        state_path = _resolve_path(self.model_dir, self.meta_info.get("token_embedding_file"))
        if state_path is None or not state_path.exists():
            raise FileNotFoundError("token_embedding_file is missing from metadata or does not exist")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        weight = state["weight"] if isinstance(state, dict) and "weight" in state else state
        embedding = nn.Embedding(weight.shape[0], weight.shape[1])
        if isinstance(state, dict) and "weight" in state:
            embedding.load_state_dict(state)
        else:
            embedding.weight.data.copy_(weight)
        return embedding.to(device=self._exec_device, dtype=self._dtype).eval()

    def _load_ctc_tokenizer(self, vocab_path: Optional[str]):
        from funasr.register import tables

        name = self.meta_info.get("ctc_tokenizer", "SenseVoiceTokenizer")
        conf = dict(self.meta_info.get("ctc_tokenizer_conf", {}))
        if vocab_path is not None:
            conf.setdefault("vocab_path", vocab_path)
        conf.setdefault("is_multilingual", True)
        tokenizer_cls = tables.tokenizer_classes.get(name)
        if tokenizer_cls is None:
            return None
        return tokenizer_cls(**conf)

    def _load_ctc_tokenizer_from_funasr_model(self):
        from funasr import AutoModel

        model_dir = str(_resolve_path(self.model_dir, self.meta_info.get("funasr_model_dir")) or self.model_dir)
        model, _ = AutoModel.build_model(model=model_dir, device="cpu", trust_remote_code=False)
        return getattr(model, "ctc_tokenizer", None)

    def _ctc_postprocess(self, ctc_logits: torch.Tensor, ctc_lens: torch.Tensor, result: dict[str, Any]) -> None:
        if self.ctc_tokenizer is None or self.blank_id < 0:
            result["ctc_logits"] = ctc_logits
            result["ctc_lens"] = ctc_lens
            return

        x = ctc_logits[0, : int(ctc_lens[0].item()), :]
        yseq = x.argmax(dim=-1)
        yseq = torch.unique_consecutive(yseq, dim=-1)
        mask = yseq != self.blank_id
        token_int = yseq[mask].tolist()
        ctc_text = self.ctc_tokenizer.decode(token_int).replace("<|nospeech|>", "")
        result["ctc_text"] = ctc_text
        result["ctc_logits"] = x

        ctc_target_ids = torch.tensor(self.ctc_tokenizer.encode(ctc_text), dtype=torch.int64, device=x.device)
        result["ctc_timestamps"] = forced_align(x, ctc_target_ids, self.blank_id)
        text_target_ids = torch.tensor(
            self.ctc_tokenizer.encode(result.get("text", "")), dtype=torch.int64, device=x.device
        )
        result["timestamps"] = forced_align(x, text_target_ids, self.blank_id)
        for timestamps in [result["timestamps"], result["ctc_timestamps"]]:
            for timestamp in timestamps:
                timestamp["token"] = self.ctc_tokenizer.decode([timestamp["token"]])
                timestamp["start_time"] = timestamp["start_time"] * 6 * 10 / 1000
                timestamp["end_time"] = timestamp["end_time"] * 6 * 10 / 1000

    def _extract_fbank(self, audio: TensorLikeAudio) -> tuple[torch.Tensor, torch.Tensor]:
        from funasr import AutoModel
        from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

        model_dir = str(_resolve_path(self.model_dir, self.meta_info.get("funasr_model_dir")) or self.model_dir)
        _, kwargs = AutoModel.build_model(model=model_dir, device="cpu", trust_remote_code=False)
        frontend = kwargs.get("frontend")
        if frontend is None:
            raise RuntimeError("FunASR frontend is not available from model metadata")

        if isinstance(audio, str):
            wav = load_audio_text_image_video(audio, fs=frontend.fs)
        elif isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio).float()
        elif isinstance(audio, torch.Tensor):
            wav = audio.float()
        else:
            raise TypeError(f"unsupported audio input type: {type(audio)}")
        speech, speech_lengths = extract_fbank(wav, data_type="sound", frontend=frontend, is_final=True)
        if self.max_frames > 0:
            cur_frames = int(speech.shape[1])
            if cur_frames < self.max_frames:
                speech = torch.nn.functional.pad(speech, (0, 0, 0, self.max_frames - cur_frames))
                speech_lengths = torch.tensor([self.max_frames], dtype=torch.int32)
            elif cur_frames > self.max_frames:
                speech = speech[:, : self.max_frames, :]
                speech_lengths = torch.tensor([self.max_frames], dtype=torch.int32)
        return speech.to(self._exec_device, dtype=torch.float32), speech_lengths.to(self._exec_device)

    def encode_audio(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run encoder -> adaptor and optionally CTC branch."""
        speech_mask = _sequence_mask(speech_lengths.to(torch.int32), int(speech.shape[1])).to(self._dtype)
        speech_att_mask = _attention_additive_mask(speech_mask).to(self._dtype)
        enc_inputs = self._run_by_names(
            self.encoder_session,
            {
                "speech": speech.to(self._dtype),
                "speech_mask": speech_mask,
                "speech_att_mask": speech_att_mask,
                "speech_lengths": speech_lengths.to(torch.int32),
                "input_features": speech.transpose(1, 2).to(self._dtype),
                "feature_lens": speech_lengths.to(torch.int32),
            },
        )
        encoder_out, encoder_out_lens = self._split_outputs(enc_inputs)

        adaptor_out = encoder_out.to(self._dtype)
        adaptor_lens = encoder_out_lens.to(torch.int32)
        if self.adaptor_session is not None:
            adaptor_rate = int(self.meta_info.get("audio_adaptor_downsample_rate", 1) or 1)
            adaptor_lens_in = _downsample_lengths(encoder_out_lens.to(torch.int32), adaptor_rate)
            adaptor_mask = _sequence_mask(
                adaptor_lens_in, _downsample_time(int(encoder_out.shape[1]), adaptor_rate)
            ).to(self._dtype)
            adaptor_att_mask = _attention_additive_mask(adaptor_mask).to(self._dtype)
            adaptor_out, adaptor_lens = self._split_outputs(
                self._run_by_names(
                    self.adaptor_session,
                    {
                        "encoder_out": encoder_out.to(self._dtype),
                        "adaptor_mask": adaptor_mask,
                        "adaptor_att_mask": adaptor_att_mask,
                        "encoder_out_lens": encoder_out_lens.to(torch.int32),
                        "speech_lengths": encoder_out_lens.to(torch.int32),
                    },
                )
            )

        ctc_logits = None
        ctc_session = self.ctc_branch_session or self.ctc_decoder_session
        if ctc_session is not None:
            ctc_rate = int(self.meta_info.get("ctc_decoder_downsample_rate", 1) or 1)
            ctc_lens_in = _downsample_lengths(encoder_out_lens.to(torch.int32), ctc_rate)
            ctc_mask = _sequence_mask(ctc_lens_in, _downsample_time(int(encoder_out.shape[1]), ctc_rate)).to(self._dtype)
            ctc_att_mask = _attention_additive_mask(ctc_mask).to(self._dtype)
            ctc_decoder_out, ctc_decoder_lens = self._split_outputs(
                self._run_by_names(
                    ctc_session,
                    {
                        "encoder_out": encoder_out.to(self._dtype),
                        "ctc_mask": ctc_mask,
                        "ctc_att_mask": ctc_att_mask,
                        "encoder_out_lens": encoder_out_lens.to(torch.int32),
                    },
                )
            )
            if self.ctc_branch_session is not None:
                ctc_logits = ctc_decoder_out
            elif self.ctc_head_session is not None:
                ctc_logits = self._first_tensor(
                    self._run_by_names(self.ctc_head_session, {"encoder_out": ctc_decoder_out})
                )
            else:
                ctc_logits = ctc_decoder_out
            encoder_out_lens = ctc_decoder_lens

        return adaptor_out, adaptor_lens, ctc_logits, encoder_out_lens

    def _run_by_names(self, session: HMONNXInference, candidates: dict[str, torch.Tensor]) -> Any:
        names = []
        try:
            names = list(session.get_input_names())
        except Exception:
            pass
        if names:
            feed = {name: candidates[name] for name in names if name in candidates}
            if len(feed) == len(names):
                return session.run(feed)
        ordered = [v for k, v in candidates.items() if k in candidates]
        return session(*ordered)

    @staticmethod
    def _first_tensor(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        if isinstance(outputs, dict):
            return next(iter(outputs.values()))
        raise TypeError(f"unsupported HMONNX output type: {type(outputs)}")

    def _split_outputs(self, outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(outputs, dict):
            values = list(outputs.values())
        elif isinstance(outputs, (list, tuple)):
            values = list(outputs)
        else:
            values = [outputs]
        data = values[0]
        if len(values) > 1:
            lens = values[1]
        else:
            lens = torch.tensor([data.shape[1]], dtype=torch.int32, device=data.device)
        return data, lens

    def _build_prompt_text(
        self, hotwords: Optional[Sequence[str]] = None, language: Optional[str] = None, itn: bool = True
    ) -> str:
        hotwords = list(hotwords or [])
        prompt = ""
        if hotwords:
            prompt += (
                "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n"
                "**上下文信息：**\n\n\n"
            )
            prompt += f"热词列表：[{', '.join(hotwords)}]\n"
        prompt += "语音转写" if language is None else f"语音转写成{language}"
        if not itn:
            prompt += "，不进行文本规整"
        return prompt + "："

    def build_input_embeds(
        self,
        audio_embeds: torch.Tensor,
        audio_lens: torch.Tensor,
        hotwords: Optional[Sequence[str]] = None,
        language: Optional[str] = None,
        itn: bool = True,
        system_prompt: str = "You are a helpful assistant.",
    ) -> torch.Tensor:
        prompt = self._build_prompt_text(hotwords, language, itn)
        prefix = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|startofspeech|>"
        suffix = "<|endofspeech|><|im_end|>\n<|im_start|>assistant\n"
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        ids = torch.tensor([prefix_ids + suffix_ids], dtype=torch.long, device=self._exec_device)
        split = len(prefix_ids)
        text_embeds = self.embed_tokens(ids)[0]
        audio_len = int(audio_lens[0].item())
        return torch.cat(
            [text_embeds[:split], audio_embeds[0, :audio_len, :].to(text_embeds.dtype), text_embeds[split:]],
            dim=0,
        ).unsqueeze(0)

    def _new_caches(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        k_caches = [
            CacheTensor(torch.zeros(self.kv_cache_shape, dtype=self._dtype, device=self._exec_device))
            for _ in range(self.num_hidden_layers)
        ]
        v_caches = [
            CacheTensor(torch.zeros(self.kv_cache_shape, dtype=self._dtype, device=self._exec_device))
            for _ in range(self.num_hidden_layers)
        ]
        return k_caches, v_caches

    def _pad_prefill(self, inputs_embeds: torch.Tensor) -> tuple[torch.Tensor, int]:
        seq_len = int(inputs_embeds.shape[1])
        target_len = self.max_prefill_length or seq_len
        if seq_len > target_len:
            inputs_embeds = inputs_embeds[:, :target_len, :]
            seq_len = target_len
        elif seq_len < target_len:
            pad_ids = torch.full(
                (1, target_len - seq_len), self.pad_token_id, dtype=torch.long, device=self._exec_device
            )
            pad_embeds = self.embed_tokens(pad_ids).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, pad_embeds], dim=1)
        return inputs_embeds.to(self._dtype), seq_len

    def _decoder_feed(
        self,
        session: HMONNXInference,
        inputs_embeds: torch.Tensor,
        valid_length: torch.Tensor,
        current_length: torch.Tensor,
        k_caches: Sequence[torch.Tensor],
        v_caches: Sequence[torch.Tensor],
    ) -> Any:
        names = []
        try:
            names = list(session.get_input_names())
        except Exception:
            pass
        if names:
            feed = {
                "input_embeds": inputs_embeds,
                "inputs_embeds": inputs_embeds,
                "past_seq_length": valid_length,
                "valid_length": valid_length,
                "current_input_length": current_length,
                "current_length": current_length,
            }
            inputs = {name: feed[name] for name in names if name in feed}
            for i in range(self.num_hidden_layers):
                key_candidates = [
                    f"past_key_cache_{i}",
                    f"model_layers_{i}_self_attn_kcache_input",
                    f"past_k_cache_{i}",
                ]
                val_candidates = [
                    f"past_value_cache_{i}",
                    f"model_layers_{i}_self_attn_vcache_input",
                    f"past_v_cache_{i}",
                ]
                for name in key_candidates:
                    if name in names:
                        inputs[name] = k_caches[i]
                for name in val_candidates:
                    if name in names:
                        inputs[name] = v_caches[i]
            if len(inputs) == len(names):
                return session.run(inputs)
        return session(inputs_embeds, valid_length, current_length, *k_caches, *v_caches)

    def _decode_logits(self, outputs: Any) -> torch.Tensor:
        logits = self._first_tensor(outputs)
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        return logits

    @torch.no_grad()
    def generate(
        self,
        inputs: Union[TensorLikeAudio, Sequence[TensorLikeAudio]],
        hotwords: Optional[Sequence[str]] = None,
        language: Optional[str] = None,
        itn: bool = True,
        max_new_tokens: int = 512,
        eos_token_id: Optional[int] = None,
        **_: Any,
    ) -> List[dict[str, Any]]:
        if isinstance(inputs, (str, np.ndarray, torch.Tensor)):
            inputs = [inputs]
        eos_token_id = int(eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id)
        results = []
        for index, audio in enumerate(inputs):
            speech, speech_lengths = self._extract_fbank(audio)
            audio_embeds, audio_lens, ctc_logits, ctc_lens = self.encode_audio(speech, speech_lengths)
            inputs_embeds = self.build_input_embeds(audio_embeds, audio_lens, hotwords, language, itn)
            inputs_embeds, seq_len = self._pad_prefill(inputs_embeds)
            k_caches, v_caches = self._new_caches()

            valid_length = torch.tensor([0], dtype=torch.int32, device=self._exec_device)
            current_length = torch.tensor([seq_len], dtype=torch.int32, device=self._exec_device)
            outputs = self._decoder_feed(
                self.prefill_session, inputs_embeds, valid_length, current_length, k_caches, v_caches
            )
            next_token = int(torch.argmax(self._decode_logits(outputs), dim=-1).item())
            generated = [next_token]
            valid_length = torch.tensor([seq_len], dtype=torch.int32, device=self._exec_device)
            current_length = torch.tensor([1], dtype=torch.int32, device=self._exec_device)

            for _step in range(max_new_tokens - 1):
                token_tensor = torch.tensor([[generated[-1]]], dtype=torch.long, device=self._exec_device)
                token_embed = self.embed_tokens(token_tensor).to(self._dtype)
                outputs = self._decoder_feed(
                    self.decode_session, token_embed, valid_length, current_length, k_caches, v_caches
                )
                next_token = int(torch.argmax(self._decode_logits(outputs), dim=-1).item())
                generated.append(next_token)
                valid_length = valid_length + 1
                if next_token == eos_token_id:
                    break

            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            text = re.sub(r"<[^>]*>", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            key = os.path.splitext(os.path.basename(audio))[0] if isinstance(audio, str) else f"sample_{index}"
            result: dict[str, Any] = {"key": key, "text": text, "token_ids": generated}
            if ctc_logits is not None:
                self._ctc_postprocess(ctc_logits, ctc_lens, result)
            results.append(result)
        return results
