# Copyright 2025 HOUMO AI
#
# File: _model.py
# Description:
#   Export wrappers for FunAudioChat split deployment.
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

from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from xhquant import nn as xhnn
from xhquant.api import Config, ConfigDict, HMONNXInference
from xhquant.nn import MaskedSoftmax
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES, wrap_llm_model
from .modeling_funaudiochat import (
    FunAudioChatAudioAttention,
    FunAudioChatAudioEncoderLayer,
    FunAudioChatDecoder,
    FunAudioChatDiscreteEncoder,
    FunAudioChatForConditionalGeneration,
)


def _patched_prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs):
    speech_ids = kwargs.pop("speech_ids", None)
    single_modal = kwargs.pop("single_modal", False)
    speech_attention_mask = kwargs.pop("speech_attention_mask", None)

    model_inputs = super(FunAudioChatHMONNXForConditionalGeneration, self).prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )

    if not self.is_prefill:
        text_features = self.get_input_embeddings()(input_ids[:, -1]).unsqueeze(1)
        model_inputs.update({"text_embeds": text_features})

    if not any(self.generate_speech) and self.audio_invert_tower is not None:
        self.audio_invert_tower.crq_grobal_step = 0

    self.generate_speech |= input_ids[:, -1] == self.config.text_config.audio_bos_index
    if any(self.generate_speech) and speech_ids is not None and speech_ids.shape[-1] != 0:
        last_group_speech_ids = speech_ids[:, -self.config.audio_config.group_size:]
        if getattr(self, "use_hmonnx_audio_tower", False) and getattr(self, "hmonnx_audio_tower_session", None) is not None:
            runtime_device = (
                self.hmonnx_audio_tower_session._device
                if hasattr(self.hmonnx_audio_tower_session, "_device")
                else torch.device("cuda:0")
            )
            audio_features = self.hmonnx_audio_tower_session(last_group_speech_ids.to(dtype=torch.int32, device=runtime_device))
            if isinstance(audio_features, tuple):
                audio_features = audio_features[0]
            audio_features = audio_features.to(input_ids.device)
        else:
            audio_features = self.audio_tower(last_group_speech_ids)[0]
        text_features = self.get_input_embeddings()(input_ids[:, -1]).unsqueeze(1)
        if not single_modal:
            audio_features = (text_features + audio_features) / 2
        inputs_embeds = torch.where(self.generate_speech[:, None].unsqueeze(1), audio_features, text_features)
        model_inputs.update({"input_ids": None, "inputs_embeds": inputs_embeds})

    model_inputs.update(
        {
            "speech_attention_mask": speech_attention_mask,
            "speech_ids": speech_ids,
        }
    )
    return model_inputs


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        FunAudioChatAudioAttention: "FunAudioChatAudioAttention",
    }
)
class _FunAudioChatAudioAttention(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)

        bsz, seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(bsz, seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(bsz, seq_length, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(bsz, seq_length, self.num_heads, self.head_dim)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states = query_states * self.kv_scale
        key_states = key_states.transpose(2, 3)
        attn_weights = torch.matmul(query_states, key_states)

        if attention_mask is not None:
            attn_weights = self.maskedadd(attn_weights, attention_mask)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        # if attn_output.shape[0] == 1:
        attn_output = attn_output.squeeze(0)
        return attn_output

    def _setup(self, cfg: Optional[Dict[str, Any]] = None):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        self.kv_scale = 1 / math.sqrt(self.head_dim)
        self.maskedadd = xhnn.MaskedAdd()
        # self.masked_softmax = MaskedSoftmax(dim=-1)
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        FunAudioChatAudioEncoderLayer: "FunAudioChatAudioEncoderLayer",
    }
)
class _FunAudioChatAudioEncoderLayer(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = torch.clamp(hidden_states, min=-64504.0, max=64504.0)

        return (hidden_states,)

    def _setup(self, cfg: Optional[Dict[str, Any]] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        FunAudioChatDiscreteEncoder: "FunAudioChatDiscreteEncoder",
    }
)
class _FunAudioChatDiscreteEncoder(DynamicModule):
    def forward(
        self,
        audio_ids,
        inputs_embeds=None,
        continuous_audio_features=None,
        continuous_audio_valid_mask=None,
        continuous_audio_output_lengths=None,
        feature_exist_mask=None,
        return_dict=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(audio_ids)
        continuous_audio_hidden_states = None

        inputs_embeds = inputs_embeds.reshape(inputs_embeds.shape[0], -1, self.group_size * self.hidden_size)

        if continuous_audio_features is not None:
            continuous_audio_features = continuous_audio_features.reshape(
                continuous_audio_features.shape[0], -1, self.group_size, self.hidden_size
            )
            continuous_audio_features = continuous_audio_features.mean(dim=2)
            # if continuous_audio_valid_mask is not None:
            #     if continuous_audio_valid_mask.dim() == 4:
            #         pooled_valid_mask = (continuous_audio_valid_mask.sum(dim=2) > 0).to(continuous_audio_features.dtype)
            #     else:
            #         pooled_valid_mask = continuous_audio_valid_mask.to(continuous_audio_features.dtype)
            #     continuous_audio_features = continuous_audio_features * pooled_valid_mask
            continuous_audio_hidden_states = self.continual_output_matching(continuous_audio_features)
            hidden_states = continuous_audio_hidden_states
        else:
            hidden_states = self.output_matching(
                inputs_embeds.reshape(inputs_embeds.shape[0], -1, self.group_size, self.hidden_size).mean(dim=2)
            )

        encoder_states = (inputs_embeds, hidden_states, continuous_audio_hidden_states)
        all_attentions = None
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return hidden_states

    def _setup(self, cfg: Optional[Dict[str, Any]] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3ForCausalLM: "Qwen3ForCausalLM",
    }
)
class _FunAudioChatQwen3ForCausalLM(DynamicModule):
    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_seq_length: Optional[torch.Tensor] = None,
        current_input_length: Optional[torch.Tensor] = None,
        past_key_cache: Optional[list[torch.Tensor]] = None,
        past_value_cache: Optional[list[torch.Tensor]] = None,
    ):
        batch_size = inputs_embeds.shape[0]
        if current_input_length is None:
            current_input_length = torch.full(
                (batch_size,),
                inputs_embeds.shape[1],
                dtype=torch.int32,
                device=inputs_embeds.device,
            )
        if past_seq_length is None:
            past_seq_length = torch.zeros((batch_size,), dtype=torch.int32, device=inputs_embeds.device)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        last_hidden_state = outputs.last_hidden_state
        logits = self.lm_head(last_hidden_state)
        return logits, last_hidden_state

    def _setup(self, cfg: Optional[Dict[str, Any]] = None):
        return self


class FunAudioChatAudioEncoderExportWrapper(nn.Module):
    """Build LM-ready embeddings from FunAudioChat frontend inputs.

    This wrapper covers the frontend path before the Qwen3 language model:
    continuous audio tower + discrete audio tower + optional transcript fusion +
    scatter into token embeddings.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_lens: torch.LongTensor,
        aftercnn_lens: torch.LongTensor,
        speech_maxlen: int,
        feature_exist_mask: torch.Tensor,
        audio_token_positions: torch.LongTensor,
    ):
        super().__init__()
        self.model = model
        self.speech_maxlen = speech_maxlen
        self.register_buffer("feature_lens", feature_lens.to(torch.long), persistent=False)
        self.register_buffer("aftercnn_lens", aftercnn_lens.to(torch.long), persistent=False)
        self.register_buffer("feature_exist_mask", feature_exist_mask.to(torch.bool), persistent=False)
        self.register_buffer("audio_token_positions", audio_token_positions.to(torch.long), persistent=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        speech_ids: torch.LongTensor,
        input_features: torch.FloatTensor,
        text_ids: torch.LongTensor,
        text_attention_mask: torch.Tensor,
    ):
        target_device = input_ids.device
        input_ids = input_ids.to(target_device)
        speech_ids = speech_ids.to(target_device)
        input_features = input_features.to(target_device)
        text_ids = text_ids.to(target_device)
        text_attention_mask = text_attention_mask.to(target_device).bool()

        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        _, continuous_audio_output_lengths = self.model.audio_tower._get_feat_extract_output_lengths(
            torch.full_like(self.feature_lens, self.speech_maxlen)
        )
        audio_outputs = self.model.continuous_audio_tower(
            input_features,
            feature_lens=self.feature_lens,
            aftercnn_lens=self.aftercnn_lens,
            speech_maxlen=self.speech_maxlen,
        )
        continuous_audio_features = audio_outputs.last_hidden_state.to(inputs_embeds.device, inputs_embeds.dtype)

        audio_features, audio_embeds, _ = self.model.audio_tower(
            speech_ids,
            continuous_audio_features=continuous_audio_features,
            continuous_audio_output_lengths=continuous_audio_output_lengths,
            feature_exist_mask=self.feature_exist_mask,
        )

        text_ids = text_ids.clone()
        text_ids[text_ids == self.model.config.text_config.eos_token_id] = self.model.config.text_config.sil_index
        if self.model.config.text_config.pad_token_id is not None:
            text_ids[text_ids == self.model.config.text_config.pad_token_id] = self.model.config.text_config.sil_index

        text_features = self.model.get_input_embeddings()(text_ids)
        text_features = text_features * text_attention_mask[:, :, None].to(text_features.dtype)
        mixed_audio_features = (text_features + audio_features) / 2
        mix_mask = text_attention_mask[:, :, None].to(audio_features.dtype)
        audio_features = mixed_audio_features * mix_mask + audio_features * (1.0 - mix_mask)

        flat_audio_features = audio_features[:, : self.audio_token_positions.numel(), :]
        flat_text_features = text_features[:, : self.audio_token_positions.numel(), :]

        merged_inputs_embeds = inputs_embeds.clone()
        merged_inputs_embeds[:, self.audio_token_positions, :] = flat_audio_features.to(inputs_embeds.device, inputs_embeds.dtype)

        decoder_text_embeds = self.model.get_input_embeddings()(input_ids)
        decoder_text_embeds[:, self.audio_token_positions, :] = flat_text_features.to(
            decoder_text_embeds.device,
            decoder_text_embeds.dtype,
        )

        return merged_inputs_embeds, decoder_text_embeds, audio_embeds


class FunAudioChatDecoderExportWrapper(nn.Module):
    """Export wrapper for FunAudioChat audio decoder tower."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        speech_inputs_embeds: torch.FloatTensor,
        audio_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        out = self.decoder(
            inputs_embeds=speech_inputs_embeds,
            audio_embeds=audio_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        return out.logits


class FunAudioChatHMONNXForConditionalGeneration(FunAudioChatForConditionalGeneration):
    """FunAudioChat validation model with HMONNX audio-decoder runtime.

    The frontend encoder path and text Qwen path stay in PyTorch, while the
    audio decoder path can be executed with the exported prefill/decode HMONNX
    models for end-to-end validation.
    """

    def __init__(self, config):
        super().__init__(config)
        self.warp_context_length = 256
        self.warp_input_sequence_length = 256
        self.language_warp_past_key_caches = None
        self.language_warp_past_value_caches = None
        self.language_warp_past_seq_length = 0
        self.hmonnx_work_dir: Optional[Path] = None
        self.hmonnx_export_meta: Optional[dict[str, Any]] = None
        self.hmonnx_audio_encoder_session: Optional[HMONNXInference] = None
        self.hmonnx_audio_tower_session: Optional[HMONNXInference] = None
        self.hmonnx_language_prefill_session: Optional[HMONNXInference] = None
        self.hmonnx_language_decode_session: Optional[HMONNXInference] = None
        self.hmonnx_decoder_prefill_session: Optional[HMONNXInference] = None
        self.hmonnx_decoder_decode_session: Optional[HMONNXInference] = None
        self.use_hmonnx_encoder = False
        self.use_hmonnx_decoder = False
        self.use_hmonnx_language = False

    def load_hmonnx_runtime(
        self,
        work_dir: str | Path,
        device: str = "cuda:0",
        execution_device: str = "cuda:0",
        input_sequence_length: int = 256,
    ):
        work_dir = Path(work_dir)
        export_meta_path = work_dir / "export_meta.json"
        if export_meta_path.exists():
            export_meta = json.loads(export_meta_path.read_text(encoding="utf-8"))
        else:
            audio_meta_path = work_dir / "audio_encoder" / "meta.json"
            if not audio_meta_path.exists():
                raise FileNotFoundError(f"missing {export_meta_path} and {audio_meta_path}")
            audio_meta = json.loads(audio_meta_path.read_text(encoding="utf-8"))
            quant_type = audio_meta.get("audio_quant_type", "w8a8h1_sefp")
            export_meta = {
                "hf_model_path": "/data01/datasets/Fun-Audio-Chat-8B",
                "audio_encoder_hmonnx": f"audio_encoder/hmonnx/funaudiochat_audio_encoder_{quant_type}.onnx",
                "audio_encoder_meta": "audio_encoder/meta.json",
                "input_sequence_length": input_sequence_length,
            }
        self.hmonnx_work_dir = work_dir
        self.hmonnx_export_meta = export_meta

        self.hmonnx_audio_encoder_session = None
        self.hmonnx_audio_tower_session = None
        self.hmonnx_language_prefill_session = None
        self.hmonnx_language_decode_session = None
        self.hmonnx_decoder_prefill_session = None
        self.hmonnx_decoder_decode_session = None

        if export_meta.get("audio_encoder_hmonnx"):
            self.hmonnx_audio_encoder_session = HMONNXInference(
                str(work_dir / export_meta["audio_encoder_hmonnx"])
            )
            self.hmonnx_audio_encoder_session.exec_device = torch.device(execution_device)
            self.hmonnx_audio_encoder_session.to(torch.device(device))

        if export_meta.get("audio_tower_hmonnx"):
            self.hmonnx_audio_tower_session = HMONNXInference(
                str(work_dir / export_meta["audio_tower_hmonnx"])
            )
            self.hmonnx_audio_tower_session.exec_device = torch.device(execution_device)
            self.hmonnx_audio_tower_session.to(torch.device(device))

        qwen_meta_rel = export_meta.get("qwen3_meta")
        qwen_meta_path = work_dir / qwen_meta_rel if qwen_meta_rel else work_dir / "qwen3" / "meta.json"
        if qwen_meta_path.exists():
            qwen_meta = json.loads(qwen_meta_path.read_text(encoding="utf-8"))
            prefill_rel = qwen_meta.get("prefill_onnx")
            if prefill_rel:
                self.hmonnx_language_prefill_session = HMONNXInference(str(qwen_meta_path.parent / prefill_rel))
                self.hmonnx_language_prefill_session.exec_device = torch.device(execution_device)
                self.hmonnx_language_prefill_session.to(torch.device(device))
            decode_rel = qwen_meta.get("decode_onnx")
            if decode_rel:
                self.hmonnx_language_decode_session = HMONNXInference(str(qwen_meta_path.parent / decode_rel))
                self.hmonnx_language_decode_session.exec_device = torch.device(execution_device)
                self.hmonnx_language_decode_session.to(torch.device(device))

        if export_meta.get("audio_decoder_prefill_hmonnx") and export_meta.get("audio_decoder_decode_hmonnx"):
            self.hmonnx_decoder_prefill_session = HMONNXInference(
                str(work_dir / export_meta["audio_decoder_prefill_hmonnx"])
            )
            self.hmonnx_decoder_prefill_session.exec_device = torch.device(execution_device)
            self.hmonnx_decoder_prefill_session.to(torch.device(device))

            self.hmonnx_decoder_decode_session = HMONNXInference(
                str(work_dir / export_meta["audio_decoder_decode_hmonnx"])
            )
            self.hmonnx_decoder_decode_session.exec_device = torch.device(execution_device)
            self.hmonnx_decoder_decode_session.to(torch.device(device))

        self.hmonnx_input_sequence_length = int(input_sequence_length)
        self.hmonnx_decoder_total_length = int(input_sequence_length) * int(self.audio_invert_tower.group_size)
        self.use_hmonnx_encoder = self.hmonnx_audio_encoder_session is not None
        self.use_hmonnx_audio_tower = self.hmonnx_audio_tower_session is not None
        self.use_hmonnx_decoder = (
            self.hmonnx_decoder_prefill_session is not None and self.hmonnx_decoder_decode_session is not None
        )
        self.use_hmonnx_language = False
        return self

    def enable_hmonnx_encoder(self, enabled: bool = True):
        self.use_hmonnx_encoder = bool(enabled) and self.hmonnx_audio_encoder_session is not None
        return self

    def enable_hmonnx_decoder(self, enabled: bool = True):
        self.use_hmonnx_decoder = (
            bool(enabled)
            and self.hmonnx_decoder_prefill_session is not None
            and self.hmonnx_decoder_decode_session is not None
        )
        return self

    def enable_hmonnx_language(self, enabled: bool = True):
        self.use_hmonnx_language = (
            bool(enabled)
            and self.hmonnx_language_prefill_session is not None
            and self.hmonnx_language_decode_session is not None
        )
        return self

    @staticmethod
    def _move_to_device(x: Any, device: torch.device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, list):
            return [FunAudioChatHMONNXForConditionalGeneration._move_to_device(v, device) for v in x]
        if isinstance(x, tuple):
            return tuple(FunAudioChatHMONNXForConditionalGeneration._move_to_device(v, device) for v in x)
        if hasattr(x, "to"):
            return x.to(device)
        return x

    @staticmethod
    def _select_next_tokens(logits: torch.Tensor, do_sample: bool) -> torch.Tensor:
        next_token_logits = logits[:, -1, :]
        if do_sample:
            return torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
        return torch.argmax(next_token_logits, dim=-1)

    @staticmethod
    def _get_export_helper_module():
        module_name = "funaudiochat_xh2a_export_hmonnx_runtime"
        if module_name in sys.modules:
            return sys.modules[module_name]

        helper_path = Path(__file__).resolve().parents[4] / "examples" / "audio" / "fun_audio_chat" / "funaudiochat_xh2a_export_hmonnx.py"
        spec = importlib.util.spec_from_file_location(module_name, helper_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to load export helper module from {helper_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _run_audio_encoder_warp_pt(self, split_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        from ._static_export import FunAudioChatStaticEncoderWarp

        encoder_wrapper = FunAudioChatStaticEncoderWarp(
            self,
            feature_lens=split_inputs["feature_lens"],
            aftercnn_lens=split_inputs["aftercnn_lens"],
            speech_maxlen=int(split_inputs["speech_maxlen"]),
            feature_exist_mask=split_inputs["feature_exist_mask"],
            audio_token_positions=split_inputs["audio_token_positions"],
            chunk_lengths=split_inputs["chunk_lengths"],
            pooled_lengths=split_inputs["pooled_lengths"],
        ).eval()
        return encoder_wrapper(
            split_inputs["speech_ids"],
            split_inputs["audio_inputs_embeds"],
            split_inputs["padded_input_features"],
            split_inputs["chunk_padded_mask"],
            split_inputs["aftercnn_valid_mask"],
            split_inputs["audio_attention_mask"],
            split_inputs["continuous_audio_valid_mask"],
        )

    def _run_audio_encoder_hmonnx(self, split_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.hmonnx_audio_encoder_session is not None
        runtime_device = (
            self.hmonnx_audio_encoder_session._device
            if hasattr(self.hmonnx_audio_encoder_session, "_device")
            else torch.device("cuda:0")
        )
        input_args = (
            split_inputs["speech_ids"],
            split_inputs["audio_inputs_embeds"],
            split_inputs["padded_input_features"],
            split_inputs["chunk_padded_mask"],
            split_inputs["aftercnn_valid_mask"],
            split_inputs["audio_attention_mask"],
            split_inputs["continuous_audio_valid_mask"],
        )
        outputs = self.hmonnx_audio_encoder_session(*self._move_to_device(input_args, runtime_device))
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs.to(split_inputs["inputs_embeds"].device)

    def _run_audio_encoder(self, split_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_hmonnx_encoder and self.hmonnx_audio_encoder_session is not None:
            return self._run_audio_encoder_hmonnx(split_inputs)
        return self._run_audio_encoder_warp_pt(split_inputs)

    def _audio_invert_tower_generate_forward(
        self,
        decoder: FunAudioChatDecoder,
        *args,
        **kwargs,
    ):
        if self.use_hmonnx_decoder and self.hmonnx_decoder_prefill_session is not None and self.hmonnx_decoder_decode_session is not None:
            return self._audio_invert_tower_hmonnx_forward(decoder, *args, **kwargs)
        return self._audio_invert_tower_warp_forward(decoder, *args, **kwargs)

    @classmethod
    def from_float_model(cls, model: FunAudioChatForConditionalGeneration):
        if isinstance(model, cls):
            hmonnx_model = model
        else:
            model.__class__ = cls
            hmonnx_model = model

        hmonnx_model.warp_context_length = getattr(hmonnx_model, "warp_context_length", 256)
        hmonnx_model.warp_input_sequence_length = getattr(hmonnx_model, "warp_input_sequence_length", 256)
        hmonnx_model.language_warp_past_key_caches = None
        hmonnx_model.language_warp_past_value_caches = None
        hmonnx_model.language_warp_past_seq_length = 0
        hmonnx_model.hmonnx_work_dir = None
        hmonnx_model.hmonnx_export_meta = None
        hmonnx_model.hmonnx_audio_encoder_session = None
        hmonnx_model.hmonnx_audio_tower_session = None
        hmonnx_model.hmonnx_language_prefill_session = None
        hmonnx_model.hmonnx_language_decode_session = None
        hmonnx_model.hmonnx_decoder_prefill_session = None
        hmonnx_model.hmonnx_decoder_decode_session = None
        hmonnx_model.use_hmonnx_encoder = False
        hmonnx_model.use_hmonnx_audio_tower = False
        hmonnx_model.use_hmonnx_decoder = False
        hmonnx_model.use_hmonnx_language = False

        hmonnx_model.prepare_inputs_for_generation = types.MethodType(_patched_prepare_inputs_for_generation, hmonnx_model)

        hmonnx_model.eval()
        return hmonnx_model

    @staticmethod
    def _build_language_attention_mask(
        valid_key_length: int,
        total_query_length: int,
        total_key_length: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        mask_value = torch.finfo(dtype).min
        attention_mask = torch.zeros((total_query_length, total_key_length), dtype=dtype, device=device)

        past_key_length = total_key_length - total_query_length
        query_positions = past_key_length + torch.arange(total_query_length, device=device)[:, None]
        key_positions = torch.arange(total_key_length, device=device)[None, :]
        attention_mask = attention_mask.masked_fill(key_positions > query_positions, mask_value)

        if valid_key_length < total_key_length:
            attention_mask[:, valid_key_length:total_key_length] = mask_value
            attention_mask[query_positions.squeeze(-1) >= valid_key_length, :] = mask_value
        return attention_mask

    @staticmethod
    def _get_language_last_hidden_state(outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs[1]

    def _run_language_prefill(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_length: int,
        past_key_caches: list[torch.Tensor],
        past_value_caches: list[torch.Tensor],
    ) -> torch.Tensor:
        total_length = int(inputs_embeds.shape[1])
        valid_text_length = int(attention_mask.sum().item())

        if total_length <= chunk_length:
            padded_inputs_embeds = inputs_embeds
            if total_length < chunk_length:
                pad_shape = list(inputs_embeds.shape)
                pad_shape[1] = chunk_length - total_length
                padded_inputs_embeds = torch.cat(
                    [inputs_embeds, torch.zeros(pad_shape, dtype=inputs_embeds.dtype, device=inputs_embeds.device)],
                    dim=1,
                )
            past_seq_length = torch.zeros((inputs_embeds.shape[0],), dtype=torch.int32, device=inputs_embeds.device)
            current_input_length = torch.full(
                (inputs_embeds.shape[0],),
                valid_text_length,
                dtype=torch.int32,
                device=inputs_embeds.device,
            )
            language_attention_mask = self._build_language_attention_mask(
                valid_text_length,
                chunk_length,
                chunk_length,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            outputs = self.language_model(
                padded_inputs_embeds,
                language_attention_mask,
                past_seq_length,
                current_input_length,
                past_key_caches,
                past_value_caches,
            )
            return self._get_language_last_hidden_state(outputs)[:, :valid_text_length, :]

        hidden_states = []
        processed_length = 0

        for start in range(0, total_length, chunk_length):
            end = min(start + chunk_length, total_length)
            current_length = end - start
            current_embeds = inputs_embeds[:, start:end, :]
            chunk_valid_length = int(attention_mask[:, start:end].sum().item())
            padded_embeds = current_embeds
            if current_length < chunk_length:
                pad_shape = list(current_embeds.shape)
                pad_shape[1] = chunk_length - current_length
                padded_embeds = torch.cat(
                    [current_embeds, torch.zeros(pad_shape, dtype=current_embeds.dtype, device=current_embeds.device)],
                    dim=1,
                )

            chunk_attention_mask = self._build_language_attention_mask(
                processed_length + chunk_valid_length,
                chunk_length,
                processed_length + chunk_length,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            past_seq_length = torch.full(
                (inputs_embeds.shape[0],),
                processed_length,
                dtype=torch.int32,
                device=inputs_embeds.device,
            )
            current_input_length = torch.full(
                (inputs_embeds.shape[0],),
                chunk_valid_length,
                dtype=torch.int32,
                device=inputs_embeds.device,
            )

            outputs = self.language_model(
                padded_embeds,
                chunk_attention_mask,
                past_seq_length,
                current_input_length,
                past_key_caches,
                past_value_caches,
            )
            hidden_states.append(self._get_language_last_hidden_state(outputs)[:, :chunk_valid_length, :])
            processed_length += chunk_valid_length

        return torch.cat(hidden_states, dim=1)[:, :valid_text_length, :]

    def _run_language_model_warp_prefill(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
    ):
        helper = self._get_export_helper_module()

        if not getattr(self, "_language_model_warp_wrapped", False):
            register_wrap_modules(self)
            self.language_model = wrap_llm_model(
                self.language_model,
                Config(
                    dict(
                        max_sequence_length=self.warp_context_length,
                        input_sequence_length=self.warp_input_sequence_length,
                        use_cache=True,
                        num_logits_to_keep=0,
                        kv_cache=dict(
                            cache_axis=2,
                        ),
                    )
                ),
            )
            self._language_model_warp_wrapped = True

        language_past_key_caches, language_past_value_caches = helper.build_kv_caches(
            self.language_model,
            self.warp_context_length,
        )
        self.language_warp_past_key_caches = language_past_key_caches
        self.language_warp_past_value_caches = language_past_value_caches
        last_hidden_state = self._run_language_prefill(
            inputs_embeds,
            attention_mask.to(torch.int32),
            self.warp_input_sequence_length,
            language_past_key_caches,
            language_past_value_caches,
        )
        self.language_warp_past_seq_length = int(attention_mask.sum().item())
        lm_head = self.language_model.get_output_embeddings()
        logits = lm_head(last_hidden_state)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(last_hidden_state,) if output_hidden_states else None,
            past_key_values=(True,),
            attentions=None,
            loss=None,
            aux_loss=None,
        )

    def _run_language_model_hmonnx_prefill(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
    ):
        assert self.hmonnx_language_prefill_session is not None

        # pt_outputs = self._run_language_model_warp_prefill(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     output_hidden_states=output_hidden_states,
        # )

        helper = self._get_export_helper_module()
        runtime_device = (
            self.hmonnx_language_prefill_session._device
            if hasattr(self.hmonnx_language_prefill_session, "_device")
            else torch.device("cuda:0")
        )
        total_length = int(self.hmonnx_input_sequence_length)
        valid_text_length = int(attention_mask.sum().item())
        padded_inputs_embeds = helper.pad_seq_dim(inputs_embeds, total_length, 0.0)
        padded_attention_mask = helper.build_decoder_attention_mask(
            valid_text_length,
            total_length,
            total_length,
            dtype=inputs_embeds.dtype,
        ).to(inputs_embeds.device)
        self.language_warp_past_key_caches, self.language_warp_past_value_caches = helper.build_kv_caches(
            self.language_model,
            self.warp_context_length,
        )
        hmonnx_input_args = (
            padded_inputs_embeds,
            padded_attention_mask.unsqueeze(0).unsqueeze(0),
            torch.zeros((inputs_embeds.shape[0],), dtype=torch.int32, device=inputs_embeds.device),
            torch.full(
                (inputs_embeds.shape[0],),
                valid_text_length,
                dtype=torch.int32,
                device=inputs_embeds.device,
            ),
            *self.language_warp_past_key_caches,
            *self.language_warp_past_value_caches,
        )
        self.language_warp_past_seq_length = int(attention_mask.sum().item())
        hmonnx_outputs = self.hmonnx_language_prefill_session(*self._move_to_device(hmonnx_input_args, runtime_device))
        if isinstance(hmonnx_outputs, tuple):
            hmonnx_logits = hmonnx_outputs[0]
            hmonnx_last_hidden_state = hmonnx_outputs[1] # if len(hmonnx_outputs) > 1 else pt_outputs.hidden_states[-1]
        else:
            hmonnx_logits = hmonnx_outputs
            hmonnx_last_hidden_state = None # pt_outputs.hidden_states[-1]
        hmonnx_logits = hmonnx_logits[:, :valid_text_length, :]
        hmonnx_last_hidden_state = hmonnx_last_hidden_state[:, :valid_text_length, :]
        hmonnx_logits = hmonnx_logits.to(inputs_embeds.device)
        hmonnx_last_hidden_state = hmonnx_last_hidden_state.to(inputs_embeds.device)

        return SimpleNamespace(
            logits=hmonnx_logits,
            hidden_states=(hmonnx_last_hidden_state,) if output_hidden_states else None,
            # past_key_values=pt_outputs.past_key_values,
            attentions=None,
            loss=None,
            aux_loss=None,
        )

    def _run_language_model_hmonnx_decode(
        self,
        inputs_embeds: torch.Tensor,
        output_hidden_states: bool = True,
    ):
        assert self.hmonnx_language_decode_session is not None

        if self.language_warp_past_key_caches is None or self.language_warp_past_value_caches is None:
            dummy_attention_mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                dtype=torch.int32,
                device=inputs_embeds.device,
            )
            return self._run_language_model_hmonnx_prefill(
                inputs_embeds=inputs_embeds,
                attention_mask=dummy_attention_mask,
                output_hidden_states=output_hidden_states,
            )

        helper = self._get_export_helper_module()
        runtime_device = (
            self.hmonnx_language_decode_session._device
            if hasattr(self.hmonnx_language_decode_session, "_device")
            else torch.device("cuda:0")
        )

        current_input_length = torch.full(
            (inputs_embeds.shape[0],),
            inputs_embeds.shape[1],
            dtype=torch.int32,
            device=inputs_embeds.device,
        )
        past_seq_length = torch.full(
            (inputs_embeds.shape[0],),
            int(self.language_warp_past_seq_length),
            dtype=torch.int32,
            device=inputs_embeds.device,
        )
        decode_attention_mask = helper.build_decoder_attention_mask(
            self.language_warp_past_seq_length + inputs_embeds.shape[1],
            inputs_embeds.shape[1],
            int(self.hmonnx_input_sequence_length),
            dtype=inputs_embeds.dtype,
        ).to(inputs_embeds.device)
        hmonnx_input_args = (
            inputs_embeds,
            decode_attention_mask.unsqueeze(0).unsqueeze(0),
            past_seq_length,
            current_input_length,
            *self.language_warp_past_key_caches,
            *self.language_warp_past_value_caches,
        )
        hmonnx_outputs = self.hmonnx_language_decode_session(*self._move_to_device(hmonnx_input_args, runtime_device))
        if isinstance(hmonnx_outputs, tuple):
            hmonnx_logits = hmonnx_outputs[0]
            hmonnx_last_hidden_state = hmonnx_outputs[1]
        else:
            hmonnx_logits = hmonnx_outputs
            hmonnx_last_hidden_state = inputs_embeds

        hmonnx_logits = hmonnx_logits.to(inputs_embeds.device)
        hmonnx_last_hidden_state = hmonnx_last_hidden_state.to(inputs_embeds.device)
        self.language_warp_past_seq_length += int(inputs_embeds.shape[1])
        return SimpleNamespace(
            logits=hmonnx_logits,
            hidden_states=(hmonnx_last_hidden_state,) if output_hidden_states else None,
            past_key_values=(True,),
            attentions=None,
            loss=None,
            aux_loss=None,
        )

    def _run_language_model_warp_decode(
        self,
        inputs_embeds: torch.Tensor,
        output_hidden_states: bool = True,
    ):
        if self.language_warp_past_key_caches is None or self.language_warp_past_value_caches is None:
            dummy_attention_mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                dtype=torch.int32,
                device=inputs_embeds.device,
            )
            return self._run_language_model_warp_prefill(
                inputs_embeds=inputs_embeds,
                attention_mask=dummy_attention_mask,
                output_hidden_states=output_hidden_states,
            )

        helper = self._get_export_helper_module()

        current_input_length = torch.full(
            (inputs_embeds.shape[0],),
            inputs_embeds.shape[1],
            dtype=torch.int32,
            device=inputs_embeds.device,
        )
        past_seq_length = torch.full(
            (inputs_embeds.shape[0],),
            int(self.language_warp_past_seq_length),
            dtype=torch.int32,
            device=inputs_embeds.device,
        )
        decode_attention_mask = helper.build_decoder_attention_mask(
            self.language_warp_past_seq_length + inputs_embeds.shape[1],
            inputs_embeds.shape[1],
            self.language_warp_past_seq_length + inputs_embeds.shape[1],
            dtype=inputs_embeds.dtype,
        ).to(inputs_embeds.device)
        outputs = self.language_model(
            inputs_embeds,
            decode_attention_mask,
            past_seq_length,
            current_input_length,
            self.language_warp_past_key_caches,
            self.language_warp_past_value_caches,
        )
        last_hidden_state = self._get_language_last_hidden_state(outputs)
        self.language_warp_past_seq_length += int(inputs_embeds.shape[1])
        logits = self.language_model.get_output_embeddings()(last_hidden_state)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(last_hidden_state,) if output_hidden_states else None,
            past_key_values=(True,),
            attentions=None,
            loss=None,
            aux_loss=None,
        )

    def _prepare_warp_embeddings_from_forward_inputs(
        self,
        input_ids: torch.LongTensor,
        input_features: torch.FloatTensor,
        speech_ids: torch.LongTensor,
        text_ids: Optional[torch.LongTensor],
        attention_mask: torch.Tensor,
        speech_attention_mask: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        feature_exist_mask: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
    ):
        helper = self._get_export_helper_module()

        raw_inputs = {
            "input_ids": input_ids,
            "input_features": input_features,
            "speech_ids": speech_ids,
            "attention_mask": attention_mask,
            "speech_attention_mask": speech_attention_mask,
            "feature_attention_mask": feature_attention_mask,
            "feature_exist_mask": feature_exist_mask,
            "disable_text_branch": True,
        }

        split_inputs = helper.preprocess_static_inputs(self, raw_inputs)
        audio_features = self._run_audio_encoder(split_inputs)

        audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(speech_attention_mask.sum(-1))[1]

        num_audios, max_audio_tokens, _ = audio_features.shape
        audio_features_mask = torch.arange(max_audio_tokens, device=audio_output_lengths.device)[None, :]
        audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
        flat_audio_features = audio_features[audio_features_mask]

        merged_inputs_embeds = split_inputs["inputs_embeds"].clone()
        special_audio_mask = (input_ids == self.config.audio_token_index).to(merged_inputs_embeds.device).unsqueeze(-1)
        merged_inputs_embeds = merged_inputs_embeds.masked_scatter(
            special_audio_mask.expand_as(merged_inputs_embeds),
            flat_audio_features.to(merged_inputs_embeds.device, merged_inputs_embeds.dtype),
        )

        # decoder_text_embeds = self.get_input_embeddings()(input_ids)
        return merged_inputs_embeds #, decoder_text_embeds

    def _audio_invert_tower_warp_forward(
        self,
        decoder: FunAudioChatDecoder,
        inputs_embeds=None,
        audio_embeds=None,
        labels=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        from ._static_export import (
            FunAudioChatStaticDecoderDecodeWarp,
            FunAudioChatStaticDecoderPrefillWarp,
        )
        helper = self._get_export_helper_module()

        decoder_prefill_wrapper = FunAudioChatStaticDecoderPrefillWarp(decoder).eval()
        decoder_decode_wrapper = FunAudioChatStaticDecoderDecodeWarp(decoder).eval()

        inputs_embeds = decoder.pre_matching(inputs_embeds)
        bs, slen, _ = inputs_embeds.shape
        hidden_states = inputs_embeds.reshape(bs, slen * decoder.group_size, -1)
        decoder.crq_audio_embeds = (
            decoder.get_embeddings(decoder.config.bos_token_id)[None, None, :]
            .repeat(bs, 1, 1)
            .to(dtype=hidden_states.dtype, device=hidden_states.device)
            if decoder.crq_audio_embeds is None
            else decoder.crq_audio_embeds.unsqueeze(1)
        )

        decoder_prefill_valid_length = hidden_states.shape[1] - (decoder.group_size - 1)
        padded_attention_mask = helper.build_decoder_attention_mask(
            decoder_prefill_valid_length,
            hidden_states.shape[1],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
        )
        decoder.crq_generate_tokens = []
        all_logits = []

        prefill_logits = decoder_prefill_wrapper(
            hidden_states,
            torch.zeros((bs,), dtype=torch.int32, device=hidden_states.device),
            torch.full((bs,), decoder_prefill_valid_length, dtype=torch.int32, device=hidden_states.device),
            padded_attention_mask,
        )
        prefill_logits = prefill_logits[:, :decoder_prefill_valid_length, :]
        crq_audio_tokens, prefill_logits = decoder.sampling_step(prefill_logits)
        decoder.crq_generate_tokens.append(crq_audio_tokens.unsqueeze(1))
        all_logits.append(prefill_logits)
        decoder.crq_audio_embeds = decoder.get_embeddings(crq_audio_tokens)

        for step_idx in range(decoder.group_size - 1):
            current_hidden = hidden_states[
                :,
                decoder_prefill_valid_length + step_idx : decoder_prefill_valid_length + step_idx + 1,
                :,
            ]
            step_attention_mask = helper.build_decoder_attention_mask(
                decoder_prefill_valid_length + step_idx + 1,
                1,
                hidden_states.shape[1],
                dtype=hidden_states.dtype,
            )
            step_logits = decoder_decode_wrapper(
                current_hidden,
                torch.full((bs,), decoder_prefill_valid_length + step_idx, dtype=torch.int32, device=hidden_states.device),
                torch.ones((bs,), dtype=torch.int32, device=hidden_states.device),
                step_attention_mask,
            )
            crq_audio_tokens, step_logits = decoder.sampling_step(step_logits)
            decoder.crq_generate_tokens.append(crq_audio_tokens.unsqueeze(1))
            all_logits.append(step_logits[:, -1:, :])
            decoder.crq_audio_embeds = decoder.get_embeddings(crq_audio_tokens)

        decoder.crq_generate_tokens = torch.cat(decoder.crq_generate_tokens, dim=1)
        logits = torch.cat(all_logits, dim=1)
        encoder_hidden_states = (hidden_states,) if output_hidden_states else None
        if not return_dict:
            return tuple(v for v in [None, logits, encoder_hidden_states, None] if v is not None)
        return CausalLMOutput(loss=None, logits=logits, hidden_states=encoder_hidden_states, attentions=None)

    def _audio_invert_tower_hmonnx_forward(
        self,
        decoder: FunAudioChatDecoder,
        inputs_embeds=None,
        audio_embeds=None,
        labels=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if not hasattr(decoder, "crq_do_sample"):
            return FunAudioChatDecoder.forward(
                decoder,
                inputs_embeds=inputs_embeds,
                audio_embeds=audio_embeds,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        assert self.hmonnx_decoder_prefill_session is not None
        assert self.hmonnx_decoder_decode_session is not None

        helper = self._get_export_helper_module()

        runtime_device = self.hmonnx_decoder_prefill_session._device if hasattr(self.hmonnx_decoder_prefill_session, "_device") else torch.device("cuda:0")

        if decoder.crq_past_key_values is None or not hasattr(decoder, "hmonnx_past_key_caches"):
            prefill_past_key_caches, prefill_past_value_caches = helper.build_kv_caches(
                decoder.crq_transformer,
                self.hmonnx_decoder_total_length,
            )
            decoder.hmonnx_past_key_caches = self._move_to_device(prefill_past_key_caches, runtime_device)
            decoder.hmonnx_past_value_caches = self._move_to_device(prefill_past_value_caches, runtime_device)
            decoder.hmonnx_past_seq_length = 0

        inputs_embeds = decoder.pre_matching(inputs_embeds)
        bs, slen, _ = inputs_embeds.shape
        hidden_states = inputs_embeds.reshape(bs, slen * decoder.group_size, -1)

        decoder.crq_audio_embeds = (
            decoder.get_embeddings(decoder.config.bos_token_id)[None, None, :]
            .repeat(bs, 1, 1)
            .to(dtype=hidden_states.dtype, device=hidden_states.device)
            if decoder.crq_audio_embeds is None
            else decoder.crq_audio_embeds.unsqueeze(1)
        )

        decoder.crq_generate_tokens = []
        all_logits = []
        use_prefill = slen > 1
        decode_start_idx = 0

        if use_prefill:
            decoder_prefill_valid_length = hidden_states.shape[1] - (decoder.group_size - 1)
            prefill_step_hidden_states = hidden_states + decoder.crq_audio_embeds
            padded_decoder_prefill_hidden_states = helper.pad_seq_dim(
                prefill_step_hidden_states,
                self.hmonnx_decoder_total_length,
                0.0,
            )

            decoder_prefill_past_seq_length = torch.full(
                (bs,),
                int(decoder.hmonnx_past_seq_length),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            decoder_prefill_current_input_length = torch.full(
                (bs,),
                decoder_prefill_valid_length,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            padded_attention_mask = helper.build_decoder_attention_mask(
                int(decoder.hmonnx_past_seq_length) + decoder_prefill_valid_length,
                self.hmonnx_decoder_total_length,
                int(decoder.hmonnx_past_seq_length) + self.hmonnx_decoder_total_length,
                dtype=hidden_states.dtype,
            )
            prefill_input_args = (
                padded_decoder_prefill_hidden_states,
                decoder_prefill_past_seq_length,
                decoder_prefill_current_input_length,
                padded_attention_mask,
                *decoder.hmonnx_past_key_caches,
                *decoder.hmonnx_past_value_caches,
            )

            prefill_logits = self.hmonnx_decoder_prefill_session(*self._move_to_device(prefill_input_args, runtime_device))
            prefill_logits = prefill_logits[:, :decoder_prefill_valid_length, :]
            crq_audio_tokens, prefill_logits = decoder.sampling_step(prefill_logits)
            decoder.crq_generate_tokens.append(crq_audio_tokens.unsqueeze(1))
            all_logits.append(prefill_logits)
            decoder.crq_audio_embeds = decoder.get_embeddings(crq_audio_tokens)
            decoder.hmonnx_past_seq_length += decoder_prefill_valid_length
            decoder.crq_past_key_values = True
            decode_start_idx = decoder_prefill_valid_length

        decode_step_start = 1 if use_prefill else 0
        decode_step_end = decoder.group_size

        for decode_step_idx in range(decode_step_start, decode_step_end):
            if decode_step_idx == 0:
                step_inputs = hidden_states[:, :1, :] + decoder.crq_audio_embeds
            else:
                step_hidden_index = slen * decoder.group_size - (decoder.group_size - decode_step_idx)
                step_inputs = hidden_states[:, step_hidden_index : step_hidden_index + 1, :] + decoder.crq_audio_embeds
            step_past_seq_length = torch.full(
                (bs,),
                int(decoder.hmonnx_past_seq_length),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            step_current_input_length = torch.full(
                (bs,),
                1,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            step_attention_mask = helper.build_decoder_attention_mask(
                int(decoder.hmonnx_past_seq_length) + 1,
                1,
                self.hmonnx_decoder_total_length,
                dtype=hidden_states.dtype,
            )
            step_logits = self.hmonnx_decoder_decode_session(
                step_inputs.to(runtime_device),
                step_past_seq_length.to(runtime_device),
                step_current_input_length.to(runtime_device),
                step_attention_mask.to(runtime_device),
                *decoder.hmonnx_past_key_caches,
                *decoder.hmonnx_past_value_caches,
            )
            crq_audio_tokens, step_logits = decoder.sampling_step(step_logits)
            decoder.crq_generate_tokens.append(crq_audio_tokens.unsqueeze(1))
            all_logits.append(step_logits[:, -1:, :])
            decoder.crq_audio_embeds = decoder.get_embeddings(crq_audio_tokens)
            decoder.hmonnx_past_seq_length += 1

        decoder.crq_generate_tokens = torch.cat(decoder.crq_generate_tokens, dim=1)
        logits = torch.cat(all_logits, dim=1)
        encoder_hidden_states = (hidden_states,) if output_hidden_states else None
        if not return_dict:
            return tuple(v for v in [None, logits, encoder_hidden_states, None] if v is not None)
        return CausalLMOutput(loss=None, logits=logits, hidden_states=encoder_hidden_states, attentions=None)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        speech_ids: Optional[torch.LongTensor] = None,
        text_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speech_attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        feature_exist_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        text_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_warp_encoder = (
            inputs_embeds is None
            and input_ids is not None
            and input_features is not None
            and speech_ids is not None
            and attention_mask is not None
            and speech_attention_mask is not None
            and feature_attention_mask is not None
            and feature_exist_mask is not None
            and input_ids.shape[1] != 1
        )
        if use_warp_encoder:
            merged_inputs_embeds = self._prepare_warp_embeddings_from_forward_inputs(
                input_ids=input_ids,
                input_features=input_features,
                speech_ids=speech_ids,
                text_ids=text_ids,
                attention_mask=attention_mask,
                speech_attention_mask=speech_attention_mask,
                feature_attention_mask=feature_attention_mask,
                feature_exist_mask=feature_exist_mask,
                text_attention_mask=text_attention_mask,
            )
            inputs_embeds = merged_inputs_embeds
        else:
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

                # assert speech_ids is None, "speech_ids must be None"
            

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_prefill_warp = getattr(self, "is_prefill", True) or past_key_values is None or inputs_embeds.shape[1] > 1
        if self.use_hmonnx_language:
            if use_prefill_warp:
                self.language_warp_past_key_caches = None
                self.language_warp_past_value_caches = None
                self.language_warp_past_seq_length = 0
                outputs = self._run_language_model_hmonnx_prefill(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            else:
                outputs = self._run_language_model_hmonnx_decode(
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True,
                )
        else:
            outputs = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )
        aux_loss = outputs.aux_loss if hasattr(outputs, "aux_loss") else None
        speech_loss = None
        speech_logits = None
        if self.audio_invert_tower is not None:
            last_hidden_state = outputs.hidden_states[-1]

            audio_embeds = None
            if not self.sp_gen_kwargs['disable_speech']:
                speech_inputs_embeds = last_hidden_state
                if text_embeds is None:
                    text_embeds = self.get_input_embeddings()(input_ids)
                speech_inputs_embeds = speech_inputs_embeds + text_embeds.detach()
                if self.use_hmonnx_decoder:
                    speech_output = self._audio_invert_tower_hmonnx_forward(
                        self.audio_invert_tower,
                        audio_embeds=audio_embeds,
                        inputs_embeds=speech_inputs_embeds,
                        labels=None,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        return_dict=True,
                    )
                else:
                    speech_output = self.audio_invert_tower(
                        audio_embeds=audio_embeds,
                        inputs_embeds=speech_inputs_embeds,
                        labels=None,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        return_dict=True,
                    )
                speech_loss = speech_output.loss if hasattr(speech_output, "loss") else None
                speech_logits = speech_output.logits

        if not return_dict:
            output = (outputs.logits, speech_logits, outputs.hidden_states, outputs.attentions, outputs.past_key_values)
            return output

        from .modeling_funaudiochat import FunAudioChatCausalLMOutputWithPast
        return FunAudioChatCausalLMOutputWithPast(
            loss=None,
            aux_loss=aux_loss,
            text_loss=None,
            speech_loss=speech_loss,
            text_logits=outputs.logits,
            speech_logits=speech_logits,
            # past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
        )


def register_wrap_modules(hf_model=None):
    """Register xhquant traceable modules for the inner Qwen3 language model."""
    from ._model_qwen3 import (
        _Qwen3Attention,
        _Qwen3DecoderLayer,
        # _Qwen3ForCausalLM,
        _Qwen3Model,
        _Qwen3RMSNorm,
        _Qwen3RotaryEmbedding,
        register_wrap_modules as qwen3_register_wrap_modules,
    )

    if hf_model is not None and hasattr(hf_model, "language_model"):
        qwen3_register_wrap_modules(hf_model.language_model)
        if getattr(hf_model, "audio_invert_tower", None) is not None and hasattr(hf_model.audio_invert_tower, "crq_transformer"):
            qwen3_register_wrap_modules(hf_model.audio_invert_tower.crq_transformer)
    else:
        qwen3_register_wrap_modules(hf_model)

    # touch qwen3 legacy trace modules so inner qwen3 blocks are also registered
    _ = _Qwen3RotaryEmbedding
    _ = _Qwen3Attention
    _ = _Qwen3DecoderLayer
    _ = _Qwen3Model
    _ = _Qwen3RMSNorm
    # _ = _Qwen3ForCausalLM

    # touch local audio trace modules so registry is populated for export
    _ = _FunAudioChatAudioAttention
    _ = _FunAudioChatAudioEncoderLayer
    _ = _FunAudioChatDiscreteEncoder
    _ = _FunAudioChatQwen3ForCausalLM
