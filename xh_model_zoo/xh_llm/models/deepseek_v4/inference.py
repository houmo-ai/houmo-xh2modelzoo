# ================================================================== #
#  File: inference.py                                                 #
#  Description:                                                       #
#    DeepSeek-V4 ONNX inference engine with MLA KV-cache support.    #
#                                                                     #
#    MLA attention uses shared K==V projection, so each layer has    #
#    only one KV cache (not separate K/V). Prefill and decode run   #
#    on separate HMONNX sessions.                                    #
# ================================================================== #

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer

from xhquant.api import CacheTensor, GoldenMixin, HMONNXInference

from ....utils import DeviceDtypeMixin


class DeepseekV4Inference(DeviceDtypeMixin):
    """DeepSeek-V4 HMONNX inference: prefill + decode with shared KV cache."""

    def __init__(
        self,
        model_config_file: str,
        fast_mode: bool = True,
        device: str = "cuda",
        execution_device: str = "cuda",
        pipeline_devices=None,
    ):
        super().__init__()

        self.fast_mode = fast_mode
        self._device = torch.device(device)
        self._set_exec_device(torch.device(execution_device))

        # Pipeline-parallel (layer-split model parallel). When set, decoder
        # layers are binned across these GPUs (weights resident, hidden
        # states shipped at stage boundaries). Either pass explicitly or
        # fall back to HMONNX_PIPELINE_DEVICES env var (handled inside
        # HMONNXInference at parse time).
        import os

        if pipeline_devices is None:
            pipeline_devices = os.getenv("HMONNX_PIPELINE_DEVICES")
        self._pipeline_devices = pipeline_devices
        self._pp_enabled = bool(pipeline_devices)

        model_dir = Path(model_config_file).parent
        meta_info = json.load(open(model_config_file, "r"))
        self.meta_info = meta_info

        # -- HMONNX file paths --
        self.prefill_onnx_file = model_dir / meta_info["prefill_onnx"]
        self.decode_onnx_file = model_dir / meta_info["decode_onnx"]

        # -- Shared KV cache (MLA: K==V, one per layer) --
        kv_cache_shape = meta_info["kv_cache"]["shape"]
        num_decoder_layers = meta_info["kv_cache"]["num_decoder_layers"]

        self.past_kv_caches: List[CacheTensor] = []
        for _ in range(num_decoder_layers):
            self.past_kv_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))

        # -- Tokenizer --
        hf_config_dir = str(model_dir / meta_info["hf_config"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_config_dir,
            trust_remote_code=True,
        )

        # -- Token embedding --
        tok_state = torch.load(
            model_dir / meta_info["token_embedding_file"],
            map_location="cpu",
            weights_only=True,
        )
        self.token_embedding = nn.Embedding(
            tok_state["weight"].shape[0],
            tok_state["weight"].shape[1],
        ).to(torch.float16)
        self.token_embedding.load_state_dict(tok_state)

        # -- Runtime config --
        self.batch_size = 1
        self.prefill_input_sequence_length = meta_info["wrap_cfg"]["input_sequence_length"]
        self.input_sequence_length = self.prefill_input_sequence_length
        self.pad_token_id = self.tokenizer.eos_token_id
        self._phase_prefill = True

        self.prefill_session: Optional[HMONNXInference] = None
        self.decode_session: Optional[HMONNXInference] = None

        # -- Compressed KV cache（全量列表，与 decode 输入对齐） --
        # 非 compressor 层用零张量占位，compressor 层 prefill 后填充
        ckv_meta = meta_info.get("compressed_kv_cache", {})
        self.compressor_layer_indices = ckv_meta.get("layer_indices", [])
        ckv_shapes = ckv_meta.get("shapes", {})
        head_dim = kv_cache_shape[-1]
        self.compressed_kv_caches: List[Tensor] = []
        for i in range(num_decoder_layers):
            shape = ckv_shapes.get(str(i), [1, 1, 1, head_dim])
            self.compressed_kv_caches.append(torch.zeros(shape, dtype=torch.float16))

    # -------------------------------------------------------------- #
    #  Phase management                                                #
    # -------------------------------------------------------------- #

    def set_phase_prefill(self, prefill: bool):
        self._phase_prefill = prefill
        if prefill:
            if self.prefill_session is None:
                self.init_prefill()
            self.input_sequence_length = self.prefill_input_sequence_length
        else:
            self.prefill_session = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.decode_session is None:
                self.init_decode()
            self.input_sequence_length = 1

    def _configure_session_pipeline(self, session):
        """Apply pipeline-parallel config to a freshly built HMONNX session."""
        if not self._pp_enabled:
            return
        if not session.pipeline_enabled:
            session.configure_pipeline(self._pipeline_devices)

    def init_prefill(self):
        if self.prefill_session is not None:
            return
        from .deepseek_v4_converter import _register_std_parsers

        _register_std_parsers()
        self.prefill_session = HMONNXInference(str(self.prefill_onnx_file))
        if self.fast_mode:
            self.prefill_session.to_fast_mode()
        self.prefill_session.exec_device = self.execution_device
        self._configure_session_pipeline(self.prefill_session)
        self.prefill_session.to(self._device)
        self._ensure_aligned_mode(self.prefill_session)

    def _ensure_aligned_mode(self, session):
        """确保 session 及其子模块的 fast_mode 与 self.fast_mode 一致。

        HMONNX 模块（如 LLMCache）默认 fast_mode=True，
        即使 inference engine 设 fast_mode=False，
        模块内部仍可能在 fast mode 下运行（返回截断 tensor），
        导致与固定 shape 的 attention_mask 不匹配。
        """
        if not self.fast_mode and hasattr(session, "to_aligned_mode"):
            session.to_aligned_mode()

    def init_decode(self):
        if self.decode_session is not None:
            return
        from .deepseek_v4_converter import _register_std_parsers

        _register_std_parsers()
        self.decode_session = HMONNXInference(str(self.decode_onnx_file))
        if self.fast_mode:
            self.decode_session.to_fast_mode()
        self.decode_session.exec_device = self.execution_device
        self._configure_session_pipeline(self.decode_session)
        self.decode_session.to(self._device)
        self._ensure_aligned_mode(self.decode_session)

    def get_input_sequence_length(self):
        return self.input_sequence_length

    def set_input_sequence_length(self, length: int):
        self.input_sequence_length = length

    # -------------------------------------------------------------- #
    #  Input preparation                                               #
    # -------------------------------------------------------------- #

    def prepare_inputs(
        self,
        data,
        input_sequence_length,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List[CacheTensor]]:
        input_ids = data["input_ids"]
        seq_length = input_ids.shape[1]
        input_ids = input_ids.to(self.execution_device)

        assert seq_length <= input_sequence_length, f"Input too long: {seq_length} > {input_sequence_length}"

        # -- Pad to input_sequence_length --
        if input_sequence_length > seq_length:
            pad = torch.full(
                (1, input_sequence_length - seq_length),
                self.pad_token_id,
                dtype=torch.long,
                device=self.execution_device,
            )
            input_ids = torch.cat([input_ids, pad], dim=-1)

        # -- HMONNX requires int32 for index tensors --
        input_ids = input_ids.to(torch.int32)

        inputs_embeds = self.token_embedding.to(self.execution_device)(input_ids)
        past_seq_length = data["past_seq_length"]

        position_ids = torch.arange(
            past_seq_length,
            past_seq_length + input_sequence_length,
            dtype=torch.int32,
            device=self.execution_device,
        ).unsqueeze(0)

        return (
            inputs_embeds.to(self.execution_device),
            position_ids,
            torch.tensor([past_seq_length], dtype=torch.int32, device=self.execution_device),
            torch.tensor([seq_length], dtype=torch.int32, device=self.execution_device),
            input_ids,
            self.past_kv_caches,
        )

    # -------------------------------------------------------------- #
    #  Forward (dispatch to prefill/decode session)                    #
    # -------------------------------------------------------------- #

    def _input_device(self):
        """Device for model-level inputs (embed/lm_head live on stage 0).

        In pipeline-parallel mode that is the first pipeline device; in
        single-device mode it is the configured execution device.
        """
        if self._pp_enabled and self.prefill_session is not None and self.prefill_session.pipeline_enabled:
            return self.prefill_session.pipeline_devices[0]
        if self._pp_enabled and self.decode_session is not None and self.decode_session.pipeline_enabled:
            return self.decode_session.pipeline_devices[0]
        return self.execution_device

    def forward(
        self,
        inputs_embeds: Tensor,
        position_ids: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        input_ids: Tensor,
        past_kv_caches: List[CacheTensor],
    ) -> Tensor:
        session = self.prefill_session if self._phase_prefill else self.decode_session
        assert session is not None, "Session not initialized"

        # Model-level inputs land on stage 0 (PP) / execution_device (single).
        # Per-module routing inside the session ships hidden states onward.
        if self._pp_enabled:
            in_dev = self._input_device()
        else:
            in_dev = self._device

        if self._phase_prefill:
            out = session(
                inputs_embeds.to(in_dev),
                position_ids.to(in_dev),
                past_seq_length.to(in_dev),
                current_input_length.to(in_dev),
                input_ids.to(in_dev),
                *past_kv_caches,
            )
            # Prefill 输出: (logits, compressed_kv_2, ...)
            # 按 layer index 写入对应的 compressed_kv_caches 槽位
            if isinstance(out, tuple):
                logits = out[0]
                for k, li in enumerate(self.compressor_layer_indices):
                    self.compressed_kv_caches[li] = out[1 + k]
            else:
                logits = out
        else:
            # Decode: 把 compressed_kv 打包进 past_kv_caches
            # past_key_caches = [kv_0..kv_N, ckv_0..ckv_N]
            combined_caches = list(past_kv_caches) + self.compressed_kv_caches
            out = session(
                inputs_embeds.to(in_dev),
                position_ids.to(in_dev),
                past_seq_length.to(in_dev),
                current_input_length.to(in_dev),
                input_ids.to(in_dev),
                *combined_caches,
            )
            logits = out

        if isinstance(session, GoldenMixin):
            session.update_step()
        return logits

    # -------------------------------------------------------------- #
    #  NOTE: 死代码，待删除。                                          #
    #  早期 e2e 实现，已被 DeepseekV4HFCompatible.forward() +          #
    #  generate() 替代。且有 bug：decode 没传 compressed_kv_caches。   #
    # -------------------------------------------------------------- #

    # @torch.no_grad()
    # def _forward(self, messages, enable_thinking=False):
    #     assert self.batch_size == 1 and len(messages) == self.batch_size
    #
    #     texts = self.tokenizer.apply_chat_template(
    #         messages, tokenize=False,
    #         enable_thinking=enable_thinking,
    #         add_generation_prompt=True,
    #     )
    #
    #     batch_input_ids = []
    #     for text in texts:
    #         model_inputs = self.tokenizer([text], padding=False, return_tensors="pt")
    #         batch_input_ids.append(model_inputs.input_ids.cpu().tolist()[0])
    #
    #     data_prefill = {
    #         "input_ids": torch.tensor(batch_input_ids),
    #         "past_seq_length": 0,
    #     }
    #     device = torch.device(self.device)
    #     exec_dev = torch.device(self.execution_device)
    #
    #     # -- Prefill --
    #     prefill_inputs = self.prepare_inputs(
    #         data_prefill, self.prefill_input_sequence_length,
    #     )
    #     (
    #         inputs_embeds, position_ids, past_sl, cur_il, iid, kv_caches,
    #     ) = prefill_inputs
    #
    #     prefill_session = HMONNXInference(str(self.prefill_onnx_file))
    #     prefill_session.to(device)
    #     prefill_session.exec_device = exec_dev
    #
    #     prefill_logits = prefill_session(
    #         inputs_embeds, position_ids, past_sl, cur_il, iid, *kv_caches,
    #     )
    #     next_id, _ = decode_next_token(self.tokenizer, prefill_logits)
    #
    #     del prefill_session
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #
    #     # -- Decode --
    #     past_seq_len = [len(ids) for ids in batch_input_ids]
    #     decode_ids = next_id.cpu().tolist()
    #
    #     data_decode = {
    #         "input_ids": torch.tensor(decode_ids),
    #         "past_seq_length": past_seq_len[0],
    #     }
    #     decode_inputs = self.prepare_inputs(data_decode, 1)
    #     (
    #         inputs_embeds, position_ids, past_sl, cur_il, iid, kv_caches,
    #     ) = decode_inputs
    #
    #     decode_session = HMONNXInference(str(self.decode_onnx_file))
    #     decode_session.to(device)
    #     decode_session.exec_device = exec_dev
    #
    #     decode_logits = decode_session(
    #         inputs_embeds, position_ids, past_sl, cur_il, iid, *kv_caches,
    #     )
    #     decode_id, _ = decode_next_token(self.tokenizer, decode_logits)
    #
    #     generate_ids = torch.cat([next_id, decode_id], dim=1)
    #     generate_text = self.tokenizer.batch_decode(
    #         generate_ids, skip_special_tokens=True,
    #     )
    #     return generate_ids, generate_text
