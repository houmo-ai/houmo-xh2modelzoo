# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import AutoProcessor

from ...hmonnx.hmonnx_model import HMONNXModel
from ...hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
from ...types import VLLMModelMeta
from ...utils import unfold_args
from .data_preprocess import HunyuanOCRPrefillPlan, HunyuanOCRTextDataPreprocess
from .dflash_draft import HunyuanOCRDraftCacheController
from .hunyuan_ocr_processor import HunyuanOCRMultiBucketProcessor
from .hunyuan_ocr_speculative_runtime import (
    HunyuanOCRSpeculativeDecoder,
    HunyuanOCRSpeculativeRuntimeError,
    load_draft_graphs,
    resolve_draft_runtime_contract,
)
from .target_verify_runtime import HunyuanOCRTargetVerifyController, TargetVerifyResult


class XHHunYuanOCRHMONNXModel(VisonLLMHMONNXModel):
    """Autoregressive HMONNX runtime for exported HunyuanOCR bundles."""

    def __init__(self, meta_info: VLLMModelMeta, **kwargs: Any) -> None:
        super().__init__(meta_info, **kwargs)
        self.visual_meta = meta_info.visual_config
        self.visual_buckets = dict(getattr(meta_info, "visual_buckets", {}) or {})
        if self.visual_meta is None and self.visual_buckets:
            self.visual_meta = next(iter(self.visual_buckets.values()))
        if self.visual_meta is None:
            raise ValueError("HunyuanOCR HMONNX metadata must include visual_config or visual_buckets")
        self.visual = HMONNXModel(self.visual_meta.hmonnx, device_map=self._valid_devices)
        self.visual_by_bucket = {
            bucket_id: (
                self.visual
                if visual_meta.hmonnx == self.visual_meta.hmonnx
                else HMONNXModel(visual_meta.hmonnx, device_map=self._valid_devices)
            )
            for bucket_id, visual_meta in self.visual_buckets.items()
        }
        self.last_request_summary: dict[str, Any] | None = None
        self._logical_past_length = 0
        self._rope_delta: torch.Tensor | None = None
        self._spec_decode = dict(getattr(meta_info, "spec_decode", {}) or {})
        verify = dict(self._spec_decode.get("verify", {}) or {})
        verify_input_length = int(verify.get("input_length", 0))
        verify_hmonnx = getattr(meta_info, "verify_hmonnx", "")
        self.verify_model = (
            HMONNXModel(verify_hmonnx, device_map=self._valid_devices)
            if verify_input_length > 1 and verify_hmonnx
            else None
        )
        generation_eos = getattr(meta_info, "generation_eos_token_id", ())
        if type(generation_eos) is int:
            generation_eos = (generation_eos,)
        self._verify_controller = (
            HunyuanOCRTargetVerifyController(
                max_sequence_length=int(meta_info.model_config.context_max_length),
                verify_input_length=verify_input_length,
                pad_token_id=int(meta_info.pad_token_id),
                generation_eos_token_ids=generation_eos,
            )
            if self.verify_model is not None
            else None
        )
        draft_fields = (
            "dflash_context_hmonnx",
            "dflash_context_decode_hmonnx",
            "dflash_decode_hmonnx",
        )
        from .hunyuan_ocr_llm_model import HunyuanOCRTextExportMeta

        self._speculative_runtime_ready = (
            HunyuanOCRTextExportMeta._runtime_ready(meta_info)
            and self.verify_model is not None
            and all(bool(getattr(meta_info, field, "")) for field in draft_fields)
        )
        self._draft_contract: dict[str, Any] | None = None
        self._draft_graphs = None
        self._draft_cache: HunyuanOCRDraftCacheController | None = None
        self._speculative_decoder: HunyuanOCRSpeculativeDecoder | None = None

    def _set_device(self, device):
        super()._set_device(device)
        self.visual.to(device)
        for visual in self.visual_by_bucket.values():
            visual.to(device)
        if self.verify_model is not None:
            self.verify_model.to(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        self.visual._set_dtype(dtype)
        for visual in self.visual_by_bucket.values():
            visual._set_dtype(dtype)
        if self.verify_model is not None:
            self.verify_model._set_dtype(dtype)
        return self

    def to_fast(self):
        self.visual.to_fast()
        for visual in self.visual_by_bucket.values():
            visual.to_fast()
        if self.verify_model is not None:
            self.verify_model.to_fast()
        super().to_fast()
        return self

    def run_visual_bucket(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        grid = torch.as_tensor(image_grid_thw, dtype=torch.long)
        candidates = self.visual_buckets or {"default": self.visual_meta}
        matches = [
            (bucket_id, visual_meta)
            for bucket_id, visual_meta in candidates.items()
            if grid.shape == (1, 3)
            and tuple(int(value) for value in grid[0].tolist())
            == tuple(int(value) for value in visual_meta.image_grid_thw)
        ]
        if len(matches) != 1:
            supported = {
                bucket_id: list(visual_meta.image_grid_thw)
                for bucket_id, visual_meta in candidates.items()
            }
            raise ValueError(
                "HunyuanOCR image_grid_thw does not match exactly one approved bucket: "
                f"got={grid.tolist()}, supported={supported}"
            )
        bucket_id, visual_meta = matches[0]
        visual = self.visual_by_bucket.get(bucket_id, self.visual)
        pixels = torch.as_tensor(pixel_values, device=self.device, dtype=self.dtype)
        expected_input_shape = tuple(int(value) for value in visual_meta.input_shape)
        if tuple(pixels.shape) != expected_input_shape:
            raise ValueError(
                "HunyuanOCR visual input does not match bucket metadata: "
                f"bucket={bucket_id}, expected={expected_input_shape}, got={tuple(pixels.shape)}"
            )
        features = visual(pixels)
        if isinstance(features, (tuple, list)):
            if len(features) != 1:
                raise ValueError(
                    f"HunyuanOCR visual graph must return one output, got {len(features)}"
                )
            features = features[0]
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        expected_output_shape = tuple(int(value) for value in visual_meta.output_shape)
        if tuple(features.shape) != expected_output_shape:
            raise ValueError(
                "HunyuanOCR visual output does not match bucket metadata: "
                f"bucket={bucket_id}, expected={expected_output_shape}, got={tuple(features.shape)}"
            )
        return features, visual_meta

    def generate(self, *args, **kwargs):
        dflash_enabled = kwargs.pop("dflash_enabled", False)
        requested_draft_tokens = kwargs.pop("dflash_num_draft_tokens", None)
        if not dflash_enabled:
            return super().generate(*args, **kwargs)
        if not self._speculative_runtime_ready:
            raise HunyuanOCRSpeculativeRuntimeError(
                "HunyuanOCR DFlash requires a runtime-ready bundle with verify and draft artifacts"
            )
        self._validate_speculative_generation_kwargs(kwargs)
        supplied_eos = kwargs.get("eos_token_id")
        if supplied_eos is not None:
            expected_eos = getattr(self.meta_info, "generation_eos_token_id", None)
            if isinstance(supplied_eos, torch.Tensor):
                supplied_eos = supplied_eos.flatten().tolist()
            if isinstance(expected_eos, torch.Tensor):
                expected_eos = expected_eos.flatten().tolist()
            if isinstance(supplied_eos, (list, tuple)):
                supplied_eos = list(supplied_eos)
            if isinstance(expected_eos, (list, tuple)):
                expected_eos = list(expected_eos)
            if supplied_eos != expected_eos:
                raise HunyuanOCRSpeculativeRuntimeError(
                    "HunyuanOCR DFlash only supports the metadata generation_eos_token_id"
                )
        if requested_draft_tokens is not None and type(requested_draft_tokens) is not int:
            raise HunyuanOCRSpeculativeRuntimeError("dflash_num_draft_tokens must be an integer")
        self._requested_draft_tokens = requested_draft_tokens
        return self._generate_speculative(args, kwargs)

    @staticmethod
    def _validate_speculative_generation_kwargs(kwargs: dict[str, Any]) -> None:
        if kwargs.get("return_dict_in_generate"):
            raise HunyuanOCRSpeculativeRuntimeError(
                "HunyuanOCR DFlash does not support return_dict_in_generate"
            )
        if int(kwargs.get("num_return_sequences", 1) or 1) != 1:
            raise HunyuanOCRSpeculativeRuntimeError(
                "HunyuanOCR DFlash supports num_return_sequences=1 only"
            )
        repetition_penalty = kwargs.get("repetition_penalty")
        if repetition_penalty is not None and float(repetition_penalty) != 1.0:
            raise HunyuanOCRSpeculativeRuntimeError(
                "HunyuanOCR DFlash requires repetition_penalty=1.0"
            )
        if kwargs.get("stopping_criteria"):
            raise HunyuanOCRSpeculativeRuntimeError(
                "HunyuanOCR DFlash does not support stopping_criteria"
            )
        if kwargs.get("streamer") is not None:
            raise HunyuanOCRSpeculativeRuntimeError(
                "HunyuanOCR DFlash does not support streamer"
            )
        min_new_tokens = kwargs.get("min_new_tokens")
        if min_new_tokens not in (None, 0):
            raise HunyuanOCRSpeculativeRuntimeError(
                "HunyuanOCR DFlash does not support min_new_tokens"
            )

    def _generate_speculative(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
        self._reset_speculative_state()
        self._prepare_speculative_decoder()
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            raise HunyuanOCRSpeculativeRuntimeError("HunyuanOCR DFlash generation requires input_ids")
        input_ids = torch.as_tensor(input_ids)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise HunyuanOCRSpeculativeRuntimeError("HunyuanOCR DFlash supports batch_size=1 only")
        max_new_tokens = kwargs.get("max_new_tokens")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise HunyuanOCRSpeculativeRuntimeError("HunyuanOCR DFlash requires positive max_new_tokens")
        if kwargs.get("do_sample") or int(kwargs.get("num_beams", 1) or 1) != 1:
            raise HunyuanOCRSpeculativeRuntimeError("HunyuanOCR DFlash supports greedy decoding only")

        prefill_data: dict[str, Any] = {"input_ids": input_ids}
        for name in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if kwargs.get(name) is not None:
                prefill_data[name] = kwargs[name]
        prefill_output = self.run_generation_step(prefill_data)
        first_token_id = int(self._extract_logits(prefill_output)[0, -1].argmax(dim=-1).item())
        decoder = self._speculative_decoder
        if decoder is None:
            raise HunyuanOCRSpeculativeRuntimeError("HunyuanOCR speculative decoder was not prepared")
        produced = decoder.run(first_token_id=first_token_id, max_new_tokens=max_new_tokens)
        self.last_request_summary = {"spec_decode": decoder.stats.as_summary(enabled=True)}
        generated = torch.tensor([produced], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat((input_ids, generated), dim=1)

    def _reset_speculative_state(self) -> None:
        self._logical_past_length = 0
        self._rope_delta = None
        self.last_request_summary = None
        if self._verify_controller is not None:
            self._verify_controller.reset_generation_state()
        if self._speculative_decoder is not None:
            self._speculative_decoder.reset()
        self.set_prefill()

    def _prepare_speculative_decoder(self) -> None:
        if self._draft_contract is None:
            self._draft_contract = resolve_draft_runtime_contract(self._spec_decode)
        if self._draft_graphs is None:
            self._draft_graphs = load_draft_graphs(
                self.meta_info,
                self._draft_contract,
                device=self.device,
            )
        if self._draft_cache is None:
            self._draft_cache = HunyuanOCRDraftCacheController(
                capacity=int(self._draft_contract["capacity"]),
                block_size=int(self._draft_contract["block_size"]),
            )
        requested_draft_tokens = getattr(self, "_requested_draft_tokens", None)
        num_draft_tokens = int(self._draft_contract["num_draft_tokens"])
        if requested_draft_tokens is not None:
            if not 1 <= requested_draft_tokens <= num_draft_tokens:
                raise HunyuanOCRSpeculativeRuntimeError(
                    "dflash_num_draft_tokens must be within the exported draft contract: "
                    f"1 <= value <= {num_draft_tokens}"
                )
            num_draft_tokens = requested_draft_tokens
        self._speculative_decoder = HunyuanOCRSpeculativeDecoder(
            graphs=self._draft_graphs,
            draft_cache=self._draft_cache,
            embedding_weight=self.get_input_embeddings().weight.detach().to(torch.float16),
            mask_token_id=int(self._draft_contract["mask_token_id"]),
            generation_eos_token_ids=self._draft_contract["generation_eos_token_ids"],
            block_size=int(self._draft_contract["block_size"]),
            num_draft_tokens=num_draft_tokens,
            run_target_verify=self.run_target_verify,
            commit_verify_prefix=self.commit_verify_prefix,
            discard_verify_result=self.discard_verify_result,
            run_decode_step=self._run_decode_step,
            target_logical_past_length=lambda: self._logical_past_length,
        )
        self._speculative_decoder.reset()

    def get_tf_processor(self):
        processor = AutoProcessor.from_pretrained(self.hf_model_dir, backend="pil")
        manifest = getattr(self.meta_info, "resolution_bucket_manifest", None)
        if isinstance(manifest, dict) and manifest.get("status") == "approved":
            return HunyuanOCRMultiBucketProcessor(processor, manifest)
        return processor

    def forward(self, *args):
        normalized_args = [
            arg.to(torch.int32) if isinstance(arg, torch.Tensor) and arg.dtype == torch.int64 else arg
            for arg in args
        ]
        return super().forward(*normalized_args)

    @staticmethod
    def _extract_logits(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise TypeError("HunyuanOCR target graph output must expose logits")

    @staticmethod
    def _extract_target_hidden(output: Any) -> torch.Tensor:
        if isinstance(output, (tuple, list)) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
            return output[1]
        raise TypeError("HunyuanOCR DFlash target graph must expose target_hidden")

    def run_generation_step(self, data: dict[str, Any]) -> Any:
        past_seq_length = int(data.get("past_seq_length", self._logical_past_length))
        if past_seq_length > 0:
            return self._run_decode_step(int(torch.as_tensor(data["input_ids"])[0, -1].item()))

        plan_data = dict(data)
        pixel_values = plan_data.pop("pixel_values", None)
        if pixel_values is not None:
            features, visual_meta = self.run_visual_bucket(pixel_values, plan_data["image_grid_thw"])
            plan_data["image_embeds"] = features
            plan_data["metadata_image_token_count"] = int(visual_meta.image_token_count)
        plan = self.get_data_preprocessor().build_prefill_plan(plan_data)
        return self._run_prefill_plan(plan)

    def _run_prefill_plan(self, plan: HunyuanOCRPrefillPlan) -> Any:
        chunk_size = int(self.meta_info.model_config.prefill_chunk_length)
        processor = self.get_data_preprocessor()
        processor.input_sequence_length = chunk_size
        self.set_prefill()
        output = None
        for start in range(0, plan.valid_length, chunk_size):
            valid_length = min(chunk_size, plan.valid_length - start)
            processed = processor(
                {
                    "inputs_embeds": plan.inputs_embeds[:, start : start + valid_length],
                    "position_ids": plan.position_ids[:, :, start : start + valid_length],
                    "past_seq_length": start,
                    "current_input_length": valid_length,
                }
            )
            output = self.forward(*processed)
            decoder = self._speculative_decoder
            if decoder is not None:
                decoder.append_context(self._extract_target_hidden(output)[:, :valid_length])
            self._logical_past_length = start + valid_length
        if output is None:
            raise ValueError("HunyuanOCR prefill must contain at least one token")
        self._rope_delta = plan.rope_delta
        return output

    def _run_decode_step(self, token_id: int) -> torch.Tensor:
        if self._rope_delta is None:
            raise RuntimeError("HunyuanOCR DFlash decode requires a completed prefill")
        self.set_decode()
        processor = self.get_data_preprocessor()
        processor.input_sequence_length = 1
        processed = processor(
            {
                "input_ids": torch.tensor([[token_id]], dtype=torch.long),
                "past_seq_length": self._logical_past_length,
                "rope_deltas": self._rope_delta,
            }
        )
        output = self.forward(*processed)
        self._logical_past_length += 1
        return self._extract_logits(output)

    def run_target_verify(
        self,
        *,
        current_token_id: int,
        draft_token_ids: Sequence[int],
    ) -> TargetVerifyResult:
        if self.verify_model is None or self._verify_controller is None or self._rope_delta is None:
            raise RuntimeError("HunyuanOCR target verify runtime is not ready")
        rope_delta = int(torch.as_tensor(self._rope_delta).flatten().item())
        self._verify_controller.restore_request_state(
            past_seq_length=self._logical_past_length,
            rope_delta=rope_delta,
        )
        return self._verify_controller.run_target_verify(
            current_token_id=current_token_id,
            draft_token_ids=draft_token_ids,
            executor=self._execute_target_verify,
        )

    def _execute_target_verify(
        self,
        *,
        input_token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_seq_length: int,
        current_input_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processor = self.get_data_preprocessor()
        previous_length = processor.input_sequence_length
        processor.input_sequence_length = int(input_token_ids.shape[1])
        try:
            processed = processor(
                {
                    "input_ids": input_token_ids,
                    "position_ids": position_ids,
                    "past_seq_length": past_seq_length,
                    "current_input_length": current_input_length,
                }
            )
            graph_args = [
                arg.to(torch.int32) if isinstance(arg, torch.Tensor) and arg.dtype == torch.int64 else arg
                for arg in unfold_args(processed)
            ]
            output = self.verify_model(*graph_args)
        finally:
            processor.input_sequence_length = previous_length
        return self._extract_logits(output), self._extract_target_hidden(output)

    def commit_verify_prefix(self, *, transaction_id: int, accepted_draft_count: int) -> int:
        if self._verify_controller is None:
            raise RuntimeError("HunyuanOCR target verify controller is not ready")
        self._logical_past_length = self._verify_controller.commit_verify_prefix(
            transaction_id=transaction_id,
            accepted_draft_count=accepted_draft_count,
        )
        return self._logical_past_length

    def discard_verify_result(self, *, transaction_id: int) -> None:
        if self._verify_controller is None:
            raise RuntimeError("HunyuanOCR target verify controller is not ready")
        self._verify_controller.discard_verify_result(transaction_id=transaction_id)

    def _get_data_preprocessor(self) -> HunyuanOCRTextDataPreprocess:
        config = self.meta_info.model_config
        data_preprocess = HunyuanOCRTextDataPreprocess(
            token_embedding=self.get_input_embeddings(),
            input_sequence_length=self.get_input_sequence_length(),
            context_max_length=int(config.context_max_length),
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            pad_token_id=int(self.meta_info.pad_token_id),
            image_token_id=int(getattr(self.meta_info, "image_token_id", 120120)),
            image_start_token_id=int(getattr(config, "image_start_token_id", 120118)),
            image_end_token_id=int(getattr(config, "image_end_token_id", 120119)),
            spatial_merge_size=int(config.visual_config.spatial_merge_size),
        )
        data_preprocess.to(self._device, self._dtype)
        return data_preprocess


__all__ = ["XHHunYuanOCRHMONNXModel"]
