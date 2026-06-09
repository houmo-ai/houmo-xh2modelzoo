import json
import types
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from xhquant.api import HMONNXGoldenInference

from .text_encoder_wrapper import Flux2KleinWrappedTextEncoderModel


DEFAULT_TEXT_ENCODER_OUT_LAYERS = (9, 18, 27)


def build_qwen3_text_inputs(
    tokenizer,
    prompt: Union[str, Sequence[str]],
    max_sequence_length: int,
) -> Dict[str, torch.Tensor]:
    prompt_list = [prompt] if isinstance(prompt, str) else list(prompt)
    all_input_ids = []
    all_attention_masks = []

    for single_prompt in prompt_list:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": single_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        all_input_ids.append(inputs["input_ids"])
        all_attention_masks.append(inputs["attention_mask"])

    return {
        "input_ids": torch.cat(all_input_ids, dim=0),
        "attention_mask": torch.cat(all_attention_masks, dim=0),
    }


def build_qwen3_prompt_embeds(
    text_encoder: nn.Module,
    tokenizer,
    prompt: Union[str, Sequence[str]],
    max_sequence_length: int,
    hidden_states_layers: Tuple[int, ...] = DEFAULT_TEXT_ENCODER_OUT_LAYERS,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    model_inputs = build_qwen3_text_inputs(tokenizer, prompt, max_sequence_length)
    model_device = text_encoder.device if device is None else device
    model_dtype = text_encoder.dtype if dtype is None else dtype

    input_ids = model_inputs["input_ids"].to(model_device)
    attention_mask = model_inputs["attention_mask"].to(model_device)

    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=model_dtype, device=model_device)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

    return prompt_embeds, model_inputs


class Flux2KleinTextEncoderExportWrapper(nn.Module):
    def __init__(
        self,
        text_encoder: nn.Module,
        hidden_states_layers: Tuple[int, ...] = DEFAULT_TEXT_ENCODER_OUT_LAYERS,
        output_dtype: torch.dtype = torch.float16,
        max_sequence_length: int = 512,
        batch_size: int = 1,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.hidden_states_layers = tuple(hidden_states_layers)
        self.output_dtype = output_dtype
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.hidden_size = int(self.text_encoder.config.hidden_size)
        self.num_hidden_layers = len(self.hidden_states_layers)
        self.wrapped_model = Flux2KleinWrappedTextEncoderModel(
            self.text_encoder,
            input_sequence_length=self.max_sequence_length,
        )

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        collected = self.wrapped_model(
            inputs_embeds.to(self.output_dtype),
            attention_mask=attention_mask,
        )
        out = torch.stack([collected[k] for k in self.hidden_states_layers], dim=1)
        out = out.to(dtype=self.output_dtype)
        return out.permute(0, 2, 1, 3).reshape(
            self.batch_size,
            self.max_sequence_length,
            self.num_hidden_layers * self.hidden_size,
        )


class Flux2KleinTextEncoderInference(nn.Module):
    def __init__(
        self,
        hmonnx_path: Union[str, Path],
        tokenizer,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int, ...] = DEFAULT_TEXT_ENCODER_OUT_LAYERS,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.runtime = HMONNXGoldenInference(str(hmonnx_path))
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.text_encoder_out_layers = tuple(text_encoder_out_layers)
        self._dtype = dtype
        self._device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.token_embedding: Optional[nn.Embedding] = None
        self.runtime.exec_device = self._device

    @classmethod
    def from_meta(
        cls,
        meta_path: Union[str, Path],
        tokenizer=None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> "Flux2KleinTextEncoderInference":
        meta_file = Path(meta_path)
        meta_info = json.load(open(meta_file, "r"))
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(meta_info["tokenizer_dir"], trust_remote_code=True)
        inference = cls(
            hmonnx_path=meta_file.parent / meta_info["hmonnx_file"],
            tokenizer=tokenizer,
            max_sequence_length=int(meta_info["max_sequence_length"]),
            text_encoder_out_layers=tuple(meta_info["text_encoder_out_layers"]),
            device=device,
            dtype=dtype,
        )
        token_embedding_path = meta_info.get("token_embedding_file")
        if token_embedding_path is not None:
            token_embedding_state_dict = torch.load(
                meta_file.parent / token_embedding_path,
                map_location="cpu",
                weights_only=True,
            )
            token_embedding = nn.Embedding(
                token_embedding_state_dict["weight"].shape[0],
                token_embedding_state_dict["weight"].shape[1],
            ).to(dtype)
            token_embedding.load_state_dict(token_embedding_state_dict)
            inference.token_embedding = token_embedding.to(inference.device)
        return inference

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._device = torch.device(device)
            self.runtime.exec_device = self._device
            if self.token_embedding is not None:
                self.token_embedding = self.token_embedding.to(self._device)
        if dtype is not None:
            self._dtype = dtype
            if self.token_embedding is not None:
                self.token_embedding = self.token_embedding.to(dtype=dtype)
        return self

    @staticmethod
    def _prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = prompt_embeds.shape
        out_ids = []
        for _ in range(batch_size):
            coords = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(seq_len))
            out_ids.append(coords)
        return torch.stack(out_ids)

    def prepare_inputs(self, prompt: Union[str, Sequence[str]], max_sequence_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        inputs = build_qwen3_text_inputs(
            self.tokenizer,
            prompt,
            self.max_sequence_length if max_sequence_length is None else max_sequence_length,
        )
        inputs["attention_mask"] = ( 1 - inputs["attention_mask"] ) * -65504
        return {
            "input_ids": inputs["input_ids"].to(device=self.device, dtype=torch.int32),
            "attention_mask": inputs["attention_mask"].to(device=self.device, dtype=torch.int32),
        }

    def prepare_runtime_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.token_embedding is None:
            raise ValueError("token_embedding is not initialized")
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        attention_mask = attention_mask.to(device=self.device, dtype=torch.int32)
        inputs_embeds = self.token_embedding(input_ids).to(device=self.device, dtype=self.dtype)
        return inputs_embeds, attention_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        inputs_embeds, runtime_attention_mask = self.prepare_runtime_inputs(input_ids, attention_mask)
        outputs = self.runtime(
            inputs_embeds,
            runtime_attention_mask,
        )
        prompt_embeds = outputs[0] if isinstance(outputs, tuple) else outputs
        return prompt_embeds.to(device=self.device, dtype=self.dtype)

    def get_prompt_embeds(
        self,
        prompt: Union[str, Sequence[str]],
        max_sequence_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        model_inputs = self.prepare_inputs(prompt, max_sequence_length=max_sequence_length)
        prompt_embeds = self.forward(model_inputs["input_ids"], model_inputs["attention_mask"])
        return prompt_embeds, model_inputs

    def encode_prompt(
        self,
        prompt: Union[str, Sequence[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: Optional[int] = None,
        text_encoder_out_layers: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del text_encoder_out_layers
        encode_device = self.device if device is None else torch.device(device)
        if prompt is None:
            prompt = ""

        prompt_list = [prompt] if isinstance(prompt, str) else list(prompt)

        if prompt_embeds is None:
            prompt_embeds, _ = self.get_prompt_embeds(prompt_list, max_sequence_length=max_sequence_length)

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self._prepare_text_ids(prompt_embeds).to(encode_device)
        prompt_embeds = prompt_embeds.to(encode_device, dtype=self.dtype)
        return prompt_embeds, text_ids


def attach_hmonnx_text_encoder(pipe, text_encoder: Flux2KleinTextEncoderInference):
    def _encode_prompt(
        self,
        prompt,
        device=None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int, ...] = DEFAULT_TEXT_ENCODER_OUT_LAYERS,
    ):
        return text_encoder.encode_prompt(
            prompt=prompt,
            device=self._execution_device if device is None else device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

    pipe.text_encoder = text_encoder
    pipe.encode_prompt = types.MethodType(_encode_prompt, pipe)
    return pipe