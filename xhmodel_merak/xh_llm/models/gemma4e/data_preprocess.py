from pathlib import Path

import torch
import torch.nn as nn

from ...llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor


def _aligned(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align


class _Gemma4PerLayerInputRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normed_output = hidden_states.float()
        normed_output = normed_output * torch.rsqrt(normed_output.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed_output = normed_output * self.weight.float()
        return normed_output.to(input_dtype)


class Gemma4PerLayerInputBuilder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size_per_layer_input: int,
        num_hidden_layers: int,
        hidden_size: int,
        hidden_size_per_layer_input: int,
        pad_token_id: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.embed_scale = hidden_size_per_layer_input**0.5
        self.per_layer_input_scale = 2.0**-0.5
        self.per_layer_model_projection_scale = hidden_size**-0.5
        self.embed_tokens_per_layer = nn.Embedding(
            vocab_size_per_layer_input,
            num_hidden_layers * hidden_size_per_layer_input,
            pad_token_id,
        )
        self.per_layer_model_projection = nn.Linear(
            hidden_size,
            num_hidden_layers * hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection_norm = _Gemma4PerLayerInputRMSNorm(hidden_size_per_layer_input, eps=rms_norm_eps)

    @classmethod
    def from_language_model(cls, language_model) -> "Gemma4PerLayerInputBuilder":
        config = language_model.config
        builder = cls(
            vocab_size_per_layer_input=config.vocab_size_per_layer_input,
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            hidden_size_per_layer_input=config.hidden_size_per_layer_input,
            pad_token_id=config.pad_token_id,
            rms_norm_eps=config.rms_norm_eps,
        )
        builder.embed_tokens_per_layer.load_state_dict(language_model.embed_tokens_per_layer.state_dict())
        builder.per_layer_model_projection.load_state_dict(language_model.per_layer_model_projection.state_dict())
        per_layer_norm_state = {
            name: tensor
            for name, tensor in language_model.per_layer_projection_norm.state_dict().items()
            if not name.startswith("norm.")
        }
        builder.per_layer_projection_norm.load_state_dict(per_layer_norm_state)
        builder.to(dtype=language_model.per_layer_model_projection.weight.dtype)
        return builder.eval()

    @classmethod
    def from_artifact(cls, artifact_path: str | Path, text_config) -> "Gemma4PerLayerInputBuilder":
        builder = cls(
            vocab_size_per_layer_input=text_config.vocab_size_per_layer_input,
            num_hidden_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            hidden_size_per_layer_input=text_config.hidden_size_per_layer_input,
            pad_token_id=text_config.pad_token_id,
            rms_norm_eps=text_config.rms_norm_eps,
        )
        try:
            saved = torch.load(str(artifact_path), map_location="cpu", weights_only=True)
        except Exception:
            saved = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
        state_dict = saved["state_dict"] if isinstance(saved, dict) and "state_dict" in saved else saved
        builder.load_state_dict(state_dict)
        sample_tensor = next(iter(state_dict.values()))
        builder.to(dtype=sample_tensor.dtype)
        return builder.eval()

    def save_artifact(self, artifact_path: str | Path):
        state_dict = {name: tensor.detach().cpu() for name, tensor in self.state_dict().items()}
        torch.save({"state_dict": state_dict}, str(artifact_path))

    def forward(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        compute_dtype = self.per_layer_model_projection.weight.dtype
        per_layer_inputs = self.embed_tokens_per_layer(input_ids).to(compute_dtype) * self.embed_scale
        per_layer_inputs = per_layer_inputs.reshape(
            batch_size,
            seq_length,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_inputs = per_layer_inputs.permute(0, 2, 1, 3).contiguous()
        per_layer_projection = self.per_layer_model_projection(inputs_embeds.to(compute_dtype))
        per_layer_projection = per_layer_projection * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            batch_size,
            seq_length,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = per_layer_projection.permute(0, 2, 1, 3).contiguous()
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        return ((per_layer_projection + per_layer_inputs) * self.per_layer_input_scale).to(inputs_embeds.dtype)


class Gemma4InputProcessorConfig(BaseInputProcessorConfig):
    def __init__(
        self,
        embed_tokens,
        input_sequence_length: int = 256,
        past_key_caches=None,
        past_value_caches=None,
        per_layer_input_builder: Gemma4PerLayerInputBuilder | None = None,
        *,
        context_max_length: int = 2048,
        sliding_window: int = 512,
        use_explicit_attention_mask: bool = True,
        image_token_id: int = -1,
        audio_token_id: int = -1,
        video_token_id: int = -1,
        **kwargs,
    ):
        super().__init__(
            embed_tokens=embed_tokens,
            input_sequence_length=input_sequence_length,
            past_key_caches=past_key_caches,
            past_value_caches=past_value_caches,
            **kwargs,
        )
        self.per_layer_input_builder = per_layer_input_builder
        self.context_max_length = context_max_length
        self.sliding_window = sliding_window
        self.use_explicit_attention_mask = use_explicit_attention_mask
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.video_token_id = video_token_id


class Gemma4DataPreprocess(BaseLLMInputProcessor):
    def __init__(self, config: Gemma4InputProcessorConfig):
        super().__init__(config)
        self.context_max_length = config.context_max_length
        self.sliding_window = config.sliding_window
        self.use_explicit_attention_mask = config.use_explicit_attention_mask
        self.image_token_id = config.image_token_id
        self.audio_token_id = config.audio_token_id
        self.video_token_id = config.video_token_id
        self.per_layer_input_builder = config.per_layer_input_builder

    def to(self, *args, **kwargs) -> "Gemma4DataPreprocess":
        super().to(*args, **kwargs)
        if self.per_layer_input_builder is not None:
            self.per_layer_input_builder.to(device=self._device, dtype=self._dtype)
        return self

    def _build_llm_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        llm_input_ids = input_ids.clone()
        if self.image_token_id >= 0:
            llm_input_ids[input_ids == self.image_token_id] = self.pad_token_id
        if self.audio_token_id >= 0:
            llm_input_ids[input_ids == self.audio_token_id] = self.pad_token_id
        if self.video_token_id >= 0:
            llm_input_ids[input_ids == self.video_token_id] = self.pad_token_id
        return llm_input_ids

    def _pad_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[1]
        if seq_length > self.input_sequence_length:
            raise ValueError(
                f"Input sequence length is too long. max input sequence length is {self.input_sequence_length} but got {seq_length}"
            )
        if seq_length == self.input_sequence_length:
            return input_ids

        padding_input_ids = torch.full(
            (input_ids.shape[0], self.input_sequence_length - seq_length),
            self.pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, padding_input_ids], dim=-1)

    @staticmethod
    def _flatten_features(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            return features
        return features.reshape(-1, features.shape[-1])

    def _scatter_features(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        token_id: int,
        features: torch.Tensor | None,
    ) -> torch.Tensor:
        if token_id < 0 or features is None:
            return inputs_embeds

        flat_features = self._flatten_features(features).to(inputs_embeds.device, inputs_embeds.dtype)
        token_positions = (input_ids == token_id).nonzero(as_tuple=False)
        if token_positions.numel() == 0:
            if flat_features.shape[0] != 0:
                raise ValueError(f"Received features for token id {token_id}, but prompt does not contain that token.")
            return inputs_embeds
        if token_positions.shape[0] != flat_features.shape[0]:
            raise ValueError(
                f"Feature count does not match token count for token id {token_id}: "
                f"{flat_features.shape[0]} vs {token_positions.shape[0]}"
            )

        updated = inputs_embeds.clone()
        updated[token_positions[:, 0], token_positions[:, 1]] = flat_features
        return updated

    def _build_attention_masks(
        self,
        *,
        current_input_length: int,
        past_seq_length: int,
        mm_token_type_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_len = self.input_sequence_length
        neg = torch.tensor(torch.finfo(self._dtype).min, dtype=self._dtype, device=device)
        full_ctx = self.context_max_length
        sw = self.sliding_window

        full_mask = torch.full((1, 1, q_len, full_ctx), neg, dtype=self._dtype, device=device)
        valid_k = min(full_ctx, max(1, past_seq_length + current_input_length))
        for q in range(q_len):
            if q < current_input_length:
                full_end = min(valid_k, past_seq_length + q + 1)
                full_mask[0, 0, q, :full_end] = 0
            else:
                full_mask[0, 0, q, 0] = 0

        slide_ctx = full_ctx if sw is None else min(full_ctx, _aligned(sw + q_len - 1, 16))
        clamped_past = min(past_seq_length, sw - 1) if sw is not None and sw > 0 else past_seq_length
        sliding_mask = torch.full((1, 1, q_len, slide_ctx), neg, dtype=self._dtype, device=device)
        for q in range(q_len):
            if q < current_input_length:
                causal_end = min(slide_ctx, clamped_past + q + 1)
                sw_start = max(0, clamped_past + q - sw + 1) if sw is not None else 0
                sliding_mask[0, 0, q, sw_start:causal_end] = 0
            else:
                sliding_mask[0, 0, q, 0] = 0

        return full_mask, sliding_mask

    def forward(self, data: dict | tuple | list) -> list[torch.Tensor]:
        assert isinstance(data, dict)

        input_ids = data.get("input_ids")
        if input_ids is None:
            raise ValueError("Gemma4DataPreprocess requires input_ids.")
        input_ids = input_ids.to(self._device)
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."

        current_input_length = input_ids.shape[1]
        input_ids = self._pad_input_ids(input_ids)
        llm_input_ids = self._build_llm_input_ids(input_ids)
        inputs_embeds = self.embed_tokens(llm_input_ids)

        inputs_embeds = self._scatter_features(
            inputs_embeds,
            input_ids,
            token_id=self.image_token_id,
            features=data.get("image_embeds"),
        )
        inputs_embeds = self._scatter_features(
            inputs_embeds,
            input_ids,
            token_id=self.audio_token_id,
            features=data.get("audio_embeds"),
        )
        inputs_embeds = self._scatter_features(
            inputs_embeds,
            input_ids,
            token_id=self.video_token_id,
            features=data.get("video_embeds"),
        )
        per_layer_inputs = None
        if self.per_layer_input_builder is not None:
            per_layer_inputs = self.per_layer_input_builder(llm_input_ids.to(torch.long), inputs_embeds)

        past_seq_length = int(data.get("past_seq_length", 0))
        mm_token_type_ids = data.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros((1, current_input_length), dtype=torch.long, device=self._device)
        else:
            mm_token_type_ids = mm_token_type_ids.to(self._device)
        if mm_token_type_ids.shape[-1] < self.input_sequence_length:
            mm_token_type_ids = torch.cat(
                [
                    mm_token_type_ids,
                    torch.zeros(
                        (mm_token_type_ids.shape[0], self.input_sequence_length - mm_token_type_ids.shape[-1]),
                        dtype=mm_token_type_ids.dtype,
                        device=mm_token_type_ids.device,
                    ),
                ],
                dim=-1,
            )
        else:
            mm_token_type_ids = mm_token_type_ids[:, : self.input_sequence_length]

        output = [
            inputs_embeds.to(self._device),
            torch.tensor([past_seq_length], dtype=torch.int32, device=self._device),
            torch.tensor([current_input_length], dtype=torch.int32, device=self._device),
            self.past_key_caches,
            self.past_value_caches,
        ]
        if self.use_explicit_attention_mask:
            global_attention_mask, local_attention_mask = self._build_attention_masks(
                current_input_length=current_input_length,
                past_seq_length=past_seq_length,
                mm_token_type_ids=mm_token_type_ids[0],
                device=self._device,
            )
            output.insert(3, local_attention_mask)
            output.insert(4, global_attention_mask)
        if per_layer_inputs is not None:
            output.insert(0, per_layer_inputs.to(self._device, inputs_embeds.dtype))
        return tuple(output)
