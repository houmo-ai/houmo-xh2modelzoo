import torch
import torch.nn.functional as F

from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel, Gemma4VisionPatchEmbedder
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


def _build_patch_position_embeddings(
    position_embedding_table: torch.Tensor,
    pixel_position_ids: torch.Tensor,
    padding_positions: torch.Tensor,
) -> torch.Tensor:
    clamped_positions = pixel_position_ids.to(position_embedding_table.dtype).clamp(min=0)
    one_hot = F.one_hot(clamped_positions.long(), num_classes=position_embedding_table.shape[1])
    one_hot = one_hot.permute(0, 2, 1, 3).to(position_embedding_table)
    position_embeddings = one_hot @ position_embedding_table
    position_embeddings = position_embeddings.sum(dim=1)
    position_embeddings = torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)
    return position_embeddings


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4VisionPatchEmbedder: "Gemma4VisionPatchEmbedder"})
class _Gemma4VisionPatchEmbedder(DynamicModule):
    def _setup(self, cfg=None):
        return None

    def _position_embeddings(self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor) -> torch.Tensor:
        return _build_patch_position_embeddings(self.position_embedding_table, pixel_position_ids, padding_positions)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        position_embeddings = self._position_embeddings(pixel_position_ids, padding_positions)
        return hidden_states + position_embeddings


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4VisionModel: "Gemma4VisionModel"})
class _Gemma4VisionModel(DynamicModule):
    def _setup(self, cfg=None):
        return None

    @staticmethod
    def _build_bidirectional_attention_mask(valid_positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = valid_positions.shape
        return valid_positions[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)

    @staticmethod
    def _avg_pool_by_positions(
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        output_length: int,
        pooling_kernel_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooling_kernel_area = pooling_kernel_size * pooling_kernel_size
        input_seq_len = hidden_states.shape[1]
        if pooling_kernel_area * output_length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {output_length}: "
                f"{pooling_kernel_size=}^2 times {output_length=} must be {input_seq_len}."
            )

        clamped_positions = pixel_position_ids.to(hidden_states.dtype).clamp(min=0)
        pooled_positions = (clamped_positions / float(pooling_kernel_size)).to(pixel_position_ids.dtype)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        pooled_width = (max_x / float(pooling_kernel_size)).to(pixel_position_ids.dtype)
        kernel_idxs = pooled_positions[..., 0] + pooled_width * pooled_positions[..., 1]
        weights = F.one_hot(kernel_idxs.long(), output_length).float() / float(pooling_kernel_area)
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = weights.sum(dim=1) > 0
        return output.type_as(hidden_states), mask

    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length}) than there are patches"
                f" ({hidden_states.shape[1]}). Change the value of `num_soft_tokens` when processing."
            )

        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        valid_mask = ~padding_positions
        if hidden_states.shape[1] != output_length:
            hidden_states, valid_mask = self._avg_pool_by_positions(
                hidden_states,
                pixel_position_ids,
                output_length,
                self.config.pooling_kernel_size,
            )
        return hidden_states * self.pooler.root_hidden_size, valid_mask

    def forward(self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor):
        pooling_kernel_size = self.config.pooling_kernel_size
        output_length = pixel_values.shape[-2] // (pooling_kernel_size * pooling_kernel_size)

        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        attention_mask = self._build_bidirectional_attention_mask(~padding_positions)
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_position_ids=pixel_position_ids,
        )
        hidden_states = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, "last_hidden_state") else encoder_outputs
        hidden_states, pooler_mask = self._pool_hidden_states(hidden_states, pixel_position_ids, padding_positions, output_length)

        if self.config.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale
        return hidden_states, pooler_mask


def register_wrap_modules(hf_model=None):
    return None


register_wrap_cls = register_wrap_modules
