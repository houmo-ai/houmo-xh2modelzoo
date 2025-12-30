import torch, torch.nn as nn, torch.nn.functional as F
from torch.fx import GraphModule
from typing import Optional, List, Tuple, Union
from transformers import AutoModelForCausalLM
from xh_model_zoo_new.xh_llm.base_llm_infer_adapter import (
    BaseLLMHFCompatible,
    _LLMInfer,
    BaseLLMInferHMONNXImpl,
    BaseLLMInferQModelImpl,
)
from transformers.configuration_utils import PretrainedConfig
from xhquant.api import ConfigDict, QuantGraph, FrontendGraph
from abc import abstractmethod
from xh_model_zoo_new.xh_llm.models.qwen2_5_vl.device_dtype_mixin import DeviceDtypeMixin
from xhquant.api import HMONNXInference
from xhquant.xhonnxruntime.hmonnx_graph_inference import HMONNXGrapInference
from xh_model_zoo_new.xh_llm.models.builder import LLM_DYNAMIC_MODULES
from transformers.generation.utils import GenerationMixin
from transformers import AutoTokenizer


from .modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast


class _Qwen2_5_VLVisionInfer(nn.Module):

    def __init__(self, wrap_cfg: ConfigDict, hf_config: PretrainedConfig):
        super().__init__()
        self.wrap_cfg = ConfigDict(wrap_cfg)
        self.hf_config = hf_config

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward must be implemented in subclass")

    @property
    def is_window_optimizer_enabled(self):  # 根据它判断是否使用 window_mask
        window_size = self.hf_config.vision_config.window_size
        max_size_w, max_size_h = self.wrap_cfg.vision.max_size_w, self.wrap_cfg.vision.max_size_h
        return max_size_w % window_size and max_size_h % window_size

    def prepare_inputs(self, pixel_values: torch.Tensor, image_grid_thw):
        pixel_values = pixel_values.half().unsqueeze(2).repeat(1, 1, self.wrap_cfg.vision.temporal_patch_size, 1, 1)
        # image_grid_thw[0][0] = (
        #     self.wrap_cfg.vision.image_max_size_t // self.wrap_cfg.vision.temporal_patch_size
        # )  # TODO ?

        device = pixel_values.device
        window_index, cu_window_seqlens = self.get_window_index(image_grid_thw)

        inputs = [
            pixel_values,
            window_index.to(device=device, dtype=torch.int32),
        ]
        if self.is_window_optimizer_enabled:
            cu_window_seqlens = torch.tensor(cu_window_seqlens, device=device, dtype=torch.int32)
            cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
            seq_len = cu_window_seqlens[-1]
            attention_mask = torch.full(
                [1, seq_len, seq_len],
                torch.finfo(torch.float16).min,
                device=pixel_values.device,
                dtype=torch.float16,
            )
            for i in range(1, len(cu_window_seqlens)):
                attention_mask[
                    ...,
                    cu_window_seqlens[i - 1] : cu_window_seqlens[i],
                    cu_window_seqlens[i - 1] : cu_window_seqlens[i],
                ] = 0
            inputs.append(attention_mask)
        return inputs

    def get_window_index(self, grid_thw: torch.Tensor):
        hf_config = self.hf_config
        window_size, spatial_merge_size, patch_size = (
            hf_config.vision_config.window_size,
            hf_config.vision_config.spatial_merge_size,
            hf_config.vision_config.patch_size,
        )
        spatial_merge_unit = spatial_merge_size * spatial_merge_size

        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = window_size // spatial_merge_size // patch_size
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // spatial_merge_size,
                grid_w // spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens


class Qwen2_5_VLVisionInferQModelImpl(_Qwen2_5_VLVisionInfer):
    def __init__(
        self,
        vision_model: Union[nn.Module, QuantGraph, FrontendGraph],
        wrap_cfg: ConfigDict,
        hf_config: PretrainedConfig,
    ):
        super().__init__(wrap_cfg, hf_config)
        self.vision_model = vision_model

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        inputs = self.prepare_inputs(pixel_values, image_grid_thw)
        return self.vision_model(*inputs)


class Qwen2_5_VLVisionInferHMONNXImpl(_Qwen2_5_VLVisionInfer, DeviceDtypeMixin):
    def __init__(self, vision_onnx_file: str, wrap_cfg: ConfigDict, hf_config: PretrainedConfig):
        super().__init__(wrap_cfg, hf_config)
        self.wrap_cfg = ConfigDict(wrap_cfg)
        self.hf_config = hf_config
        self.vision_model = HMONNXGrapInference(vision_onnx_file)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        inputs = self.prepare_inputs(pixel_values, image_grid_thw)
        image_embeds = self.vision_model(*inputs)
        return image_embeds


class Qwen2_5_VLLLMInferQModelImpl(BaseLLMInferQModelImpl):
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        time_position_ids: torch.Tensor,
        height_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        past_key_caches: List[torch.Tensor],
        past_value_caches: List[torch.Tensor],
    ):
        self.set_input_sequence_length(inputs_embeds.shape[-2])
        if isinstance(self.model, GraphModule):
            return self.model(
                inputs_embeds,
                time_position_ids,
                height_position_ids,
                width_position_ids,
                past_seq_length,
                current_input_length,
                *past_key_caches,
                *past_value_caches,
            )
        else:
            return self.model(
                inputs_embeds,
                time_position_ids,
                height_position_ids,
                width_position_ids,
                past_seq_length,
                current_input_length,
                past_key_caches,
                past_value_caches,
            )


class Qwen2_5_VLLLMInferHMONNXImpl(BaseLLMInferHMONNXImpl):
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        time_position_ids: torch.Tensor,
        height_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        past_key_caches: List[torch.Tensor],
        past_value_caches: List[torch.Tensor],
    ):
        token_length = inputs_embeds.shape[-2]
        if token_length == 1:
            return self.decode(
                inputs_embeds,
                time_position_ids,
                height_position_ids,
                width_position_ids,
                past_seq_length,
                current_input_length,
                past_key_caches,
                past_value_caches,
            )
        return self.prefill(
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        )

    def prefill(
        self,
        inputs_embeds: torch.Tensor,
        time_position_ids: torch.Tensor,
        height_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        past_key_caches: List[torch.Tensor],
        past_value_caches: List[torch.Tensor],
    ):
        seq_length = inputs_embeds.shape[-2]
        input_step = self.get_input_sequence_length()
        steps = (seq_length + input_step - 1) // input_step
        padding_len = steps * input_step - seq_length
        if padding_len:
            pad = torch.zeros(
                inputs_embeds.shape[0],
                padding_len,
                inputs_embeds.shape[-1],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            inputs_embeds = torch.cat([inputs_embeds, pad], dim=1)
            time_position_ids = torch.cat(
                [
                    time_position_ids,
                    torch.zeros(padding_len, dtype=time_position_ids.dtype, device=time_position_ids.device),
                ]
            )
            height_position_ids = torch.cat(
                [
                    height_position_ids,
                    torch.zeros(padding_len, dtype=height_position_ids.dtype, device=height_position_ids.device),
                ]
            )
            width_position_ids = torch.cat(
                [
                    width_position_ids,
                    torch.zeros(padding_len, dtype=width_position_ids.dtype, device=width_position_ids.device),
                ]
            )

        out = None
        for i in range(steps):
            start = i * input_step
            end = (i + 1) * input_step
            sub_inputs = inputs_embeds[:, start:end, :]
            sub_time = time_position_ids[start:end]
            sub_height = height_position_ids[start:end]
            sub_width = width_position_ids[start:end]
            sub_past_seq_length = past_seq_length + start
            sub_current_len = torch.tensor(
                [min(end, seq_length) - start], dtype=current_input_length.dtype, device=current_input_length.device
            )
            out = self.prefill_session(
                sub_inputs,
                sub_time,
                sub_height,
                sub_width,
                sub_past_seq_length,
                sub_current_len,
                *past_key_caches,
                *past_value_caches,
            )
        return out

    def decode(
        self,
        inputs_embeds: torch.Tensor,
        time_position_ids: torch.Tensor,
        height_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        past_key_caches: List[torch.Tensor],
        past_value_caches: List[torch.Tensor],
    ):
        return self.decoder_session(
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_input_length,
            *past_key_caches,
            *past_value_caches,
        )


class Qwen2_5_VLHFCompatible(BaseLLMHFCompatible):
    def _setup(
        self,
        llm_model: _LLMInfer,
        vision_model: _Qwen2_5_VLVisionInfer,
        embed_tokens: nn.Embedding,
        wrap_cfg: ConfigDict,
        hf_config: PretrainedConfig,
        processor: Optional=None,
        is_native_vision_model: bool = False,
    ):
        super()._setup(llm_model, embed_tokens, wrap_cfg, hf_config, processor.tokenizer)
        self.vision_model = vision_model
        self.processor = processor
        self.is_native_vision_model = is_native_vision_model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        hm_pixel_values: Optional[torch.Tensor] = None,
    ) -> Union[Tuple]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # if hm_pixel_values is not None:
        #     pixel_values = hm_pixel_values

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if pixel_values is not None:
                # pixel_values = pixel_values
                if self.is_native_vision_model:
                    image_embeds = self.vision_model.forward_ori(pixel_values, image_grid_thw)
                else:
                    image_embeds = self.vision_model(hm_pixel_values, image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0] if self.is_native_vision_model else image_embeds.shape[1]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                raise NotImplementedError("video inputs are not supported in demo inference yet.")

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or not hasattr(self, "rope_deltas")
                or self.rope_deltas is None
                or (past_key_values is None or self.past_seq_length == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if self._prefill:  # TODO 用来区分prefill 和 decode 貌似没啥必要？
            past_seq_length = torch.tensor([0], dtype=torch.int32).to(inputs_embeds.device)
            self._prefill = False
        else:
            past_seq_length = torch.tensor([self.past_seq_length], dtype=torch.int32).to(inputs_embeds.device)

        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        current_input_length = torch.tensor([seq_length], dtype=torch.int32).to(inputs_embeds.device)

        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches
        if past_key_values is None or (hasattr(self, "use_cache") and not self.use_cache):
            past_key_caches = [torch.tensor([])] * len(past_key_caches)
            past_value_caches = [torch.tensor([])] * len(past_value_caches)

        time_position_ids = position_ids[0, 0, :].to(torch.int32)
        height_position_ids = position_ids[1, 0, :].to(torch.int32)
        width_position_ids = position_ids[2, 0, :].to(torch.int32)
        outputs = self.llm_model(
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        )

        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

        num_logits_to_keep = self.llm_model.get_num_logits_to_keep()
        if num_logits_to_keep != 0:
            logits = logits[:, :seq_length, :]

        self.past_seq_length += seq_length
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return Qwen2_5_VLCausalLMOutputWithPast(logits=logits)

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def load_and_process_image(self, image_path):
        """
        Loads an image from the given path, converts to RGB, resizes proportionally if needed,
        and pads to (self.image_size_w, self.image_size_h) with (114,114,114) background.
        Returns the processed PIL image.
        """
        from PIL import Image, ImageOps

        target_w, target_h = self.vision_model.wrap_cfg.vision.max_size_w, self.vision_model.wrap_cfg.vision.max_size_h
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        if (orig_w, orig_h) != (target_w, target_h):
            # Resize while keeping aspect ratio
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)
            # Pad to target size
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            left = 0
            top = 0
            right = pad_w
            bottom = pad_h
            image = ImageOps.expand(image, border=(left, top, right, bottom), fill=(114, 114, 114))
        return image

    def load_and_process_image_v2(self, image_path):
        """
        Loads an image from the given path, converts to RGB, resizes proportionally if needed,
        and pads to (self.image_size_w, self.image_size_h) with (114,114,114) background.
        Returns the processed PIL image.
        """
        from PIL import Image, ImageOps

        target_w, target_h = self.vision_model.wrap_cfg.vision.max_size_w, self.vision_model.wrap_cfg.vision.max_size_h
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        if (orig_w, orig_h) != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.BICUBIC)
        return image

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen2-5-VL position_ids are prepared with rope_deltas
        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            if cache_position[0] == 0 or self.model.rope_deltas is None:
                vision_positions, rope_deltas = self.model.get_rope_index(
                    model_inputs.get("input_ids", None),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            elif "position_ids" in model_inputs:
                position_ids = model_inputs["position_ids"][None, ...]
                delta = self.model.rope_deltas
                delta = delta.repeat_interleave(position_ids.shape[1] // delta.shape[0], dim=0)
                vision_positions = position_ids + delta.expand_as(position_ids)
                vision_positions = vision_positions.expand(3, vision_positions.shape[1], -1)

            # Concatenate "text + vision" positions into [4, bs, seq-len]
            if "position_ids" not in model_inputs:
                text_positions = torch.arange(input_ids, device=input_ids.device)[None, None, :]
            else:
                text_positions = model_inputs["position_ids"][None, ...]
            model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)

        if cache_position[0] != 0:
            model_inputs["hm_pixel_values"] = None
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def generate(self, *args, **kwargs):
        self._prefill = True
        self.past_seq_length = 0
        return super().generate(*args, **kwargs)

    @classmethod
    def from_hmonnx(
        cls,
        vision_path: str,
        prefill_path: str,
        decoder_path: str,
        wrap_cfg: ConfigDict,
        hf_config: PretrainedConfig,
        token_embedding: nn.Embedding,
        native_model_or_path: Union[str, nn.Module],
        processor: "Qwen2_5_VLProcessor",
    ):
        # 1. create infer
        llm_infer_impl = Qwen2_5_VLLLMInferHMONNXImpl(
            prefill_path=prefill_path,
            decoder_path=decoder_path,
            wrap_cfg=wrap_cfg,
        )
        vision_model = Qwen2_5_VLVisionInferHMONNXImpl(vision_path, wrap_cfg, hf_config)

        # 2. create native model
        if isinstance(native_model_or_path, str):
            native_model = AutoModelForCausalLM.from_config(hf_config)
        elif isinstance(native_model_or_path, nn.Module):
            native_model = native_model_or_path
        else:
            raise ValueError(f"Invalid type for native_model_or_path: {type(native_model_or_path)}")

        # 3. register and convert
        cls.register(native_model, LLM_DYNAMIC_MODULES)
        return LLM_DYNAMIC_MODULES.convert(
            native_model,
            llm_model=llm_infer_impl,
            vision_model=vision_model,
            embed_tokens=token_embedding,
            wrap_cfg=wrap_cfg,
            hf_config=hf_config,
            processor=processor,
        )

    @classmethod
    def from_qmodel(
        cls,
        llm_model: Union[QuantGraph, FrontendGraph, nn.Module],
        vision_model: Union[nn.Module, QuantGraph, FrontendGraph],
        hf_config: "PretrainedConfig",
        wrap_cfg: "ConfigDict",
        token_embedding: nn.Embedding,
        native_model_or_path: Optional[Union[AutoModelForCausalLM, str]],  # TODO 用 Config 来避免加载全量权重
        processor: "Qwen2_5_VLProcessor",
        is_native_vision_model: bool = False,
    ):
        """"""
        # 1. Create Infer
        llm_model = Qwen2_5_VLLLMInferQModelImpl(llm_model, wrap_cfg, hf_config)
        if not is_native_vision_model:
            vision_model = Qwen2_5_VLVisionInferQModelImpl(vision_model, wrap_cfg, hf_config)

        # 2. Create HFCompatible Model
        if native_model_or_path is None:
            from transformers.modeling_utils import no_init_weights

            with no_init_weights():
                native_model = AutoModelForCausalLM.from_config(hf_config)
        elif isinstance(native_model_or_path, str):
            native_model = AutoModelForCausalLM.from_pretrained(native_model_or_path, trust_remote_code=True)
        elif isinstance(native_model_or_path, nn.Module):
            native_model = native_model_or_path
        else:
            raise ValueError(f"Invalid type: {type(native_model_or_path)}")

        # 3. register and convert
        cls.register(native_model, LLM_DYNAMIC_MODULES)
        return LLM_DYNAMIC_MODULES.convert(
            native_model,
            llm_model=llm_model,
            vision_model=vision_model,
            embed_tokens=token_embedding,
            hf_config=hf_config,
            wrap_cfg=wrap_cfg,
            processor=processor,
            is_native_vision_model=is_native_vision_model,
        )

    @torch.inference_mode()
    def demo(self, prompt: str, image_path: Optional[str] = None, max_generation_length: int = 32) -> str:
        device = self.embed_tokens.weight.device
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if image_path is not None:
            if self.is_native_vision_model:
                from qwen_vl_utils import process_vision_info

                image_inputs, _ = process_vision_info(messages)
            else:
                image_inputs = self.load_and_process_image_v2(image_path)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt").to(
            device
        )
        # Trim prompt tokens from generation
        from transformers.generation.streamers import TextStreamer

        streamer = TextStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generated_ids = self.generate(**inputs, max_new_tokens=max_generation_length, streamer=streamer)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
