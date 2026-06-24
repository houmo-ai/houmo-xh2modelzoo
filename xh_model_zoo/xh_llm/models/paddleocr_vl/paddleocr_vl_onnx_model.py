import torch
from PIL import Image
from torch import Tensor, nn
from xhquant.api import HMONNXGoldenInference as HMONNXInference
from xhquant.core import CacheTensor

from ..builder import MODELS
from ..device_dtype_mixin import DeviceDtypeMixin
from ..llm_onnx_model import LLMONNXModel
from .modeling_paddleocr_vl import PaddleOCRVLForConditionalGeneration
from .processing_paddleocr_vl import PaddleOCRVLProcessor


def decode_next_token(tokenizer, logits: torch.Tensor, do_sample: bool = False):
    """Decode next token from logits.

    logits: [batch, seq_len, vocab_size] or [batch, vocab_size]
    Always uses the LAST position and float32 for numerical stability.
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]  # [batch, vocab_size]
    logits = logits.float()  # fp32 for numerical stability
    if do_sample:
        probs = nn.functional.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        next_tokens = next_tokens.unsqueeze(0)
    else:
        next_tokens = torch.argmax(logits, dim=-1)  # [batch]
        if next_tokens.dim() == 1:
            next_tokens = next_tokens.unsqueeze(0)  # [1, batch]
    next_token_str = tokenizer.batch_decode(next_tokens, skip_special_tokens=True)
    return next_tokens, next_token_str


@MODELS.register_module()
class PaddleOCRVLONNXModel(LLMONNXModel, DeviceDtypeMixin):
    def __init__(
        self,
        hf_model_dir: str,
        image_feature,
        prefill,
        decode,
        kv_cache,
        cache_len: int = 2048,
    ):
        super().__init__(prefill, decode, kv_cache)
        self._device = torch.device("cpu")
        self._dtype = torch.float16
        self._exec_device = torch.device("cpu")

        self.prefill_config = prefill
        self.decode_config = decode
        self.image_feature_config = image_feature

        self.prefill_session = HMONNXInference(self.prefill_config.onnx)
        self.decode_session = HMONNXInference(self.decode_config.onnx)
        self.visual_session = HMONNXInference(self.image_feature_config.onnx)
        self.visual_onnx_path = self.image_feature_config.onnx

        self.kv_cache = kv_cache
        self.num_hidden_layers = kv_cache.num_hidden_layers
        self.kv_cache_shape = kv_cache.shape
        self.cache_len = cache_len

        for i in range(self.num_hidden_layers):
            self.register_buffer(
                f"past_k_cache_{i}",
                CacheTensor(torch.zeros(self.kv_cache_shape, dtype=torch.float16)),
                persistent=False,
            )
            self.register_buffer(
                f"past_v_cache_{i}",
                CacheTensor(torch.zeros(self.kv_cache_shape, dtype=torch.float16)),
                persistent=False,
            )

        # Load HF model for mlp_AR + config/token ids + rope index
        self.hf_model = PaddleOCRVLForConditionalGeneration.from_pretrained(
            hf_model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",
        ).eval()
        self.model_config = self.hf_model.config
        self.mlp_AR = self.hf_model.mlp_AR

        self.pad_token_id = self.model_config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0
        self.image_token_id = self.model_config.image_token_id

        # EOS token IDs for stopping generation
        # config.eos_token_id may be None; fall back to generation_config / tokenizer default (2 = </s>)
        eos_id = self.model_config.eos_token_id
        if eos_id is None:
            gen_cfg = getattr(self.model_config, "generation_config", None)
            if gen_cfg is not None:
                eos_id = getattr(gen_cfg, "eos_token_id", None)
        if eos_id is None:
            eos_id = 2  # </s> fallback
        if isinstance(eos_id, int):
            self.eos_token_id = [eos_id]
        elif isinstance(eos_id, (list, tuple)):
            self.eos_token_id = list(eos_id)
        else:
            self.eos_token_id = [2]

        self.rope_deltas = None
        self._prefill_position_ids = None

        self.token_embedding = None

        # Keep visual_session for ONNX vision inference
        # The re-exported ONNX now includes proper position embeddings and RoPE

        # Free heavy HF model components to reduce memory & speed up
        import gc

        del self.hf_model.visual
        del self.hf_model.model
        del self.hf_model.lm_head
        gc.collect()

    def _set_exec_device(self, device):
        super()._set_exec_device(device)
        if self.prefill_session is not None:
            self.prefill_session.exec_device = device
        if self.decode_session is not None:
            self.decode_session.exec_device = device
        if self.visual_session is not None:
            self.visual_session.exec_device = device
        self.mlp_AR.to(device)

    def _set_device(self, device):
        super()._set_device(device)
        if self.prefill_session is not None:
            self.prefill_session.to(device)
        if self.decode_session is not None:
            self.decode_session.to(device)
        if self.visual_session is not None:
            self.visual_session.to(device)
        if self.token_embedding is not None:
            self.token_embedding.to(device)
        for i in range(self.num_hidden_layers):
            setattr(
                self, f"past_k_cache_{i}", getattr(self, f"past_k_cache_{i}").to(device)
            )
            setattr(
                self, f"past_v_cache_{i}", getattr(self, f"past_v_cache_{i}").to(device)
            )
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        if self.token_embedding is not None:
            self.token_embedding.to(dtype)
        return self

    def set_input_embeddings(self, value):
        self.token_embedding = value
        self.token_embedding.to(torch.float16)

    def _get_past_caches(self):
        past_key_caches, past_value_caches = [], []
        for i in range(self.num_hidden_layers):
            past_key_caches.append(getattr(self, f"past_k_cache_{i}"))
            past_value_caches.append(getattr(self, f"past_v_cache_{i}"))
        return past_key_caches, past_value_caches

    def _vision_forward(self, pixel_values: Tensor, image_grid_thw: Tensor):
        """
        ONNX vision encoder + native mlp_AR to produce image embeddings.

        The ONNX vision model expects a full image [B, 1, C, H, W].
        Processor now outputs full resized image [B, C, H, W] (no patches).

        Input: pixel_values – full image [B, C, H, W] or [B, 1, C, H, W]
        Output: image_embeds after mlp_AR, plus deepstack copies
        """
        # Add temporal dim if needed: [B, C, H, W] → [B, 1, C, H, W]
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)

        # Run ONNX vision model (includes Conv2d patchify + position embeddings + RoPE)
        full_image = pixel_values.to(self.exec_device).half()
        vision_output = self.visual_session(full_image)

        # vision_output is a tuple: (last_hidden_state, pooler_output)
        last_hidden_state = vision_output[0]  # [1, num_patches, embed_dim]

        # Wrap as list for mlp_AR (which expects vision_return_embed_list format)
        image_embeds = [
            last_hidden_state.squeeze(0)
        ]  # list of [num_patches, embed_dim]

        image_embeds = self.mlp_AR(image_embeds, image_grid_thw)
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds

    def _build_inputs_embeds(self, input_ids: Tensor, image_embeds: Tensor):
        input_ids = input_ids.to(self.exec_device)
        seq_length = input_ids.shape[1]
        if self.token_embedding is None:
            raise RuntimeError("token_embedding is not set")
        if seq_length < 1:
            raise RuntimeError("empty input_ids")
        inputs_embeds = self.token_embedding.to(self.exec_device)(input_ids)

        n_image_tokens = torch.sum(input_ids == self.image_token_id).item()
        if n_image_tokens > 0:
            if image_embeds is None or image_embeds.numel() == 0:
                raise ValueError("image tokens present but image_embeds is empty")
            if image_embeds.dim() == 3 and image_embeds.shape[0] == 1:
                image_embeds = image_embeds.squeeze(0)
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"image tokens/features mismatch: tokens={n_image_tokens}, features={n_image_features}"
                )
            image_mask = (
                (input_ids == self.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        return inputs_embeds

    def _get_position_ids(
        self, input_ids: Tensor, image_grid_thw: Tensor, past_seq_length: int
    ):
        if past_seq_length == 0:
            position_ids, rope_deltas = self.hf_model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=None,
            )
            self.rope_deltas = rope_deltas
            self._prefill_position_ids = position_ids
        else:
            if self.rope_deltas is None:
                raise RuntimeError("rope_deltas is None, run prefill first")
            batch_size, seq_length = input_ids.shape
            delta = past_seq_length + self.rope_deltas.to(input_ids.device)
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        time_position_ids = position_ids[0, 0].to(torch.float16)
        hight_position_ids = position_ids[1, 0].to(torch.float16)
        width_position_ids = position_ids[2, 0].to(torch.float16)
        return time_position_ids, hight_position_ids, width_position_ids

    def _get_prefill_position_ids(
        self, input_ids: Tensor, image_grid_thw: Tensor, input_sequence_length: int
    ):
        if self._prefill_position_ids is None or self.rope_deltas is None:
            seq_length = input_ids.shape[1]
            padded_length = (
                (seq_length + input_sequence_length - 1) // input_sequence_length
            ) * input_sequence_length
            if padded_length > seq_length:
                pad_len = padded_length - seq_length
                pad_ids = torch.full(
                    (input_ids.shape[0], pad_len),
                    self.pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                input_ids = torch.cat([input_ids, pad_ids], dim=-1)

            position_ids, rope_deltas = self.hf_model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=None,
            )
            self.rope_deltas = rope_deltas
            self._prefill_position_ids = position_ids
        return self._prefill_position_ids

    def prefill(self, input_ids: Tensor, image_embeds: Tensor, image_grid_thw: Tensor):
        input_sequence_length = self.prefill_config.input_sequence_length
        seq_length = input_ids.shape[1]
        original_seq_length = seq_length  # save the real (un-padded) length
        padded_length = (
            (seq_length + input_sequence_length - 1) // input_sequence_length
        ) * input_sequence_length
        if padded_length > seq_length:
            pad_len = padded_length - seq_length
            pad_ids = torch.full(
                (input_ids.shape[0], pad_len),
                self.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, pad_ids], dim=-1)
        steps = padded_length // input_sequence_length

        past_key_caches, past_value_caches = self._get_past_caches()
        full_position_ids = self._get_prefill_position_ids(
            input_ids, image_grid_thw, input_sequence_length
        )

        # Build inputs_embeds and deepstack for the FULL padded sequence first,
        # then slice per chunk.  This matches the export's test_step flow where
        # masked_scatter distributes ALL image embeddings across the entire
        # sequence before chunking.  The previous per-chunk scatter was wrong:
        # masked_scatter restarts from image_embeds[0] for every chunk, so only
        # the first chunk received correct image features.
        full_inputs_embeds = self._build_inputs_embeds(input_ids, image_embeds)

        # Track cumulative past_seq_length across chunks (like test_step does)
        past_seq_length = 0

        for i in range(steps):
            start = i * input_sequence_length
            end = (i + 1) * input_sequence_length
            # content_length: actual number of real (non-padded) tokens in this chunk
            content_length = min(end, original_seq_length) - start
            pos_chunk = full_position_ids[:, 0, start:end]
            time_pos = pos_chunk[0].to(torch.float16)
            h_pos = pos_chunk[1].to(torch.float16)
            w_pos = pos_chunk[2].to(torch.float16)
            inputs = [
                full_inputs_embeds[:, start:end, :],
                time_pos,
                h_pos,
                w_pos,
                torch.tensor([past_seq_length], dtype=torch.int32).to(self.exec_device),
                torch.tensor([content_length], dtype=torch.int32).to(self.exec_device),
            ]
            inputs += past_key_caches
            inputs += past_value_caches
            output = self.prefill_session(*inputs)

            # Accumulate past_seq_length like test_step does
            # ONNX modifies past_key_caches and past_value_caches in-place
            past_seq_length += content_length

        # Return output logits AND updated KV caches for use in decode loop
        return output, past_key_caches, past_value_caches

    def decode(
        self,
        input_ids: Tensor,
        past_seq_length: int,
        image_grid_thw: Tensor,
        past_key_caches=None,
        past_value_caches=None,
    ):
        # If KV caches not provided, create new empty ones (for first decode call after prefill)
        if past_key_caches is None or past_value_caches is None:
            past_key_caches, past_value_caches = self._get_past_caches()

        inputs_embeds = self._build_inputs_embeds(input_ids, torch.empty(0))
        time_pos, h_pos, w_pos = self._get_position_ids(
            input_ids, image_grid_thw, past_seq_length=past_seq_length
        )
        inputs = [
            inputs_embeds,
            time_pos,
            h_pos,
            w_pos,
            torch.tensor([past_seq_length], dtype=torch.int32).to(self.exec_device),
            torch.tensor([1], dtype=torch.int32).to(self.exec_device),
        ]
        inputs += past_key_caches
        inputs += past_value_caches
        # Return output logits AND updated KV caches for next decode step
        output = self.decode_session(*inputs)
        return output, past_key_caches, past_value_caches

    def chat(
        self,
        prompt: str,
        image_path: str,
        processor: PaddleOCRVLProcessor,
        logger,
        max_new_tokens: int = 64,
        do_sample: bool = False,
    ):
        if image_path is None:
            raise ValueError("image_path is required")
        image_obj = image_path
        if isinstance(image_path, str):
            image_obj = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_obj},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values", None)
        if pixel_values is None and "hm_pixel_values" in inputs:
            pixel_values = inputs["hm_pixel_values"][0]
        if pixel_values is None:
            raise KeyError("pixel_values/hm_pixel_values not found in processor output")
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)
        if (
            pixel_values.dim() == 5
            and pixel_values.shape[2] != 3
            and pixel_values.shape[1] == 3
        ):
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        image_grid_thw = inputs["image_grid_thw"].to(torch.long)
        print(
            f"-------Input ids shape: {input_ids.shape}, pixel_values shape: {pixel_values.shape}, image_grid_thw shape: {image_grid_thw.shape}"
        )

        image_embeds = self._vision_forward(pixel_values, image_grid_thw)

        # Prefill: get logits from the last prefill step AND the accumulated KV caches
        prefill_output, past_key_caches, past_value_caches = self.prefill(
            input_ids, image_embeds, image_grid_thw
        )
        prefill_logits = prefill_output[0]  # [1, 1, vocab_size] (num_logits_to_keep=1)

        next_token_ids, next_token_text = decode_next_token(
            processor.tokenizer, prefill_logits, do_sample=do_sample
        )

        generated = [next_token_ids.item()]
        if logger is not None:
            logger.info(
                f"Prefill next token: {next_token_ids.item()} {next_token_text}"
            )

        # Check EOS after prefill
        if next_token_ids.item() in self.eos_token_id:
            output_text = processor.tokenizer.decode(
                generated, skip_special_tokens=True
            )
            if logger is not None:
                logger.info(output_text)
            return output_text

        past_seq_length = input_ids.shape[1]

        # Decode loop - use the KV caches from prefill
        for _ in range(max_new_tokens - 1):
            output, past_key_caches, past_value_caches = self.decode(
                next_token_ids,
                past_seq_length,
                image_grid_thw,
                past_key_caches,
                past_value_caches,
            )
            logits = output[0]  # Extract logits from output tuple

            next_token_ids, next_token_text = decode_next_token(
                processor.tokenizer, logits, do_sample=do_sample
            )

            generated.append(next_token_ids.item())
            past_seq_length += 1

            if logger is not None:
                logger.info(
                    f"Decode next token: {next_token_ids.item()} {next_token_text}"
                )

            # Print incremental output for progress visibility
            if logger is not None and len(generated) % 10 == 0:
                partial = processor.tokenizer.decode(
                    generated, skip_special_tokens=True
                )
                logger.info(f"Output so far: {partial}")

            # Stop at EOS
            if next_token_ids.item() in self.eos_token_id:
                break

        output_text = processor.tokenizer.decode(generated, skip_special_tokens=True)
        if logger is not None:
            logger.info(f"Final output: {output_text}")
        return output_text
