import torch
from torch import Tensor

from xhquant.core import CacheTensor

from ...builder import register_llm_model
from .minicpmo_base_model import XHMiniCPMOBaseModel


@register_llm_model("MiniCPMO45AudioModel", master=False)
class XHMiniCPMOAudioModel(XHMiniCPMOBaseModel):
    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type,
        allow_quant=True,
        export_cfg=None,
    ):
        super().__init__(
            hf_model,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            frontend_type=frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )
        self.streaming = False

    def get_hf_model(self, device_map="cpu", **kwargs):
        hf_model = super().get_hf_model(device_map=device_map, **kwargs)
        self.audio_encoder_layer = hf_model.audio_encoder_layer
        self.audio_projection_layer = hf_model.audio_projection_layer
        self.audio_avg_pooler = hf_model.audio_avg_pooler
        return hf_model

    def init_wrap_model(
        self,
        hf_model=None,
        *,
        streaming: bool = False,
        prefix_extra_frames: int = 0,
        suffix_extra_frames: int = 0,
        input_frame_capacity: int | None = None,
    ):
        if hf_model is None:
            hf_model = self.get_hf_model()
        from ._audio_model_impl import register_wrap_cls as audio_register_wrap_cls  # noqa F401

        audio_register_wrap_cls(hf_model)
        apm = hf_model.apm

        wraped_apm = self._init_wrap_model_with_llm_registry(apm)
        wraped_apm.audio_projection_layer = hf_model.audio_projection_layer
        wraped_apm.audio_encoder_layer = hf_model.audio_encoder_layer
        wraped_apm.audio_avg_pooler = hf_model.audio_avg_pooler
        self.streaming = streaming
        if streaming:
            from ._audio_streaming import StreamingAudioExportAdapter

            if input_frame_capacity is None:
                raise ValueError("Streaming Audio export requires input_frame_capacity")
            config = hf_model.apm.config
            num_layers = int(config.encoder_layers)
            head_dim = int(config.d_model) // int(config.encoder_attention_heads)
            cache_capacity = int(self.wrap_cfg.cache_capacity)
            cache_shape = (1, int(config.encoder_attention_heads), cache_capacity, head_dim)
            self.past_key_caches = [
                CacheTensor(torch.zeros(cache_shape, dtype=torch.float16)) for _ in range(num_layers)
            ]
            self.past_value_caches = [
                CacheTensor(torch.zeros(cache_shape, dtype=torch.float16)) for _ in range(num_layers)
            ]
            wraped_apm.streaming_prefix_extra_frames = prefix_extra_frames
            wraped_apm.streaming_suffix_extra_frames = suffix_extra_frames
            wraped_apm.streaming_output_capacity = (
                (input_frame_capacity + 1) // 2 - (prefix_extra_frames + 1) // 2 - (suffix_extra_frames + 1) // 2
            )
            wraped_apm.streaming_cache_capacity = cache_capacity
            self._wrap_model = StreamingAudioExportAdapter(wraped_apm, num_layers)
        return self._wrap_model

    def _normalize_audio_output(self, output) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if getattr(output, "hidden_states", None) is not None:
            audio_states = output.hidden_states[self.audio_encoder_layer]
        elif getattr(output, "last_hidden_state", None) is not None:
            audio_states = output.last_hidden_state
        elif isinstance(output, (tuple, list)) and output:
            audio_states = output[0]
        else:
            raise TypeError(f"Unsupported audio encoder output: {type(output)}")

        audio_embeds = self.audio_projection_layer(audio_states)
        audio_embeds = self.audio_avg_pooler(audio_embeds.transpose(1, 2))
        return audio_embeds.transpose(1, 2)

    def _set_device(self, device: torch.device) -> None:
        super()._set_device(device)

    def prepare_inputs_for_graph(self, data) -> tuple[Tensor, ...]:
        if self.streaming:
            inputs = self.prepare_inputs(data)
            return (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], *inputs[5], *inputs[6])
        return self.prepare_inputs(data)

    def prepare_inputs(self, data) -> tuple[Tensor, Tensor]:
        if self.streaming:
            execution_device = self.__dict__.get("_exec_device") or self.__dict__.get("_device", torch.device("cpu"))
            return (
                data["input_features"].to(execution_device),
                torch.tensor([int(data["valid_mel_length"])], dtype=torch.int32, device=execution_device),
                torch.tensor([int(data["past_seq_length"])], dtype=torch.int32, device=execution_device),
                torch.tensor([int(data["current_input_length"])], dtype=torch.int32, device=execution_device),
                data["attention_mask"].to(execution_device),
                self.past_key_caches,
                self.past_value_caches,
            )
        if isinstance(data, dict):
            input_features = data["input_features"]
            audio_attention_mask = data["audio_attention_mask"]
        else:
            input_features, audio_attention_mask = data

        execution_device = self.__dict__.get("_exec_device") or self.__dict__.get("_device", input_features.device)
        return (input_features.to(execution_device), audio_attention_mask.to(execution_device))

    def _forward(
        self,
        input_features: Tensor,
        audio_attention_mask: Tensor,
        **kwargs,
    ) -> Tensor:
        output = self(input_features, audio_attention_mask)
        if self.streaming:
            return output
        return self._normalize_audio_output(output)
