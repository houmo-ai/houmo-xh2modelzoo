import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding
from xhquant.api import get_xhquant_logger
from xhquant.common import PrecisionMode

from xhmodel_merak.xh_llm.utils import unfold_args

from ...hmonnx.hmonnx_model import HMONNXModel, HMONNXGolden
from ...hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
from ...kv_cache_mixin import KVCacheMixin
from ...types import KVCacheConfig, LLMModelMeta
from .data_preprocess import Gemma4DataPreprocess, Gemma4InputProcessorConfig, Gemma4PerLayerInputBuilder
from .gemma4_processor import XHGemma4Processor, configure_gemma4_visual_processor


def _cast_hmonnx_int_args(args):
    return [arg.to(torch.int32) if hasattr(arg, "dtype") and arg.dtype == torch.int64 else arg for arg in unfold_args(args)]


class VisualHMONNXModel(HMONNXModel):
    def __init__(self, hmonnx: str, *, export_mode: str = "full", output_scale: float = 1.0):
        super().__init__(hmonnx)
        self.export_mode = export_mode
        self.output_scale = float(output_scale)

    @staticmethod
    def _patch_visual_session_precision(graph_session) -> int:
        node_modules = getattr(graph_session, "node_modules", None)
        if node_modules is None:
            return 0

        patched = 0
        for module in node_modules:
            op_type = getattr(module, "op_type", None)
            if op_type == "ReduceSum":
                module.precision_mode = PrecisionMode.FAST
                patched += 1
            elif op_type == "OneHot":
                module.precision_mode = PrecisionMode.ALIGNED
        return patched

    def forward(self, *args):
        if getattr(self, "export_mode", "full") == "compact" and len(args) > 1:
            args = args[:1]
        args = _cast_hmonnx_int_args(args)
        self.hmonnx_session._set_session_env()
        graph_session = getattr(self.hmonnx_session, "_session", None)
        if graph_session is None:
            raise RuntimeError("Gemma4 visual HMONNX session failed to initialize.")
        self._patch_visual_session_precision(graph_session)
        expected_inputs = len(graph_session.inputs)
        if len(args) > expected_inputs:
            args = args[:expected_inputs]
        out = graph_session.forward(*args)
        if isinstance(out, (tuple, list)) and len(out) == 1:
            out = out[0]
        output_scale = getattr(self, "output_scale", 1.0)
        if output_scale != 1.0:
            if isinstance(out, tuple):
                return (out[0] * output_scale, *out[1:])
            if isinstance(out, list):
                return [out[0] * output_scale, *out[1:]]
            return out * output_scale
        return out


class AudioHMONNXModel(HMONNXModel):
    def __init__(self, hmonnx: str, input_feature_length: int = 0):
        self._enable_cuda_graph = False
        self.hmonnx_session = HMONNXGolden(hmonnx)
        self._enable_auto_offload = False
        self._dtype = torch.float16
        self._device = torch.device("cpu")
        self._enable_golden = False
        self.input_feature_length = input_feature_length

    def forward(self, *args):
        args = list(args)
        if self.input_feature_length > 0 and len(args) >= 2:
            input_features, input_features_mask = args[0], args[1]
            seq_len = input_features.shape[1]
            target_len = self.input_feature_length
            if seq_len > target_len:
                input_features = input_features[:, :target_len, :]
                input_features_mask = input_features_mask[:, :target_len]
            elif seq_len < target_len:
                pad_len = target_len - seq_len
                input_features = torch.nn.functional.pad(input_features, (0, 0, 0, pad_len))
                input_features_mask = torch.nn.functional.pad(input_features_mask, (0, pad_len), value=0)
            args[0], args[1] = input_features, input_features_mask
        out = super().forward(*_cast_hmonnx_int_args(args))
        if isinstance(out, (tuple, list)) and len(out) == 1:
            return out[0]
        return out


class AudioONNXModel:
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self._dtype = torch.float16
        self._device = torch.device("cpu")
        self._session = None
        self._session_providers = None

    def _set_device(self, device):
        self._device = torch.device(device)
        self._session = None
        self._session_providers = None
        return self

    def _set_dtype(self, dtype):
        self._dtype = dtype
        return self

    def to(self, device):
        return self._set_device(device)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def _get_providers(self):
        providers: list = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if self._device.type == "cuda" and "CUDAExecutionProvider" in available:
            device_id = self._device.index or 0
            providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        return providers

    def _get_session(self):
        providers = self._get_providers()
        if self._session is None or self._session_providers != providers:
            self._session = ort.InferenceSession(self.onnx_path, providers=providers)
            self._session_providers = providers
        return self._session

    def forward(self, *args):
        input_features, input_features_mask = _cast_hmonnx_int_args(args)
        session = self._get_session()
        outputs = session.run(
            None,
            {
                "input_features": input_features.detach().float().cpu().numpy().astype(np.float32, copy=False),
                "input_features_mask": input_features_mask.detach().cpu().numpy(),
            },
        )
        audio_embeds = torch.from_numpy(outputs[0]).to(self._device, self._dtype)
        audio_embeds_mask = torch.from_numpy(outputs[1]).to(self._device)
        return audio_embeds, audio_embeds_mask


class AudioPyTorchModel:
    """PyTorch-based audio model that supports dynamic input length."""

    def __init__(self, hf_model_dir: str):
        from transformers import AutoModelForImageTextToText

        hf_model = AutoModelForImageTextToText.from_pretrained(
            hf_model_dir, dtype=torch.float16, device_map="cpu"
        )
        self._audio_tower = hf_model.model.audio_tower.eval()
        self._embed_audio = hf_model.model.embed_audio.eval()
        del hf_model.model.language_model
        del hf_model.lm_head
        del hf_model
        self._dtype = torch.float16
        self._device = torch.device("cpu")

    def _set_device(self, device):
        self._device = torch.device(device)
        self._audio_tower.to(self._device)
        self._embed_audio.to(self._device)
        return self

    def _set_dtype(self, dtype):
        self._dtype = dtype
        return self

    def to(self, device):
        return self._set_device(device)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @torch.no_grad()
    def forward(self, input_features, input_features_mask):
        audio_outputs = self._audio_tower(
            input_features=input_features,
            attention_mask=input_features_mask,
        )
        if hasattr(audio_outputs, "last_hidden_state"):
            hidden_states = audio_outputs.last_hidden_state
            output_mask = getattr(audio_outputs, "attention_mask", None)
        elif isinstance(audio_outputs, (tuple, list)):
            hidden_states = audio_outputs[0]
            output_mask = audio_outputs[1] if len(audio_outputs) > 1 else None
        else:
            hidden_states = audio_outputs
            output_mask = None

        audio_embeds = self._embed_audio(inputs_embeds=hidden_states)
        if isinstance(audio_embeds, (tuple, list)):
            audio_embeds = audio_embeds[0]

        if output_mask is None:
            output_mask = torch.ones(
                audio_embeds.shape[:2], dtype=torch.bool, device=audio_embeds.device
            )
        return audio_embeds, output_mask


class VisualONNXModel:
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self._dtype = torch.float16
        self._device = torch.device("cpu")
        self._session = None
        self._session_providers = None

    def _set_device(self, device):
        self._device = torch.device(device)
        self._session = None
        self._session_providers = None
        return self

    def _set_dtype(self, dtype):
        self._dtype = dtype
        return self

    def to(self, device):
        return self._set_device(device)

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def _get_providers(self):
        providers: list = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if self._device.type == "cuda" and "CUDAExecutionProvider" in available:
            device_id = self._device.index or 0
            providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        return providers

    def _get_session(self):
        providers = self._get_providers()
        if self._session is None or self._session_providers != providers:
            self._session = ort.InferenceSession(self.onnx_path, providers=providers)
            self._session_providers = providers
        return self._session

    def forward(self, *args):
        session = self._get_session()
        input_metas = session.get_inputs()
        if len(args) < len(input_metas):
            raise ValueError(f"Gemma4 visual ONNX expects {len(input_metas)} inputs, got {len(args)}.")

        feeds = {}
        for value, input_meta in zip(args, input_metas):
            array = value.detach().cpu().numpy()
            if "float" in input_meta.type:
                array = array.astype(np.float32, copy=False)
            elif input_meta.type == "tensor(int32)" and array.dtype == np.int64:
                array = array.astype(np.int32, copy=False)
            feeds[input_meta.name] = array

        outputs = []
        for output in session.run(None, feeds):
            tensor = torch.from_numpy(output)
            if tensor.is_floating_point():
                tensor = tensor.to(self._device, self._dtype)
            else:
                tensor = tensor.to(self._device)
            outputs.append(tensor)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class Gemma4KVCacheMixinHMONNX(KVCacheMixin):
    def __init__(self, kv_cache_config: KVCacheConfig, layer_kv_shapes: list[list[int]]):
        super().__init__(kv_cache_config)
        self.layer_kv_shapes = layer_kv_shapes

    def prepare_kv_cache(self, dtype=torch.float16):
        if not self.use_cache:
            return
        self.past_key_caches.clear()
        self.past_value_caches.clear()
        for shape in self.layer_kv_shapes:
            self.past_key_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(shape, dtype=dtype)))
            self.past_value_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(shape, dtype=dtype)))


class XHGemma4_HMONNXModel(VisonLLMHMONNXModel):
    @staticmethod
    def _get_path_str(value) -> str | None:
        if isinstance(value, (str, os.PathLike)):
            return os.fspath(value)
        return None

    @staticmethod
    def _build_audio_runtime(audio_meta, hf_model_dir=None):
        force_backend = (os.environ.get("XHMODEL_GEMMA4_AUDIO_BACKEND") or "").lower()
        if force_backend == "pytorch":
            if hf_model_dir is None or not Path(hf_model_dir).exists():
                raise ValueError(
                    "XHMODEL_GEMMA4_AUDIO_BACKEND=pytorch requires a valid hf_model directory; "
                    f"got {hf_model_dir!r}."
                )
            get_xhquant_logger().warning(
                "Forcing Gemma4 audio runtime to PyTorch (XHMODEL_GEMMA4_AUDIO_BACKEND=pytorch)."
            )
            return AudioPyTorchModel(hf_model_dir)
        hmonnx_path = XHGemma4_HMONNXModel._get_path_str(getattr(audio_meta, "hmonnx", None))
        if force_backend != "onnx" and hmonnx_path is not None:
            if not Path(hmonnx_path).exists():
                raise FileNotFoundError(f"Gemma4 audio HMONNX artifact not found: {hmonnx_path}")
            input_feature_length = int(getattr(audio_meta, "input_feature_length", 0) or 0)
            return AudioHMONNXModel(hmonnx_path, input_feature_length=input_feature_length)
        onnx_path = XHGemma4_HMONNXModel._get_path_str(getattr(audio_meta, "onnx", None))
        if onnx_path is not None:
            if not Path(onnx_path).exists():
                raise FileNotFoundError(f"Gemma4 audio ONNX artifact not found: {onnx_path}")
            return AudioONNXModel(onnx_path)
        if hf_model_dir is not None and Path(hf_model_dir).exists():
            return AudioPyTorchModel(hf_model_dir)
        raise ValueError("Gemma4 audio runtime requires an exported HMONNX or ONNX artifact, or a valid HF model directory.")

    @staticmethod
    def _build_visual_runtime(visual_meta):
        export_mode = getattr(visual_meta, "export_mode", "full")
        hmonnx_path = XHGemma4_HMONNXModel._get_path_str(getattr(visual_meta, "hmonnx", None))
        if hmonnx_path is not None:
            if not Path(hmonnx_path).exists():
                raise FileNotFoundError(f"Gemma4 visual HMONNX artifact not found: {hmonnx_path}")
            output_scale = float(getattr(visual_meta, "output_scale", 1.0) or 1.0)
            return VisualHMONNXModel(hmonnx_path, export_mode=export_mode, output_scale=output_scale)
        raise ValueError("Gemma4 visual runtime requires an exported HMONNX artifact.")

    @staticmethod
    def _disable_kvcache_fast_mode(hmonnx_model: HMONNXModel) -> int:
        hmonnx_session = hmonnx_model.hmonnx_session
        hmonnx_session.initialize()
        graph_session = getattr(hmonnx_session, "_session", None)
        node_modules = getattr(graph_session, "node_modules", None)
        if node_modules is None:
            return 0

        patched = 0
        for module in node_modules:
            if getattr(module, "op_type", None) != "KVcache" or not hasattr(module, "fast_mode"):
                continue
            module.fast_mode = False
            patched += 1
        return patched

    @staticmethod
    def _build_embed_tokens_from_meta(meta: LLMModelMeta) -> Gemma4TextScaledWordEmbedding:
        quant_embedding_path = Path(meta.quant_embedding)
        try:
            loaded = torch.load(str(quant_embedding_path), map_location="cpu", weights_only=True)
        except Exception:
            loaded = torch.load(str(quant_embedding_path), map_location="cpu", weights_only=False)

        if isinstance(loaded, Gemma4TextScaledWordEmbedding):
            return loaded

        state_dict = loaded
        text_config = AutoConfig.from_pretrained(meta.hf_config, trust_remote_code=True).get_text_config()
        embed_tokens = Gemma4TextScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            text_config.pad_token_id,
            embed_scale=text_config.hidden_size**0.5,
        )
        sample_tensor = next(iter(state_dict.values()))
        embed_tokens.to(dtype=sample_tensor.dtype)
        embed_tokens.load_state_dict(state_dict)

        logger = get_xhquant_logger()
        if quant_embedding_path.exists():
            logger.info(f"Loaded quantized embedding from {quant_embedding_path}")
        else:
            logger.warning(f"Quantized embedding file not found: {quant_embedding_path}")
        return embed_tokens

    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        super().__init__(meta_info, **kwargs)
        self.visual_meta = getattr(meta_info, "visual_config", None)
        self.audio_meta = getattr(meta_info, "audio_config", None)
        self.visual = self._build_visual_runtime(self.visual_meta) if self.visual_meta else None
        hf_model_path = getattr(meta_info.model_config, "hf_model", None) or self.hf_model_dir
        self.audio = self._build_audio_runtime(self.audio_meta, hf_model_path) if self.audio_meta else None
        layer_kv_shapes = getattr(meta_info, "layer_kv_shapes", [])
        if layer_kv_shapes:
            self._kvcache_mixin = Gemma4KVCacheMixinHMONNX(self.kvcache_config, layer_kv_shapes)
        self.per_layer_input_builder: Gemma4PerLayerInputBuilder | None = None
        self._aligned_text_cache_modes: set[str] = set()

    def _ensure_text_cache_alignment(self, mode: str, hmonnx_model: HMONNXModel) -> None:
        if mode in self._aligned_text_cache_modes:
            return
        patched = self._disable_kvcache_fast_mode(hmonnx_model)
        logger = get_xhquant_logger()
        logger.info(f"Gemma4 runtime disabled fast KVcache mode for {patched} {mode} text nodes")
        self._aligned_text_cache_modes.add(mode)

    def _set_device(self, device):
        super()._set_device(device)
        if self.visual is not None:
            self.visual.to(device)
        if self.audio is not None:
            self.audio.to(device)
        if self.per_layer_input_builder is not None:
            self.per_layer_input_builder.to(device=device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        if self.visual is not None:
            self.visual._set_dtype(dtype)
        if self.audio is not None:
            self.audio._set_dtype(dtype)
        if self.per_layer_input_builder is not None:
            self.per_layer_input_builder.to(dtype=dtype)
        return self

    def set_prefill(self):
        super().set_prefill()
        self._ensure_text_cache_alignment("prefill", self.prefill_model)

    def set_decode(self):
        super().set_decode()
        self._ensure_text_cache_alignment("decode", self.decode_model)

    def get_tf_processor(self):
        processor = XHGemma4Processor.from_pretrained(self.hf_model_dir)
        model_config = self.meta_info.model_config
        if getattr(model_config, "visual_config", None) is not None:
            processor = configure_gemma4_visual_processor(
                processor,
                export_mode=getattr(model_config.visual_config, "export_mode", "full"),
                max_size_w=model_config.visual_config.max_size_w,
                max_size_h=model_config.visual_config.max_size_h,
                patch_size=model_config.visual_config.patch_size,
                image_seq_length=model_config.visual_config.image_seq_length,
            )
        if getattr(model_config, "audio_config", None) is not None:
            processor.config.sampling_rate = model_config.audio_config.sampling_rate
            input_feature_length = int(getattr(model_config.audio_config, "input_feature_length", 0) or 0)
            if input_feature_length > 0:
                processor.config.audio_feature_length = input_feature_length
        return processor

    def forward(self, *args):
        if self.is_prefill():
            self._ensure_text_cache_alignment("prefill", self.prefill_model)
        else:
            self._ensure_text_cache_alignment("decode", self.decode_model)
        return super().forward(*_cast_hmonnx_int_args(args))

    def _get_data_preprocessor(self) -> Gemma4DataPreprocess:
        text_config = AutoConfig.from_pretrained(self.hf_model_dir).get_text_config()
        if getattr(text_config, "hidden_size_per_layer_input", 0) and self.per_layer_input_builder is None:
            artifact_path = getattr(self.meta_info, "per_layer_input_builder", None)
            if not artifact_path:
                raise ValueError("Gemma4 HMONNX runtime requires per_layer_input_builder artifact.")
            self.per_layer_input_builder = Gemma4PerLayerInputBuilder.from_artifact(artifact_path, text_config)
        config = Gemma4InputProcessorConfig(
            embed_tokens=self.get_input_embeddings(),
            input_sequence_length=self.get_input_sequence_length(),
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            per_layer_input_builder=self.per_layer_input_builder,
            pad_token_id=self.pad_token_id,
            context_max_length=self.meta_info.model_config.context_max_length,
            sliding_window=text_config.sliding_window,
            use_explicit_attention_mask=getattr(self.meta_info.model_config, "use_explicit_attention_mask", True),
            image_token_id=getattr(self.meta_info.model_config, "image_token_id", -1),
            audio_token_id=getattr(self.meta_info.model_config, "audio_token_id", -1),
            video_token_id=getattr(self.meta_info.model_config, "video_token_id", -1),
        )
        return Gemma4DataPreprocess(config)
