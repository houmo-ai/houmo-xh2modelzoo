import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import Qwen3OmniMoeForConditionalGeneration

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
from .qwen3_omni_convert_config import Qwen3OmniMoeConvertConfig

from xhquant.api import (  # isort:skip
    CacheTensor,
    Config,
    ConfigDict,
    DeviceType,
    convert_fx_model_to_hmonnx,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
    is_ssfp_quant_config,
)


class Qwen3OmniMoeConverterXH2a(HFTransfromersConverter):
    """Qwen3Omni converter (thinker text path) for XH2a HMONNX export via FX."""

    target_device = DeviceType.XH2a

    def __init__(self, config: Qwen3OmniMoeConvertConfig):
        super().__init__()
        self.config = config

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(hf_model_dir, **kwargs)
        model.eval()
        return model

    def _convert(self, hf_model_path: str, output_dir: str):
        logger = get_root_logger()
        config = self.config

        # ---- 1. Load model ----
        native_model = self.load_hf_model(
            hf_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
            attn_implementation="eager",
        )

        # Extract components for later use
        thinker = native_model.thinker
        audio_tower = thinker.audio_tower if hasattr(thinker, 'audio_tower') else None
        visual = thinker.visual if hasattr(thinker, 'visual') else None
        talker = native_model.talker if hasattr(native_model, 'talker') else None
        token2wav = native_model.token2wav if hasattr(native_model, 'token2wav') else None

        # Don't delete yet - we'll use them for export
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Load quantisation weights if available
        resume_from = config.quant_weight
        if resume_from is not None:
            self.load_quant_weight(resume_from, thinker)

        lm_head = thinker.lm_head
        if not hasattr(lm_head, "quant_weight"):
            config.quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"

        model_name = Path(hf_model_path).name
        target_device = config.quant_scheme.target_device
        batch_size = config.batch_size
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length
        assert target_device == DeviceType.XH2a, f"Only support XH2a, got {target_device}"
        quant_type = config.quant_scheme.quant_type
        quant_config = create_quant_config(config.quant_scheme)
        quant_config = ConfigDict(quant_config)

        # ---- 2. Metadata skeleton ----
        meta_info: Dict[str, Any] = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            device=str(target_device),
            model_name=model_name,
            hf_model_path=hf_model_path,
            quant_scheme=config.quant_scheme.to_dict(),
            quant_weight=resume_from,
        )

        work_dir = Path(output_dir)

        # Copy HF config files
        hf_config_dir = work_dir / "hf_config"
        hf_config_dir.mkdir(exist_ok=True, parents=True)
        for cfg_file in [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "tokenizer.json",
        ]:
            src = Path(hf_model_path) / cfg_file
            if src.exists():
                shutil.copyfile(src, hf_config_dir / cfg_file)
            else:
                logger.warning(f"{src} not exists, skip copy")
        meta_info["hf_config"] = str(hf_config_dir.relative_to(work_dir))

        # Save token embedding
        token_embedding = thinker.model.get_input_embeddings()
        token_embedding_file = work_dir / "token_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file))
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        # ---- 3. Register wrapper modules (side-effect import) ----
        from ._text_model import register_wrap_modules as text_register_wrap_modules
        text_register_wrap_modules()

        # ---- 4. Wrap model for FX tracing ----
        wrap_cfg = Config(
            dict(
                batch_size=batch_size,
                max_sequence_length=context_length,
                input_sequence_length=input_sequence_length,
                use_cache=True,
                num_logits_to_keep=config.num_logits_to_keep,
                kv_cache=dict(cache_axis=2),
            )
        )
        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        wrapped_model = wrap_llm_model(thinker, wrap_cfg)

        # ---- 5. Setup KV cache ----
        num_hidden_layers = wrapped_model.model.config.num_hidden_layers
        head_dim = wrapped_model.model.layers[0].self_attn.head_dim
        num_key_value_heads = wrapped_model.model.config.num_key_value_heads
        num_decoder_layers = num_hidden_layers

        kv_cache_shape = [1, num_key_value_heads, context_length, head_dim]
        meta_info["kv_cache"] = dict(
            shape=kv_cache_shape,
            num_decoder_layers=num_decoder_layers,
        )

        past_key_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            for _ in range(num_decoder_layers)
        ]
        past_value_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            for _ in range(num_decoder_layers)
        ]

        # ---- 6. Prepare prefill inputs ----
        input_ids_t = torch.randint(0, 1000, (1, input_sequence_length), dtype=torch.long)
        inputs_embeds = token_embedding(input_ids_t)
        deepstack_visual_embed_0 = torch.zeros_like(inputs_embeds)
        deepstack_visual_embed_1 = torch.zeros_like(inputs_embeds)
        deepstack_visual_embed_2 = torch.zeros_like(inputs_embeds)

        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor([input_sequence_length], dtype=torch.int32)

        inputs = (
            inputs_embeds,
            past_seq_length_t,
            current_input_length_t,
            deepstack_visual_embed_0,
            deepstack_visual_embed_1,
            deepstack_visual_embed_2,
            past_key_caches,
            past_value_caches,
        )

        input_names = [
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "deepstack_visual_embed_0",
            "deepstack_visual_embed_1",
            "deepstack_visual_embed_2",
        ]
        for i in range(num_decoder_layers):
            input_names.append(f"past_key_cache_{i}")
        for i in range(num_decoder_layers):
            input_names.append(f"past_value_cache_{i}")
        output_names = ["logits"]

        prefix = f"{model_name}-{target_device}-{context_length // 1024}k-{quant_type}"

        # ---- 7. Export prefill HMONNX (FX path) ----
        prefill_onnx_file = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))

        logger.info("Start exporting prefill model (FX path)")
        quanted_model = convert_fx_model_to_quanted_model(
            wrapped_model,
            inputs,
            target_device,
            quant_config=quant_config,
        )

        compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(
            quanted_model, inputs, str(prefill_onnx_file), compatible_names, output_names
        )
        logger.info(f"Export prefill model to {prefill_onnx_file}")

        # ---- 8. Export decode HMONNX ----
        decode_inputs = (
            inputs_embeds[:, :1, :],
            past_seq_length_t,
            torch.ones_like(current_input_length_t),
            deepstack_visual_embed_0[:, :1, :],
            deepstack_visual_embed_1[:, :1, :],
            deepstack_visual_embed_2[:, :1, :],
            past_key_caches,
            past_value_caches,
        )

        wrap_cfg.input_sequence_length = 1
        quanted_model.update_cfg(wrap_cfg)

        decode_onnx_file = work_dir / "hmonnx" / "decode" / f"{prefix}_decoder.onnx"
        decode_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["decode_onnx"] = str(decode_onnx_file.relative_to(work_dir))

        logger.info("Start exporting decode model")
        compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(
            quanted_model, decode_inputs, str(decode_onnx_file), compatible_names, output_names
        )
        logger.info(f"Export decode model to {decode_onnx_file}")

        # ---- 9. Export Audio Encoder (if exists) ----
        if config.export_audio_encoder and audio_tower is not None:
            logger.info("Start exporting audio encoder...")
            audio_meta = self._export_audio_encoder(
                audio_tower, work_dir, meta_info, config, quant_config, model_name, target_device, quant_type
            )
            meta_info.update(audio_meta)
            logger.info("Audio encoder export complete")

        # ---- 10. Export Vision Encoder (if exists) ----
        if config.export_vision_encoder and visual is not None:
            logger.info("Start exporting vision encoder...")
            vision_meta = self._export_vision_encoder(
                visual, work_dir, meta_info, config, quant_config, model_name, target_device, quant_type
            )
            meta_info.update(vision_meta)
            logger.info("Vision encoder export complete")

        # ---- 11. Export Talker LM (if exists) ----
        if config.export_talker_model and talker is not None:
            logger.info("Start exporting talker model...")
            talker_meta = self._export_talker_lm(
                talker, work_dir, meta_info, config, quant_config, model_name, target_device, quant_type
            )
            meta_info.update(talker_meta)
            logger.info("Talker model export complete")

            # ---- 12. Export Talker Prediction (if exists) ----
            code_predictor = getattr(talker, "code_predictor", None)
            if config.export_talker_prediction and code_predictor is not None:
                logger.info("Start exporting talker prediction model...")
                talker_pred_meta = self._export_talker_prediction(
                    code_predictor,
                    work_dir,
                    meta_info,
                    config,
                    quant_config,
                    model_name,
                    target_device,
                    quant_type,
                )
                meta_info.update(talker_pred_meta)
                logger.info("Talker prediction export complete")

        # ---- 13. Save metadata ----
        with open(work_dir / "meta.json", "w") as f:
            json.dump(meta_info, f, indent=4)
        logger.info(f"Conversion complete. Artifacts in {work_dir}")

    def _export_audio_encoder(self, audio_tower, work_dir, meta_info, config, quant_config, 
                             model_name, target_device, quant_type):
        """Export audio encoder module."""
        logger = get_root_logger()
        from ._audio_model import register_wrap_modules as audio_register_wrap_modules
        
        audio_register_wrap_modules()
        
        # Prepare wrapped audio encoder for export to avoid raw HF forward Python control-flow.
        audio_tower = audio_tower.to(torch.float16).cpu()
        wrapped_audio = wrap_llm_model(audio_tower, Config(dict()))

        # After FX trace, only padded_feature and cu_seqlens have graph users
        # (padded_mask_after_cnn is declared in forward but unused)
        batch_size = 1
        mel_bins = int(getattr(audio_tower.config, "num_mel_bins", 128))
        mel_length = 100
        cnn_steps = 13
        dummy_feature = torch.randn(batch_size, mel_bins, mel_length, dtype=torch.float16)
        dummy_cu = torch.tensor([0, cnn_steps], dtype=torch.int32)

        inputs = (dummy_feature, dummy_cu)
        input_names = ["padded_feature", "cu_seqlens"]
        output_names = ["audio_embeds"]
        
        # Export audio encoder HMONNX
        audio_dir = work_dir / "hmonnx" / "audio"
        audio_dir.mkdir(exist_ok=True, parents=True)
        audio_onnx_file = audio_dir / f"{model_name}-{target_device}-audio_encoder.onnx"
        
        try:
            logger.info(f"Exporting audio encoder to {audio_onnx_file}")
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_fx_model_to_hmonnx(
                wrapped_audio,
                inputs,
                target_device,
                audio_onnx_file,
                input_names=compatible_names,
                output_names=output_names,
            )
            logger.info(f"Audio encoder export successful: {audio_onnx_file}")
            return {
                "audio_encoder_onnx": str(audio_onnx_file.relative_to(work_dir)),
                "audio_mel_dim": mel_bins,
                "audio_max_length": mel_length,
                "audio_batch_size": batch_size,
            }
        except Exception as e:
            logger.warning(f"Audio encoder export failed: {e}")
            raise RuntimeError(f"Audio encoder export failed: {e}") from e

    def _export_vision_encoder(self, visual, work_dir, meta_info, config, quant_config,
                              model_name, target_device, quant_type):
        """Export vision encoder module."""
        logger = get_root_logger()
        from ._vision_model import register_wrap_modules as vision_register_wrap_modules
        
        vision_register_wrap_modules()
        
        # Prepare wrapped vision encoder for export to align with existing XH wrapper path.
        visual = visual.to(torch.float16).cpu()
        vision_wrap_cfg = Config(
            dict(
                max_size_w=224,
                max_size_h=224,
                max_size_t=2,
                temporal_patch_size=2,
                patch_size=16,
                only_first_block=False,
            )
        )
        wrapped_visual = wrap_llm_model(visual, vision_wrap_cfg)

        patch_size = 16
        channels = 3
        height = 224
        width = 224
        frames = 2
        dummy_pixels = torch.randn(1, channels, frames, height, width, dtype=torch.float16)

        inputs = (dummy_pixels,)
        input_names = ["pixel_values"]
        output_names = ["vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"]
        
        # Export vision encoder HMONNX
        vision_dir = work_dir / "hmonnx" / "vision"
        vision_dir.mkdir(exist_ok=True, parents=True)
        vision_onnx_file = vision_dir / f"{model_name}-{target_device}-vision_encoder.onnx"
        
        try:
            logger.info(f"Exporting vision encoder to {vision_onnx_file}")
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_fx_model_to_hmonnx(
                wrapped_visual,
                inputs,
                target_device,
                vision_onnx_file,
                input_names=compatible_names,
                output_names=output_names,
            )
            logger.info(f"Vision encoder export successful: {vision_onnx_file}")
            return {
                "vision_encoder_onnx": str(vision_onnx_file.relative_to(work_dir)),
                "vision_patch_size": patch_size,
                "vision_input_size": [height, width],
                "vision_channels": channels,
            }
        except Exception as e:
            logger.warning(f"Vision encoder export failed: {e}")
            raise RuntimeError(f"Vision encoder export failed: {e}") from e

    def _export_talker_lm(self, talker, work_dir, meta_info, config, quant_config,
                         model_name, target_device, quant_type):
        """Export talker LM module (similar to thinker but for audio code generation)."""
        logger = get_root_logger()
        from ._talker_model import register_wrap_modules as talker_register_wrap_modules
        
        talker_register_wrap_modules()
        
        # Prepare talker for export
        talker = talker.to(torch.float16).cpu()
        
        batch_size = config.batch_size
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length
        
        # Get embedding from talker
        talker_embedding = talker.model.get_input_embeddings()
        
        wrap_cfg = Config(
            dict(
                batch_size=batch_size,
                max_sequence_length=context_length,
                input_sequence_length=input_sequence_length,
                use_cache=True,
                num_logits_to_keep=config.num_logits_to_keep,
                kv_cache=dict(cache_axis=2),
            )
        )
        
        wrapped_talker = wrap_llm_model(talker, wrap_cfg)
        
        # Setup KV cache
        num_hidden_layers = wrapped_talker.model.config.num_hidden_layers
        head_dim = wrapped_talker.model.layers[0].self_attn.head_dim
        num_key_value_heads = wrapped_talker.model.config.num_key_value_heads
        
        kv_cache_shape = [1, num_key_value_heads, context_length, head_dim]
        past_key_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            for _ in range(num_hidden_layers)
        ]
        past_value_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            for _ in range(num_hidden_layers)
        ]
        
        # Prepare prefill inputs
        input_ids_t = torch.randint(0, 1000, (1, input_sequence_length), dtype=torch.long)
        inputs_embeds = talker_embedding(input_ids_t)
        
        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor([input_sequence_length], dtype=torch.int32)
        
        inputs = (
            inputs_embeds,
            past_seq_length_t,
            current_input_length_t,
            past_key_caches,
            past_value_caches,
        )
        
        input_names = ["inputs_embeds", "past_seq_length", "current_input_length"]
        for i in range(num_hidden_layers):
            input_names.append(f"past_key_cache_{i}")
        for i in range(num_hidden_layers):
            input_names.append(f"past_value_cache_{i}")
        output_names = ["logits"]
        
        talker_meta = {}
        
        try:
            # Export talker prefill
            talker_dir = work_dir / "hmonnx" / "talker"
            talker_dir.mkdir(exist_ok=True, parents=True)
            talker_prefill_file = talker_dir / f"{model_name}-{target_device}-talker_prefill.onnx"
            
            logger.info(f"Exporting talker prefill to {talker_prefill_file}")
            quanted_talker = convert_fx_model_to_quanted_model(
                wrapped_talker,
                inputs,
                target_device,
                quant_config=quant_config,
            )
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(
                quanted_talker, inputs, str(talker_prefill_file), compatible_names, output_names
            )
            logger.info(f"Talker prefill export successful: {talker_prefill_file}")
            talker_meta["talker_prefill_onnx"] = str(talker_prefill_file.relative_to(work_dir))
            
            # Export talker decode
            decode_inputs = (
                inputs_embeds[:, :1, :],
                past_seq_length_t,
                torch.ones_like(current_input_length_t),
                past_key_caches,
                past_value_caches,
            )
            wrap_cfg.input_sequence_length = 1
            quanted_talker.update_cfg(wrap_cfg)
            
            talker_decode_file = talker_dir / f"{model_name}-{target_device}-talker_decode.onnx"
            logger.info(f"Exporting talker decode to {talker_decode_file}")
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(
                quanted_talker, decode_inputs, str(talker_decode_file), compatible_names, output_names
            )
            logger.info(f"Talker decode export successful: {talker_decode_file}")
            talker_meta["talker_decode_onnx"] = str(talker_decode_file.relative_to(work_dir))
            talker_meta["talker_kv_cache"] = {
                "shape": kv_cache_shape,
                "num_decoder_layers": num_hidden_layers,
            }
            talker_meta["talker_hidden_size"] = int(inputs_embeds.shape[-1])
            talker_meta["talker_input_sequence_length"] = int(input_sequence_length)
            
        except Exception as e:
            logger.warning(f"Talker export failed: {e}")
            raise RuntimeError(f"Talker export failed: {e}") from e

        return talker_meta

    def _export_talker_prediction(
        self,
        talker_prediction,
        work_dir,
        meta_info,
        config,
        quant_config,
        model_name,
        target_device,
        quant_type,
    ):
        """Export talker code-predictor module (prefill/decode)."""
        logger = get_root_logger()
        from ._talker_prediction import register_wrap_modules as talker_prediction_register_wrap_modules

        talker_prediction_register_wrap_modules()

        talker_prediction = talker_prediction.to(torch.float16).cpu()

        batch_size = config.batch_size
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length

        codec_embedding = talker_prediction.model.get_input_embeddings()
        if isinstance(codec_embedding, (list, tuple, nn.ModuleList)):
            codec_embedding = codec_embedding[0]

        wrap_cfg = Config(
            dict(
                batch_size=batch_size,
                max_sequence_length=context_length,
                input_sequence_length=input_sequence_length,
                use_cache=True,
                num_logits_to_keep=config.num_logits_to_keep,
                kv_cache=dict(cache_axis=2),
            )
        )

        wrapped_model = wrap_llm_model(talker_prediction, wrap_cfg)

        num_hidden_layers = wrapped_model.model.config.num_hidden_layers
        head_dim = wrapped_model.model.layers[0].self_attn.head_dim
        num_key_value_heads = wrapped_model.model.config.num_key_value_heads

        kv_cache_shape = [1, num_key_value_heads, context_length, head_dim]
        past_key_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            for _ in range(num_hidden_layers)
        ]
        past_value_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            for _ in range(num_hidden_layers)
        ]

        input_ids_t = torch.randint(0, 1000, (1, input_sequence_length), dtype=torch.long)
        inputs_embeds = codec_embedding(input_ids_t)
        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor([input_sequence_length], dtype=torch.int32)

        inputs = (
            inputs_embeds,
            past_seq_length_t,
            current_input_length_t,
            past_key_caches,
            past_value_caches,
        )

        input_names = ["inputs_embeds", "past_seq_length", "current_input_length"]
        for i in range(num_hidden_layers):
            input_names.append(f"past_key_cache_{i}")
        for i in range(num_hidden_layers):
            input_names.append(f"past_value_cache_{i}")
        output_names = ["logits"]

        meta = {}
        try:
            out_dir = work_dir / "hmonnx" / "talker_prediction"
            out_dir.mkdir(exist_ok=True, parents=True)

            prefill_file = out_dir / f"{model_name}-{target_device}-talker_prediction_prefill.onnx"
            logger.info(f"Exporting talker prediction prefill to {prefill_file}")

            quanted_model = convert_fx_model_to_quanted_model(
                wrapped_model,
                inputs,
                target_device,
                quant_config=quant_config,
            )
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(
                quanted_model,
                inputs,
                str(prefill_file),
                compatible_names,
                output_names,
            )

            decode_inputs = (
                inputs_embeds[:, :1, :],
                past_seq_length_t,
                torch.ones_like(current_input_length_t),
                past_key_caches,
                past_value_caches,
            )
            wrap_cfg.input_sequence_length = 1
            quanted_model.update_cfg(wrap_cfg)

            decode_file = out_dir / f"{model_name}-{target_device}-talker_prediction_decode.onnx"
            logger.info(f"Exporting talker prediction decode to {decode_file}")
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(
                quanted_model,
                decode_inputs,
                str(decode_file),
                compatible_names,
                output_names,
            )

            meta["talker_prediction_prefill_onnx"] = str(prefill_file.relative_to(work_dir))
            meta["talker_prediction_decode_onnx"] = str(decode_file.relative_to(work_dir))
            meta["talker_prediction_kv_cache"] = {
                "shape": kv_cache_shape,
                "num_decoder_layers": num_hidden_layers,
            }
            meta["talker_prediction_hidden_size"] = int(inputs_embeds.shape[-1])
            meta["talker_prediction_input_sequence_length"] = int(input_sequence_length)
        except Exception as e:
            logger.warning(f"Talker prediction export failed: {e}")
            raise RuntimeError(f"Talker prediction export failed: {e}") from e

        return meta

    @classmethod
    def convert(cls, hf_model_path: str, config: Qwen3OmniMoeConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp:
            assert config.quant_weight is not None and Path(config.quant_weight).exists()
        cls(config)._convert(hf_model_path, output_dir)
