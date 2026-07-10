import copy
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult


_DEFAULT_TEXT = "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地"


class Qwen3TTSWorkflow(BaseOtherModelWorkflow):
    SUPPORTED_COMPONENTS = {
        "talker",
        "code_predictor",
        "text_projection",
        "speech_tokenizer",
        "base_frontend",
        "stateful_decoder",
    }

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise NotImplementedError(f"{type(self).__name__} does not support quantization")
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        from xhquant.api import set_random_seed

        set_random_seed(self.seed)
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_model_dir = self._resolve_export_model_dir(quant_result)
        export_cfg = workflow_config.build_export_dict()
        target_device = str(export_cfg.get("target_device", "XH2a"))
        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))

        components = _normalize_components(export_cfg.get("components"))
        unsupported = [name for name in components if name not in self.SUPPORTED_COMPONENTS]
        if unsupported:
            raise ValueError(f"Unsupported Qwen3-TTS component(s): {unsupported}")

        runtime_cfg = _build_runtime_cfg(export_cfg, export_model_dir, device, target_device, config_file, work_dir)
        quant_cfg = export_cfg.get("quant_types") or {}
        if not isinstance(quant_cfg, Mapping):
            raise TypeError("Qwen3-TTS export.quant_types must be a mapping when provided")

        meta: dict[str, Any] = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": str(Path(config_file).relative_to(work_dir)),
            "hf_model": export_model_dir,
            "target_device": target_device,
            "tts_mode": runtime_cfg["tts_mode"],
            "components": {},
        }
        for key in ("tts_text", "tts_speaker", "tts_instruct", "ref_audio", "ref_text"):
            if key in runtime_cfg:
                meta[key] = runtime_cfg[key]

        if "talker" in components:
            talker_cfg = copy.deepcopy(export_cfg["model"])
            talker_cfg["hf_model"] = export_model_dir
            _ensure_quant_config(talker_cfg, target_device, str(quant_cfg.get("talker", "w8a8h1_sefp")))
            component_dir = work_dir / "Talker"
            result = export_qwen3_tts_talker(
                model_cfg=talker_cfg,
                runtime_cfg=runtime_cfg,
                component_dir=component_dir,
                target_device=target_device,
                quant_type=str(quant_cfg.get("talker", "w8a8h1_sefp")),
            )
            meta["components"]["talker"] = _component_meta(work_dir, result)

        if "code_predictor" in components:
            model_cfg = copy.deepcopy(export_cfg["code_predictor_model"])
            model_cfg["hf_model"] = export_model_dir
            _ensure_quant_config(model_cfg, target_device, str(quant_cfg.get("code_predictor", "w8a8h1_sefp")))
            component_dir = work_dir / "CodePredictor"
            result = export_qwen3_tts_code_predictor(
                model_cfg=model_cfg,
                runtime_cfg=runtime_cfg,
                component_dir=component_dir,
                target_device=target_device,
                quant_type=str(quant_cfg.get("code_predictor", "w8a8h1_sefp")),
            )
            meta["components"]["code_predictor"] = _component_meta(work_dir, result)

        if "text_projection" in components:
            component_dir = work_dir / "TextProjection"
            result = export_qwen3_tts_text_projection(
                runtime_cfg=runtime_cfg,
                component_dir=component_dir,
                target_device=target_device,
                quant_type=str(quant_cfg.get("text_projection", "w8a8h1_sefp")),
            )
            meta["components"]["text_projection"] = _component_meta(work_dir, result)

        if "speech_tokenizer" in components:
            component_dir = work_dir / "SpeechTokenizer"
            result = export_qwen3_tts_speech_tokenizer(
                runtime_cfg=runtime_cfg,
                component_dir=component_dir,
                target_device=target_device,
                quant_type=str(quant_cfg.get("speech_tokenizer", "w8a8h1_sefp")),
                component_cfg=export_cfg.get("speech_tokenizer") or {},
            )
            meta["components"]["speech_tokenizer"] = _component_meta(work_dir, result)

        if "base_frontend" in components:
            component_dir = work_dir / "BaseFrontend"
            result = export_qwen3_tts_base_frontend(
                runtime_cfg=runtime_cfg,
                component_dir=component_dir,
                target_device=target_device,
                quant_type=str(quant_cfg.get("base_frontend", "w8a8h1_sefp")),
                component_cfg=export_cfg.get("base_frontend") or {},
            )
            meta["components"]["base_frontend"] = _component_meta(work_dir, result)

        if "stateful_decoder" in components:
            component_dir = work_dir / "StatefulDecoder"
            result = export_qwen3_tts_stateful_decoder(
                runtime_cfg=runtime_cfg,
                component_dir=component_dir,
                target_device=target_device,
                quant_type=str(quant_cfg.get("stateful_decoder", "w8a8h1_sefp")),
                component_cfg=export_cfg.get("stateful_decoder") or {},
            )
            meta["components"]["stateful_decoder"] = _component_meta(work_dir, result)

        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(json.dumps(_jsonable(meta), indent=4, ensure_ascii=False), encoding="utf-8")
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        import torch
        from xhquant.core import CacheTensor

        work_dir = Path(export_result.work_dir)
        meta = json.loads((work_dir / "export_meta_info.json").read_text(encoding="utf-8"))
        torch_device = _select_torch_device(device)
        golden_meta: dict[str, Any] = {
            "work_dir": str(work_dir),
            "device": torch_device,
            "input_messages": repr(input_messages),
            "components": {},
        }

        for name in ("talker", "code_predictor"):
            comp = meta["components"].get(name)
            if not comp:
                continue
            comp_dir = work_dir / comp["component_dir"]
            comp_meta = json.loads((work_dir / comp["meta_file"]).read_text(encoding="utf-8"))
            token_state = torch.load(comp_dir / comp_meta["token_embedding_file"], map_location="cpu")
            state_key = "weight" if "weight" in token_state else "0.weight"
            hidden_size = int(token_state[state_key].shape[-1])
            input_len = int(comp_meta["wrap_cfg"]["input_sequence_length"])
            kv_shape = tuple(int(dim) for dim in comp_meta["kv_cache_shape"])
            layers = int(comp_meta["num_hidden_layers"])
            generate_steps = name == "code_predictor"
            prefill_path = comp_dir / comp_meta["prefill_onnx_file"]
            prefill_inputs = _build_llm_hmonnx_inputs(
                torch.randn((1, input_len, hidden_size), dtype=torch.float16, device=torch_device),
                past_seq_length=0,
                current_input_length=input_len,
                kv_cache_shape=kv_shape,
                num_hidden_layers=layers,
                torch_device=torch_device,
                cache_tensor_cls=CacheTensor,
                generate_steps=0 if generate_steps else None,
            )
            prefill_golden = prefill_path.parent / "golden"
            _run_hmonnx_golden(prefill_path, prefill_golden, torch_device, prefill_inputs)

            decode_path = comp_dir / comp_meta["decode_onnx_file"]
            decode_inputs = _build_llm_hmonnx_inputs(
                torch.randn((1, 1, hidden_size), dtype=torch.float16, device=torch_device),
                past_seq_length=input_len,
                current_input_length=1,
                kv_cache_shape=kv_shape,
                num_hidden_layers=layers,
                torch_device=torch_device,
                cache_tensor_cls=CacheTensor,
                generate_steps=0 if generate_steps else None,
            )
            decode_golden = decode_path.parent / "golden"
            _run_hmonnx_golden(decode_path, decode_golden, torch_device, decode_inputs)
            golden_meta["components"][name] = {
                "prefill_golden_dir": str(prefill_golden),
                "decode_golden_dir": str(decode_golden),
            }

        simple_inputs = {
            "text_projection": lambda comp_meta: (
                [torch.randn((1, 1, int(comp_meta["feature_dim"])), dtype=torch.float16, device=torch_device)]
            ),
            "speech_tokenizer": lambda comp_meta: (
                [torch.randint(0, 100, tuple(comp_meta["input_shape"]), dtype=torch.int32, device=torch_device)]
            ),
        }
        for name, input_builder in simple_inputs.items():
            comp = meta["components"].get(name)
            if not comp:
                continue
            comp_dir = work_dir / comp["component_dir"]
            comp_meta = json.loads((work_dir / comp["meta_file"]).read_text(encoding="utf-8"))
            hmonnx_key = "hmonnx"
            hmonnx_file = comp_dir / comp_meta[hmonnx_key]
            golden_dir = hmonnx_file.parent / "golden"
            _run_hmonnx_golden(hmonnx_file, golden_dir, torch_device, input_builder(comp_meta))
            golden_meta["components"][name] = {"golden_dir": str(golden_dir)}

        comp = meta["components"].get("base_frontend")
        if comp:
            comp_dir = work_dir / comp["component_dir"]
            comp_meta = json.loads((work_dir / comp["meta_file"]).read_text(encoding="utf-8"))
            batch_size = int(comp_meta["batch_size"])
            audio_samples = int(comp_meta["audio_samples"])
            mel_frames = int(comp_meta["mel_frames"])
            mel_dim = int(comp_meta["mel_dim"])

            encode_hmonnx = comp_dir / comp_meta["speech_tokenizer_encode_hmonnx"]
            encode_golden = comp_dir / "golden" / "speech_tokenizer_encode"
            _run_hmonnx_golden(
                encode_hmonnx,
                encode_golden,
                torch_device,
                [
                    torch.zeros((batch_size, audio_samples), dtype=torch.float16, device=torch_device),
                    torch.ones((batch_size, audio_samples), dtype=torch.int32, device=torch_device),
                ],
            )

            speaker_hmonnx = comp_dir / comp_meta["speaker_encoder_hmonnx"]
            speaker_golden = comp_dir / "golden" / "speaker_encoder"
            _run_hmonnx_golden(
                speaker_hmonnx,
                speaker_golden,
                torch_device,
                [torch.zeros((batch_size, mel_frames, mel_dim), dtype=torch.float16, device=torch_device)],
            )
            golden_meta["components"]["base_frontend"] = {
                "speech_tokenizer_encode_golden_dir": str(encode_golden),
                "speaker_encoder_golden_dir": str(speaker_golden),
            }

        comp = meta["components"].get("stateful_decoder")
        if comp:
            comp_dir = work_dir / comp["component_dir"]
            comp_meta = json.loads((work_dir / comp["meta_file"]).read_text(encoding="utf-8"))
            stateful_hmonnx = comp_dir / comp_meta["stateful_hmonnx"]
            stateful_golden = comp_dir / "golden" / "stateful_decoder"
            _run_hmonnx_golden(
                stateful_hmonnx,
                stateful_golden,
                torch_device,
                _build_stateful_decoder_hmonnx_inputs(comp_meta, torch_device),
            )
            golden_meta["components"]["stateful_decoder"] = {"stateful_golden_dir": str(stateful_golden)}

        golden_meta_file = work_dir / "golden_meta_info.json"
        golden_meta_file.write_text(json.dumps(_jsonable(golden_meta), indent=4, ensure_ascii=False), encoding="utf-8")
        return str(golden_meta_file)


def export_qwen3_tts_talker(
    *,
    model_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    component_dir: Path,
    target_device: str,
    quant_type: str,
) -> dict[str, Any]:
    import torch
    from xhquant.api import ConfigDict, PrecisionMode, get_root_logger, ptq_quantize
    from xhquant.utils.time_profiler import time_profiler

    from xhmodel_merak.xh_other_model.builder import MODELS
    from xhmodel_merak.xh_other_model.eval_model_type import EvalModelType

    from . import XHQwen3TTSModel, XHQwen3TTSTalker

    component_dir.mkdir(parents=True, exist_ok=True)
    cfg_name = f"{runtime_cfg['model_name']}_talker_{target_device}_{quant_type}"
    cfg = ConfigDict(dict(runtime_cfg))
    cfg.model = ConfigDict(copy.deepcopy(dict(model_cfg)))
    logger = get_root_logger()
    dtype = getattr(torch, cfg.dtype)
    exec_device = torch.device(cfg.exec_device)

    xh_model = MODELS.build(cfg.model)
    assert isinstance(xh_model, XHQwen3TTSTalker)
    hf_model = xh_model.get_hf_model(device_map="cpu", dtype=torch.float16)
    assert isinstance(hf_model, XHQwen3TTSModel)
    hf_model = cast(XHQwen3TTSModel, hf_model)

    feature_dim = _capture_talker_feature_dim(hf_model, cfg, logger, "talker")
    token_embedding_file = component_dir / "token_embedding.pt"
    text_embedding_file = component_dir / "text_embedding.pt"
    torch.save(hf_model.model.talker.get_input_embeddings().state_dict(), str(token_embedding_file))
    torch.save(hf_model.model.talker.get_text_embeddings().state_dict(), str(text_embedding_file))

    xh_model.init_wrap_model(hf_model)
    xh_model.to(dtype=dtype)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            model_name=cfg_name,
            hf_model=cfg.hf_model_dir,
            target_device=target_device,
            quant_type=quant_type,
            wrap_cfg=xh_model.wrap_cfg.to_dict(),
            token_embedding_file=token_embedding_file.name,
            text_embedding_file=text_embedding_file.name,
        )
    )
    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)
    del hf_model

    input_sequence_length = xh_model.wrap_cfg.input_sequence_length
    data_batch = {
        "inputs_embeds": torch.randn(1, input_sequence_length, feature_dim),
        "past_seq_length": 0,
    }
    xh_model.convert_to_fronted_graph(data_batch)
    _empty_cuda_cache()
    xh_model.change_eval_type(eval_type=EvalModelType.FRONTEND)
    xh_model.convert_to_quant_graph(target_device)
    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()

    calib_data = _flatten_inputs(xh_model.prepare_inputs(data_batch))
    with time_profiler() as timer:
        ptq_quantize(
            xh_model.quanted_model,
            [calib_data],
            PrecisionMode.ALIGNED,
            [exec_device],
            auto_release_unused_parameters=True,
        )
        logger.info(f"Talker PTQ Quantize time: {timer():.04f} s")
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)

    prefill_dir = component_dir / "Prefill"
    decode_dir = component_dir / "Decoder"
    prefill_dir.mkdir(exist_ok=True, parents=True)
    decode_dir.mkdir(exist_ok=True, parents=True)
    prefill_onnx_file = xhmodel_export_onnx(xh_model, data_batch, str(prefill_dir), f"{cfg_name}_prefill", logger)
    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(component_dir))
    xh_model.release_exported_model()

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(dtype)
    data_batch = {"inputs_embeds": torch.randn(1, 1, feature_dim), "past_seq_length": 0}
    _empty_cuda_cache()
    xh_model.set_input_sequence_length(1)
    xh_model.to("cpu")
    decode_onnx_file = xhmodel_export_onnx(xh_model, data_batch, str(decode_dir), f"{cfg_name}_decode", logger)
    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(component_dir))
    xh_model.release_exported_model()

    meta_file = component_dir / "meta.json"
    meta_file.write_text(json.dumps(_jsonable(meta_info), indent=4, ensure_ascii=False), encoding="utf-8")
    return {"component_dir": component_dir, "meta_file": meta_file, "quant_type": quant_type}


def export_qwen3_tts_code_predictor(
    *,
    model_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    component_dir: Path,
    target_device: str,
    quant_type: str,
) -> dict[str, Any]:
    import torch
    from xhquant.api import ConfigDict, PrecisionMode, get_root_logger, ptq_quantize
    from xhquant.utils.time_profiler import time_profiler

    from xhmodel_merak.xh_other_model.builder import MODELS
    from xhmodel_merak.xh_other_model.eval_model_type import EvalModelType

    from . import XHQwen3TTSCodePredictor, XHQwen3TTSModel

    component_dir.mkdir(parents=True, exist_ok=True)
    cfg_name = f"{runtime_cfg['model_name']}_code_predictor_{target_device}_{quant_type}"
    cfg = ConfigDict(dict(runtime_cfg))
    cfg.model = ConfigDict(copy.deepcopy(dict(model_cfg)))
    logger = get_root_logger()
    dtype = getattr(torch, cfg.dtype)
    exec_device = torch.device(cfg.exec_device)

    xh_model = MODELS.build(cfg.model)
    assert isinstance(xh_model, XHQwen3TTSCodePredictor)
    hf_model = xh_model.get_hf_model(device_map="cpu", dtype=torch.float16)
    assert isinstance(hf_model, XHQwen3TTSModel)
    hf_model = cast(XHQwen3TTSModel, hf_model)
    feature_dim = _capture_talker_feature_dim(hf_model, cfg, logger, "code_predictor")

    token_embedding_file = component_dir / "token_embedding.pt"
    torch.save(hf_model.model.talker.code_predictor.get_input_embeddings().state_dict(), str(token_embedding_file))
    xh_model.init_wrap_model(hf_model)
    xh_model.to(dtype=dtype)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            model_name=cfg_name,
            hf_model=cfg.hf_model_dir,
            target_device=target_device,
            quant_type=quant_type,
            wrap_cfg=xh_model.wrap_cfg.to_dict(),
            token_embedding_file=token_embedding_file.name,
        )
    )
    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)
    del hf_model

    input_sequence_length = xh_model.wrap_cfg.input_sequence_length
    data_batch = {
        "inputs_embeds": torch.randn(1, input_sequence_length, feature_dim),
        "past_seq_length": 0,
        "generate_steps": 0,
    }
    xh_model.convert_to_fronted_graph(data_batch)
    _empty_cuda_cache()
    xh_model.change_eval_type(eval_type=EvalModelType.FRONTEND)
    xh_model.convert_to_quant_graph(target_device)
    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    calib_data = _flatten_inputs(xh_model.prepare_inputs(data_batch))
    with time_profiler() as timer:
        ptq_quantize(
            xh_model.quanted_model,
            [calib_data],
            PrecisionMode.ALIGNED,
            [exec_device],
            auto_release_unused_parameters=True,
        )
        logger.info(f"CodePredictor PTQ Quantize time: {timer():.04f} s")
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)

    prefill_dir = component_dir / "Prefill"
    decode_dir = component_dir / "Decoder"
    prefill_dir.mkdir(exist_ok=True, parents=True)
    decode_dir.mkdir(exist_ok=True, parents=True)
    prefill_onnx_file = xhmodel_export_onnx(xh_model, data_batch, str(prefill_dir), f"{cfg_name}_prefill", logger)
    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(component_dir))
    xh_model.release_exported_model()

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(dtype)
    data_batch = {
        "inputs_embeds": torch.randn(1, 1, feature_dim),
        "past_seq_length": 0,
        "generate_steps": 0,
    }
    _empty_cuda_cache()
    xh_model.set_input_sequence_length(1)
    xh_model.to("cpu")
    decode_onnx_file = xhmodel_export_onnx(xh_model, data_batch, str(decode_dir), f"{cfg_name}_decode", logger)
    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(component_dir))
    xh_model.release_exported_model()

    meta_file = component_dir / "meta.json"
    meta_file.write_text(json.dumps(_jsonable(meta_info), indent=4, ensure_ascii=False), encoding="utf-8")
    return {"component_dir": component_dir, "meta_file": meta_file, "quant_type": quant_type}


def export_qwen3_tts_text_projection(
    *,
    runtime_cfg: Mapping[str, Any],
    component_dir: Path,
    target_device: str,
    quant_type: str,
) -> dict[str, Any]:
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from xhquant.api import QuantScheme, convert_onnx_to_hmonnx, create_quant_config, get_root_logger

    component_dir.mkdir(parents=True, exist_ok=True)
    logger = get_root_logger()
    hf_model = Qwen3TTSModel.from_pretrained(
        runtime_cfg["hf_model_dir"],
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    feature_dim = 2048

    def _hook(module, inputs):
        nonlocal feature_dim
        feature_dim = int(inputs[0].shape[-1])
        return inputs

    hook = hf_model.model.talker.text_projection.register_forward_pre_hook(_hook)
    wavs, sr = _run_generate(hf_model, runtime_cfg)
    sf.write(component_dir / f"output_{runtime_cfg['tts_mode']}.wav", wavs[0], sr)
    hook.remove()

    device = torch.device(runtime_cfg["exec_device"])
    dtype = getattr(torch, runtime_cfg["dtype"])
    example_input = torch.randn(1, 1, feature_dim, device=device, dtype=dtype)
    onnx_dir = component_dir / "onnx"
    hmonnx_dir = component_dir / "hmonnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    onnx_file = onnx_dir / "text_projection.onnx"
    hmonnx_file = hmonnx_dir / f"text_projection_{target_device}_{quant_type}.onnx"
    torch.onnx.export(
        hf_model.model.talker.text_projection.float().cpu(),
        (example_input.float().cpu(),),
        onnx_file,
        input_names=["inputs_embeds"],
        output_names=["outputs"],
    )
    convert_onnx_to_hmonnx(
        onnx_file,
        [example_input.float().cpu()],
        _to_xh_device_type(target_device),
        hmonnx_file,
        quant_config=_build_quant_config(target_device, quant_type),
    )
    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hf_model": runtime_cfg["hf_model_dir"],
        "target_device": target_device,
        "quant_type": quant_type,
        "feature_dim": feature_dim,
        "onnx": str(onnx_file.relative_to(component_dir)),
        "hmonnx": str(hmonnx_file.relative_to(component_dir)),
        "act": type(hf_model.model.talker.text_projection.act_fn).__name__,
    }
    meta_file = component_dir / "meta.json"
    meta_file.write_text(json.dumps(_jsonable(meta), indent=4, ensure_ascii=False), encoding="utf-8")
    logger.info(f"text_projection converted to hmonnx and saved to {hmonnx_file}")
    return {"component_dir": component_dir, "meta_file": meta_file, "quant_type": quant_type}


def export_qwen3_tts_speech_tokenizer(
    *,
    runtime_cfg: Mapping[str, Any],
    component_dir: Path,
    target_device: str,
    quant_type: str,
    component_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    import onnx
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderTransformerModel
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from xhquant.api import convert_onnx_to_hmonnx, get_root_logger
    from xhquant.export.onnx.transforms import hmonnx_transforms

    class XHQwen3TTSTokenizerV2DecoderTransformerModel(Qwen3TTSTokenizerV2DecoderTransformerModel):
        def _setup(self, chunk_size=300):
            self.chunk_size = chunk_size
            cache_position = torch.arange(0, self.chunk_size)
            position_ids = cache_position.unsqueeze(0)
            inputs_embeds = torch.randn(1, self.chunk_size, self.config.hidden_size)
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": None,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            self.register_buffer("full_attention", create_causal_mask(**mask_kwargs), persistent=False)
            if self.has_sliding_layers:
                self.register_buffer(
                    "sliding_attention",
                    create_sliding_window_causal_mask(**mask_kwargs),
                    persistent=False,
                )
            hidden_states = torch.randn(1, self.chunk_size, self.config.hidden_size, dtype=torch.float16)
            if torch.cuda.is_available():
                hidden_states = hidden_states.cuda()
            cos, sin = self.rotary_emb(hidden_states, position_ids.to(hidden_states.device))
            self.register_buffer("cos_cached", cos.cpu(), persistent=False)
            self.register_buffer("sin_cached", sin.cpu(), persistent=False)

        def forward(self, inputs_embeds=None) -> BaseModelOutputWithPast:
            inputs_embeds = self.input_proj(inputs_embeds)
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = cache_position.unsqueeze(0)
            causal_mask_mapping = {
                "full_attention": self.full_attention,
                "sliding_attention": self.sliding_attention if self.has_sliding_layers else None,
            }
            hidden_states = inputs_embeds
            position_embeddings = (self.cos_cached, self.sin_cached)
            for decoder_layer in self.layers[: self.config.num_hidden_layers]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            hidden_states = self.norm(hidden_states)
            hidden_states = self.output_proj(hidden_states)
            return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    component_dir.mkdir(parents=True, exist_ok=True)
    logger = get_root_logger()
    hf_model = Qwen3TTSModel.from_pretrained(
        runtime_cfg["hf_model_dir"],
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    hf_model = cast(Qwen3TTSModel, hf_model)
    input_shape = None
    input_dtype = None

    def _hook(module, inputs):
        nonlocal input_shape, input_dtype
        input_shape = list(inputs[0].shape)
        input_dtype = inputs[0].dtype
        return inputs

    decode_hook = hf_model.model.speech_tokenizer.model.decoder.register_forward_pre_hook(_hook)
    hf_model.model.speech_tokenizer.model.decoder.pre_transformer.config._attn_implementation = "eager"
    wavs, sr = _run_generate(hf_model, runtime_cfg)
    sf.write(component_dir / f"output_{runtime_cfg['tts_mode']}.wav", wavs[0], sr)
    decode_hook.remove()
    if input_shape is None or input_dtype is None:
        raise RuntimeError("Failed to capture speech tokenizer input shape")

    chunk_size = int(component_cfg.get("chunk_size", 300))
    gt_shapes = {}
    for seq_len_in in range(1, chunk_size + 1):
        dummy_input_shape = list(input_shape)
        dummy_input_shape[-1] = seq_len_in
        gt_shapes["_".join(map(str, dummy_input_shape))] = [dummy_input_shape[0], 1, seq_len_in * 1920]
    shapes_file = component_dir / "decode_padding_shapes.json"
    shapes_file.write_text(json.dumps(gt_shapes, indent=2), encoding="utf-8")

    decoder = hf_model.model.speech_tokenizer.model.decoder.float().cpu().eval()
    _empty_cuda_cache()
    input_shape[-1] = chunk_size
    example_input = torch.randint(0, 100, input_shape, device=runtime_cfg["exec_device"], dtype=input_dtype)
    onnx_dir = component_dir / "onnx"
    hmonnx_dir = component_dir / "hmonnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    onnx_file = onnx_dir / "speech_tokenizer.onnx"
    pre_transformer = decoder.pre_transformer
    pre_transformer.__class__ = XHQwen3TTSTokenizerV2DecoderTransformerModel
    pre_transformer = cast(XHQwen3TTSTokenizerV2DecoderTransformerModel, pre_transformer)
    pre_transformer._setup(chunk_size=chunk_size)
    torch.onnx.export(
        decoder,
        example_input.to(torch.int32).cpu(),
        onnx_file,
        input_names=["codes"],
        output_names=["wav"],
        dynamo=True,
    )
    onnx_model = onnx.load(onnx_file)
    hmonnx_transforms(onnx_model)
    onnx.save(onnx_model, onnx_file)
    hmonnx_file = hmonnx_dir / f"speech_tokenizer_{target_device}_{quant_type}.onnx"
    convert_onnx_to_hmonnx(
        onnx_file,
        [example_input.to(torch.int32).cpu()],
        _to_xh_device_type(target_device),
        hmonnx_file,
        quant_config=_build_quant_config(target_device, quant_type),
    )
    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hf_model": runtime_cfg["hf_model_dir"],
        "target_device": target_device,
        "quant_type": quant_type,
        "input_shape": input_shape,
        "decode_padding_shapes": shapes_file.name,
        "onnx": str(onnx_file.relative_to(component_dir)),
        "hmonnx": str(hmonnx_file.relative_to(component_dir)),
    }
    meta_file = component_dir / "meta.json"
    meta_file.write_text(json.dumps(_jsonable(meta), indent=4, ensure_ascii=False), encoding="utf-8")
    logger.info(f"speech_tokenizer converted to hmonnx and saved to {hmonnx_file}")
    return {"component_dir": component_dir, "meta_file": meta_file, "quant_type": quant_type}


def export_qwen3_tts_base_frontend(
    *,
    runtime_cfg: Mapping[str, Any],
    component_dir: Path,
    target_device: str,
    quant_type: str,
    component_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
    from xhquant.api import convert_onnx_to_hmonnx

    from ._export_utils import (
        SpeakerEncoderWrapper,
        SpeechTokenizerEncodeWrapper,
        export_onnx,
        patch_mimi_codebook_matmul_distance,
        patch_mimi_conv1d_static_padding,
    )

    component_dir.mkdir(parents=True, exist_ok=True)
    model_dir = runtime_cfg["hf_model_dir"]
    tokenizer_dir = _resolve_tokenizer_dir(model_dir)
    tokenizer_model = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_dir).float().cpu().eval()
    tokenizer_model.config._attn_implementation = "eager"
    tokenizer_model.encoder.config._attn_implementation = "eager"
    tokenizer_model.encoder.encoder_transformer.config._attn_implementation = "eager"
    patch_mimi_conv1d_static_padding(tokenizer_model.encoder)
    patch_mimi_codebook_matmul_distance(tokenizer_model.encoder.quantizer)
    encode_model = SpeechTokenizerEncodeWrapper(tokenizer_model).float().cpu().eval()

    batch_size = int(component_cfg.get("batch_size", 1))
    audio_samples = int(component_cfg.get("audio_samples", 101760))
    mel_frames = int(component_cfg.get("mel_frames", 400))
    mel_dim = int(component_cfg.get("mel_dim", 128))
    opset = int(component_cfg.get("opset", 18))
    input_values = torch.zeros(batch_size, audio_samples, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, audio_samples, dtype=torch.int32)
    encode_inputs = (input_values, padding_mask)
    onnx_dir = component_dir / "onnx"
    hmonnx_dir = component_dir / "hmonnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_dir.mkdir(parents=True, exist_ok=True)
    encode_onnx = onnx_dir / "speech_tokenizer_encode.onnx"
    encode_hmonnx = hmonnx_dir / f"speech_tokenizer_encode_{target_device}_{quant_type}.onnx"
    export_onnx(
        encode_model,
        encode_inputs,
        encode_onnx,
        ["input_values", "padding_mask"],
        ["audio_codes", "valid_frames"],
        opset,
        use_dynamo=False,
    )
    convert_onnx_to_hmonnx(
        encode_onnx,
        [x.cpu() for x in encode_inputs],
        _to_xh_device_type(target_device),
        encode_hmonnx,
        quant_config=_build_quant_config(target_device, quant_type),
    )

    tts_model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).model.float().cpu().eval()
    if tts_model.speaker_encoder is None:
        raise ValueError(f"model does not have speaker_encoder: {model_dir}")
    speaker_model = SpeakerEncoderWrapper(tts_model.speaker_encoder).float().cpu().eval()
    speaker_inputs = (torch.zeros(batch_size, mel_frames, mel_dim, dtype=torch.float32),)
    speaker_onnx = onnx_dir / "speaker_encoder.onnx"
    speaker_hmonnx = hmonnx_dir / f"speaker_encoder_{target_device}_{quant_type}.onnx"
    export_onnx(speaker_model, speaker_inputs, speaker_onnx, ["mels"], ["speaker_embedding"], opset)
    convert_onnx_to_hmonnx(
        speaker_onnx,
        [x.cpu() for x in speaker_inputs],
        _to_xh_device_type(target_device),
        speaker_hmonnx,
        quant_config=_build_quant_config(target_device, quant_type),
    )
    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hf_model": model_dir,
        "tokenizer_dir": tokenizer_dir,
        "target_device": target_device,
        "quant_type": quant_type,
        "speech_tokenizer_encode_onnx": str(encode_onnx.relative_to(component_dir)),
        "speech_tokenizer_encode_hmonnx": str(encode_hmonnx.relative_to(component_dir)),
        "speaker_encoder_onnx": str(speaker_onnx.relative_to(component_dir)),
        "speaker_encoder_hmonnx": str(speaker_hmonnx.relative_to(component_dir)),
        "batch_size": batch_size,
        "audio_samples": audio_samples,
        "input_sample_rate": int(tokenizer_model.input_sample_rate),
        "encode_downsample_rate": int(tokenizer_model.encode_downsample_rate),
        "encoder_valid_num_quantizers": int(tokenizer_model.encoder_valid_num_quantizers),
        "mel_frames": mel_frames,
        "mel_dim": mel_dim,
        "speaker_encoder_sample_rate": int(tts_model.speaker_encoder_sample_rate),
    }
    meta_file = component_dir / "meta.json"
    meta_file.write_text(json.dumps(_jsonable(meta), indent=4, ensure_ascii=False), encoding="utf-8")
    return {"component_dir": component_dir, "meta_file": meta_file, "quant_type": quant_type}


def export_qwen3_tts_stateful_decoder(
    *,
    runtime_cfg: Mapping[str, Any],
    component_dir: Path,
    target_device: str,
    quant_type: str,
    component_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    import onnx
    import torch
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
    from xhquant.api import convert_onnx_to_hmonnx
    from xhquant.export.onnx.transforms import hmonnx_transforms

    from ._export_utils import (
        StatefulDecoderDynamoCombined,
        make_stateful_decoder_dummy_inputs,
        stateful_decoder_input_output_names,
    )

    component_dir.mkdir(parents=True, exist_ok=True)
    model_dir = runtime_cfg["hf_model_dir"]
    tokenizer_dir = _resolve_tokenizer_dir(model_dir)
    model = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_dir).float().cpu().eval()
    head_dim_arg = int(component_cfg.get("head_dim", 64))
    if hasattr(model.config, "decoder_config"):
        model.config.decoder_config._attn_implementation = "eager"
        model.config.decoder_config.head_dim = head_dim_arg
    if hasattr(model.decoder.pre_transformer, "config"):
        model.decoder.pre_transformer.config._attn_implementation = "eager"
        model.decoder.pre_transformer.config.head_dim = head_dim_arg
    chunk_size = int(component_cfg.get("chunk_size", 12))
    wrapper = StatefulDecoderDynamoCombined(model.decoder, chunk_size=chunk_size).float().cpu().eval()
    cfg = model.decoder.config
    num_layers = int(wrapper.num_layers)
    num_heads = int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 16)))
    head_dim = int(getattr(cfg, "head_dim", head_dim_arg))

    class Args:
        pass

    args = Args()
    args.dummy_batch = int(component_cfg.get("dummy_batch", 1))
    args.chunk_size = chunk_size
    args.dummy_history = int(component_cfg.get("dummy_history", 0))
    args.dummy_valid_frames = int(component_cfg.get("dummy_valid_frames", chunk_size))
    args.head_dim = head_dim_arg
    dummy_inputs = make_stateful_decoder_dummy_inputs(wrapper, num_heads, head_dim, args)
    input_names, output_names = stateful_decoder_input_output_names(num_layers)
    onnx_dir = component_dir / "onnx"
    hmonnx_dir = component_dir / "hmonnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_file = onnx_dir / "qwen3_tts_decoder_stateful_static.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model=wrapper,
            args=dummy_inputs,
            f=str(onnx_file),
            input_names=input_names,
            output_names=output_names,
            opset_version=int(component_cfg.get("opset", 18)),
            dynamo=True,
        )
    onnx_model = onnx.load(str(onnx_file))
    hmonnx_transforms(onnx_model)
    onnx.save(onnx_model, str(onnx_file))
    hmonnx_file = hmonnx_dir / f"qwen3_tts_decoder_stateful_static_{target_device}_{quant_type}.onnx"
    convert_onnx_to_hmonnx(
        str(onnx_file),
        [x.cpu() for x in dummy_inputs],
        _to_xh_device_type(target_device),
        str(hmonnx_file),
        quant_config=_build_quant_config(target_device, quant_type),
    )
    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hf_model": model_dir,
        "tokenizer_dir": tokenizer_dir,
        "target_device": target_device,
        "quant_type": quant_type,
        "stateful_onnx": str(onnx_file.relative_to(component_dir)),
        "stateful_hmonnx": str(hmonnx_file.relative_to(component_dir)),
        "stateful_static_buffers": True,
        "stateful_num_layers": num_layers,
        "stateful_num_heads": num_heads,
        "stateful_head_dim": head_dim,
        "stateful_kv_cache_window": int(wrapper.kv_cache_window),
        "stateful_chunk_size": int(wrapper.chunk_size),
        "stateful_samples_per_frame": int(wrapper.samples_per_frame),
        "stateful_initial_output_skip_frames": int(wrapper.part3.lookahead_frames),
        "stateful_dynamic_batch": False,
        "input_names": input_names,
        "output_names": output_names,
    }
    meta_file = component_dir / "meta.json"
    meta_file.write_text(json.dumps(_jsonable(meta), indent=4, ensure_ascii=False), encoding="utf-8")
    return {"component_dir": component_dir, "meta_file": meta_file, "quant_type": quant_type}


def xhmodel_export_onnx(xh_model, data_batch, onnx_output_dir: str, cfg_name: str, logger) -> str:
    from xhmodel_merak.xh_other_model.eval_model_type import EvalModelType

    logger.info("Start exporting")
    xh_model.to("cpu")
    _empty_cuda_cache()
    xh_model.convert_to_export_graph(data_batch)
    xh_model.change_eval_type(EvalModelType.EXPORTED)
    xh_model.to("cpu")
    _empty_cuda_cache()
    return xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]


def _capture_talker_feature_dim(hf_model, cfg: Mapping[str, Any], logger, component: str) -> int:
    feature_dim = None

    def _hook(module, args, kwargs):
        nonlocal feature_dim
        inputs_embeds = kwargs.get("inputs_embeds", None)
        feature_dim = inputs_embeds.shape[-1] if inputs_embeds is not None else None
        logger.info(f"{component} feature_dim: {feature_dim}")
        raise RuntimeError("Stop forward after getting feature_dim for export")

    module = hf_model.model.talker if component == "talker" else hf_model.model.talker.code_predictor
    hook = module.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        _run_generate(hf_model, cfg)
    except RuntimeError:
        pass
    finally:
        hook.remove()
    if feature_dim is None:
        raise RuntimeError(f"Failed to get {component} feature_dim")
    return int(feature_dim)


def _run_generate(hf_model, cfg: Mapping[str, Any]):
    mode = cfg.get("tts_mode", "custom_voice")
    text = cfg.get("tts_text", _DEFAULT_TEXT)
    if mode == "voice_design":
        return hf_model.generate_voice_design(
            text=text,
            language="Chinese",
            instruct=cfg.get("tts_instruct", ""),
        )
    if mode == "voice_clone":
        ref_audio = cfg.get("ref_audio", "/tmp/clone_1.wav")
        if not Path(ref_audio).exists():
            raise FileNotFoundError(f"missing reference audio {ref_audio}")
        return hf_model.generate_voice_clone(
            text=text,
            language="Chinese",
            ref_audio=ref_audio,
            ref_text=cfg.get("ref_text", ""),
        )
    return hf_model.generate_custom_voice(
        text=text,
        language="Chinese",
        speaker=cfg.get("tts_speaker", "vivian"),
    )


def _normalize_components(components_cfg: Any) -> list[str]:
    if components_cfg is None:
        return ["talker", "code_predictor", "text_projection", "speech_tokenizer", "stateful_decoder"]
    if isinstance(components_cfg, Sequence) and not isinstance(components_cfg, (str, bytes)):
        return [str(item) for item in components_cfg]
    if isinstance(components_cfg, Mapping):
        return [str(name) for name, cfg in components_cfg.items() if cfg is not False and cfg is not None]
    raise TypeError("Qwen3-TTS export.components must be a list or mapping")


def _build_runtime_cfg(
    export_cfg: Mapping[str, Any],
    model_dir: str,
    device: str,
    target_device: str,
    config_file: str,
    work_dir: Path,
) -> dict[str, Any]:
    runtime_cfg: dict[str, Any] = {
        "hf_model_dir": model_dir,
        "target_device": target_device,
        "device": _select_torch_device(device),
        "exec_device": _select_torch_device(device),
        "dtype": str(export_cfg.get("dtype", "float16")),
        "debug": False,
        "config_file": Path(config_file),
        "work_dir": work_dir,
        "model_name": str(export_cfg.get("model_name") or Path(os.path.normpath(model_dir)).name),
        "tts_mode": str(export_cfg.get("tts_mode", "custom_voice")),
        "tts_text": str(export_cfg.get("tts_text", _DEFAULT_TEXT)),
    }
    for key in ("tts_speaker", "tts_instruct", "ref_audio", "ref_text"):
        if key in export_cfg:
            runtime_cfg[key] = export_cfg[key]
    return runtime_cfg


def _ensure_quant_config(model_cfg: dict[str, Any], target_device: str, quant_type: str) -> None:
    from xhquant.api import ConfigDict

    if model_cfg.get("quant_config"):
        return
    model_cfg["quant_config"] = ConfigDict(_build_quant_config(target_device, quant_type))


def _build_quant_config(target_device: str, quant_type: str):
    from xhquant.api import QuantScheme, create_quant_config

    quant_scheme = QuantScheme(target_device=_to_xh_device_type(target_device), quant_type=quant_type)
    return create_quant_config(quant_scheme)


def _to_xh_device_type(target_device: str):
    from xhquant.api import DeviceType

    if target_device != "XH2a":
        raise ValueError(f"Qwen3-TTS workflow currently supports target_device='XH2a', got {target_device!r}")
    return DeviceType.XH2a


def _component_meta(work_dir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    component_dir = Path(result["component_dir"])
    meta_file = Path(result["meta_file"])
    return {
        "component_dir": str(component_dir.relative_to(work_dir)),
        "meta_file": str(meta_file.relative_to(work_dir)),
        "quant_type": result["quant_type"],
    }


def _resolve_tokenizer_dir(model_dir: str) -> str:
    speech_tokenizer_dir = Path(model_dir) / "speech_tokenizer"
    return str(speech_tokenizer_dir if speech_tokenizer_dir.exists() else Path(model_dir))


def _flatten_inputs(inputs: Any) -> list[Any]:
    flattened = []
    for arg in inputs:
        if isinstance(arg, (list, tuple)):
            flattened.extend(arg)
        else:
            flattened.append(arg)
    return flattened


def _build_llm_hmonnx_inputs(
    inputs_embeds: Any,
    *,
    past_seq_length: int,
    current_input_length: int,
    kv_cache_shape: tuple[int, ...],
    num_hidden_layers: int,
    torch_device: str,
    cache_tensor_cls: type,
    generate_steps: int | None = None,
) -> list[Any]:
    import torch

    past_seq_length_tensor = torch.tensor([past_seq_length], dtype=torch.int32, device=torch_device)
    current_input_length_tensor = torch.tensor([current_input_length], dtype=torch.int32, device=torch_device)
    past_key_caches = [
        cache_tensor_cls(torch.zeros(kv_cache_shape, dtype=torch.float16, device=torch_device))
        for _ in range(num_hidden_layers)
    ]
    past_value_caches = [
        cache_tensor_cls(torch.zeros(kv_cache_shape, dtype=torch.float16, device=torch_device))
        for _ in range(num_hidden_layers)
    ]
    inputs = [inputs_embeds, past_seq_length_tensor, current_input_length_tensor, *past_key_caches, *past_value_caches]
    if generate_steps is not None:
        inputs.append(torch.tensor([generate_steps], dtype=torch.int32, device=torch_device))
    return inputs


def _build_stateful_decoder_hmonnx_inputs(comp_meta: Mapping[str, Any], torch_device: str) -> list[Any]:
    import torch

    batch = 1
    frames = int(comp_meta["stateful_chunk_size"])
    num_layers = int(comp_meta["stateful_num_layers"])
    num_heads = int(comp_meta["stateful_num_heads"])
    head_dim = int(comp_meta["stateful_head_dim"])
    kv_cache_window = int(comp_meta["stateful_kv_cache_window"])
    return [
        torch.zeros((batch, frames, 16), dtype=torch.int32, device=torch_device),
        torch.zeros((batch, 512, 2), dtype=torch.float16, device=torch_device),
        torch.zeros((batch, 1024, 4), dtype=torch.float16, device=torch_device),
        torch.zeros((batch, 1024, 4), dtype=torch.float16, device=torch_device),
        torch.tensor([0.0], dtype=torch.float16, device=torch_device),
        torch.tensor([0], dtype=torch.int32, device=torch_device),
        torch.tensor([frames], dtype=torch.int32, device=torch_device),
        *[
            torch.zeros((batch, num_heads, kv_cache_window, head_dim), dtype=torch.float16, device=torch_device)
            for _ in range(num_layers * 2)
        ],
    ]


def _run_hmonnx_golden(hmonnx_file: Path, golden_dir: Path, device: str, inputs: Sequence[Any]) -> None:
    import shutil

    from xhquant.api import HMONNXGoldenInference

    if golden_dir.exists():
        shutil.rmtree(golden_dir)
    golden_dir.mkdir(parents=True, exist_ok=True)
    session = HMONNXGoldenInference(str(hmonnx_file))
    session.initialize()
    _patch_duplicate_tensor_info(session, hmonnx_file)
    session.to(device)
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    session.step = 0
    session(*inputs)


def _patch_duplicate_tensor_info(session: Any, hmonnx_file: Path) -> None:
    import math

    import onnx
    import torch
    from xhquant.xhonnxruntime.hmonnx_inference import TensorInfo

    inner_session = getattr(session, "_session", None)
    if inner_session is None:
        return

    model = onnx.load(str(hmonnx_file), load_external_data=False)
    candidates: dict[str, list[tuple[torch.Size, Any]]] = {}
    for value_info in [*model.graph.value_info, *model.graph.input, *model.graph.output]:
        tensor_type = value_info.type.tensor_type
        if not value_info.name or not tensor_type.elem_type:
            continue
        dims = [dim.dim_value for dim in tensor_type.shape.dim]
        if not dims or any(dim <= 0 for dim in dims):
            continue
        candidates.setdefault(value_info.name, []).append(
            (torch.Size(dims), _onnx_elem_type_to_torch_dtype(tensor_type.elem_type))
        )

    for name, infos in candidates.items():
        if len(infos) < 2 or name not in inner_session.tensors:
            continue
        shape, dtype = max(infos, key=lambda item: math.prod(item[0]))
        inner_session.tensors[name] = TensorInfo(name=name, shape=shape, dtype=dtype)


def _onnx_elem_type_to_torch_dtype(elem_type: int) -> Any:
    import onnx
    import torch

    mapping = {
        onnx.TensorProto.FLOAT: torch.float32,
        onnx.TensorProto.FLOAT16: torch.float16,
        onnx.TensorProto.INT32: torch.int32,
        onnx.TensorProto.INT64: torch.int64,
        onnx.TensorProto.BOOL: torch.bool,
    }
    return mapping.get(elem_type, torch.float32)


def _select_torch_device(device: str) -> str:
    import torch

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return str(device)
    return "cpu"


def _empty_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
