import copy
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult


GB = int(2**30)
_ENCODER_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 0.1)


class Qwen3ASRWorkflow(BaseOtherModelWorkflow):
    SUPPORTED_COMPONENTS = {"encoder", "prefill_decode"}

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
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_model_dir = self._resolve_export_model_dir(quant_result)
        export_cfg = workflow_config.build_export_dict()
        model_cfg = copy.deepcopy(export_cfg["model"])
        model_cfg["hf_model"] = export_model_dir
        # hf_model修改后不需要写回
        # workflow_config.data["export"]["model"] = model_cfg

        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        target_device = str(export_cfg.get("target_device", "XH2a"))
        # 不需要写回
        # workflow_config.data["export"]["target_device"] = target_device
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))
        components_cfg = export_cfg.get("components") or {}
        if not isinstance(components_cfg, Mapping):
            raise TypeError("Qwen3-ASR export.components must be a mapping")
        unsupported = [name for name in components_cfg if name not in self.SUPPORTED_COMPONENTS]
        if unsupported:
            raise ValueError(
                "Unsupported Qwen3-ASR export component(s): "
                f"{unsupported}. Supported components: {sorted(self.SUPPORTED_COMPONENTS)}"
            )

        encoder_cfg = components_cfg.get("encoder", {})
        prefill_decode_cfg = components_cfg.get("prefill_decode", {})
        run_encoder = encoder_cfg is not False and (not isinstance(encoder_cfg, Mapping) or encoder_cfg.get("enabled", True))
        run_prefill_decode = (
            prefill_decode_cfg is not False
            and (not isinstance(prefill_decode_cfg, Mapping) or prefill_decode_cfg.get("enabled", True))
        )
        if not run_encoder and not run_prefill_decode:
            raise ValueError("Qwen3-ASR export must enable at least one component")

        audio_cfg = export_cfg.get("audio") or {}
        if not isinstance(audio_cfg, Mapping):
            raise TypeError("Qwen3-ASR export.audio must be a mapping")
        max_audio_length = int(audio_cfg.get("max_audio_length", 1500))

        meta: dict[str, Any] = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": str(Path(config_file).relative_to(work_dir)),
            "hf_model": export_model_dir,
            "target_device": target_device,
            "components": [],
        }

        if run_encoder:
            if not isinstance(encoder_cfg, Mapping):
                raise TypeError("Qwen3-ASR export.components.encoder must be a mapping or false")
            meta["encoder"] = export_qwen3_asr_encoder(
                model_dir=export_model_dir,
                work_dir=work_dir,
                target_device=target_device,
                quant_type=str(encoder_cfg.get("quant_type", "w8a8_sefp")),
                max_audio_length=max_audio_length,
            )
            meta["components"].append("encoder")

        if run_prefill_decode:
            if not isinstance(prefill_decode_cfg, Mapping):
                raise TypeError("Qwen3-ASR export.components.prefill_decode must be a mapping or false")
            meta.update(
                export_qwen3_asr_prefill_decode(
                    model_dir=export_model_dir,
                    work_dir=work_dir,
                    target_device=target_device,
                    model_cfg=model_cfg,
                    device=device,
                    quant_type=str(prefill_decode_cfg.get("quant_type", "w8a8_sefp")),
                    max_audio_length=max_audio_length,
                    prefix_token_budget=int(prefill_decode_cfg.get("prefix_token_budget", 512)),
                )
            )
            meta["components"].append("prefill_decode")

        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(json.dumps(_jsonable(meta), indent=4), encoding="utf-8")
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
        meta_file = work_dir / "export_meta_info.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"export_meta_info.json not found under {work_dir}")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

        torch_device = _select_torch_device(device)
        golden_meta: dict[str, Any] = {
            "work_dir": str(work_dir),
            "device": torch_device,
            "input_messages": repr(input_messages),
            "components": {},
        }

        if "encoder" in meta:
            encoder_path = work_dir / meta["encoder"]["hmonnx_file"]
            encoder_cfg = meta["encoder"]["model_cfg"]
            max_audio_length = int(encoder_cfg["fixed_max_audio_length"])
            num_mel_bins = int(encoder_cfg["num_mel_bins"])
            input_features = torch.randn(1, num_mel_bins, max_audio_length, dtype=torch.float16, device=torch_device)
            feature_lens = torch.tensor([max_audio_length], dtype=torch.int32, device=torch_device)
            golden_dir = encoder_path.parent / "golden"
            _run_hmonnx_golden(encoder_path, golden_dir, torch_device, [input_features, feature_lens])
            golden_meta["components"]["encoder"] = {
                "hmonnx_file": str(encoder_path),
                "golden_dir": str(golden_dir),
            }

        token_embedding_state = torch.load(work_dir / meta["token_embedding_file"], map_location="cpu")
        hidden_size = int(token_embedding_state["weight"].shape[1])
        prefill_length = int(meta["prefill_input_sequence_length"])
        kv_cache_shape = tuple(int(dim) for dim in meta["kv_cache_shape"])
        num_hidden_layers = int(meta["num_hidden_layers"])

        prefill_path = work_dir / meta["prefill_onnx_file"]
        prefill_inputs_embeds = torch.randn((1, prefill_length, hidden_size), dtype=torch.float16, device=torch_device)
        prefill_inputs = _build_llm_hmonnx_inputs(
            prefill_inputs_embeds,
            past_seq_length=0,
            current_input_length=prefill_length,
            kv_cache_shape=kv_cache_shape,
            num_hidden_layers=num_hidden_layers,
            torch_device=torch_device,
            cache_tensor_cls=CacheTensor,
        )
        prefill_golden_dir = prefill_path.parent / "hmonnx" / "golden"
        _run_hmonnx_golden(prefill_path, prefill_golden_dir, torch_device, prefill_inputs)
        golden_meta["components"]["prefill"] = {
            "hmonnx_file": str(prefill_path),
            "golden_dir": str(prefill_golden_dir),
        }

        decode_path = work_dir / meta["decode_onnx_file"]
        decode_inputs_embeds = torch.randn((1, 1, hidden_size), dtype=torch.float16, device=torch_device)
        decode_inputs = _build_llm_hmonnx_inputs(
            decode_inputs_embeds,
            past_seq_length=prefill_length,
            current_input_length=1,
            kv_cache_shape=kv_cache_shape,
            num_hidden_layers=num_hidden_layers,
            torch_device=torch_device,
            cache_tensor_cls=CacheTensor,
        )
        decode_golden_dir = decode_path.parent / "hmonnx" / "golden"
        _run_hmonnx_golden(decode_path, decode_golden_dir, torch_device, decode_inputs)
        golden_meta["components"]["decode"] = {
            "hmonnx_file": str(decode_path),
            "golden_dir": str(decode_golden_dir),
        }

        golden_meta_file = work_dir / "golden_meta_info.json"
        golden_meta_file.write_text(json.dumps(_jsonable(golden_meta), indent=4), encoding="utf-8")
        return str(golden_meta_file)


def export_qwen3_asr_encoder(
    *,
    model_dir: str,
    work_dir: Path,
    target_device: str,
    quant_type: str,
    max_audio_length: int,
) -> dict[str, Any]:
    import onnx
    import onnxsim
    import torch
    from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config, get_root_logger
    from xhquant.patch.core import RewriterContext

    from .modeling_qwen3_asr import Qwen3ASRForConditionalGeneration

    device_type = _to_xh_device_type(target_device, DeviceType)
    model_name = Path(os.path.normpath(model_dir)).name
    model = Qwen3ASRForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    model.thinker.audio_tower.eval()
    logger = get_root_logger()

    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"
    text_cfg = model.config.thinker_config.text_config
    audio_cfg = model.config.thinker_config.audio_config
    audio_cfg.fixed_max_audio_length = int(max_audio_length)

    encoder_dir = work_dir / "Encoder"
    encoder_dir.mkdir(exist_ok=True, parents=True)
    onnx_file = encoder_dir / f"{model_name}_Encoder.onnx"
    hmonnx_file = encoder_dir / "hmonnx" / f"{model_name}_Encoder_{str(target_device)}_{quant_type}.onnx"

    input_features = torch.randn(1, audio_cfg.num_mel_bins, int(max_audio_length)).to(model.device).to(model.dtype)
    feature_lens = torch.tensor([int(max_audio_length)], dtype=torch.int32).to(model.device)

    if not onnx_file.exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            with RewriterContext(None, backend="onnxruntime"):
                temp_onnx_file = str(Path(tmp_dir) / onnx_file.name)
                torch.onnx.export(
                    model.thinker.audio_tower,
                    (input_features, feature_lens),
                    temp_onnx_file,
                    input_names=["input_features", "feature_lens"],
                    output_names=["hidden_state"],
                )
                onnx_model = onnx.load(temp_onnx_file)
                skipped_optimizers = [
                    "fuse_pad_into_conv",
                    "fuse_consecutive_slices",
                    "eliminate_common_subexpression",
                    "fuse_qkv",
                ]
                if onnx_model.ByteSize() <= _ENCODER_LARGE_MODEL_SIZE_THRESHOLD:
                    onnx_model_sim, checked = onnxsim.simplify(onnx_model, skipped_optimizers=skipped_optimizers)
                else:
                    from xhquant.utils.onnxsim_large_model import simplify_large_onnx

                    onnx_model_sim, checked = simplify_large_onnx(
                        onnx_model,
                        skipped_optimizers=skipped_optimizers,
                    )
                if checked:
                    onnx_model = onnx_model_sim
        onnx.save(
            onnx_model,
            onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{onnx_file.stem}_external_data",
        )
    else:
        onnx_model = onnx.load(onnx_file)

    logger.info(f"ONNX model saved: {onnx_file}, size: {onnx_model.ByteSize() / GB:.2f} GB")
    if not hmonnx_file.exists():
        quant_scheme = QuantScheme(target_device=device_type, quant_type=quant_type)
        convert_onnx_to_hmonnx(
            str(onnx_file),
            [input_features, feature_lens],
            device_type,
            hmonnx_file,
            quant_config=create_quant_config(quant_scheme),
            input_names=["input_features", "feature_lens"],
            output_names=["hidden_state"],
        )

    return {
        "onnx_file": str(onnx_file.relative_to(work_dir)),
        "hmonnx_file": str(hmonnx_file.relative_to(work_dir)),
        "quant_type": quant_type,
        "model_cfg": {
            "head_dim": text_cfg.head_dim,
            "num_heads": text_cfg.num_attention_heads,
            "num_key_value_heads": text_cfg.num_key_value_heads,
            "embed_dim": text_cfg.hidden_size,
            "max_source_positions": audio_cfg.max_source_positions,
            "num_decode_layers": text_cfg.num_hidden_layers,
            "num_mel_bins": audio_cfg.num_mel_bins,
            "fixed_max_audio_length": int(max_audio_length),
        },
    }


def export_qwen3_asr_prefill_decode(
    *,
    model_dir: str,
    work_dir: Path,
    target_device: str,
    model_cfg: Mapping[str, Any],
    device: str,
    quant_type: str,
    max_audio_length: int,
    prefix_token_budget: int,
) -> dict[str, Any]:
    import torch
    from xhquant.api import (
        ConfigDict,
        DeviceType,
        PrecisionMode,
        QuantScheme,
        create_quant_config,
        get_root_logger,
        ptq_quantize,
    )

    from xhmodel_merak.xh_other_model.builder import MODELS
    from xhmodel_merak.xh_other_model.eval_model_type import EvalModelType

    cfg_name = f"{Path(os.path.normpath(model_dir)).name}_{target_device}_{quant_type}"
    model_cfg = copy.deepcopy(dict(model_cfg))
    if not model_cfg.get("quant_config"):
        quant_scheme = QuantScheme(
            target_device=_to_xh_device_type(target_device, DeviceType),
            quant_type=quant_type,
        )
        model_cfg["quant_config"] = ConfigDict(create_quant_config(quant_scheme))

    cfg = ConfigDict(
        dict(
            device=_select_torch_device(device),
            exec_device=_select_torch_device(device),
            dtype="float16",
            model=model_cfg,
        )
    )
    logger = get_root_logger()
    torch_device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)

    xh_model = MODELS.build(cfg.model)
    model = xh_model.get_hf_model()
    xh_model.init_wrap_model(model.thinker.model)
    xh_model.wrap_model.lm_head = model.thinker.lm_head
    xh_model.wrap_model.lm_head.to(torch_device)
    xh_model.wrap_model.lm_head.to(dtype)

    prefill_dir = work_dir / "Prefill"
    decode_dir = work_dir / "Decoder"
    prefill_dir.mkdir(exist_ok=True, parents=True)
    decode_dir.mkdir(exist_ok=True, parents=True)

    config_dir = work_dir / "ConfigFiles"
    config_dir.mkdir(exist_ok=True, parents=True)
    _copy_hf_config_files(Path(model_dir), config_dir)

    token_embedding_file = work_dir / "token_embedding.pt"
    torch.save(xh_model.token_embedding.state_dict(), str(token_embedding_file))

    xh_model.to(torch_device)
    xh_model.to(dtype)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
    model.to(torch_device)
    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"

    text_config = model.config.thinker_config.text_config
    hidden_size = text_config.hidden_size
    num_hidden_layers = text_config.num_hidden_layers
    embed_lengths = _get_feat_extract_output_lengths(int(max_audio_length))
    text_embed_lengths = int(embed_lengths) + 21 + int(prefix_token_budget)

    final_inputs_embeds = torch.randn((1, text_embed_lengths, hidden_size), device=torch_device, dtype=torch.float16)
    data_batch = {
        "input_embeds": final_inputs_embeds.half(),
        "past_seq_length": [0],
    }

    xh_model.set_input_sequence_length(text_embed_lengths)
    with torch.no_grad():
        xh_model.test_step(data_batch)

    xh_model.interactive_mode = True
    xh_model.convert_to_fronted_graph(data_batch)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model.convert_to_quant_graph(target_device)
    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    xh_model.to(dtype)
    xh_model.to(torch_device)

    calib_data = _flatten_inputs(xh_model.prepare_inputs(data_batch))
    ptq_quantize(xh_model.quanted_model, [calib_data], PrecisionMode.ALIGNED, [exec_device])

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(torch_device)
    xh_model.to(dtype)
    xh_model = xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    base_inputs = ["input_embeds", "past_seq_length", "current_input_length"]
    key_names = [f"past_key_cache_{i}" for i in range(num_hidden_layers)]
    value_names = [f"past_value_cache_{i}" for i in range(num_hidden_layers)]
    xh_model.export_cfg = ConfigDict(dict(input_names=base_inputs + key_names + value_names, output_names=["last_hidden_state"]))

    prefill_onnx_file = xhmodel_export_onnx(
        xh_model,
        data_batch,
        str(prefill_dir),
        f"{cfg_name}_prefill",
        logger,
    )
    xh_model.release_exported_model()

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(torch_device)
    xh_model.to(dtype)
    data_batch["input_embeds"] = data_batch["input_embeds"].to(torch_device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model.set_input_sequence_length(1)
    past_seq_len = final_inputs_embeds.shape[1]
    data_batch = {
        "input_embeds": final_inputs_embeds[:, -1:, :].to(torch_device),
        "past_seq_length": [past_seq_len],
    }
    xh_model = xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    decode_onnx_file = xhmodel_export_onnx(
        xh_model,
        data_batch,
        str(decode_dir),
        f"{cfg_name}_decode",
        logger,
    )

    meta: dict[str, Any] = {
        "prefill_decode": {
            "quant_type": quant_type,
            "prefix_token_budget": int(prefix_token_budget),
        },
        "wrap_cfg": xh_model.wrap_cfg.to_dict(),
        "hf_config": str(config_dir.relative_to(work_dir)),
        "token_embedding_file": str(token_embedding_file.relative_to(work_dir)),
        "max_audio_length": int(max_audio_length),
        "audio_embed_lengths": int(embed_lengths),
        "prefix_token_budget": int(prefix_token_budget),
        "prefill_input_sequence_length": int(text_embed_lengths),
        "prefill_onnx_file": str(Path(prefill_onnx_file).relative_to(work_dir)),
        "decode_onnx_file": str(Path(decode_onnx_file).relative_to(work_dir)),
    }
    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta["use_cache"] = True
        meta["kv_cache_shape"] = xh_model.past_key_caches[0].shape
        meta["num_hidden_layers"] = len(xh_model.past_key_caches)
    return meta


def xhmodel_export_onnx(xh_model, data_batch, onnx_output_dir: str, cfg_name: str, logger) -> str:
    from xhmodel_merak.xh_other_model.eval_model_type import EvalModelType

    logger.info("Start exporting")
    xh_model.to("cpu")
    if _torch_cuda_available():
        import torch

        torch.cuda.empty_cache()
    xh_model.convert_to_export_graph(data_batch)
    xh_model.change_eval_type(EvalModelType.EXPORTED)
    xh_model.to("cpu")
    if _torch_cuda_available():
        import torch

        torch.cuda.empty_cache()
    return xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]


def _get_feat_extract_output_lengths(input_lengths: int) -> int:
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def _flatten_inputs(inputs: Any) -> list[Any]:
    flattened = []
    for arg in inputs:
        if isinstance(arg, (list, tuple)):
            flattened.extend(arg)
        else:
            flattened.append(arg)
    return flattened


def _copy_hf_config_files(src_dir: Path, dst_dir: Path) -> None:
    for cfg_file in [
        "chat_template.json",
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
    ]:
        src = src_dir / cfg_file
        if src.exists():
            shutil.copyfile(src, dst_dir / cfg_file)


def _build_llm_hmonnx_inputs(
    inputs_embeds: Any,
    *,
    past_seq_length: int,
    current_input_length: int,
    kv_cache_shape: tuple[int, ...],
    num_hidden_layers: int,
    torch_device: str,
    cache_tensor_cls: type,
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
    return [inputs_embeds, past_seq_length_tensor, current_input_length_tensor, *past_key_caches, *past_value_caches]


def _run_hmonnx_golden(hmonnx_file: Path, golden_dir: Path, device: str, inputs: Sequence[Any]) -> None:
    from xhquant.api import HMONNXGoldenInference

    session = HMONNXGoldenInference(str(hmonnx_file))
    session.to(device)
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    session.step = 0
    session(*inputs)


def _select_torch_device(device: str) -> str:
    import torch

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return str(device)
    return "cpu"


def _to_xh_device_type(target_device: str, device_type_cls: Any) -> Any:
    if target_device != "XH2a":
        raise ValueError(f"Qwen3-ASR workflow currently supports target_device='XH2a', got {target_device!r}")
    return device_type_cls.XH2a


def _torch_cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


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
