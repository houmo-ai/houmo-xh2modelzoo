import copy
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


GB = int(2**30)
_ENCODER_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 0.1)


def export_qwen3_forcealigner_encoder(
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
    from xhquant.api import (
        DeviceType,
        QuantScheme,
        convert_onnx_to_hmonnx,
        create_quant_config,
        get_root_logger,
    )
    from xhquant.patch.core import RewriterContext

    from .model import Qwen3ASRForConditionalGeneration

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
    hmonnx_file = (
        encoder_dir
        / "hmonnx"
        / f"{model_name}_Encoder_{target_device}_{quant_type}.onnx"
    )

    input_features = torch.randn(
        1,
        audio_cfg.num_mel_bins,
        int(max_audio_length),
    ).to(model.device, dtype=model.dtype)
    feature_lens = torch.tensor(
        [int(max_audio_length)],
        dtype=torch.int32,
        device=model.device,
    )

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
                    onnx_model_sim, checked = onnxsim.simplify(
                        onnx_model,
                        skipped_optimizers=skipped_optimizers,
                    )
                else:
                    from xhquant.utils.onnxsim_large_model import (
                        simplify_large_onnx,
                    )

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

    logger.info(
        f"ONNX model saved: {onnx_file}, "
        f"size: {onnx_model.ByteSize() / GB:.2f} GB"
    )
    if not hmonnx_file.exists():
        quant_scheme = QuantScheme(
            target_device=device_type,
            quant_type=quant_type,
        )
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


def export_qwen3_forcealigner_prefill(
    *,
    model_dir: str,
    work_dir: Path,
    target_device: str,
    model_cfg: Mapping[str, Any],
    device: str,
    quant_type: str,
    sequence_length: int,
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

    model_cfg = copy.deepcopy(dict(model_cfg))
    model_cfg["wrap_cfg"]["input_sequence_length"] = int(sequence_length)
    model_cfg["wrap_cfg"]["num_logits_to_keep"] = 0
    if not model_cfg.get("quant_config"):
        scheme = QuantScheme(
            target_device=_to_xh_device_type(target_device, DeviceType),
            quant_type=quant_type,
        )
        model_cfg["quant_config"] = ConfigDict(create_quant_config(scheme))

    torch_device = torch.device(_select_torch_device(device))
    dtype = torch.float16
    logger = get_root_logger()
    xh_model = MODELS.build(ConfigDict(model_cfg))
    model = xh_model.get_hf_model()
    xh_model.init_wrap_model(model.thinker.model)
    xh_model.wrap_model.lm_head = model.thinker.lm_head.to(
        torch_device,
        dtype=dtype,
    )

    config_dir = work_dir / "ConfigFiles"
    config_dir.mkdir(exist_ok=True, parents=True)
    _copy_hf_config_files(Path(model_dir), config_dir)
    token_embedding_file = work_dir / "token_embedding.pt"
    torch.save(xh_model.token_embedding.state_dict(), token_embedding_file)

    xh_model.to(torch_device).to(dtype)
    xh_model.change_eval_type(EvalModelType.WRAPED)
    model.to(torch_device)
    hidden_size = int(model.config.thinker_config.text_config.hidden_size)
    inputs_embeds = torch.randn(
        (1, sequence_length, hidden_size),
        device=torch_device,
        dtype=dtype,
    )
    data_batch = {
        "input_embeds": inputs_embeds,
        "past_seq_length": [0],
    }
    xh_model.set_input_sequence_length(sequence_length)
    with torch.no_grad():
        xh_model.test_step(data_batch)

    xh_model.interactive_mode = True
    xh_model.convert_to_fronted_graph(data_batch)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    xh_model.convert_to_quant_graph(target_device)
    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    xh_model.to(dtype).to(torch_device)
    calib_data = _flatten_inputs(xh_model.prepare_inputs(data_batch))
    ptq_quantize(
        xh_model.quanted_model,
        [calib_data],
        PrecisionMode.ALIGNED,
        [torch_device],
    )

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(dtype).to("cpu")
    num_layers = int(model.config.thinker_config.text_config.num_hidden_layers)
    base_inputs = [
        "input_embeds",
        "past_seq_length",
        "current_input_length",
    ]
    cache_inputs = [
        f"past_key_cache_{index}" for index in range(num_layers)
    ] + [
        f"past_value_cache_{index}" for index in range(num_layers)
    ]
    xh_model.export_cfg = ConfigDict(
        dict(
            input_names=base_inputs + cache_inputs,
            output_names=["logits"],
        )
    )
    cfg_name = (
        f"{Path(os.path.normpath(model_dir)).name}_"
        f"{target_device}_{quant_type}_prefill_fullseq"
    )
    prefill_dir = work_dir / "Prefill"
    prefill_dir.mkdir(exist_ok=True, parents=True)
    prefill_file = xhmodel_export_onnx(
        xh_model,
        data_batch,
        str(prefill_dir),
        cfg_name,
        logger,
    )

    meta: dict[str, Any] = {
        "prefill": {
            "quant_type": quant_type,
            "sequence_length": int(sequence_length),
        },
        "wrap_cfg": xh_model.wrap_cfg.to_dict(),
        "hf_config": str(config_dir.relative_to(work_dir)),
        "token_embedding_file": str(
            token_embedding_file.relative_to(work_dir)
        ),
        "prefill_input_sequence_length": int(sequence_length),
        "prefill_onnx_file": str(
            Path(prefill_file).relative_to(work_dir)
        ),
    }
    if xh_model.past_key_caches:
        meta["kv_cache_shape"] = xh_model.past_key_caches[0].shape
        meta["num_hidden_layers"] = len(xh_model.past_key_caches)
    return meta


def xhmodel_export_onnx(
    xh_model,
    data_batch,
    onnx_output_dir: str,
    cfg_name: str,
    logger,
) -> str:
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
    return xh_model.to_export_onnx(
        data_batch,
        onnx_output_dir,
        cfg_name,
    )[0]


def _flatten_inputs(inputs: Any) -> list[Any]:
    flattened = []
    for arg in inputs:
        if isinstance(arg, (list, tuple)):
            flattened.extend(arg)
        else:
            flattened.append(arg)
    return flattened


def _copy_hf_config_files(src_dir: Path, dst_dir: Path) -> None:
    for name in [
        "chat_template.json",
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
    ]:
        src = src_dir / name
        if src.is_file():
            shutil.copyfile(src, dst_dir / name)


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

    past_seq_length_tensor = torch.tensor(
        [past_seq_length],
        dtype=torch.int32,
        device=torch_device,
    )
    current_input_length_tensor = torch.tensor(
        [current_input_length],
        dtype=torch.int32,
        device=torch_device,
    )
    past_key_caches = [
        cache_tensor_cls(
            torch.zeros(
                kv_cache_shape,
                dtype=torch.float16,
                device=torch_device,
            )
        )
        for _ in range(num_hidden_layers)
    ]
    past_value_caches = [
        cache_tensor_cls(
            torch.zeros(
                kv_cache_shape,
                dtype=torch.float16,
                device=torch_device,
            )
        )
        for _ in range(num_hidden_layers)
    ]
    return [
        inputs_embeds,
        past_seq_length_tensor,
        current_input_length_tensor,
        *past_key_caches,
        *past_value_caches,
    ]


def _run_hmonnx_golden(
    hmonnx_file: Path,
    golden_dir: Path,
    device: str,
    inputs: Sequence[Any],
) -> None:
    from xhquant.api import HMONNXGoldenInference

    session = HMONNXGoldenInference(str(hmonnx_file))
    session.to(device)
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    session.step = 0
    session(*inputs)


def _reset_golden_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _select_torch_device(device: str) -> str:
    import torch

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return str(device)
    return "cpu"


def _to_xh_device_type(target_device: str, device_type_cls: Any) -> Any:
    if target_device != "XH2a":
        raise ValueError(
            "Qwen3-ForceAligner workflow currently supports "
            f"target_device='XH2a', got {target_device!r}"
        )
    return device_type_cls.XH2a


def _torch_cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
