"""Standalone Qwen3-0.6B LLM HMONNX export for FunASR-Nano.

This script does not call qwen3_legacy converter examples. It builds the Qwen3
text decoder directly, registers the xhquant trace substitutes, exports prefill
and decode HMONNX, and updates FunASR-Nano ``export_meta_info.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from xh_model_zoo.xh_llm.models.builder import wrap_llm_model  # noqa: E402
from xh_model_zoo.xh_llm.models.qwen3_legacy._model import register_wrap_modules  # noqa: E402,F401
from xh_model_zoo.xh_llm.models.base_converter import BaseConverter  # noqa: E402
from xhquant.api import (  # noqa: E402
    CacheTensor,
    Config,
    ConfigDict,
    DeviceType,
    QuantScheme,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


def _copy_hf_config(hf_model_dir: Path, out_dir: Path) -> Path:
    cfg_dir = out_dir / "ConfigFiles"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "tokenizer.json",
        "merges.txt",
        "chat_template.json",
        "chat_template.jinja",
    ):
        src = hf_model_dir / name
        if src.exists():
            shutil.copyfile(src, cfg_dir / name)
    return cfg_dir


def _load_qwen3(model_dir: Path) -> Qwen3ForCausalLM:
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    if not isinstance(model, Qwen3ForCausalLM):
        raise TypeError(f"expected Qwen3ForCausalLM, got {type(model)}")
    if model.config.tie_word_embeddings:
        old_torchscript = model.config.torchscript
        model.config.torchscript = True
        model.tie_weights()
        model.config.tie_word_embeddings = False
        model.config.torchscript = old_torchscript
    model.eval()
    return model


def _prepare_qwen3_from_funasr(model_dir: Path) -> Path:
    """Create a local HF Qwen3 dir by extracting ``llm.*`` tensors from model.pt."""
    if (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists():
        return model_dir

    funasr_dir = model_dir.parent if model_dir.name == "Qwen3-0.6B" else model_dir
    qwen_cfg_dir = funasr_dir / "Qwen3-0.6B"
    model_pt = funasr_dir / "model.pt"
    if not qwen_cfg_dir.exists() or not model_pt.exists():
        return model_dir

    out_dir = funasr_dir / "Qwen3-0.6B-xhquant"
    if (out_dir / "model.safetensors").exists() or (out_dir / "pytorch_model.bin").exists():
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    for src in qwen_cfg_dir.iterdir():
        if src.is_file():
            shutil.copyfile(src, out_dir / src.name)

    checkpoint = torch.load(model_pt, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    llm_state = {k[len("llm."):]: v for k, v in state_dict.items() if k.startswith("llm.")}
    if not llm_state:
        raise RuntimeError(f"No llm.* tensors found in {model_pt}")

    try:
        from safetensors.torch import save_file

        save_file(llm_state, out_dir / "model.safetensors")
        index = {
            "metadata": {"total_size": sum(v.numel() * v.element_size() for v in llm_state.values())},
            "weight_map": {k: "model.safetensors" for k in llm_state},
        }
        (out_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        torch.save(llm_state, out_dir / "pytorch_model.bin")
    return out_dir


def _flatten_inputs(inputs: tuple[Any, ...]) -> list[Any]:
    flat: list[Any] = []
    for item in inputs:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _build_input_names(num_layers: int) -> list[str]:
    names = ["inputs_embeds", "past_seq_length", "current_input_length"]
    names.extend(f"past_key_cache_{i}" for i in range(num_layers))
    names.extend(f"past_value_cache_{i}" for i in range(num_layers))
    return BaseConverter.xh1_hmonnx_compatible(names)


def _export_one(
    quanted_model: torch.nn.Module,
    inputs: tuple[Any, ...],
    out_file: Path,
    input_names: list[str],
    output_names: list[str],
    logger,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        logger.warning(f"{out_file} already exists, skip export")
        return
    convert_quanted_model_to_hmonnx(quanted_model, inputs, str(out_file), input_names, output_names)
    logger.info(f"Exported {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", default="/data01/datasets/Funasr/Fun-ASR-Nano-2512/Qwen3-0.6B")
    parser.add_argument("--work-dir", default="work_dirs/funasr_nano_xh2a")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--input-sequence-length", type=int, default=256)
    parser.add_argument("--quant-type", default="w8a8h1_sefp")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "export_qwen3_llm_hmonnx.log"), debug=bool(args.debug))
    logger = get_root_logger()

    hf_model_dir = _prepare_qwen3_from_funasr(Path(args.model_dir).expanduser().resolve())
    model = _load_qwen3(hf_model_dir)
    token_embedding = model.model.get_input_embeddings()

    cfg_dir = _copy_hf_config(hf_model_dir, work_dir)
    token_embedding_file = work_dir / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), token_embedding_file)

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"
    quant_config = ConfigDict(create_quant_config(quant_scheme))

    wrap_cfg = Config(
        dict(
            max_sequence_length=int(args.context_length),
            input_sequence_length=int(args.input_sequence_length),
            use_cache=True,
            num_logits_to_keep=1,
            kv_cache=dict(cache_axis=2),
        )
    )
    register_wrap_modules(model)
    wrapped_model = wrap_llm_model(model, wrap_cfg)

    num_layers = int(wrapped_model.model.config.num_hidden_layers)
    num_kv_heads = int(wrapped_model.model.config.num_key_value_heads)
    head_dim = int(wrapped_model.model.layers[0].self_attn.head_dim)
    hidden_size = int(wrapped_model.model.config.hidden_size)
    kv_cache_shape = [1, num_kv_heads, int(args.context_length), head_dim]

    past_key_caches: List[CacheTensor] = [
        CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_layers)
    ]
    past_value_caches: List[CacheTensor] = [
        CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_layers)
    ]

    input_ids = torch.randint(0, min(1000, wrapped_model.config.vocab_size), (1, int(args.input_sequence_length)))
    inputs_embeds = token_embedding(input_ids).detach().to(torch.float16)
    past_seq_length = torch.tensor([0], dtype=torch.int32)
    current_input_length = torch.tensor([int(args.input_sequence_length)], dtype=torch.int32)
    prefill_inputs = (inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches)
    input_names = _build_input_names(num_layers)
    output_names = ["logits"]

    logger.info("Converting Qwen3 to quant graph")
    quanted_model = convert_fx_model_to_quanted_model(
        wrapped_model,
        prefill_inputs,
        DeviceType.XH2a,
        quant_config=quant_config,
    )

    prefix = f"funasr_nano_qwen3_0_6b_xh2a_{args.quant_type}"
    prefill_file = work_dir / "Prefill" / "hmonnx" / f"{prefix}_prefill.onnx"
    _export_one(quanted_model, prefill_inputs, prefill_file, input_names, output_names, logger)

    logger.info("Updating Qwen3 graph config for decode")
    wrap_cfg.input_sequence_length = 1
    quanted_model.update_cfg(wrap_cfg)
    decode_inputs = (
        inputs_embeds[:, :1, :],
        torch.tensor([int(args.input_sequence_length)], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        past_key_caches,
        past_value_caches,
    )
    decode_file = work_dir / "Decoder" / "hmonnx" / f"{prefix}_decode.onnx"
    _export_one(quanted_model, decode_inputs, decode_file, input_names, output_names, logger)

    meta_file = work_dir / "export_meta_info.json"
    meta: Dict[str, Any] = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    meta.update(
        {
            "create_time": meta.get("create_time", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            "hf_model": str(hf_model_dir),
            "hf_config": str(cfg_dir.relative_to(work_dir)),
            "token_embedding_file": str(token_embedding_file.relative_to(work_dir)),
            "prefill_hmonnx_file": str(prefill_file.relative_to(work_dir)),
            "decode_hmonnx_file": str(decode_file.relative_to(work_dir)),
            "prefill_onnx_file": str(prefill_file.relative_to(work_dir)),
            "decode_onnx_file": str(decode_file.relative_to(work_dir)),
            "prefill_input_sequence_length": int(args.input_sequence_length),
            "kv_cache_shape": kv_cache_shape,
            "num_hidden_layers": num_layers,
            "cache_axis": 2,
            "hidden_size": hidden_size,
            "wrap_cfg": wrap_cfg.to_dict(),
            "qwen3_quant_scheme": quant_scheme.to_dict(),
        }
    )
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Updated metadata: {meta_file}")


if __name__ == "__main__":
    main()
