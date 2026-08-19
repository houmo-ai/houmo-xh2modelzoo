"""GPTQ quantization of the MiniCPM-V-4.5 LLM backbone (Qwen3-8B base)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_REPO_RESOURCE_PREFIXES = ("xh2modelzoo://", "repo://")


def _load_calibration_samples(
    calibration_jsonl: str,
    *,
    nsamples: int,
    sampling_seed: int | None,
) -> list[str]:
    records: list[str] = []
    with open(calibration_jsonl, encoding="utf-8") as handle:
        for line in handle:
            text = json.loads(line).get("text", "")
            if text:
                records.append(text)

    if not records:
        raise ValueError(f"calibration jsonl has no usable 'text' samples: {calibration_jsonl}")
    if sampling_seed is None or len(records) <= nsamples:
        return records[:nsamples]

    import numpy as np

    indices = np.random.default_rng(sampling_seed).choice(
        len(records),
        size=nsamples,
        replace=False,
    )
    return [records[int(index)] for index in indices]


def _resolve_calibration_path(calibration_jsonl: str) -> str:
    """Resolve the calibration path following the repo-wide convention
    (gemma4_series / qwen3_next quant_adapter):

    - ``xh2modelzoo://`` / ``repo://`` scheme prefixes address repo resources;
    - candidates: ``XH2MODELZOO_DATA_ROOT`` -> ``XH2MODELZOO_ROOT`` -> cwd ->
      repo root;
    - plain relative paths and absolute paths are also accepted.
    """
    if os.path.isabs(calibration_jsonl) and os.path.isfile(calibration_jsonl):
        return calibration_jsonl
    relative_path = calibration_jsonl
    for prefix in _REPO_RESOURCE_PREFIXES:
        if relative_path.startswith(prefix):
            relative_path = relative_path.removeprefix(prefix)
            break
    relative_path = relative_path.lstrip("/")
    repo_root = Path(__file__).resolve().parents[4]

    candidates: list[Path] = []
    data_root = os.environ.get("XH2MODELZOO_DATA_ROOT")
    if data_root:
        root = Path(data_root).expanduser()
        candidates.append(root / relative_path)
        if relative_path.startswith("data/"):
            candidates.append(root / relative_path.removeprefix("data/"))
    env_root = os.environ.get("XH2MODELZOO_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser() / relative_path)
    candidates.append(Path.cwd() / relative_path)
    candidates.append(repo_root / relative_path)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return str(candidate.resolve())
    raise ValueError(
        f"calibration jsonl not found: {calibration_jsonl} "
        "(checked XH2MODELZOO_DATA_ROOT, XH2MODELZOO_ROOT, cwd and repo root)"
    )


def quantize_minicpm_llm_gptq(
    *,
    model_dir: str,
    output_dir: str,
    quant_cfg: Mapping[str, Any],
    device: str,
) -> str:
    """GPTQ-quantize the MiniCPM-V-4.5 LLM backbone (Qwen3-8B base) with GPTQModel.

    The backbone is extracted from the MiniCPM host as a standalone Qwen3
    checkpoint (Qwen3Config + Qwen3ForCausalLM state dict), quantized with a
    text-only calibration corpus, and dequantized back to float ``nn.Linear``
    weights so the HMONNX export can load them without a gptqmodel dependency.
    The multi-modal Vision branches keep static PTQ.
    """

    calibration_jsonl = str(quant_cfg.get("calibration_jsonl") or "")
    if not calibration_jsonl:
        raise ValueError("minicpm GPTQ requires quant.calibration_jsonl (text-only corpus)")
    calibration_jsonl = _resolve_calibration_path(calibration_jsonl)

    gptqmodel_source = quant_cfg.get("gptqmodel_source") or os.environ.get("GPTQMODEL_SOURCE")
    if gptqmodel_source and gptqmodel_source not in sys.path:
        sys.path.insert(0, gptqmodel_source)

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import QuantizeConfig

    nsamples = int(quant_cfg.get("nsamples", 128))
    seqlen = int(quant_cfg.get("seqlen", 1024))
    sampling_seed = quant_cfg.get("sampling_seed")
    samples = _load_calibration_samples(
        calibration_jsonl,
        nsamples=nsamples,
        sampling_seed=None if sampling_seed is None else int(sampling_seed),
    )
    if len(samples) < 16:
        print(
            f"[GPTQ] warning: calibration set has only {len(samples)} sample(s); "
            "more samples generally improve quantization quality"
        )

    group_size = int(quant_cfg.get("group_size", 64))
    if group_size != 64:
        raise ValueError(f"MiniCPM-V-4.5 GPTQ requires group_size=64 for XH2A, got {group_size}")

    quantize_config = QuantizeConfig(
        bits=int(quant_cfg.get("bits", 4)),
        group_size=group_size,
        desc_act=False,
        damp_percent=float(quant_cfg.get("damp_percent", 0.01)),
        offload_to_disk_path=str(Path(output_dir) / "gptqmodel_offload"),
    )

    backbone_dir = Path(output_dir) / "llm_backbone"
    prebuilt = quant_cfg.get("llm_backbone_dir")
    if prebuilt and Path(prebuilt).is_dir() and (Path(prebuilt) / "pytorch_model.bin").is_file():
        # The backbone must be extracted with transformers==4.57 (the MiniCPM
        # remote code is incompatible with newer transformers); reuse a prebuilt
        # directory instead of re-extracting here.
        shutil.copytree(Path(prebuilt), backbone_dir, dirs_exist_ok=True)
    else:
        _export_llm_backbone(model_dir, backbone_dir)

    model = GPTQModel.from_pretrained(
        str(backbone_dir),
        quantize_config=quantize_config,
        trust_remote_code=True,
    )
    model.quantize(samples, batch_size=1, calibration_concat_size=seqlen)

    quanted_model_dir = str(Path(output_dir) / "gptq_llm")
    model.save_quantized(quanted_model_dir)

    # Dequantize the GPTQ weights back to float nn.Linear so the MiniCPM export
    # can load them without a gptqmodel dependency (the backbone is re-attached
    # to the MiniCPM host by load_state_dict in export).
    dequant_dir = Path(output_dir) / "gptq_llm_dequant"
    _dequantize_gptq_llm(quanted_model_dir, quantize_config, dequant_dir)
    return str(dequant_dir)


def _dequantize_gptq_llm(quanted_model_dir: str, quantize_config: Any, dequant_dir: Path) -> None:
    """Load the GPTQ artifact with the Torch backend and dequantize to nn.Linear."""
    import torch
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch import dequantize_model

    if dequant_dir.exists():
        shutil.rmtree(dequant_dir)
    dequant_dir.mkdir(parents=True, exist_ok=True)

    model = GPTQModel.from_quantized(
        quanted_model_dir,
        quantize_config=quantize_config,
        trust_remote_code=True,
        backend=BACKEND.TORCH,
    )
    model = dequantize_model(model)

    # GPTQModel wraps the causal LM as Qwen3QModel -> model.model
    # (Qwen3ForCausalLM); its state dict (model.layers..., lm_head) matches the
    # MiniCPM host.llm layout.
    inner = model.model
    state = {key: value.detach().cpu().clone() for key, value in inner.state_dict().items()}
    torch.save(state, dequant_dir / "pytorch_model.bin")
    shutil.copy2(Path(quanted_model_dir) / "config.json", dequant_dir / "config.json")


def _export_llm_backbone(model_dir: str, backbone_dir: Path) -> None:
    """Extract the Qwen3ForCausalLM backbone as a standalone HF Qwen3 checkpoint.

    The MiniCPM host model wraps a standard Qwen3ForCausalLM (state dict has no
    prefix); GPTQModel cannot load the remote-code MiniCPM checkpoint, so we
    rebuild a Qwen3 directory from the backbone state dict + Qwen3Config.
    """
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer, Qwen3Config

    if backbone_dir.exists():
        shutil.rmtree(backbone_dir)
    backbone_dir.mkdir(parents=True, exist_ok=True)

    minicpm_cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.save_pretrained(str(backbone_dir))

    qwen_cfg = Qwen3Config(
        vocab_size=minicpm_cfg.vocab_size,
        hidden_size=minicpm_cfg.hidden_size,
        intermediate_size=minicpm_cfg.intermediate_size,
        num_hidden_layers=minicpm_cfg.num_hidden_layers,
        num_attention_heads=minicpm_cfg.num_attention_heads,
        num_key_value_heads=minicpm_cfg.num_key_value_heads,
        head_dim=getattr(minicpm_cfg, "head_dim", None),
        hidden_act=minicpm_cfg.hidden_act,
        max_position_embeddings=minicpm_cfg.max_position_embeddings,
        rms_norm_eps=minicpm_cfg.rms_norm_eps,
        rope_theta=getattr(minicpm_cfg, "rope_theta", 10000.0),
        tie_word_embeddings=bool(getattr(minicpm_cfg, "tie_word_embeddings", False)),
        use_cache=bool(getattr(minicpm_cfg, "use_cache", True)),
    )
    qwen_cfg.save_pretrained(str(backbone_dir))

    host = AutoModel.from_pretrained(
        model_dir,
        config=minicpm_cfg,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    ).eval()
    state = {key: value.detach().cpu().clone() for key, value in host.llm.state_dict().items()}
    torch.save(state, backbone_dir / "pytorch_model.bin")


__all__ = ["quantize_minicpm_llm_gptq"]
