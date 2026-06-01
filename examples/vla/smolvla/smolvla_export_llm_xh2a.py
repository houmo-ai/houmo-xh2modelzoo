import argparse
import json
import os.path as osp
import random
import shutil
import sys
import os
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxsim import simplify


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEROBOT_SRC = REPO_ROOT.parent / "lerobot" / "src"


def ensure_lerobot_importable(lerobot_src: str | Path | None = None) -> Path:
    lerobot_src_path = Path(lerobot_src) if lerobot_src is not None else DEFAULT_LEROBOT_SRC
    lerobot_src_path = lerobot_src_path.resolve()
    if not lerobot_src_path.exists():
        raise FileNotFoundError(
            f"LeRobot src path not found: {lerobot_src_path}. "
            "Please pass --lerobot_src to point to lerobot/src."
        )
    if str(lerobot_src_path) not in sys.path:
        sys.path.insert(0, str(lerobot_src_path))
    return lerobot_src_path


from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config


ORIGIN_WORKDIR = Path("work_dirs/smolvla_llm")
ONNX_DIR = ORIGIN_WORKDIR / "smolvla_llm_onnx"
HMONNX_DIR = ORIGIN_WORKDIR / "smolvla_llm_hmonnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)
HMONNX_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_lerobot_modules(lerobot_src: str | Path | None = None):
    ensure_lerobot_importable(lerobot_src)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    return PreTrainedConfig, SmolVLAPolicy, SmolVLMWithExpertModel


def resolve_hf_cached_model_path(model_id_or_path: str) -> str:
    model_path = Path(model_id_or_path)
    if model_path.exists():
        return str(model_path.resolve())

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_cache_dir = cache_root / f"models--{model_id_or_path.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    refs_main = repo_cache_dir / "refs" / "main"

    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = snapshots_dir / revision
        if snapshot_dir.exists():
            return str(snapshot_dir.resolve())

    if snapshots_dir.exists():
        snapshot_candidates = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()])
        if snapshot_candidates:
            return str(snapshot_candidates[-1].resolve())

    return model_id_or_path


def resolve_vlm_model_source(model_path: str) -> str:
    model_dir = Path(model_path)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return resolve_hf_cached_model_path(model_path)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception:  # noqa: BLE001
        return resolve_hf_cached_model_path(model_path)

    vlm_model_name = config_data.get("vlm_model_name")
    if isinstance(vlm_model_name, str) and vlm_model_name:
        return resolve_hf_cached_model_path(vlm_model_name)

    return resolve_hf_cached_model_path(model_path)


class SmolVLALLMPart(nn.Module):
    """Export wrapper for SmolVLA text backbone feature extraction."""

    def __init__(self, vlm_with_expert: nn.Module):
        super().__init__()
        self.text_model = vlm_with_expert.get_vlm_model().text_model
        self._force_eager_attention(self.text_model)
        self.text_model.eval()

    @staticmethod
    def _force_eager_attention(text_model: nn.Module):
        config = getattr(text_model, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"

        for module in text_model.modules():
            module_config = getattr(module, "config", None)
            if module_config is not None and hasattr(module_config, "_attn_implementation"):
                module_config._attn_implementation = "eager"

    @staticmethod
    def _build_4d_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        device = attention_mask.device

        key_padding_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
        causal_mask = causal_mask[None, None, :, :]
        full_mask = key_padding_mask & causal_mask

        min_value = torch.tensor(torch.finfo(dtype).min, device=device, dtype=dtype)
        zero_value = torch.tensor(0.0, device=device, dtype=dtype)
        full_mask = torch.where(full_mask, zero_value, min_value)
        return full_mask.expand(batch_size, 1, seq_len, seq_len).contiguous()

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        inputs_embeds = inputs_embeds.contiguous()
        attention_mask = attention_mask.contiguous().to(device=inputs_embeds.device, dtype=torch.long)

        text_dtype = next(self.text_model.parameters()).dtype
        inputs_embeds = inputs_embeds.to(dtype=text_dtype)
        causal_attention_mask = self._build_4d_causal_mask(attention_mask, text_dtype)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.last_hidden_state


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def configure_hf_offline_env():
    os.environ.setdefault("HUGGINGFACE_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_policy_config_offline(model_path: str, PreTrainedConfig):
    config = PreTrainedConfig.from_pretrained(model_path)
    if hasattr(config, "vlm_model_name") and isinstance(config.vlm_model_name, str):
        config.vlm_model_name = resolve_hf_cached_model_path(config.vlm_model_name)
    return config


@torch.no_grad()
def load_vlm_with_expert(model_path: str, device: torch.device, lerobot_src: str | Path | None = None):
    configure_hf_offline_env()
    PreTrainedConfig, SmolVLAPolicy, SmolVLMWithExpertModel = get_lerobot_modules(lerobot_src)
    errors: list[str] = []
    resolved_vlm_model_path = resolve_vlm_model_source(model_path)

    try:
        vlm_with_expert = SmolVLMWithExpertModel(
            model_id=resolved_vlm_model_path,
            load_vlm_weights=True,
            freeze_vision_encoder=True,
            train_expert_only=True,
            device=str(device),
        )
        vlm_with_expert.eval()
        if resolved_vlm_model_path != model_path:
            print(f"Resolved SmolVLM backbone from policy config: {resolved_vlm_model_path}")
            return vlm_with_expert, "vlm_from_policy_config"
        print(f"Loaded SmolVLM backbone from: {resolved_vlm_model_path}")
        return vlm_with_expert, "vlm"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"vlm load failed: {exc}")

    try:
        config = load_policy_config_offline(model_path, PreTrainedConfig)
        policy = SmolVLAPolicy.from_pretrained(model_path, config=config, strict=False)
        policy.eval()
        policy.to(device)
        print(f"Loaded SmolVLA policy from: {model_path}")
        return policy.model.vlm_with_expert, "policy"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"policy load failed: {exc}")

    raise RuntimeError(
        "Unable to load SmolVLA or SmolVLM weights from the given model_path.\n"
        + "\n".join(errors)
    )


def save_hf_artifacts(model_path: str, processor, work_dir: Path) -> str | None:
    hf_config_dir = work_dir / "hf_config"
    hf_config_dir.mkdir(parents=True, exist_ok=True)

    copied_any = False
    local_model_dir = Path(model_path)
    hf_config_files = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "merges.txt",
        "vocab.json",
    ]
    if local_model_dir.exists():
        for cfg_file in hf_config_files:
            src_file = local_model_dir / cfg_file
            if src_file.exists():
                shutil.copyfile(src_file, hf_config_dir / cfg_file)
                copied_any = True

    if processor is not None:
        try:
            processor.save_pretrained(hf_config_dir)
            copied_any = True
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to save processor artifacts: {exc}")

    if copied_any:
        return str(hf_config_dir)
    return None


@torch.no_grad()
def build_dummy_inputs(
    vlm_with_expert: nn.Module,
    input_sequence_length: int,
    prompt: str,
    device: torch.device,
    export_dtype: torch.dtype,
):
    text_model = vlm_with_expert.get_vlm_model().text_model
    token_embedding = text_model.get_input_embeddings()
    tokenizer = getattr(getattr(vlm_with_expert, "processor", None), "tokenizer", None)

    if tokenizer is not None:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=input_sequence_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device=device, dtype=torch.long)
    else:
        vocab_size = getattr(token_embedding, "num_embeddings", 32000)
        input_ids = torch.randint(0, vocab_size, (1, input_sequence_length), device=device)
        attention_mask = torch.ones((1, input_sequence_length), device=device, dtype=torch.long)

    token_embedding = token_embedding.to(device)
    inputs_embeds = token_embedding(input_ids).to(dtype=export_dtype)
    return inputs_embeds, attention_mask, token_embedding, tokenizer


def export_llm(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    export_dtype = torch.float16 if device.type == "cuda" else torch.float32

    vlm_with_expert, load_mode = load_vlm_with_expert(
        args.model_path,
        device,
        args.lerobot_src,
    )
    processor = getattr(vlm_with_expert, "processor", None)
    text_model = vlm_with_expert.get_vlm_model().text_model
    hidden_size = getattr(text_model.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Unable to infer hidden_size from SmolVLA text model config.")

    print(f"Load mode: {load_mode}")
    print(f"Export device: {device}")
    print(f"Input sequence length: {args.input_sequence_length}")
    print(f"Text hidden size: {hidden_size}")

    llm_model = SmolVLALLMPart(vlm_with_expert=vlm_with_expert)
    llm_model.eval()
    llm_model.to(device=device, dtype=export_dtype)

    inputs_embeds, attention_mask, token_embedding, _ = build_dummy_inputs(
        vlm_with_expert=vlm_with_expert,
        input_sequence_length=args.input_sequence_length,
        prompt=args.prompt,
        device=device,
        export_dtype=export_dtype,
    )

    work_dir = ORIGIN_WORKDIR
    work_dir.mkdir(parents=True, exist_ok=True)

    token_embedding_file = work_dir / "token_embedding.pt"
    token_embedding_state_dict = {key: value.detach().cpu() for key, value in token_embedding.state_dict().items()}
    torch.save(token_embedding_state_dict, token_embedding_file)

    hf_config_dir = save_hf_artifacts(args.model_path, processor, work_dir)

    temp_onnx_file = ONNX_DIR / f"{args.output_name}.onnx"
    simplified_onnx_file = ONNX_DIR / f"{args.output_name}_simplified.onnx"
    out_hmonnx_file = HMONNX_DIR / f"{args.output_name}_xh2.onnx"

    print("Exporting SmolVLA LLM backbone to ONNX...")
    torch.onnx.export(
        llm_model,
        (inputs_embeds, attention_mask),
        str(temp_onnx_file),
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["last_hidden_state"],
        opset_version=args.opset,
        verbose=False,
        do_constant_folding=True,
    )

    print("Simplifying ONNX...")
    onnx_model = onnx.load(str(temp_onnx_file))
    model_simplified, check = simplify(
        onnx_model,
        test_input_shapes={
            "inputs_embeds": list(inputs_embeds.shape),
            "attention_mask": list(attention_mask.shape),
        },
    )
    if not check:
        print("Warning: onnxsim check failed, saving simplified graph anyway.")
    onnx.save(model_simplified, str(simplified_onnx_file))

    print("Converting ONNX to HMONNX for XH2A...")
    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a,
        quant_type=args.quant_type,
    )
    quant_config = create_quant_config(quant_scheme)

    calib_inputs = (
        inputs_embeds.detach().cpu(),
        attention_mask.detach().cpu(),
    )
    convert_onnx_to_hmonnx(
        str(simplified_onnx_file),
        calib_inputs,
        out_hmonnx_file=osp.join(str(out_hmonnx_file)),
        device_type="XH2A",
        quant_config=quant_config,
    )

    meta_info = {
        "model_path": args.model_path,
        "load_mode": load_mode,
        "input_names": ["inputs_embeds", "attention_mask"],
        "output_names": ["last_hidden_state"],
        "input_sequence_length": args.input_sequence_length,
        "hidden_size": hidden_size,
        "prompt": args.prompt,
        "quant_type": args.quant_type,
        "token_embedding_file": token_embedding_file.name,
        "onnx_file": str(simplified_onnx_file.relative_to(work_dir)),
        "hmonnx_file": str(out_hmonnx_file.relative_to(work_dir)),
    }
    if hf_config_dir is not None:
        meta_info["hf_config_dir"] = str(Path(hf_config_dir).relative_to(work_dir))

    meta_info_file = work_dir / "meta_info.json"
    with open(meta_info_file, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4, ensure_ascii=False)

    print(f"Token embedding saved to: {token_embedding_file}")
    if hf_config_dir is not None:
        print(f"HF config saved to: {hf_config_dir}")
    print(f"ONNX saved to: {simplified_onnx_file}")
    print(f"HMONNX saved to: {out_hmonnx_file}")
    print(f"Meta info saved to: {meta_info_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SmolVLA text backbone to ONNX/HMONNX")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "SmolVLA policy path / repo id, or fallback SmolVLM backbone repo id. "
            "For example: lerobot/smolvla_base"
        ),
    )
    parser.add_argument(
        "--lerobot_src",
        type=str,
        default=str(DEFAULT_LEROBOT_SRC),
        help="Path to lerobot/src",
    )
    parser.add_argument("--device", type=str, default="auto", help="Export device: auto/cpu/cuda")
    parser.add_argument(
        "--input_sequence_length",
        type=int,
        default=128,
        help="Sequence length used by dummy inputs and exported ONNX",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Pick up the red block and place it in the bin.",
        help="Prompt used to generate dummy token embeddings for export/calibration",
    )
    parser.add_argument("--output_name", type=str, default="smolvla_llm", help="Output file stem")
    parser.add_argument("--quant_type", type=str, default="w8a8h1_sefp", help="xhquant quant type")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--seed", type=int, default=42)
    export_llm(parser.parse_args())
