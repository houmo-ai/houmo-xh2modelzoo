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


ORIGIN_WORKDIR = Path("work_dirs/smolvla_llm_kvcache")
PREFILL_ONNX_DIR = ORIGIN_WORKDIR / "prefill_onnx"
DECODE_ONNX_DIR = ORIGIN_WORKDIR / "decode_onnx"
PREFILL_HMONNX_DIR = ORIGIN_WORKDIR / "prefill_hmonnx"
DECODE_HMONNX_DIR = ORIGIN_WORKDIR / "decode_hmonnx"
PREFILL_ONNX_DIR.mkdir(parents=True, exist_ok=True)
DECODE_ONNX_DIR.mkdir(parents=True, exist_ok=True)
PREFILL_HMONNX_DIR.mkdir(parents=True, exist_ok=True)
DECODE_HMONNX_DIR.mkdir(parents=True, exist_ok=True)


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
        "Unable to load SmolVLA or SmolVLM weights from the given model_path.\n" + "\n".join(errors)
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


def flatten_cache_dict(past_key_values: dict[int, dict[str, torch.Tensor]], num_layers: int) -> tuple[torch.Tensor, ...]:
    flat_cache: list[torch.Tensor] = []
    for layer_idx in range(num_layers):
        layer_cache = past_key_values[layer_idx]
        flat_cache.append(layer_cache["key_states"])
        flat_cache.append(layer_cache["value_states"])
    return tuple(flat_cache)


def build_prefill_input_names() -> list[str]:
    return ["prefix_embs", "attention_mask", "position_ids"]


def build_prefill_output_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for layer_idx in range(num_layers):
        names.append(f"past_key_{layer_idx}")
        names.append(f"past_value_{layer_idx}")
    return names


def build_decode_input_names(num_layers: int) -> list[str]:
    return ["suffix_embs", "attention_mask", "position_ids", *build_prefill_output_names(num_layers)]


def build_decode_output_names() -> list[str]:
    return ["suffix_hidden_state"]


class SmolVLAPrefillPart(nn.Module):
    def __init__(self, vlm_with_expert: nn.Module):
        super().__init__()
        self.vlm_with_expert = vlm_with_expert
        self.num_layers = vlm_with_expert.num_vlm_layers

    def forward(
        self,
        prefix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        prefix_dtype = self.vlm_with_expert.get_vlm_model().text_model.layers[0].self_attn.q_proj.weight.dtype
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=attention_mask.to(device=prefix_embs.device, dtype=torch.bool),
            position_ids=position_ids.to(device=prefix_embs.device, dtype=torch.long),
            past_key_values=None,
            inputs_embeds=[prefix_embs.to(dtype=prefix_dtype), None],
            use_cache=True,
            fill_kv_cache=True,
        )
        return flatten_cache_dict(past_key_values, self.num_layers)


class SmolVLADecodePart(nn.Module):
    def __init__(self, vlm_with_expert: nn.Module):
        super().__init__()
        self.vlm_with_expert = vlm_with_expert
        self.num_layers = vlm_with_expert.num_vlm_layers

    def forward(
        self,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *flat_cache: torch.Tensor,
    ) -> torch.Tensor:
        past_key_values: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx in range(self.num_layers):
            past_key_values[layer_idx] = {
                "key_states": flat_cache[layer_idx * 2],
                "value_states": flat_cache[layer_idx * 2 + 1],
            }

        suffix_dtype = self.vlm_with_expert.lm_expert.layers[0].self_attn.q_proj.weight.dtype
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=attention_mask.to(device=suffix_embs.device, dtype=torch.bool),
            position_ids=position_ids.to(device=suffix_embs.device, dtype=torch.long),
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs.to(dtype=suffix_dtype)],
            use_cache=True,
            fill_kv_cache=False,
        )
        return outputs_embeds[1]


@torch.no_grad()
def build_dummy_prefill_inputs(
    prefix_length: int,
    prefix_hidden_size: int,
    device: torch.device,
    export_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix_embs = torch.randn((1, prefix_length, prefix_hidden_size), device=device, dtype=export_dtype)
    prefix_attention_mask = torch.ones((1, prefix_length, prefix_length), device=device, dtype=torch.long)
    prefix_position_ids = torch.arange(prefix_length, device=device, dtype=torch.long).unsqueeze(0)
    return prefix_embs, prefix_attention_mask, prefix_position_ids


@torch.no_grad()
def build_dummy_decode_inputs(
    prefix_length: int,
    suffix_length: int,
    suffix_hidden_size: int,
    device: torch.device,
    export_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    suffix_embs = torch.randn((1, suffix_length, suffix_hidden_size), device=device, dtype=export_dtype)

    prefix_block = torch.ones((1, suffix_length, prefix_length), device=device, dtype=torch.long)
    suffix_block = torch.tril(torch.ones((suffix_length, suffix_length), device=device, dtype=torch.long))
    suffix_block = suffix_block.unsqueeze(0)
    full_attention_mask = torch.cat([prefix_block, suffix_block], dim=2)
    position_ids = prefix_length + torch.arange(suffix_length, device=device, dtype=torch.long).unsqueeze(0)
    return suffix_embs, full_attention_mask, position_ids


def export_single_onnx(
    model: nn.Module,
    model_inputs: tuple[torch.Tensor, ...],
    onnx_file: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
):
    torch.onnx.export(
        model,
        model_inputs,
        str(onnx_file),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        verbose=False,
        do_constant_folding=True,
    )


def simplify_onnx_model(onnx_file: Path, simplified_onnx_file: Path, input_shapes: dict[str, list[int]]):
    onnx_model = onnx.load(str(onnx_file))
    model_simplified, check = simplify(onnx_model, test_input_shapes=input_shapes)
    if not check:
        print("Warning: onnxsim check failed, saving simplified graph anyway.")
    onnx.save(model_simplified, str(simplified_onnx_file))


def export_kvcache(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    export_dtype = torch.float16 if device.type == "cuda" else torch.float32

    vlm_with_expert, load_mode = load_vlm_with_expert(args.model_path, device, args.lerobot_src)
    processor = getattr(vlm_with_expert, "processor", None)
    text_config = vlm_with_expert.config.text_config
    prefix_hidden_size = text_config.hidden_size
    suffix_hidden_size = vlm_with_expert.expert_hidden_size
    num_layers = vlm_with_expert.num_vlm_layers
    num_key_value_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim

    print(f"Load mode: {load_mode}")
    print(f"Export device: {device}")
    print(f"Prefix length: {args.prefix_length}")
    print(f"Suffix length: {args.suffix_length}")
    print(f"Num layers: {num_layers}")

    prefill_model = SmolVLAPrefillPart(vlm_with_expert)
    decode_model = SmolVLADecodePart(vlm_with_expert)
    prefill_model.eval()
    decode_model.eval()
    prefill_model.to(device=device, dtype=export_dtype)
    decode_model.to(device=device, dtype=export_dtype)

    prefix_embs, prefix_attention_mask, prefix_position_ids = build_dummy_prefill_inputs(
        prefix_length=args.prefix_length,
        prefix_hidden_size=prefix_hidden_size,
        device=device,
        export_dtype=export_dtype,
    )
    suffix_embs, decode_attention_mask, decode_position_ids = build_dummy_decode_inputs(
        prefix_length=args.prefix_length,
        suffix_length=args.suffix_length,
        suffix_hidden_size=suffix_hidden_size,
        device=device,
        export_dtype=export_dtype,
    )

    with torch.no_grad():
        flat_cache = prefill_model(prefix_embs, prefix_attention_mask, prefix_position_ids)

    work_dir = ORIGIN_WORKDIR
    work_dir.mkdir(parents=True, exist_ok=True)

    token_embedding = vlm_with_expert.get_vlm_model().text_model.get_input_embeddings()
    token_embedding_file = work_dir / "token_embedding.pt"
    token_embedding_state_dict = {key: value.detach().cpu() for key, value in token_embedding.state_dict().items()}
    torch.save(token_embedding_state_dict, token_embedding_file)

    hf_config_dir = save_hf_artifacts(args.model_path, processor, work_dir)

    prefill_input_names = build_prefill_input_names()
    prefill_output_names = build_prefill_output_names(num_layers)
    decode_input_names = build_decode_input_names(num_layers)
    decode_output_names = build_decode_output_names()

    prefill_onnx_file = PREFILL_ONNX_DIR / f"{args.output_name}_prefill.onnx"
    prefill_simplified_onnx_file = PREFILL_ONNX_DIR / f"{args.output_name}_prefill_simplified.onnx"
    prefill_hmonnx_file = PREFILL_HMONNX_DIR / f"{args.output_name}_prefill_xh2.onnx"
    decode_onnx_file = DECODE_ONNX_DIR / f"{args.output_name}_decode.onnx"
    decode_simplified_onnx_file = DECODE_ONNX_DIR / f"{args.output_name}_decode_simplified.onnx"
    decode_hmonnx_file = DECODE_HMONNX_DIR / f"{args.output_name}_decode_xh2.onnx"

    print("Exporting SmolVLA cache prefill ONNX...")
    export_single_onnx(
        prefill_model,
        (prefix_embs, prefix_attention_mask, prefix_position_ids),
        prefill_onnx_file,
        prefill_input_names,
        prefill_output_names,
        args.opset,
    )
    simplify_onnx_model(
        prefill_onnx_file,
        prefill_simplified_onnx_file,
        {
            "prefix_embs": list(prefix_embs.shape),
            "attention_mask": list(prefix_attention_mask.shape),
            "position_ids": list(prefix_position_ids.shape),
        },
    )

    print("Exporting SmolVLA cache decode ONNX...")
    decode_inputs = (suffix_embs, decode_attention_mask, decode_position_ids, *flat_cache)
    export_single_onnx(
        decode_model,
        decode_inputs,
        decode_onnx_file,
        decode_input_names,
        decode_output_names,
        args.opset,
    )
    decode_input_shapes = {
        "suffix_embs": list(suffix_embs.shape),
        "attention_mask": list(decode_attention_mask.shape),
        "position_ids": list(decode_position_ids.shape),
    }
    for name, tensor in zip(prefill_output_names, flat_cache, strict=False):
        decode_input_shapes[name] = list(tensor.shape)
    simplify_onnx_model(decode_onnx_file, decode_simplified_onnx_file, decode_input_shapes)

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)

    print("Converting prefill ONNX to HMONNX for XH2A...")
    prefill_calib_inputs = (
        prefix_embs.detach().cpu(),
        prefix_attention_mask.detach().cpu(),
        prefix_position_ids.detach().cpu(),
    )
    convert_onnx_to_hmonnx(
        str(prefill_simplified_onnx_file),
        prefill_calib_inputs,
        out_hmonnx_file=osp.join(str(prefill_hmonnx_file)),
        device_type="XH2A",
        quant_config=quant_config,
    )

    print("Converting decode ONNX to HMONNX for XH2A...")
    decode_calib_inputs = tuple(tensor.detach().cpu() for tensor in decode_inputs)
    convert_onnx_to_hmonnx(
        str(decode_simplified_onnx_file),
        decode_calib_inputs,
        out_hmonnx_file=osp.join(str(decode_hmonnx_file)),
        device_type="XH2A",
        quant_config=quant_config,
    )

    cache_shapes = {
        name: list(tensor.shape) for name, tensor in zip(prefill_output_names, flat_cache, strict=False)
    }
    meta_info = {
        "model_path": args.model_path,
        "load_mode": load_mode,
        "prefix_length": args.prefix_length,
        "suffix_length": args.suffix_length,
        "prefix_hidden_size": prefix_hidden_size,
        "suffix_hidden_size": suffix_hidden_size,
        "num_layers": num_layers,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "quant_type": args.quant_type,
        "token_embedding_file": token_embedding_file.name,
        "prefill": {
            "input_names": prefill_input_names,
            "output_names": prefill_output_names,
            "onnx_file": str(prefill_simplified_onnx_file.relative_to(work_dir)),
            "hmonnx_file": str(prefill_hmonnx_file.relative_to(work_dir)),
        },
        "decode": {
            "input_names": decode_input_names,
            "output_names": decode_output_names,
            "onnx_file": str(decode_simplified_onnx_file.relative_to(work_dir)),
            "hmonnx_file": str(decode_hmonnx_file.relative_to(work_dir)),
        },
        "cache_shapes": cache_shapes,
        "notes": [
            "prefill 对应 modeling_smolvla.py 中 fill_kv_cache=True 的 prefix cache 构建。",
            "decode 对应 modeling_smolvla.py 中 fill_kv_cache=False 的 suffix denoise 分支。",
            "decode 消费固定 prefix cache，不返回更新后的 cache。",
            "旧的 smolvla_export_llm_xh2a.py 只是 text_model 特征导出，不等价于这里的 prefill。",
        ],
    }
    if hf_config_dir is not None:
        meta_info["hf_config_dir"] = str(Path(hf_config_dir).relative_to(work_dir))

    meta_info_file = work_dir / "meta_info.json"
    with open(meta_info_file, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4, ensure_ascii=False)

    print(f"Token embedding saved to: {token_embedding_file}")
    if hf_config_dir is not None:
        print(f"HF config saved to: {hf_config_dir}")
    print(f"Prefill ONNX saved to: {prefill_simplified_onnx_file}")
    print(f"Prefill HMONNX saved to: {prefill_hmonnx_file}")
    print(f"Decode ONNX saved to: {decode_simplified_onnx_file}")
    print(f"Decode HMONNX saved to: {decode_hmonnx_file}")
    print(f"Meta info saved to: {meta_info_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SmolVLA cache prefill/decode ONNX/HMONNX")
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
        "--prefix_length",
        type=int,
        default=256,
        help="Sequence length used by exported prefix cache model",
    )
    parser.add_argument(
        "--suffix_length",
        type=int,
        default=50,
        help="Sequence length used by exported suffix decode model",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="smolvla_llm",
        help="Output file stem",
    )
    parser.add_argument("--quant_type", type=str, default="w8a8h1_sefp", help="xhquant quant type")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--seed", type=int, default=42)
    export_kvcache(parser.parse_args())