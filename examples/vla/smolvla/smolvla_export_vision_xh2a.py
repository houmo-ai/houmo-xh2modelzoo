import argparse
import os
import os.path as osp
import random
import sys
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


ORIGIN_WORKDIR = Path("work_dirs/smolvla_vision")
ONNX_DIR = ORIGIN_WORKDIR / "smolvla_vision_onnx"
HMONNX_DIR = ORIGIN_WORKDIR / "smolvla_vision_hmonnx"
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


class SmolVLAVisionPart(nn.Module):
    """Export wrapper for SmolVLA vision tower + connector."""

    def __init__(self, vlm_with_expert: nn.Module, normalize_input: bool = True):
        super().__init__()
        vlm_model = vlm_with_expert.get_vlm_model()
        self.vision_model = vlm_model.vision_model
        self.connector = vlm_model.connector
        self.normalize_input = normalize_input

        self.vision_model.eval()
        self.connector.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.contiguous()
        if self.normalize_input:
            # LeRobot SmolVLA preprocesses images from [0, 1] to [-1, 1].
            pixel_values = pixel_values * 2.0 - 1.0

        vision_dtype = next(self.vision_model.parameters()).dtype
        pixel_values = pixel_values.to(dtype=vision_dtype)

        image_hidden_states = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=None,
        ).last_hidden_state
        image_hidden_states = self.connector(image_hidden_states)
        return image_hidden_states


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


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

    try:
        config = load_policy_config_offline(model_path, PreTrainedConfig)
        policy = SmolVLAPolicy.from_pretrained(model_path, config=config, strict=False)
        policy.eval()
        policy.to(device)
        image_size = getattr(config, "resize_imgs_with_padding", (512, 512))
        print(f"Loaded SmolVLA policy from: {model_path}")
        return policy.model.vlm_with_expert, tuple(image_size), "policy"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"policy load failed: {exc}")

    try:
        vlm_with_expert = SmolVLMWithExpertModel(
            model_id=model_path,
            load_vlm_weights=True,
            freeze_vision_encoder=True,
            train_expert_only=True,
            device=str(device),
        )
        vlm_with_expert.eval()
        image_size = (512, 512)
        print(f"Loaded SmolVLM backbone from: {model_path}")
        return vlm_with_expert, image_size, "vlm"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"vlm load failed: {exc}")

    raise RuntimeError(
        "Unable to load SmolVLA or SmolVLM weights from the given model_path.\n"
        + "\n".join(errors)
    )


def export_vision(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    export_dtype = torch.float16 if device.type == "cuda" else torch.float32

    vlm_with_expert, default_image_size, load_mode = load_vlm_with_expert(
        args.model_path,
        device,
        args.lerobot_src,
    )
    image_height = args.image_height or int(default_image_size[0])
    image_width = args.image_width or int(default_image_size[1])

    print(f"Load mode: {load_mode}")
    print(f"Export device: {device}")
    print(f"Input resolution: {image_height}x{image_width}")
    print(f"Input normalize [0,1]->[-1,1]: {not args.no_normalize_input}")

    vision_model = SmolVLAVisionPart(
        vlm_with_expert=vlm_with_expert,
        normalize_input=not args.no_normalize_input,
    )
    vision_model.eval()
    vision_model.to(device=device, dtype=export_dtype)

    temp_onnx_file = ONNX_DIR / f"{args.output_name}.onnx"
    simplified_onnx_file = ONNX_DIR / f"{args.output_name}_simplified.onnx"
    out_hmonnx_file = HMONNX_DIR / f"{args.output_name}_xh2.onnx"

    dummy_input = torch.rand(
        1,
        3,
        image_height,
        image_width,
        dtype=export_dtype,
        device=device,
    )

    print("Exporting SmolVLA vision to ONNX...")
    torch.onnx.export(
        vision_model,
        dummy_input,
        str(temp_onnx_file),
        input_names=["pixel_values"],
        output_names=["image_embeddings"],
        opset_version=args.opset,
        verbose=False,
        do_constant_folding=True,
    )

    print("Simplifying ONNX...")
    onnx_model = onnx.load(str(temp_onnx_file))
    model_simplified, check = simplify(
        onnx_model,
        test_input_shapes={"pixel_values": [1, 3, image_height, image_width]},
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

    calib_input = dummy_input.detach().cpu()
    convert_onnx_to_hmonnx(
        str(simplified_onnx_file),
        (calib_input,),
        out_hmonnx_file=osp.join(str(out_hmonnx_file)),
        device_type="XH2A",
        quant_config=quant_config,
    )

    print(f"ONNX saved to: {simplified_onnx_file}")
    print(f"HMONNX saved to: {out_hmonnx_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SmolVLA vision tower + connector to ONNX/HMONNX")
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
    parser.add_argument("--image_height", type=int, default=None, help="Override export input height")
    parser.add_argument("--image_width", type=int, default=None, help="Override export input width")
    parser.add_argument("--output_name", type=str, default="smolvla_vision", help="Output file stem")
    parser.add_argument("--quant_type", type=str, default="w8a8h1_sefp", help="xhquant quant type")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--no_normalize_input",
        action="store_true",
        help="Disable built-in [0,1] -> [-1,1] normalization before vision forward",
    )
    parser.add_argument("--seed", type=int, default=42)
    export_vision(parser.parse_args())
