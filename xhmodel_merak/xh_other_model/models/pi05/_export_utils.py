import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from onnxsim import simplify


class Siglip(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, pixel_values):
        image_features = self.model.get_image_features(pixel_values)
        if isinstance(image_features, torch.Tensor):
            return image_features
        return image_features.pooler_output * self.model.config.text_config.hidden_size**0.5


class TimeMLPWrapper(nn.Module):
    def __init__(self, time_mlp_in, time_mlp_out):
        super().__init__()
        self.time_mlp_in = time_mlp_in
        self.time_mlp_out = time_mlp_out

    def forward(self, time_emb):
        from torch.nn import functional as F

        x = self.time_mlp_in(time_emb)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        return F.silu(x)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pi05_policy(model_path: str, device: str = "cpu"):
    from lerobot.policies.pi05 import PI05Policy

    config = _load_local_pi05_config_compat(model_path)
    if config is None:
        from lerobot.configs.policies import PreTrainedConfig

        config = PreTrainedConfig.from_pretrained(model_path)
    policy = PI05Policy.from_pretrained(model_path, config=config, strict=True)
    policy.to(device)
    policy.config.device = device
    return policy.eval()


def _load_local_pi05_config_compat(model_path: str):
    config_file = Path(model_path) / "config.json"
    if not config_file.is_file():
        return None

    raw_config = json.loads(config_file.read_text(encoding="utf-8"))
    if raw_config.get("relative_exclude_joints", []) is not None:
        return None

    import draccus
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    raw_config.pop("type", None)
    raw_config["relative_exclude_joints"] = []
    return draccus.decode(PI05Config, raw_config)


def export_onnx_and_simplify(
    model: nn.Module,
    inputs: torch.Tensor | Sequence[Any],
    onnx_file: Path,
    simplified_onnx_file: Path,
    input_names: list[str],
    output_names: list[str],
    test_input_shapes: dict[str, list[int]],
    *,
    opset_version: int = 17,
    verbose: bool = False,
) -> Path:
    import onnx

    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    simplified_onnx_file.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        str(onnx_file),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        verbose=verbose,
    )
    onnx_model = onnx.load(str(onnx_file))
    model_simplified, check = simplify(onnx_model, test_input_shapes=test_input_shapes)
    if not check:
        raise RuntimeError(f"ONNX simplification failed for {onnx_file}")
    onnx.save(model_simplified, str(simplified_onnx_file))
    return simplified_onnx_file
