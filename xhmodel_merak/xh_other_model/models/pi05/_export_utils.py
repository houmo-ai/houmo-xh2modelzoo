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
        self.vision_tower = model.vision_tower.eval()
        self.multi_modal_projector = model.multi_modal_projector.eval()

    def forward(self, pixel_values):
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        return self.multi_modal_projector(selected_image_feature)


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

    policy = PI05Policy.from_pretrained(model_path, strict=True)
    policy.to(device)
    policy.config.device = device
    return policy.eval()


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
