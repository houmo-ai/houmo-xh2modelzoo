from pathlib import Path
from typing import Any

import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from ..registry import DATASETS


@DATASETS.register_module()
class LlamaDataset(data.Dataset):
    def __init__(self, root: str):
        self._hidden_states = torch.load(Path(root) / "hidden_states.pt", weights_only=True, map_location="cpu")
        self._position_embeddings = torch.load(
            Path(root) / "position_embeddings.pt", weights_only=True, map_location="cpu"
        )
        self._position_ids = torch.load(Path(root) / "position_ids.pt", weights_only=True, map_location="cpu")

    def __getitem__(self, index: int) -> Any:
        return (
            self._hidden_states[index],
            self._position_ids[index],
            (
                self._position_embeddings[0][index],
                self._position_embeddings[1][index],
            ),
        )

    def __len__(self) -> int:
        return len(self._hidden_states)
