from pathlib import Path
from typing import Any

import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from ..registry import DATASETS


@DATASETS.register_module()
class Stereo(data.Dataset):
    def __init__(self, root: str):
        super().__init__()
        root_path = Path(root)
        left_path = root_path / "left"
        right_path = root_path / "right"
        left_images = sorted(left_path.glob("*.png"))
        self.image_infos = []
        for left_image_path in left_images:
            fname = left_image_path.name
            right_image_path = right_path / fname
            self.image_infos.append((str(left_image_path), str(right_image_path)))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index: int) -> Any:
        left_image_file, right_image_file = self.image_infos[index]
        images = []
        for image_file in [left_image_file, right_image_file]:
            image = Image.open(image_file).convert("RGB")
            image = self.transform(image)
            images.append(image)
        return tuple(images)

    def __len__(self) -> int:
        return len(self.image_infos)
