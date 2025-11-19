from typing import Any

import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST

from ..registry import DATASETS


@DATASETS.register_module()
class NoxMNIST(MNIST):
    def __init__(
        self,
        root,
        train: bool = True,
    ) -> None:
        transform = transforms.ToTensor()
        super().__init__(root, train, transform, download=True)

    def __getitem__(self, index: int) -> Any:
        img, _ = super().__getitem__(index)
        return img
