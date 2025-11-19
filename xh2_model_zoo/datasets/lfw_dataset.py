import copy
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets
from xhquant.utils import fileio
from xhquant.utils.logger import get_root_logger

from ..registry import DATASETS, TRANSFORMS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class LFWDataset(datasets.ImageFolder):
    def __init__(self):
        raise NotImplementedError
