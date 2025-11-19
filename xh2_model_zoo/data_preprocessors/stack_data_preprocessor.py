import math
from numbers import Number
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F

from ..registry import MODELS
from ..structures import DataSample, MultiTaskDataSample
from .base_data_preprocessor import BaseDataPreprocessor
from .utils import batch_label_to_onehot, cat_batch_labels, stack_batch, tensor_split


@MODELS.register_module()
class StackDataPreprocessor(BaseDataPreprocessor):
    """Data pre-processor for stacking inputs.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        pad_size_divisor: int = 1,
        pad_value: Number = 0,
        to_rgb: bool = False,
        stack_dim=0,
        data_field: str = "inputs",
    ):
        super().__init__(data_field=data_field)
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.to_rgb = to_rgb

        if mean is not None:
            assert std is not None, (
                "To enable the normalization in " "preprocessing, please specify both `mean` and `std`."
            )
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer("std", torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False
        self.stack_dim = stack_dim

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        data_field = self.data_field
        if data_field is None:
            data_field = "inputs"
        imgs = data.get(data_field, None)

        def _process_img(img):
            # ------ To RGB ------
            if self.to_rgb and img.size(1) == 3:
                img = img.flip(1)

            # -- Normalization ---
            if self._enable_normalize:
                img = img.float()
                img = (img - self.mean) / self.std

            # ------ Padding -----
            if self.pad_size_divisor > 1:
                h, w = img.shape[-2:]

                target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                img = F.pad(img, (0, pad_w, 0, pad_h), "constant", self.pad_value)
            return img

        if isinstance(imgs, torch.Tensor):
            imgs = _process_img(imgs)
        elif isinstance(imgs, Sequence):
            imgs = torch.stack([_process_img(img) for img in imgs], dim=self.stack_dim)
        elif imgs is not None:
            raise ValueError(f"{type(imgs)} is not supported for imgs inputs.")

        data = {key: value for key, value in data.items()}
        data_samples = data.get("data_samples", None)
        data[data_field] = imgs
        data.setdefault("data_samples", None)
        data["data_samples"] = data_samples
        return data
        # return {"inputs": imgs, "data_samples": data_samples}
