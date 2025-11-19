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
class ClsDataPreprocessor(BaseDataPreprocessor):
    def __init__(
        self,
        mean: Optional[Sequence[Number]] = None,
        std: Optional[Sequence[Number]] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[Number, int] = 0,
        to_rgb: bool = False,
        to_onehot: bool = False,
        num_classes: Optional[int] = None,
        batch_augments: Optional[dict] = None,
        data_field: str = "inputs",
    ):
        super().__init__(data_field)
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.to_rgb = to_rgb
        self.to_onehot = to_onehot
        self.num_classes = num_classes

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

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        data_field = self.data_field
        if data_field is None:
            data_field = "inputs"
        inputs = self.cast_data(data[data_field])

        if isinstance(inputs, torch.Tensor):
            # The branch if use `default_collate` as the collate_fn in the
            # dataloader.
            # ------ To RGB ------
            if self.to_rgb and inputs.size(1) == 3:
                inputs = inputs.flip(1)

            # -- Normalization ---
            inputs = inputs.float()
            if self._enable_normalize:
                inputs = (inputs - self.mean) / self.std

            # ------ Padding -----
            if self.pad_size_divisor > 1:
                h, w = inputs.shape[-2:]

                target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                inputs = F.pad(inputs, (0, pad_w, 0, pad_h), "constant", self.pad_value)
        else:
            # The branch if use `pseudo_collate` as the collate_fn in the
            # dataloader.

            processed_inputs = []
            for input_ in inputs:
                # ------ To RGB ------
                if self.to_rgb and input_.size(0) == 3:
                    input_ = input_.flip(0)

                # -- Normalization ---
                input_ = input_.float()
                if self._enable_normalize:
                    input_ = (input_ - self.mean) / self.std

                processed_inputs.append(input_)
            # Combine padding and stack
            inputs = stack_batch(processed_inputs, self.pad_size_divisor, self.pad_value)

        data_samples = data.get("data_samples", None)
        sample_item = data_samples[0] if data_samples is not None else None

        if isinstance(sample_item, DataSample):
            batch_label = None
            batch_score = None

            if "gt_label" in sample_item:
                gt_labels = [sample.gt_label for sample in data_samples]
                batch_label, label_indices = cat_batch_labels(gt_labels)
                batch_label = batch_label.to(self.device)
            if "gt_score" in sample_item:
                gt_scores = [sample.gt_score for sample in data_samples]
                batch_score = torch.stack(gt_scores).to(self.device)
            elif self.to_onehot and "gt_label" in sample_item:
                assert batch_label is not None, "Cannot generate onehot format labels because no labels."
                num_classes = self.num_classes or sample_item.get("num_classes")
                assert num_classes is not None, (
                    "Cannot generate one-hot format labels because not set " "`num_classes` in `data_preprocessor`."
                )
                batch_score = batch_label_to_onehot(batch_label, label_indices, num_classes).to(self.device)

            # ----- Batch Augmentations ----
            if training and self.batch_augments is not None and batch_score is not None:
                inputs, batch_score = self.batch_augments(inputs, batch_score)

            # ----- scatter labels and scores to data samples ---
            if batch_label is not None:
                for sample, label in zip(data_samples, tensor_split(batch_label, label_indices)):
                    sample.set_gt_label(label)
            if batch_score is not None:
                for sample, score in zip(data_samples, batch_score):
                    sample.set_gt_score(score)
        elif isinstance(sample_item, MultiTaskDataSample):
            data_samples = self.cast_data(data_samples)

        data = {key: value for key, value in data.items()}
        data[data_field] = inputs
        data["data_samples"] = data_samples
        return data
