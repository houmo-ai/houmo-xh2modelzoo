from numbers import Number
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from xhquant.utils import is_seq_of

from ..registry import MODELS
from ..structures import DetDataSample, PixelData
from .img_data_preprocessor import ImgDataPreprocessor


@MODELS.register_module()
class KeypointDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for detection tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It supports batch augmentations.
    2. It will additionally append batch_input_shape and pad_shape
    to data_samples considering the object detection task.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size (tuple, optional): Fixed padding size.
            Expected padding shape (w, h). Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        boxtype2tensor (bool): Whether to convert the ``BaseBoxes`` type of
            bboxes data to ``Tensor`` type. Defaults to True.
        non_blocking (bool): Whether block current process
            when transferring data to device. Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
    """

    def __init__(
        self,
        mean: Optional[Sequence[Union[float, int]]] = None,
        std: Optional[Sequence[Union[float, int]]] = None,
        pad_size: Optional[Sequence[int]] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        pad_mask: bool = False,
        mask_pad_value: int = 0,
        pad_seg: bool = False,
        seg_pad_value: int = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        boxtype2tensor: bool = True,
        non_blocking: Optional[bool] = False,
        batch_augments: Optional[List[dict]] = None,
        data_field: str = "inputs",
    ):
        super().__init__(
            mean=mean,
            std=std,
            pad_size=pad_size,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking,
            data_field=data_field,
        )
        self.batch_augments: Optional[nn.ModuleList] = None
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList([MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor
        self.scale = 0.5
        self.boxsize = 368
        self.stride = 8
        self.padValue = 128

    def forward(self, in_data: dict, training: bool = False) -> dict:
        """Perform normalization,padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = super().forward(data=in_data, training=training)

        data_field = self.data_field
        if data_field is None:
            data_field = "inputs"

        assert isinstance(data, dict) and data_field in data and "data_samples" in data
        inputs, data_samples = data[data_field], data["data_samples"]

        h = inputs.shape[0]
        w = inputs.shape[1]

        pad = 4 * [None]
        pad[0] = 0  # up
        pad[1] = 0  # left
        pad[2] = 0 if (h % self.stride == 0) else self.stride - (h % self.stride)  # down
        pad[3] = 0 if (w % self.stride == 0) else self.stride - (w % self.stride)

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            # batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample in data_samples:
                data_sample.set_metainfo(
                    {
                        "pad": pad,
                    }
                )

        in_data = {key: value for key, value in in_data.items()}
        in_data[data_field] = inputs
        in_data["data_samples"] = data_samples
        return in_data

        # return {"inputs": inputs, "data_samples": data_samples}

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data["inputs"]
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                multiplier = self.scale * self.boxsize / ori_input.shape[0]
                imageToTest = cv2.resize(ori_input, (0, 0), fx=multiplier, fy=multiplier, interpolation=cv2.INTER_CUBIC)
                imageToTest_padded, pad = self.padRightDownCorner(imageToTest, self.stride, self.padValue)
                im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1))  # / 256 - 0.5
                im = np.ascontiguousarray(im)
                im = torch.tensor(im)

        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                "The input of `ImgDataPreprocessor` should be a NCHW tensor "
                "or a list of tensor, but got a tensor with shape: "
                f"{_batch_inputs.shape}"
            )
            multiplier = self.scale * self.boxsize / ori_input.shape[0]
            imageToTest = cv2.resize(ori_input, (0, 0), fx=multiplier, fy=multiplier, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = self.padRightDownCorner(imageToTest, self.stride, self.padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1))  # / 256 - 0.5
            im = np.ascontiguousarray(im)
            im = torch.tensor(im)
        else:
            raise TypeError(
                "Output of `cast_data` should be a dict "
                "or a tuple with inputs and data_samples, but got"
                f"{type(data)}: {data}"
            )
        return im, imageToTest_padded, pad

    def pad_gt_masks(self, batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if "masks" in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(data_samples.batch_input_shape, pad_val=self.mask_pad_value)

    def pad_gt_sem_seg(self, batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_sem_seg to shape of batch_input_shape."""
        if "gt_sem_seg" in batch_data_samples[0]:
            for data_samples in batch_data_samples:
                gt_sem_seg = data_samples.gt_sem_seg.sem_seg
                h, w = gt_sem_seg.shape[-2:]
                pad_h, pad_w = data_samples.batch_input_shape
                gt_sem_seg = F.pad(
                    gt_sem_seg,
                    pad=(0, max(pad_w - w, 0), 0, max(pad_h - h, 0)),
                    mode="constant",
                    value=self.seg_pad_value,
                )
                data_samples.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)

    def padRightDownCorner(self, img, stride, padValue):
        h = img.shape[0]
        w = img.shape[1]

        pad = 4 * [None]
        pad[0] = 0  # up
        pad[1] = 0  # left
        pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
        pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

        img_padded = img
        pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
        img_padded = np.concatenate((pad_up, img_padded), axis=0)
        pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
        img_padded = np.concatenate((pad_left, img_padded), axis=1)
        pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
        img_padded = np.concatenate((img_padded, pad_down), axis=0)
        pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
        img_padded = np.concatenate((img_padded, pad_right), axis=1)

        return img_padded, pad
