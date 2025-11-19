import warnings
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from xhquant.utils import is_str

from ...registry import TRANSFORMS
from ...structures import DataSample, DetDataSample, MultiTaskDataSample, SegDataSample
from ...structures.bbox import BaseBoxes
from ...structures.instance_data import InstanceData
from ...structures.pixel_data import PixelData
from .base import BaseTransform


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f"Type {type(data)} cannot be converted to tensor."
            "Supported types are: `numpy.ndarray`, `torch.Tensor`, "
            "`Sequence`, `int` and `float`"
        )


@TRANSFORMS.register_module()
class ClsPackInputs(BaseTransform):
    """Pack the inputs data.

    **Required Keys:**

    - ``input_key``
    - ``*algorithm_keys``
    - ``*meta_keys``

    **Deleted Keys:**

    All other keys in the dict.

    **Added Keys:**

    - inputs (:obj:`torch.Tensor`): The forward data of models.
    - data_samples (:obj:`~mmpretrain.structures.DataSample`): The
      annotation info of the sample.

    Args:
        input_key (str): The key of element to feed into the model forwarding.
            Defaults to 'img'.
        algorithm_keys (Sequence[str]): The keys of custom elements to be used
            in the algorithm. Defaults to an empty tuple.
        meta_keys (Sequence[str]): The keys of meta information to be saved in
            the data sample. Defaults to :attr:`ClsPackInputs.DEFAULT_META_KEYS`.

    .. admonition:: Default algorithm keys

        Besides the specified ``algorithm_keys``, we will set some default keys
        into the output data sample and do some formatting. Therefore, you
        don't need to set these keys in the ``algorithm_keys``.

        - ``gt_label``: The ground-truth label. The value will be converted
          into a 1-D tensor.
        - ``gt_score``: The ground-truth score. The value will be converted
          into a 1-D tensor.
        - ``mask``: The mask for some self-supervise tasks. The value will
          be converted into a tensor.

    .. admonition:: Default meta keys

        - ``sample_idx``: The id of the image sample.
        - ``img_path``: The path to the image file.
        - ``ori_shape``: The original shape of the image as a tuple (H, W).
        - ``img_shape``: The shape of the image after the pipeline as a
          tuple (H, W).
        - ``scale_factor``: The scale factor between the resized image and
          the original image.
        - ``flip``: A boolean indicating if image flip transform was used.
        - ``flip_direction``: The flipping direction.
    """

    DEFAULT_META_KEYS = (
        "sample_idx",
        "img_path",
        "ori_shape",
        "img_shape",
        "scale_factor",
        "flip",
        "flip_direction",
    )

    def __init__(self, input_key="img", algorithm_keys=(), meta_keys=DEFAULT_META_KEYS):
        self.input_key = input_key
        self.algorithm_keys = algorithm_keys
        self.meta_keys = meta_keys

    @staticmethod
    def format_input(input_):
        if isinstance(input_, list):
            return [ClsPackInputs.format_input(item) for item in input_]
        elif isinstance(input_, np.ndarray):
            if input_.ndim == 2:  # For grayscale image.
                input_ = np.expand_dims(input_, -1)
            if input_.ndim == 3 and not input_.flags.c_contiguous:
                input_ = np.ascontiguousarray(input_.transpose(2, 0, 1))
                input_ = to_tensor(input_)
            elif input_.ndim == 3:
                # convert to tensor first to accelerate, see
                # https://github.com/open-mmlab/mmdetection/pull/9533
                input_ = to_tensor(input_).permute(2, 0, 1).contiguous()
            else:
                # convert input with other shape to tensor without permute,
                # like video input (num_crops, C, T, H, W).
                input_ = to_tensor(input_)
        elif isinstance(input_, Image.Image):
            input_ = F.pil_to_tensor(input_)
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f"Unsupported input type {type(input_)}.")

        return input_

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = dict()
        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results["inputs"] = self.format_input(input_)

        data_sample = DataSample()

        # Set default keys
        if "gt_label" in results:
            data_sample.set_gt_label(results["gt_label"])
        if "gt_score" in results:
            data_sample.set_gt_score(results["gt_score"])
        if "mask" in results:
            data_sample.set_mask(results["mask"])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type="metainfo")

        packed_results["data_samples"] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_key='{self.input_key}', "
        repr_str += f"algorithm_keys={self.algorithm_keys}, "
        repr_str += f"meta_keys={self.meta_keys})"
        return repr_str


@TRANSFORMS.register_module()
class PackDetInputs(BaseTransform):
    """Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """

    mapping_table = {
        "gt_bboxes": "bboxes",
        "gt_bboxes_labels": "labels",
        "gt_masks": "masks",
    }

    def __init__(
        self,
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
        ),
    ):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results["inputs"] = img

        if "gt_ignore_flags" in results:
            valid_idx = np.where(results["gt_ignore_flags"] == 0)[0]
            ignore_idx = np.where(results["gt_ignore_flags"] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == "gt_masks" or isinstance(results[key], BaseBoxes):
                if "gt_ignore_flags" in results:
                    instance_data[self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if "gt_ignore_flags" in results:
                    instance_data[self.mapping_table[key]] = to_tensor(results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if "proposals" in results:
            proposals = InstanceData(
                bboxes=to_tensor(results["proposals"]),
                scores=to_tensor(results["proposals_scores"]),
            )
            data_sample.proposals = proposals

        if "gt_seg_map" in results:
            gt_sem_seg_data = dict(sem_seg=to_tensor(results["gt_seg_map"][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if "ignore_index" in results:
                metainfo = dict(ignore_index=results["ignore_index"])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str


@TRANSFORMS.register_module()
class PackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(
        self,
        meta_keys=(
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "reduce_zero_label",
        ),
    ):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results["inputs"] = img

        data_sample = SegDataSample()
        if "gt_seg_map" in results:
            if len(results["gt_seg_map"].shape) == 2:
                data = to_tensor(results["gt_seg_map"][None, ...].astype(np.int64))
            else:
                warnings.warn(
                    "Please pay attention your ground truth "
                    "segmentation map, usually the segmentation "
                    "map is 2D, but got "
                    f'{results["gt_seg_map"].shape}'
                )
                data = to_tensor(results["gt_seg_map"].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if "gt_edge_map" in results:
            gt_edge_data = dict(data=to_tensor(results["gt_edge_map"][None, ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        if "gt_depth_map" in results:
            gt_depth_data = dict(data=to_tensor(results["gt_depth_map"][None, ...]))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str


@TRANSFORMS.register_module()
class PackMultiInputs(BaseTransform):
    """Pack the inputs data.

    **Required Keys:**

    - ``input_key``
    - ``*algorithm_keys``
    - ``*meta_keys``

    **Deleted Keys:**

    All other keys in the dict.

    **Added Keys:**

    - inputs (:obj:`torch.Tensor`): The forward data of models.
    - data_samples (:obj:`~mmpretrain.structures.DataSample`): The
      annotation info of the sample.

    Args:
        input_key (str): The key of element to feed into the model forwarding.
            Defaults to 'img'.
        algorithm_keys (Sequence[str]): The keys of custom elements to be used
            in the algorithm. Defaults to an empty tuple.
        meta_keys (Sequence[str]): The keys of meta information to be saved in
            the data sample. Defaults to :attr:`PackInputs.DEFAULT_META_KEYS`.

    .. admonition:: Default algorithm keys

        Besides the specified ``algorithm_keys``, we will set some default keys
        into the output data sample and do some formatting. Therefore, you
        don't need to set these keys in the ``algorithm_keys``.

        - ``gt_label``: The ground-truth label. The value will be converted
          into a 1-D tensor.
        - ``gt_score``: The ground-truth score. The value will be converted
          into a 1-D tensor.
        - ``mask``: The mask for some self-supervise tasks. The value will
          be converted into a tensor.

    .. admonition:: Default meta keys

        - ``sample_idx``: The id of the image sample.
        - ``img_path``: The path to the image file.
        - ``ori_shape``: The original shape of the image as a tuple (H, W).
        - ``img_shape``: The shape of the image after the pipeline as a
          tuple (H, W).
        - ``scale_factor``: The scale factor between the resized image and
          the original image.
        - ``flip``: A boolean indicating if image flip transform was used.
        - ``flip_direction``: The flipping direction.
    """

    DEFAULT_META_KEYS = (
        "sample_idx",
        "img_path",
        "ori_shape",
        "img_shape",
        "scale_factor",
        "flip",
        "flip_direction",
    )

    def __init__(self, input_keys=("img",), algorithm_keys=(), meta_keys=DEFAULT_META_KEYS):
        self.input_keys = input_keys
        self.algorithm_keys = algorithm_keys
        self.meta_keys = meta_keys

    @staticmethod
    def format_input(input_):
        if isinstance(input_, list):
            return [PackMultiInputs.format_input(item) for item in input_]
        elif isinstance(input_, np.ndarray):
            if input_.ndim == 2:  # For grayscale image.
                input_ = np.expand_dims(input_, -1)
            if input_.ndim == 3 and not input_.flags.c_contiguous:
                input_ = np.ascontiguousarray(input_.transpose(2, 0, 1))
                input_ = to_tensor(input_)
            elif input_.ndim == 3:
                # convert to tensor first to accelerate, see
                # https://github.com/open-mmlab/mmdetection/pull/9533
                input_ = to_tensor(input_).permute(2, 0, 1).contiguous()
            else:
                # convert input with other shape to tensor without permute,
                # like video input (num_crops, C, T, H, W).
                input_ = to_tensor(input_)
        elif isinstance(input_, Image.Image):
            input_ = F.pil_to_tensor(input_)
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f"Unsupported input type {type(input_)}.")

        return input_

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = dict()
        for input_key in self.input_keys:
            if input_key in results:
                input_ = results[input_key]
                packed_results[input_key] = self.format_input(input_)

        # if self.input_key in results:
        #     input_ = results[self.input_key]
        #     packed_results["inputs"] = self.format_input(input_)

        data_sample = DataSample()

        # Set default keys
        if "gt_label" in results:
            data_sample.set_gt_label(results["gt_label"])
        if "gt_score" in results:
            data_sample.set_gt_score(results["gt_score"])
        if "mask" in results:
            data_sample.set_mask(results["mask"])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type="metainfo")

        packed_results["data_samples"] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_keys='{self.input_keys}', "
        repr_str += f"algorithm_keys={self.algorithm_keys}, "
        repr_str += f"meta_keys={self.meta_keys})"
        return repr_str


@TRANSFORMS.register_module()
class PackInputs(BaseTransform):
    """Pack the inputs data.

    **Required Keys:**

    - ``input_key``
    - ``*algorithm_keys``
    - ``*meta_keys``

    **Deleted Keys:**

    All other keys in the dict.

    **Added Keys:**

    - inputs (:obj:`torch.Tensor`): The forward data of models.
    - data_samples (:obj:`~mmpretrain.structures.DataSample`): The
      annotation info of the sample.

    Args:
        input_key (str): The key of element to feed into the model forwarding.
            Defaults to 'img'.
        algorithm_keys (Sequence[str]): The keys of custom elements to be used
            in the algorithm. Defaults to an empty tuple.
        meta_keys (Sequence[str]): The keys of meta information to be saved in
            the data sample. Defaults to :attr:`PackInputs.DEFAULT_META_KEYS`.

    .. admonition:: Default algorithm keys

        Besides the specified ``algorithm_keys``, we will set some default keys
        into the output data sample and do some formatting. Therefore, you
        don't need to set these keys in the ``algorithm_keys``.

        - ``gt_label``: The ground-truth label. The value will be converted
          into a 1-D tensor.
        - ``gt_score``: The ground-truth score. The value will be converted
          into a 1-D tensor.
        - ``mask``: The mask for some self-supervise tasks. The value will
          be converted into a tensor.

    .. admonition:: Default meta keys

        - ``sample_idx``: The id of the image sample.
        - ``img_path``: The path to the image file.
        - ``ori_shape``: The original shape of the image as a tuple (H, W).
        - ``img_shape``: The shape of the image after the pipeline as a
          tuple (H, W).
        - ``scale_factor``: The scale factor between the resized image and
          the original image.
        - ``flip``: A boolean indicating if image flip transform was used.
        - ``flip_direction``: The flipping direction.
    """

    DEFAULT_META_KEYS = (
        "sample_idx",
        "img_path",
        "ori_shape",
        "img_shape",
        "scale_factor",
        "flip",
        "flip_direction",
    )

    def __init__(self, input_key="img", algorithm_keys=(), meta_keys=DEFAULT_META_KEYS):
        self.input_key = input_key
        self.algorithm_keys = algorithm_keys
        self.meta_keys = meta_keys

    @staticmethod
    def format_input(input_):
        if isinstance(input_, list):
            return [PackInputs.format_input(item) for item in input_]
        elif isinstance(input_, np.ndarray):
            if input_.ndim == 2:  # For grayscale image.
                input_ = np.expand_dims(input_, -1)
            if input_.ndim == 3 and not input_.flags.c_contiguous:
                input_ = np.ascontiguousarray(input_.transpose(2, 0, 1))
                input_ = to_tensor(input_)
            elif input_.ndim == 3:
                # convert to tensor first to accelerate, see
                # https://github.com/open-mmlab/mmdetection/pull/9533
                input_ = to_tensor(input_).permute(2, 0, 1).contiguous()
            else:
                # convert input with other shape to tensor without permute,
                # like video input (num_crops, C, T, H, W).
                input_ = to_tensor(input_)
        elif isinstance(input_, Image.Image):
            input_ = F.pil_to_tensor(input_)
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f"Unsupported input type {type(input_)}.")

        return input_

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = dict()
        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results["inputs"] = self.format_input(input_)

        data_sample = DataSample()

        # Set default keys
        if "gt_label" in results:
            data_sample.set_gt_label(results["gt_label"])
        if "gt_score" in results:
            data_sample.set_gt_score(results["gt_score"])
        if "mask" in results:
            data_sample.set_mask(results["mask"])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type="metainfo")

        packed_results["data_samples"] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_key='{self.input_key}', "
        repr_str += f"algorithm_keys={self.algorithm_keys}, "
        repr_str += f"meta_keys={self.meta_keys})"
        return repr_str
