from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

if hasattr(torch, "tensor_split"):
    tensor_split = torch.tensor_split
else:
    # A simple implementation of `tensor_split`.
    def tensor_split(input: torch.Tensor, indices: list):
        outs = []
        for start, end in zip([0] + indices, indices + [input.size(0)]):
            outs.append(input[start:end])
        return outs


LABEL_TYPE = Union[torch.Tensor, np.ndarray, Sequence, int]
SCORE_TYPE = Union[torch.Tensor, np.ndarray, Sequence]
from ..structures import SegSampleList


def format_label(value: LABEL_TYPE) -> torch.Tensor:
    """Convert various python types to label-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.

    Returns:
        :obj:`torch.Tensor`: The foramtted label tensor.
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f"Type {type(value)} is not an available label type.")
    assert value.ndim == 1, f"The dims of value should be 1, but got {value.ndim}."

    return value


def format_score(value: SCORE_TYPE) -> torch.Tensor:
    """Convert various python types to score-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence): Score values.

    Returns:
        :obj:`torch.Tensor`: The foramtted score tensor.
    """

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).float()
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f"Type {type(value)} is not an available label type.")
    assert value.ndim == 1, f"The dims of value should be 1, but got {value.ndim}."

    return value


def cat_batch_labels(elements: List[torch.Tensor]):
    """Concat a batch of label tensor to one tensor.

    Args:
        elements (List[tensor]): A batch of labels.

    Returns:
        Tuple[torch.Tensor, List[int]]: The first item is the concated label
        tensor, and the second item is the split indices of every sample.
    """
    labels = []
    splits = [0]
    for element in elements:
        labels.append(element)
        splits.append(splits[-1] + element.size(0))
    batch_label = torch.cat(labels)
    return batch_label, splits[1:-1]


def batch_label_to_onehot(batch_label, split_indices, num_classes):
    """Convert a concated label tensor to onehot format.

    Args:
        batch_label (torch.Tensor): A concated label tensor from multiple
            samples.
        split_indices (List[int]): The split indices of every sample.
        num_classes (int): The number of classes.

    Returns:
        torch.Tensor: The onehot format label tensor.

    Examples:
        >>> import torch
        >>> from mmpretrain.structures import batch_label_to_onehot
        >>> # Assume a concated label from 3 samples.
        >>> # label 1: [0, 1], label 2: [0, 2, 4], label 3: [3, 1]
        >>> batch_label = torch.tensor([0, 1, 0, 2, 4, 3, 1])
        >>> split_indices = [2, 5]
        >>> batch_label_to_onehot(batch_label, split_indices, num_classes=5)
        tensor([[1, 1, 0, 0, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0]])
    """
    sparse_onehot_list = F.one_hot(batch_label, num_classes)
    onehot_list = [sparse_onehot.sum(0) for sparse_onehot in tensor_split(sparse_onehot_list, split_indices)]
    return torch.stack(onehot_list)


def label_to_onehot(label: LABEL_TYPE, num_classes: int):
    """Convert a label to onehot format tensor.

    Args:
        label (LABEL_TYPE): Label value.
        num_classes (int): The number of classes.

    Returns:
        torch.Tensor: The onehot format label tensor.

    Examples:
        >>> import torch
        >>> from mmpretrain.structures import label_to_onehot
        >>> # Single-label
        >>> label_to_onehot(1, num_classes=5)
        tensor([0, 1, 0, 0, 0])
        >>> # Multi-label
        >>> label_to_onehot([0, 2, 3], num_classes=5)
        tensor([1, 0, 1, 1, 0])
    """
    label = format_label(label)
    sparse_onehot = F.one_hot(label, num_classes)
    return sparse_onehot.sum(0)


def stack_batch(
    tensor_list: List[torch.Tensor],
    pad_size_divisor: int = 1,
    pad_value: Union[int, float] = 0,
) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the tensor to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
       Tensor: The n dim tensor.
    """
    assert isinstance(tensor_list, list), f"Expected input type to be list, but got {type(tensor_list)}"
    assert tensor_list, "`tensor_list` could not be an empty list"
    assert len({tensor.ndim for tensor in tensor_list}) == 1, (
        f"Expected the dimensions of all tensors must be the same, "
        f"but got {[tensor.ndim for tensor in tensor_list]}"
    )

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor([tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)


def seg_stack_batch(
    inputs: List[torch.Tensor],
    data_samples: Optional[SegSampleList] = None,
    size: Optional[tuple] = None,
    size_divisor: Optional[int] = None,
    pad_val: Union[int, float] = 0,
    seg_pad_val: Union[int, float] = 255,
) -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """
    assert isinstance(inputs, list), f"Expected input type to be list, but got {type(inputs)}"
    assert len({tensor.ndim for tensor in inputs}) == 1, (
        f"Expected the dimensions of all inputs must be the same, " f"but got {[tensor.ndim for tensor in inputs]}"
    )
    assert inputs[0].ndim == 3, f"Expected tensor dimension to be 3, " f"but got {inputs[0].ndim}"
    assert len({tensor.shape[0] for tensor in inputs}) == 1, (
        f"Expected the channels of all inputs must be the same, " f"but got {[tensor.shape[0] for tensor in inputs]}"
    )

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (size_divisor is not None), "only one of size and size_divisor should be valid"

    padded_inputs = []
    padded_samples = []
    inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
    max_size = np.stack(inputs_sizes).max(0)
    if size_divisor is not None and size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size + (size_divisor - 1)) // size_divisor * size_divisor

    for i in range(len(inputs)):
        tensor = inputs[i]
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        elif size_divisor is not None:
            width = max(max_size[-1] - tensor.shape[-1], 0)
            height = max(max_size[-2] - tensor.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]

        # pad img
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_inputs.append(pad_img)
        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            pad_shape = None
            if "gt_sem_seg" in data_sample:
                gt_sem_seg = data_sample.gt_sem_seg.data
                del data_sample.gt_sem_seg.data
                data_sample.gt_sem_seg.data = F.pad(gt_sem_seg, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_sem_seg.shape
            if "gt_edge_map" in data_sample:
                gt_edge_map = data_sample.gt_edge_map.data
                del data_sample.gt_edge_map.data
                data_sample.gt_edge_map.data = F.pad(gt_edge_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_edge_map.shape
            if "gt_depth_map" in data_sample:
                gt_depth_map = data_sample.gt_depth_map.data
                del data_sample.gt_depth_map.data
                data_sample.gt_depth_map.data = F.pad(gt_depth_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_depth_map.shape
            data_sample.set_metainfo(
                {
                    "img_shape": tensor.shape[-2:],
                    "pad_shape": pad_shape,
                    "padding_size": padding_size,
                }
            )
            padded_samples.append(data_sample)
        else:
            padded_samples.append(dict(img_padding_size=padding_size, pad_shape=pad_img.shape[-2:]))

    return torch.stack(padded_inputs, dim=0), padded_samples
