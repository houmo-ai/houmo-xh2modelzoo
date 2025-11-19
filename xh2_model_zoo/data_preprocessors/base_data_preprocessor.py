from typing import Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
from typing_extensions import Self

from ..registry import MODELS
from ..structures import BaseDataElement

CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str, None]


@MODELS.register_module()
class BaseDataPreprocessor(nn.Module):
    def __init__(self, data_field: str = "inputs", non_blocking: Optional[bool] = False):
        super().__init__()
        self._non_blocking = non_blocking
        self._device = torch.device("cpu")
        self._dtype = None
        self.data_field = data_field  # 处理哪个字段的数据

    def cast_data(self, data: CastData) -> CastData:
        """Copying data to the target device.

        Args:
            data (dict): Data returned by ``DataLoader``.

        Returns:
            CollatedResult: Inputs and data sample at target device.
        """
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, tuple) and hasattr(data, "_fields"):
            # namedtuple
            return type(data)(*(self.cast_data(sample) for sample in data))  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, (torch.Tensor, BaseDataElement)):
            if self._dtype is not None and isinstance(data, torch.Tensor) and data.is_floating_point:
                data = data.to(self._dtype)
            return data.to(self.device, non_blocking=self._non_blocking)
        else:
            return data

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        """Preprocesses the data into the model input format.

        After the data pre-processing of :meth:`cast_data`, ``forward``
        will stack the input tensor list to a batch tensor at the first
        dimension.

        Args:
            data (dict): Data returned by dataloader
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        return self.cast_data(data)  # type: ignore

    @property
    def device(self):
        return self._device

    def _set_dtype(self, dtype) -> Self:
        self._dtype = dtype
        return self

    def _set_device(self, device) -> Self:
        self._device = device
        return self

    def to(self, *args, **kwargs) -> Self:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """

        # Since Torch has not officially merged
        # the npu-related fields, using the _parse_to function
        # directly will cause the NPU to not be found.
        # Here, the input parameters are processed to avoid errors.
        # if args and isinstance(args[0], str) and "npu" in args[0]:
        #     args = tuple([list(args)[0].replace("npu", torch.npu.native_device)])
        # if kwargs and "npu" in str(kwargs.get("device", "")):
        #     kwargs["device"] = kwargs["device"].replace("npu", torch.npu.native_device)

        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._device = torch.device(device)
        if dtype is not None:
            self._dtype = dtype
        return super().to(*args, **kwargs)

    def cuda(self, *args, **kwargs) -> Self:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.cuda.current_device())
        return super().cuda()

    def cpu(self, *args, **kwargs) -> Self:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device("cpu")
        return super().cpu()
