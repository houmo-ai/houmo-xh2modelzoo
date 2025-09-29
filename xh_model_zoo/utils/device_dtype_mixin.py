from typing import Optional, Union

import torch
import torch.nn as nn


class DeviceDtypeMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._device = torch.device("cuda")
        self._dtype = torch.float16
        self._exec_device = torch.device("cuda")

    @property
    def device(self):
        return self._device

    @property
    def execution_device(self):
        return self._exec_device

    @property
    def dtype(self):
        return self._dtype

    def _set_device(self, device: torch.device) -> None:
        self._device = device

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self._dtype = dtype

    def _set_exec_device(self, device: torch.device) -> None:
        self._exec_device = device

    def set_exec_device(self, device) -> "DeviceDtypeMixin":
        def apply_fn(module):
            if not hasattr(module, "_set_exec_device"):
                return
            module._set_exec_device(device)

        self.apply(apply_fn)
        return self

    def to(self, *args, **kwargs) -> "DeviceDtypeMixin":
        """Overrides this method to call :meth:`BaseDataPreprocessor.to`
        additionally.

        Returns:
            DeviceDtypeMixin: The model itself.
        """

        # Since Torch has not officially merged
        # the npu-related fields, using the _parse_to function
        # directly will cause the NPU to not be found.
        # Here, the input parameters are processed to avoid errors.

        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            # self._set_device(torch.device(device))
            def apply_fn(module):
                if not hasattr(module, "_set_device"):
                    return
                module._set_device(device)

            self.apply(apply_fn)
            # return super().to(*args, **kwargs)
            return self
        if dtype is not None:
            # self._set_dtype(dtype)
            def apply_fn(module):
                if not hasattr(module, "_set_dtype"):
                    return
                module._set_dtype(dtype)

            self.apply(apply_fn)
            # return super().to(*args, **kwargs)
            return self

    def cuda(
        self,
        device: Optional[Union[int, str, torch.device]] = None,
    ) -> "DeviceDtypeMixin":
        """Overrides this method to call :meth:`BaseDataPreprocessor.cuda`
        additionally.

        Returns:
            DeviceDtypeMixin: The model itself.
        """
        if isinstance(device, int):
            device = torch.device("cuda", index=device)
        self._set_device(torch.device(device))
        return super().cuda(device)

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.cpu`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        self._set_device(torch.device("cpu"))
        return super().cpu()
