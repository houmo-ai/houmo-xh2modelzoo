# Copyright 2025 HOUMO AI
#
# File: device_dtype_mixin.py
# Description:
#   Device and dtype management mixin for models.
#   This module provides DeviceDtypeMixin class for managing
#   device and dtype properties.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union

import torch
import torch.nn as nn


class DeviceDtypeMixin(nn.Module):
    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def _set_device(self, device: torch.device) -> None:
        self._device = device

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self._dtype = dtype

    def _set_exec_device(self, device: torch.device) -> None:
        self._exec_device = device

    def set_exec_device(self, device) -> None:
        def apply_fn(module):
            if not hasattr(module, "_set_exec_device"):
                return
            module._set_exec_device(device)

        self.apply(apply_fn)
        return self

    def to(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.to`
        additionally.

        Returns:
            nn.Module: The model itself.
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
    ) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.cuda`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        if device is None or isinstance(device, int):
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
