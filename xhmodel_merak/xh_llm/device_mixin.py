import torch


class DeviceMixin:
    def _set_device(self, device: torch.device | list[torch.device] | str | list[str]) -> "DeviceMixin":
        raise NotImplementedError("DeviceMixin should implement _set_device method to set the device for the model")

    def _set_dtype(self, dtype):
        raise NotImplementedError("DeviceMixin should implement _set_dtype method to set the dtype for the model")

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            if device.index is None:
                device = torch.device(device.type, 0)
            self._set_device(device)
        if dtype is not None:
            self._set_dtype(dtype)
        return self

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def auto_offload(self, devices=None):
        raise NotImplementedError("autooffload method should be implemented in the compatible module")
