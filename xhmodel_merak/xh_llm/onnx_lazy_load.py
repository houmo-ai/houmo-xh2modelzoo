"""Lazy-loading utilities for ONNX models with external data.

When an ONNX model is saved with ``save_as_external_data=True``, the weight
tensors live in a separate binary file.  ``onnx.load()`` reads that file
eagerly, which doubles peak memory for large vision encoders.

This module provides :func:`lazy_load_onnx` which:

1. Loads only the graph structure (``load_external_data=False``).
2. Memory-maps the external data file.
3. Returns a ``LazyOnnxModel`` wrapper that intercepts tensor access
   and loads data from the mmap on demand.

After calling :meth:`LazyOnnxModel.load_all_tensors` (or letting
``to_frontend_graph`` consume the model), all tensors behave identically
to a normally-loaded ``ModelProto``.
"""

from __future__ import annotations

import mmap
import os
from pathlib import Path

import onnx
from onnx import TensorProto
from onnx.external_data_helper import ExternalDataInfo, _get_all_tensors, uses_external_data


class _TensorSlice:
    """Describes a slice of mmap data for one tensor."""

    __slots__ = ("mm", "offset", "length")

    def __init__(self, mm: mmap.mmap, offset: int, length: int):
        self.mm = mm
        self.offset = offset
        self.length = length

    def read(self) -> bytes:
        self.mm.seek(self.offset)
        return self.mm.read(self.length)


class LazyOnnxModel:
    """Wraps an ONNX ``ModelProto`` with on-demand external data loading.

    Tensors that reference external data are tracked by tensor name.
    When :meth:`load_tensor` is called (or :meth:`load_all_tensors`),
    the raw bytes are read from the mmap and materialised into the
    protobuf ``raw_data`` field.
    """

    def __init__(self, model: onnx.ModelProto, mmaps: list[mmap.mmap], lazy_map: dict[str, _TensorSlice]):
        self.model = model
        self._mmaps = mmaps
        self._lazy_map = lazy_map  # tensor.name -> _TensorSlice

    # --- Expose ModelProto interface for downstream compatibility ---

    def __getattr__(self, name: str):
        return getattr(self.model, name)

    @property
    def proto(self) -> onnx.ModelProto:
        return self.model

    # --- Lazy loading API ---

    def load_tensor(self, tensor: TensorProto) -> None:
        """Force-load one tensor. No-op if already loaded."""
        ts = self._lazy_map.pop(tensor.name, None)
        if ts is not None:
            tensor.raw_data = ts.read()
            tensor.data_location = TensorProto.DEFAULT
            del tensor.external_data[:]

    def load_all_tensors(self) -> None:
        """Force-load all remaining lazy tensors in the model."""
        for tensor in _get_all_tensors(self.model):
            if tensor.name in self._lazy_map:
                self.load_tensor(tensor)

    @property
    def pending_count(self) -> int:
        """Number of tensors not yet loaded."""
        return len(self._lazy_map)

    def close(self) -> None:
        """Close all mmap handles. Call after all tensors are loaded."""
        for mm in self._mmaps:
            mm.close()
        self._mmaps.clear()
        self._lazy_map.clear()

    def __del__(self):
        self.close()


def lazy_load_onnx(onnx_path: str | Path) -> LazyOnnxModel:
    """Load an ONNX model, deferring external tensor data via mmap.

    Parameters
    ----------
    onnx_path:
        Path to the ``.onnx`` file.

    Returns
    -------
    LazyOnnxModel wrapping the ``ModelProto``.  External tensor data is
    loaded on demand when ``load_tensor`` / ``load_all_tensors`` is called,
    or when downstream code like ``to_frontend_graph`` reads ``raw_data``.

    Usage
    -----
    >>> lm = lazy_load_onnx("model.onnx")
    >>> lm.load_all_tensors()           # materialise everything
    >>> result = to_frontend_graph(lm.model, ...)
    >>> lm.close()
    """
    onnx_path = str(onnx_path)
    base_dir = os.path.dirname(os.path.abspath(onnx_path))

    model = onnx.load(onnx_path, load_external_data=False)

    ext_tensors = [t for t in _get_all_tensors(model) if uses_external_data(t)]
    if not ext_tensors:
        return LazyOnnxModel(model, [], {})

    # Group tensors by external file location
    files: dict[str, list[TensorProto]] = {}
    for t in ext_tensors:
        info = ExternalDataInfo(t)
        files.setdefault(info.location, []).append(t)

    mmaps: list[mmap.mmap] = []
    lazy_map: dict[str, _TensorSlice] = {}

    for location, tensors in files.items():
        file_path = os.path.join(base_dir, location)
        fd = os.open(file_path, os.O_RDONLY)
        file_size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
        os.close(fd)
        mmaps.append(mm)

        for t in tensors:
            info = ExternalDataInfo(t)
            offset = info.offset or 0
            length = info.length or (file_size - offset)
            lazy_map[t.name] = _TensorSlice(mm, offset, length)

    return LazyOnnxModel(model, mmaps, lazy_map)
