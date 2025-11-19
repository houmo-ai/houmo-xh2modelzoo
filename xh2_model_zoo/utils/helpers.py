import dataclasses
import hashlib
import logging
import math
import os
import shutil
import sys
import tempfile
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import chain
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import onnx
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

_hub_dir = None
ENV_TORCH_HOME = "TORCH_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"


def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(ENV_TORCH_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "torch"))
    )
    return torch_home


def get_dir():
    r"""
    Get the Torch Hub cache directory used for storing downloaded models & weights.

    If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_HUB"):
        warnings.warn("TORCH_HUB is deprecated, please use env TORCH_HOME instead")

    if _hub_dir is not None:
        return _hub_dir
    return os.path.join(_get_torch_home(), "hub")


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

    """
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'.format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def download_ptfile_from_url(url: str, model_dir=None, progress: bool = True, file_name=None):
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (string, optional): name for the downloaded file. Filename from ``url`` will be used if not set.

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn("TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead")

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file


def load_state_dict_from_url(
    url: str, model_dir=None, map_location=None, progress: bool = True, check_hash: bool = False, file_name=None
):
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (string, optional): name for the downloaded file. Filename from ``url`` will be used if not set.

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn("TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead")

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return torch.load(cached_file, map_location=map_location)


def get_onnx_from_url(
    url: str,
    model_dir=None,
    base_folder="data/model_zoo2/",
    company="unkown",
    progress: bool = True,
    file_name=None,
    return_path=False,
):
    if model_dir is None:
        model_dir = os.path.join(str(Path(__file__).parent.resolve().parent.parent), base_folder, company)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    onnx_model = onnx.load(cached_file)

    if return_path:
        return onnx_model, cached_file

    return onnx_model


def get_pth_from_url(
    url: str, model_dir=None, base_folder="data/model_zoo2/", company="unkown", progress: bool = True, file_name=None
):
    if model_dir is None:
        model_dir = os.path.join(str(Path(__file__).parent.resolve().parent.parent), base_folder, company)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file
