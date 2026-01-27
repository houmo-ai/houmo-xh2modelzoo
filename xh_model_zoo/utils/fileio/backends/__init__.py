# Copyright 2025 HOUMO AI
#
# File: __init__.py
# Description:
#   Backends module initialization.
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

from .base import BaseStorageBackend
from .http_backend import HTTPBackend
from .lmdb_backend import LmdbBackend
from .local_backend import LocalBackend
from .memcached_backend import MemcachedBackend
from .petrel_backend import PetrelBackend
from .registry_utils import backends, prefix_to_backends, register_backend

__all__ = [
    "BaseStorageBackend",
    "LocalBackend",
    "HTTPBackend",
    "LmdbBackend",
    "MemcachedBackend",
    "PetrelBackend",
    "register_backend",
    "backends",
    "prefix_to_backends",
]
