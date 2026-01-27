# Copyright 2025 HOUMO AI
#
# File: __init__.py
# Description:
#   File I/O module initialization for xh_model_zoo.
#   This module exports file I/O backends, handlers, and utilities.
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
from .backends import (
    BaseStorageBackend,
    HTTPBackend,
    LmdbBackend,
    LocalBackend,
    MemcachedBackend,
    PetrelBackend,
    register_backend,
)
from .file_client import FileClient, HardDiskBackend
from .handlers import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler, register_handler
from .io import (
    copy_if_symlink_fails,
    copyfile,
    copyfile_from_local,
    copyfile_to_local,
    copytree,
    copytree_from_local,
    copytree_to_local,
    dump,
    exists,
    generate_presigned_url,
    get,
    get_file_backend,
    get_local_path,
    get_text,
    isdir,
    isfile,
    join_path,
    list_dir_or_file,
    load,
    put,
    put_text,
    remove,
    rmtree,
)
from .parse import dict_from_file, list_from_file

__all__ = [
    "BaseStorageBackend",
    "FileClient",
    "PetrelBackend",
    "MemcachedBackend",
    "LmdbBackend",
    "HardDiskBackend",
    "LocalBackend",
    "HTTPBackend",
    "copy_if_symlink_fails",
    "copyfile",
    "copyfile_from_local",
    "copyfile_to_local",
    "copytree",
    "copytree_from_local",
    "copytree_to_local",
    "exists",
    "generate_presigned_url",
    "get",
    "get_file_backend",
    "get_local_path",
    "get_text",
    "isdir",
    "isfile",
    "join_path",
    "list_dir_or_file",
    "put",
    "put_text",
    "remove",
    "rmtree",
    "load",
    "dump",
    "register_handler",
    "BaseFileHandler",
    "JsonHandler",
    "PickleHandler",
    "YamlHandler",
    "list_from_file",
    "dict_from_file",
    "register_backend",
]
