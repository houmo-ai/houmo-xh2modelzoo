# Copyright 2025 HOUMO AI
#
# File: misc.py
# Description:
#   Lightweight misc helpers used by config/fileio modules.
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
import importlib
import warnings
from typing import Optional, Sequence, Tuple, Type, Union


def is_str(x) -> bool:
    return isinstance(x, str)


def is_seq_of(
    seq: Sequence,
    expected_type: Union[Type, Tuple[Type, ...]],
    seq_type: Optional[Type] = None,
) -> bool:
    if seq_type is None:
        exp_seq_type = Sequence
    else:
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    return all(isinstance(item, expected_type) for item in seq)


def import_modules_from_strings(imports, allow_failed_imports: bool = False):
    """Import modules from string specs.

    Args:
        imports: module name or a list/tuple of module names.
        allow_failed_imports: return None for failed imports instead of raising.
    """
    if imports is None:
        return None
    if isinstance(imports, str):
        imports = [imports]
        single_import = True
    else:
        single_import = False

    if not isinstance(imports, (list, tuple)):
        raise TypeError(f"imports must be a str/list/tuple, but got {type(imports)}")

    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported")
        try:
            imported_tmp = importlib.import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise
        imported.append(imported_tmp)

    if single_import:
        return imported[0]
    return imported
