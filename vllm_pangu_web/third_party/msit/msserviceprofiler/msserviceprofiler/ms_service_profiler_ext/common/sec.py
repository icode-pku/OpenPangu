# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import stat
import argparse

from .constants import EXT_SIZE_MAPPING, TENSOR_MAX_SIZE
from .utils import confirmation_interaction


def _has_soft_link(path: str) -> bool:
    abs_path = os.path.abspath(path)
    norm_path = os.path.normpath(abs_path)
    
    current = os.path.sep
    for part in norm_path.split(os.sep)[1:]:
        if not part:
            continue
        current = os.path.join(current, part)
        if os.path.islink(current):
            return True
    
    return os.path.islink(norm_path)


def _check_file_size_based_on_ext(path):
    ext = os.path.splitext(path)[1]
    size = os.path.getsize(path)

    if ext in EXT_SIZE_MAPPING:
        if size > EXT_SIZE_MAPPING[ext]:
            return False
    else:
        if size > TENSOR_MAX_SIZE:
            confirmation_prompt = "The file %r is larger than expected. " \
                                "Attempting to read such a file could potentially impact system performance.\n" \
                                "Please confirm your awareness of the risks associated with this action (y/n): " % path
            return confirmation_interaction(confirmation_prompt)

    return True


def _normal_user_extra_checks(
        path: str, 
        st: os.stat_result, 
        is_dir: True, 
        require_executable: True,
        error_type: Exception
    ):
    if _has_soft_link(path):
        raise error_type(f"Path contains soft links: {path!r}")
    
    access_mode = os.X_OK if require_executable else os.R_OK
    error_msg = "Path not executable" if require_executable else "Path not readable"

    if not os.access(path, access_mode):
        raise error_type(f"{error_msg}: {path!r}")
    
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise error_type(f"Insecure group or others write permissions: {path!r}")
    
    if st.st_uid not in (0, os.geteuid()):
        raise error_type(f"Path ownership mismatch: {path!r}")
    
    if not is_dir and not _check_file_size_based_on_ext(path):
        raise error_type(f"File too large: {path!r}")
    
    return os.path.realpath(path)


def _common_security_checks(
    path: str,
    *,
    is_dir: bool,
    require_executable: bool,
    raise_argparse: bool = False
) -> str:
    if not isinstance(path, str):
        raise TypeError("Expected path to be 'str', got %r instead" % type(path))
    
    error_type = argparse.ArgumentTypeError if raise_argparse else OSError
    
    try:
        st = os.stat(path)
    except OSError as e:
        raise error_type(f"File not found or no permission: {path!r}") from e
    
    file_type_is_valid = stat.S_ISDIR(st.st_mode) if is_dir else stat.S_ISREG(st.st_mode)
    error_msg = "Not a directory" if is_dir else "Not a regular file"
    
    if not file_type_is_valid:
        raise error_type(f"{error_msg}: {path!r}")
    
    if os.geteuid() == 0:
        return os.path.realpath(path)

    return _normal_user_extra_checks(path, st, is_dir, require_executable, error_type)


def read_file_common_check(path: str, *, raise_argparse: bool = True):
    return _common_security_checks(
        path, 
        is_dir=False, 
        require_executable=False, 
        raise_argparse=raise_argparse
    )


def execute_file_common_check(path: str, *, raise_argparse: bool = True):
    return _common_security_checks(
        path, 
        is_dir=False, 
        require_executable=True, 
        raise_argparse=raise_argparse
    )
    

def list_dir_common_check(path: str, *, raise_argparse: bool = True):
    return _common_security_checks(
        path, 
        is_dir=True, 
        require_executable=False, 
        raise_argparse=raise_argparse
    )


def traverse_dir_common_check(path: str, *, raise_argparse: bool = True):
    return _common_security_checks(
        path, 
        is_dir=True, 
        require_executable=True, 
        raise_argparse=raise_argparse
    )
