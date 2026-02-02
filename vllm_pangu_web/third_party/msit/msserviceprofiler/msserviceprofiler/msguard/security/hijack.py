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

from ..constraints import PathConstraint, Rule, InvalidParameterError


def update_env_s(
        env_var: str,
        path: str,
        constraint: PathConstraint = Rule.input_file_read,
        prepend: bool = True
    ) -> None:
    """
    Add a path to an environment variable for searching.

    Parameters
    ----------
    env_var : str
        The name of the environment variable to modify.
    path : str
        The directory path to add to the environment variable.
    prepend : bool, optional
        If True, add the path to the beginning of the variable. If False, add to the end.
        Default is True.

    Raises
    ------
    TypeError
        If `env_var` is not a string.

    Notes
    -----
    If the environment variable does not exist, it will be created with the given path.
    If `path` is not absolute, it will be converted to an absolute path.
    """
    if not isinstance(env_var, str):
        raise TypeError(
            f"Expected 'env_var' to be str. Got {type(env_var).__name__} instead."
        )

    abs_path = os.path.abspath(path)
    if not constraint.is_satisfied_by(abs_path):
        raise InvalidParameterError("env_var", "update_env_s", constraint, abs_path)

    current_value = os.environ.get(env_var)

    if current_value:
        parts = [p for p in current_value.split(os.pathsep)]
        if prepend:
            parts.insert(0, abs_path)
        else:
            parts.append(abs_path)
        new_value = os.pathsep.join(parts)
    else:
        new_value = abs_path

    os.environ[env_var] = new_value
