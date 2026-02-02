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
import copy

from .builder import Path
from ..logic import where
from ..base import BaseConstraint


class _RuleDescriptor:
    def __init__(self, constraint):
        if not isinstance(constraint, BaseConstraint):
            raise TypeError("Only children of PathConstraint are allowed to be registered.")
        self._constraint = constraint

    def __get__(self, instance, owner):
        return copy.copy(self._constraint)


class Rule:
    @classmethod
    def register(cls, scenario: str, constraint: BaseConstraint):
        setattr(cls, scenario, _RuleDescriptor(constraint))


Rule.register(
    "input_file_read",
    where(
        os.getuid() == 0,
        Path.is_file(),
        Path.is_file() & ~Path.has_soft_link() &
        Path.is_readable() & ~Path.is_writable_to_group_or_others() &
        Path.is_consistent_to_current_user() & Path.is_size_reasonable(),
        description="current user is root"
    )
)

Rule.register(
    "input_file_exec",
    where(
        os.getuid() == 0,
        Path.is_file(),
        Path.is_file() & ~Path.has_soft_link() &
        Path.is_executable() & ~Path.is_writable_to_group_or_others() &
        Path.is_consistent_to_current_user() & Path.is_size_reasonable(),
        description="current user is root"
    )
)

Rule.register(
    "input_dir_traverse",
    where(
        os.getuid() == 0,
        Path.is_dir(),
        Path.is_dir() & ~Path.has_soft_link() & Path.is_readable() &
        Path.is_executable() & ~Path.is_writable_to_group_or_others() &
        Path.is_consistent_to_current_user(),
        description="current user is root"
    )
)

Rule.register(
    "output_path_create",
    ~Path.exists() & ~Path.is_name_too_long()
)

Rule.register(
    "output_path_overwrite",
    where(
        os.getuid() == 0,
        Path.is_file(),
        Path.is_file() & ~Path.has_soft_link() &
        Path.is_writable() & ~Path.is_writable_to_group_or_others() &
        Path.is_consistent_to_current_user(),
        description="current user is root"
    )
)

Rule.register(
    "output_path_write",
    where(
        Path.exists(),
        Rule.output_path_overwrite,
        ~Path.is_name_too_long() & Path.has_writable_parent_dir()
    )
)

Rule.register(
    "output_path_append",
    Rule.output_path_overwrite
)

Rule.register(
    "output_dir",
    where(
        os.getuid() == 0,
        Path.is_dir(),
        Path.is_dir() & ~Path.has_soft_link() &
        Path.is_writable() & ~Path.is_writable_to_group_or_others() &
        Path.is_consistent_to_current_user(),
        description="current user is root"
    )
)
