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

from enum import Enum, auto
from abc import ABC, abstractmethod


class ConstraintStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    SKIPPED = auto()
    

class BaseConstraint(ABC):
    def __init__(self, *, description=None):
        self.description = "" if description is None else description 
        self.status = ConstraintStatus.SKIPPED

    def __str__(self):
        return self.description

    def __and__(self, other):
        from .logic import AndConstraint, make_constraint
        return AndConstraint(self, make_constraint(other))

    def __or__(self, other):
        from .logic import OrConstraint, make_constraint
        return OrConstraint(self, make_constraint(other))

    def __invert__(self):
        from .logic import NotConstraint
        return NotConstraint(self)

    @abstractmethod
    def is_satisfied_by(self, val):
        pass
