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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict
from ..utils import get_handler, ErrorType, ErrorHandler


@dataclass
class CollectResult:
    data: Dict
    error_handler: ErrorHandler


class BaseCollector(ABC):
    def __init__(self, error_handler: ErrorHandler = None):
        self.error_handler = error_handler or get_handler(ErrorType.ERR_COLLECT)

    @abstractmethod
    def _collect_data(self) -> Dict:
        pass

    def collect(self) -> CollectResult:
        return CollectResult(self._collect_data(), self.error_handler)
