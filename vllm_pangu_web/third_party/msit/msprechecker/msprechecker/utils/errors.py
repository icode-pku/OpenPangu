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

import bisect
from enum import Enum
from functools import total_ordering
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


class ErrorType(Enum):
    ERR_COLLECT = 0
    ERR_CHECK = 1
    ERR_COMPARE = 2
    ERR_UNKNOWN = 3


@total_ordering
class ErrorSeverity(Enum):
    ERR_HIGH = "high"
    ERR_MEDIUM = "medium"
    ERR_LOW = "low"

    _ORDER_MAP = {
        "low": 0,
        "medium": 1,
        "high": 2
    }

    def __str__(self):
        reset = "\033[0m"
        return f"{self.color_code}[{self.symbol}]{reset}"
    
    def __gt__(self, other):
        order_map = self._ORDER_MAP.value
        if isinstance(other, ErrorSeverity):
            return order_map[self.value] > order_map[other.value]

        if isinstance(other, str):
            return order_map[self.value] > order_map[other.lower()]
        
        raise TypeError(f"Expected 'other' to be str or ErrorSeverity. Got {type(other).__name__} instead.")

    @property
    def symbol(self):
        return {
            ErrorSeverity.ERR_HIGH: "NOK",
            ErrorSeverity.ERR_MEDIUM: "WARNING",
            ErrorSeverity.ERR_LOW: "RECOMMEND"
        }[self]

    @property
    def color_code(self):
        return {
            ErrorSeverity.ERR_HIGH: "\033[91m",
            ErrorSeverity.ERR_MEDIUM: "\033[93m",
            ErrorSeverity.ERR_LOW: "\033[96m"
        }[self]


@dataclass
class ErrorContext:
    filename: Optional[str] = None
    function: Optional[str] = None
    lineno: Optional[int] = None
    what: Optional[str] = None
    path: Optional[str] = None
    actual: Optional[str] = None
    expected: Optional[str] = None
    start_col: Optional[int] = None
    context_lines: Optional[Dict[int, str]] = None


class BaseError(ABC):
    def __init__(self, reason: str, severity: ErrorSeverity, context: ErrorContext):
        if isinstance(severity, str):
            severity = ErrorSeverity(severity)

        self.reason = reason
        self.severity = severity
        self.context = context


class CollectError(BaseError):
    pass


class CheckError(BaseError):
    pass


class ConfigError(CheckError):
    pass


class CompareError(BaseError):
    def __init__(self,
                 key: str,
                 values: Dict[str, str],
                 reason: str = "",
                 severity: ErrorSeverity = ErrorSeverity.ERR_LOW
                ):
        context = ErrorContext()
        super().__init__(reason, severity, context)
        self.key = key
        self.values = values


class ErrorHandler(ABC):
    def __init__(self, *, severity: ErrorSeverity = None, type_: str = ""):
        self._errors: List[BaseError] = []
        self._type = type_
        self._severity = severity or ErrorSeverity.ERR_LOW
    
    def __iter__(self):
        return iter(self._errors)

    @property
    def errors(self):
        return self._errors
    
    @property
    def severity(self):
        return self._severity
    
    @property
    def type(self):
        return self._type
    
    @type.setter
    def type(self, another_type):
        self._type = another_type
    
    @abstractmethod
    def add_error(self, reason: str, severity: ErrorSeverity, **context) -> None:
        """add error to handler"""

    def empty(self) -> bool:
        return len(self._errors) == 0
    
    def extend(self, other_handler) -> None:
        if not isinstance(other_handler, ErrorHandler):
            raise TypeError
        
        self._errors.extend(other_handler.errors)
    
    def filter(self, severity: ErrorSeverity):
        return list(filter(lambda error: error.severity >= severity, self._errors))
    
    def clear(self) -> None:
        self._errors.clear()


class CollectErrorHandler(ErrorHandler):
    def add_error(self, reason: str, severity: ErrorSeverity = ErrorSeverity.ERR_MEDIUM, **context):
        if severity < self._severity:
            return

        error_context = ErrorContext(**context)
        self._errors.append(CollectError(
            reason=reason,
            severity=severity,
            context=error_context
        ))


class CheckErrorHandler(ErrorHandler):
    def add_error(self, reason: str, severity: ErrorSeverity, **context):
        if severity < self._severity:
            return

        path = context.get('path', '')
        if '%' in path:
            context['path'] = path.replace('%', '')

        error_context = ErrorContext(**context)
        self._errors.append(CheckError(
            reason=reason,
            severity=severity,
            context=error_context
        ))


class ConfigErrorHandler(ErrorHandler):
    def __init__(self, severity: ErrorSeverity = None, file_lines: List[str] = None, 
                 key_mapping: Dict[str, Tuple[int, int]] = None, 
                 context_hierarchy: List[List[int]] = None, type_: str = ""):
        super().__init__(severity=severity, type_=type_)
        self._file_lines = file_lines
        self._key_mapping = key_mapping
        self._sorted_key = sorted(key_mapping) if key_mapping else []
        self._context_hierarchy = context_hierarchy

    def add_error(self, reason: str, severity: ErrorSeverity, **context):
        if severity < self._severity:
            return

        path = context.get('path', '')
        if '%' in path:
            percent_pos = path.index('%')
            dot_pos = path[:percent_pos].rfind('.')
            path = path.replace('%', '')
            context['path'] = path[dot_pos + 1:]
        else:
            context['path'] = path.rsplit('.', 1)[-1]

        orig_lineno, shifted_lineno, start_col = self._find_lineno_and_col(path)

        context['lineno'] = shifted_lineno + 1
        context['start_col'] = start_col
        context['context_lines'] = {
            context_lineno + 1: self._file_lines[context_lineno] 
            for context_lineno in self._context_hierarchy[orig_lineno]
        }

        error_context = ErrorContext(**context)
        self._errors.append(ConfigError(
            reason=reason,
            severity=severity,
            context=error_context
        ))

    def _find_lineno_and_col(self, path):
        if path in self._key_mapping:
            lineno, start_col = self._key_mapping[path]
            return lineno, lineno, start_col
        
        nearest_path = self._find_nearest_path(path)
        lineno, start_col = self._key_mapping[nearest_path]
        lineno_shift = (len(self._errors) + 1) / len(self._file_lines)

        return lineno, lineno + lineno_shift, start_col

    def _find_nearest_path(self, path):
        path_pos = bisect.bisect_left(self._sorted_key, path)
        if path_pos == 0:
            nearest_path = self._sorted_key[path_pos]
        elif path_pos == len(self._sorted_key) - 1 or path_pos == len(self._sorted_key):
            nearest_path = self._sorted_key[path_pos - 1]
        else:
            candidate = self._sorted_key[path_pos]
            second_candidate = self._sorted_key[path_pos - 1]
            if abs(len(candidate) - len(path)) < abs(len(second_candidate) - len(path)):
                nearest_path = candidate
            else:
                nearest_path = second_candidate

        return nearest_path


class CompareErrorHandler(ErrorHandler):
    def add_error(self, reason: str, severity: ErrorSeverity, **context):
        self._errors.append(CompareError(reason, severity))


def get_handler(error_type: ErrorType) -> ErrorHandler:
    cmd_to_handler = {
        ErrorType.ERR_COLLECT: CollectErrorHandler,
        ErrorType.ERR_CHECK: CheckErrorHandler,
        ErrorType.ERR_COMPARE: CompareErrorHandler,
    }

    if not isinstance(error_type, ErrorType):
        raise TypeError

    if error_type not in cmd_to_handler:
        raise ValueError
    
    return cmd_to_handler[error_type]()
