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


class InvalidParameterError(Exception):
    # Class-level constants for colors (only computed once)
    _COLORS = {
        'RED': "\033[1;31m",
        'YELLOW': "\033[1;33m",
        'CYAN': "\033[1;36m",
        'RESET': "\033[0m"
    }

    def __init__(self, parameter_name, caller_name, constraint, parameter_value):
        self.parameter_name = parameter_name
        self.caller_name = caller_name
        self.constraint = constraint
        self.parameter_value = parameter_value

        super().__init__(self.build_error_message())

    def build_error_message(self):
        """Build the complete error message with colors and formatting"""
        message = f"Parameter validation failed in " \
                  f"{self._COLORS['CYAN']}function {self.caller_name!r}{self._COLORS['RESET']}: " \
                  f"{self._COLORS['YELLOW']}expected parameter {self.parameter_name!r} " \
                  f"{self.constraint}{self._COLORS['RESET']}, " \
                  f"{self._COLORS['RED']}but receieved {self.parameter_value!r}{self._COLORS['RESET']}"

        return message
