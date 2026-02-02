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

from .base import BaseValidator


class GreaterThanValidator(BaseValidator):
    @staticmethod
    def validate(actual_value, expected_value):
        try:
            res = actual_value > expected_value
        except Exception:
            res = False
        return res


class GEValidator(BaseValidator):
    @staticmethod
    def validate(actual_value, expected_value):
        try:
            res = actual_value >= expected_value
        except Exception:
            res = False
        return res


class EqualValidator(BaseValidator):
    @staticmethod
    def validate(actual_value, expected_value):
        try:
            res = actual_value == expected_value
        except Exception:
            res = False
        return res


class LessThanValidator(BaseValidator):
    @staticmethod
    def validate(actual_value, expected_value):
        try:
            res = actual_value < expected_value
        except Exception:
            res = False
        return res


class LEValidator(BaseValidator):
    @staticmethod
    def validate(actual_value, expected_value):
        try:
            res = actual_value <= expected_value
        except Exception:
            res = False
        return res


class NEValidator(BaseValidator):
    @staticmethod
    def validate(actual_value, expected_value):
        try:
            res = actual_value != expected_value
        except Exception:
            res = False
        return res
