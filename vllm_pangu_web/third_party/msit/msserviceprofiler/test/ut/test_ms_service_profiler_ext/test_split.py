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

import unittest

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import argparse
import pytest

from msserviceprofiler.ms_service_profiler_ext.split import (
    add_exporters, main, arg_parse, 
    check_string_valid, check_non_negative_integer
)
from msserviceprofiler.ms_service_profiler_ext.exporters.exporter_prefill import ExporterPrefill
from msserviceprofiler.ms_service_profiler_ext.exporters.exporter_decode import ExporterDecode


class TestSplitFuctions(unittest.TestCase):
    mock_args = None
    mocker = None

    @property
    def mock_args(self):
        return self.__class__.mock_args

    @mock_args.setter
    def mock_args(self, value):
        self.__class__.mock_args = value

    @property
    def mocker(self):
        return self.__class__.mocker

    @mocker.setter
    def mocker(self, value):
        self.__class__.mocker = value

    def test_add_exporters_with_prefill(self):
        args = Namespace(prefill_batch_size=4, decode_batch_size=0, prefill_rid="-1", decode_rid="-1")
        exporters = add_exporters(args)

        self.assertEqual(len(exporters), 1)
        self.assertIsInstance(exporters[0], ExporterPrefill)

    def test_add_exporters_with_decode(self):
        args = Namespace(prefill_batch_size=0, decode_batch_size=10, prefill_rid="-1", decode_rid="-1")
        exporters = add_exporters(args)

        self.assertEqual(len(exporters), 1)
        self.assertIsInstance(exporters[0], ExporterDecode)

    def test_add_exporters_with_both(self):
        args = Namespace(prefill_batch_size=4, decode_batch_size=10, prefill_rid="-1", decode_rid="-1")
        exporters = add_exporters(args)

        self.assertEqual(len(exporters), 2)
        self.assertIsInstance(exporters[0], ExporterPrefill)
        self.assertIsInstance(exporters[1], ExporterDecode)

    def test_main(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        arg_parse(subparsers)
        main(parser.parse_args())

    @pytest.fixture(autouse=True)
    def _inject_mocker(self, mocker):
        self.mocker = mocker
        self.mock_args = Namespace(input_path="/fake/input", output_path="/fake/output", log_level="info", format="csv")

        # 2. 配置全局mock
        mocker.patch("argparse.ArgumentParser.parse_args", return_value=self.mock_args)
        mocker.patch("ms_service_profiler.utils.log.set_log_level")
        mocker.patch(
            "msserviceprofiler.ms_service_profiler_ext.split.add_exporters",
            return_value=[ExporterPrefill, ExporterDecode]
        )
        mocker.patch.object(Path, "mkdir")
        mocker.patch("os.path.exists", return_value=True)
        mocker.patch("os.makedirs")

        yield
        self.mocker = None
        self.mock_args = None


class TestCheckStringValid(unittest.TestCase):
    def test_valid_string(self):
        # 测试一个有效的字符串
        result = check_string_valid("valid_string123")
        self.assertEqual(result, "valid_string123")

    def test_string_exceeds_max_length(self):
        # 测试一个超过最大长度的字符串
        long_string = "a" * 257
        with self.assertRaises(argparse.ArgumentTypeError):
            check_string_valid(long_string)

    def test_string_with_unsafe_characters(self):
        # 测试一个包含不安全字符的字符串
        unsafe_string = "invalid@string"
        with self.assertRaises(argparse.ArgumentTypeError):
            check_string_valid(unsafe_string)

    def test_empty_string(self):
        # 测试一个空字符串
        with self.assertRaises(argparse.ArgumentTypeError):
            check_string_valid("")

    def test_string_with_spaces(self):
        # 测试一个包含空格的字符串
        with self.assertRaises(argparse.ArgumentTypeError):
            check_string_valid("string with spaces")

    def test_string_with_special_characters(self):
        # 测试一个包含特殊字符的字符串
        with self.assertRaises(argparse.ArgumentTypeError):
            check_string_valid("string!with@special#characters")


class TestCheckNonNegativeInteger(unittest.TestCase):
    
    def test_positive_integer(self):
        """测试正整数输入"""
        self.assertEqual(check_non_negative_integer(5), 5)
    
    def test_zero(self):
        """测试零输入"""
        self.assertEqual(check_non_negative_integer(0), 0)
    
    def test_negative_integer(self):
        """测试负整数输入，应抛出异常"""
        with self.assertRaises(ValueError) as context:
            check_non_negative_integer(-1)
        self.assertEqual(str(context.exception), "'-1' is not a positive integer.")
    
    def test_string_input(self):
        """测试字符串输入，应抛出异常"""
        with self.assertRaises(ValueError) as context:
            check_non_negative_integer("abc")
        self.assertEqual(str(context.exception), "'abc' cannot convert to a positive integer.")
    
    def test_string_representation_of_integer(self):
        """测试整数的字符串表示输入"""
        self.assertEqual(check_non_negative_integer("10"), 10)
    
    def test_string_representation_of_negative_integer(self):
        """测试负整数的字符串表示输入，应抛出异常"""
        with self.assertRaises(ValueError) as context:
            check_non_negative_integer("-5")
        self.assertEqual(str(context.exception), "'-5' is not a positive integer.")