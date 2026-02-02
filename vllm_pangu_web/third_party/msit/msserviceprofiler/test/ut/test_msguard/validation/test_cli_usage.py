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
import argparse
import unittest
import tempfile
from unittest.mock import patch

from msserviceprofiler.msguard import validate_args, Rule, InvalidParameterError
from msserviceprofiler.msguard.security import open_s


class TestValidateArgs(unittest.TestCase):
    """Unit tests for validate_args decorator"""

    def test_validate_args_given_valid_readable_file_when_input_file_read_constraint_then_no_exception(self):
        """当提供可读文件且符合input_file_read约束时，不应抛出异常"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            os.chmod(tmp_file.name, 0o400)  # 设置为只读
            validator = validate_args(Rule.input_file_read)
            validator(tmp_file.name)  # 预期不会抛出异常

    def test_validate_args_given_invalid_readable_file_when_input_file_read_constraint_then_raise_error(self):
        """当文件不存在且不符合input_file_read约束时，应抛出ArgumentTypeError"""
        validator = validate_args(Rule.input_file_read)
        with self.assertRaises(argparse.ArgumentTypeError):
            validator("/nonexistent/file")  # 文件不存在，预期抛出异常

    def test_validate_args_given_valid_executable_file_when_input_file_exec_constraint_then_no_exception(self):
        """当提供可执行文件且符合input_file_exec约束时，不应抛出异常"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            os.chmod(tmp_file.name, 0o500)  # 设置为可读可执行
            validator = validate_args(Rule.input_file_exec)
            validator(tmp_file.name)  # 预期不会抛出异常

    @unittest.skipIf(os.geteuid() == 0, "root用户可执行任何文件")
    def test_validate_args_given_invalid_executable_file_when_input_file_exec_constraint_then_raise_error(self):
        """当文件不可执行且不符合input_file_exec约束时，应抛出ArgumentTypeError"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            os.chmod(tmp_file.name, 0o400)  # 设置为只读
            validator = validate_args(Rule.input_file_exec)
            with self.assertRaises(argparse.ArgumentTypeError):
                validator(tmp_file.name)  # 文件不可执行，预期抛出异常

    def test_validate_args_given_valid_directory_when_input_dir_traverse_constraint_then_no_exception(self):
        """当提供有效目录且符合input_dir_traverse约束时，不应抛出异常"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            validator = validate_args(Rule.input_dir_traverse)
            validator(tmp_dir)  # 预期不会抛出异常

    def test_validate_args_given_invalid_directory_when_input_dir_traverse_constraint_then_raise_error(self):
        """当目录不存在且不符合input_dir_traverse约束时，应抛出ArgumentTypeError"""
        validator = validate_args(Rule.input_dir_traverse)
        with self.assertRaises(argparse.ArgumentTypeError):
            validator("/nonexistent/directory")  # 目录不存在，预期抛出异常

    def test_validate_args_given_new_path_when_output_path_create_constraint_then_no_exception(self):
        """当路径不存在且符合output_path_create约束时，不应抛出异常"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_file = os.path.join(tmp_dir, "newfile.txt")
            validator = validate_args(Rule.output_path_create)
            validator(new_file)  # 预期不会抛出异常

    def test_validate_args_given_existing_path_when_output_path_create_constraint_then_raise_error(self):
        """当路径已存在且不符合output_path_create约束时，应抛出ArgumentTypeError"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            validator = validate_args(Rule.output_path_create)
            with self.assertRaises(argparse.ArgumentTypeError):
                validator(tmp_file.name)  # 文件已存在，预期抛出异常

    @patch('os.getuid', return_value=0)
    def test_validate_args_given_root_user_when_output_path_overwrite_constraint_then_no_exception(self, mock_getuid):
        """当root用户尝试覆盖文件且符合output_path_overwrite约束时，不应抛出异常"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            validator = validate_args(Rule.output_path_overwrite)
            validator(tmp_file.name)  # root用户可覆盖，预期不会抛出异常

    @unittest.skipIf(os.geteuid() == 0, "root用户可以覆盖任何文件")
    def test_validate_args_given_non_root_user_when_output_path_overwrite_constraint_then_raise_error(self):
        """当非root用户尝试覆盖文件且不符合output_path_overwrite约束时，应抛出ArgumentTypeError"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            os.chmod(tmp_file.name, 0o777)
            validator = validate_args(Rule.output_path_overwrite)
            with self.assertRaises(argparse.ArgumentTypeError):
                validator(tmp_file.name)  # 非root用户不可覆盖，预期抛出异常

    def test_validate_args_given_valid_path_when_output_path_write_constraint_then_no_exception(self):
        """当路径可写且符合output_path_write约束时，不应抛出异常"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_file = os.path.join(tmp_dir, "newfile.txt")
            validator = validate_args(Rule.output_path_write)
            validator(new_file)  # 预期不会抛出异常

    def test_validate_args_given_invalid_path_when_output_path_write_constraint_then_raise_error(self):
        """当路径不可写且不符合output_path_write约束时，应抛出ArgumentTypeError"""
        validator = validate_args(Rule.output_path_write)
        with self.assertRaises(argparse.ArgumentTypeError):
            validator("/nonexistent/path/to/file")  # 父目录不存在，预期抛出异常

    @patch('os.getuid', return_value=0)
    def test_validate_args_given_root_user_when_output_dir_constraint_then_no_exception(self, mock_getuid):
        """root用户只会校验目录是否存在，不应抛出异常"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            validator = validate_args(Rule.output_dir)
            validator(tmp_dir)  # root用户可创建目录，预期不会抛出异常

    @unittest.skipIf(os.geteuid() == 0, "root用户可以创建任何目录")
    def test_validate_args_given_non_root_user_when_output_dir_constraint_then_raise_error(self):
        """非root用户会校验软链接、权限等其他问题，应抛出ArgumentTypeError"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chmod(tmp_dir, 0o777)
            validator = validate_args(Rule.output_dir)
            with self.assertRaises(argparse.ArgumentTypeError):
                validator(tmp_dir)  # 非root用户不可创建目录，预期抛出异常

    def test_validate_args_given_valid_input_path_when_input_file_read_constraint_then_return(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input-path', type=validate_args(Rule.input_file_read))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "temp_file")
            with open_s(temp_file, 'w'):
                pass

            args = parser.parse_args(['--input-path', temp_file])
            self.assertEqual(args.input_path, temp_file)
