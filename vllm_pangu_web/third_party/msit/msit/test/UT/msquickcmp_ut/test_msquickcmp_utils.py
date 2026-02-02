# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
from unittest.mock import patch
from components.debug.compare.msquickcmp.common.utils import check_exec_cmd, check_dynamic_shape, \
    AccuracyCompareException


class TestCheckExecCmd(unittest.TestCase):

    @patch('components.debug.compare.msquickcmp.common.utils.check_exec_script_file')
    def test_valid_command_with_two_args(self, mock_check_exec_script_file):
        mock_check_exec_script_file.return_value = None

        command = "python script.py"
        result = check_exec_cmd(command)

        # assert right
        mock_check_exec_script_file.assert_called_with('script.py')
        self.assertTrue(result)

    @patch('components.debug.compare.msquickcmp.common.utils.check_exec_script_file')
    def test_valid_command_with_more_than_two_args(self, mock_check_exec_script_file):
        mock_check_exec_script_file.return_value = None
        command = "python script.py arg1 arg2"
        result = check_exec_cmd(command)

        # assert right
        mock_check_exec_script_file.assert_called_with('script.py')
        self.assertTrue(result)

    def test_invalid_Command(self):
        command = 'bash'
        with self.assertRaises(AccuracyCompareException):
            check_exec_cmd(command)


class TestCheckDynamicShape(unittest.TestCase):
    def test_no_dynamic_shape(self):
        shape = [1,2,3,4]
        result = check_dynamic_shape(shape)
        self.assertFalse(result, "The shape should not be dynamic")

    def test_dynamic_shape_with_none(self):
        shape = [1,2,3,None]
        result = check_dynamic_shape(shape)
        self.assertTrue(result, "The shape should be dynamic due to None")

    def test_dynamic_shape_with_string(self):
        shape = [1,2,'unknown',4]
        result = check_dynamic_shape(shape)
        self.assertTrue(result, "The shape should be dynamic due to a string")

    def test_empty_dynamic(self):
        shape = []
        result = check_dynamic_shape(shape)
        self.assertFalse(result, "An empty shape should be considered dynamic")

    def test_dynamic_shape_with_mixed_value(self):
        shape = [1,2,'dynamic',None]
        result = check_dynamic_shape(shape)
        self.assertTrue(result, "The shape should be dynamic due to None and string")