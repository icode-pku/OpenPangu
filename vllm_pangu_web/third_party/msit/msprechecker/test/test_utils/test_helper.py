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

import unittest
from unittest.mock import patch, mock_open
from msguard.security import open_s

from msprechecker.utils import is_in_container, singleton


class TestIsInContainer(unittest.TestCase):
    @patch('os.path.exists')
    def test_is_in_container_when_dockerenv_exists(self, mock_exists):
        """测试当.dockerenv文件存在时返回True"""
        mock_exists.return_value = True
        self.assertTrue(is_in_container())

    @patch('os.path.exists')
    @patch('msprechecker.utils.helper.open_s', new_callable=mock_open, read_data='systemd (1, #threads: 1)\n')
    def test_is_in_container_when_dockerenv_not_exists_and_systemd(self, mock_open_file, mock_exists):
        """测试当.dockerenv不存在且第一个进程是systemd时返回False"""
        mock_exists.return_value = False
        self.assertFalse(is_in_container())

    @patch('os.path.exists')
    @patch('msprechecker.utils.helper.open_s', new_callable=mock_open, read_data='not_systemd (1, #threads: 1)\n')
    def test_is_in_container_when_dockerenv_not_exists_and_not_systemd(self, mock_open_file, mock_exists):
        """测试当.dockerenv不存在且第一个进程不是systemd时返回True"""
        mock_exists.return_value = False
        self.assertTrue(is_in_container())

    @patch('os.path.exists')
    @patch('msprechecker.utils.helper.open_s', side_effect=Exception("File not found"))
    def test_is_in_container_when_sched_file_read_fails(self, mock_open_file, mock_exists):
        """测试当读取/proc/1/sched失败时返回True"""
        mock_exists.return_value = False
        self.assertTrue(is_in_container())


class TestSingletonDecorator(unittest.TestCase):
    def test_singleton_decorator_creates_only_one_instance(self):
        """测试单例装饰器只创建一个实例"""
        
        @singleton
        class TestClass:
            def __init__(self, value):
                self.value = value
        
        instance1 = TestClass(1)
        instance2 = TestClass(2)
        
        self.assertIs(instance1, instance2)
        self.assertEqual(instance1.value, 1)  # 第二次初始化参数被忽略
        self.assertEqual(instance2.value, 1)
