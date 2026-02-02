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
import shutil
import tempfile
import unittest

from msserviceprofiler.msguard.security import (
    walk_s, WalkLimitError, open_s, touch_s, mkdir_s
)


class TestWalkS(unittest.TestCase):
    def test_walk_s_given_valid_dir_when_no_rules_then_yield_all_files_and_dirs(self):
        """测试给定有效目录且无规则限制时，应返回所有文件和目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            sub_dir = os.path.join(temp_dir, "subdir")
            os.mkdir(sub_dir, 0o750)

            file1 = os.path.join(temp_dir, "file1.txt")
            with open_s(file1, 'w') as f:
                f.write("test")

            file2 = os.path.join(sub_dir, "file2.txt")
            with open_s(file2, 'w') as f:
                f.write("test")

            result = list(walk_s(temp_dir, dir_rule=None, file_rule=None))
            self.assertIn(file1, result, "应包含根目录下的文件")
            self.assertIn(sub_dir, result, "应包含子目录")
            self.assertIn(file2, result, "应包含子目录中的文件")

    def test_walk_s_given_valid_dir_when_max_files_exceeded_then_raise_error(self):
        """测试当扫描文件数超过max_files时，应抛出WalkLimitError异常"""
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(3):
                with open_s(os.path.join(temp_dir, f"file{i}.txt"), 'w') as f:
                    f.write("test")

            with self.assertRaises(WalkLimitError, msg="文件数超过限制时应抛出异常"):
                list(walk_s(temp_dir, max_files=1))

    def test_walk_s_given_valid_dir_when_max_depths_exceeded_then_raise_error(self):
        """测试当扫描深度超过max_depths时，应抛出WalkLimitError异常"""
        with tempfile.TemporaryDirectory() as temp_dir:
            sub_dir = os.path.join(temp_dir, "subdir")
            os.mkdir(sub_dir, 0o750)

            with self.assertRaises(WalkLimitError, msg="扫描深度超过限制时应抛出异常"):
                list(walk_s(temp_dir, max_depths=0))

    def test_walk_s_given_invalid_dir_when_input_then_raise_exception(self):
        """测试给定无效目录时，应抛出异常"""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = os.path.join(temp_dir, "nonexistent")
            with self.assertRaises(Exception, msg="无效目录路径时应抛出异常"):
                list(walk_s(invalid_path))


class TestOpenS(unittest.TestCase):
    def test_open_s_given_valid_file_when_read_mode_then_return_file_object(self):
        """测试以读模式打开有效文件时，应返回可读的文件对象"""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        try:
            with open_s(temp_file_path, 'r') as f:
                self.assertTrue(hasattr(f, 'read'), "文件对象应具有read方法")
        finally:
            os.unlink(temp_file_path)

    def test_open_s_given_valid_file_when_write_mode_then_return_file_object(self):
        """测试以写模式打开有效文件时，应返回可写的文件对象"""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        try:
            with open_s(temp_file_path, 'w') as f:
                self.assertTrue(hasattr(f, 'write'), "文件对象应具有write方法")
        finally:
            os.unlink(temp_file_path)

    def test_open_s_given_nonexistent_file_when_write_mode_then_create_file(self):
        """测试以写模式打开不存在的文件时，应创建新文件"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "new_file.txt")
            with open_s(file_path, 'w') as f:
                self.assertTrue(hasattr(f, 'write'), "应成功创建文件并返回可写对象")
            self.assertTrue(os.path.exists(file_path), "文件应被成功创建")

    def test_open_s_given_nonexistent_file_when_read_mode_then_raise_exception(self):
        """测试以读模式打开不存在的文件时，应抛出异常"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "nonexistent.txt")
            with self.assertRaises(Exception, msg="读取不存在的文件时应抛出异常"):
                open_s(file_path, 'r')

    def test_open_s_given_invalid_mode_when_input_then_raise_exception(self):
        """测试使用无效模式时，应抛出ValueError异常"""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        try:
            with self.assertRaises(ValueError, msg="无效文件模式时应抛出异常"):
                open_s(temp_file_path, 'invalid')
        finally:
            os.unlink(temp_file_path)

    def test_open_s_given_existing_file_when_exclusive_mode_then_raise_exception(self):
        """测试以独占模式打开已存在文件时，应抛出异常"""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name

        try:
            with self.assertRaises(Exception, msg="独占模式打开已存在文件时应抛出异常"):
                open_s(temp_file_path, 'x')
        finally:
            os.unlink(temp_file_path)

    def test_touch_s_should_raise_value_error_given_empty_path(self):
        self.assertRaises(ValueError, touch_s, '')

    def test_touch_s_should_raise_file_exists_error_given_existent_path(self):
        self.assertRaises(FileExistsError, touch_s, __file__, exist_ok=False)
        touch_s(__file__, exist_ok=True)

    def test_touch_s_should_create_a_file_with_default_mode_given_non_existent_path(self):
        file_name = "non-existent-file"
        touch_s(file_name)
        self.assertTrue(os.path.exists(file_name))
        self.assertEqual(os.stat(file_name).st_mode, os.st.S_IFREG | 0o640)
        os.remove(file_name)

    def test_mkdir_s_should_raise_value_error_given_empty_path(self):
        self.assertRaises(ValueError, mkdir_s, '')

    def test_mkdir_s_should_raise_file_exists_error_given_existent_path(self):
        self.assertRaises(FileExistsError, mkdir_s, "..", exist_ok=False)
        mkdir_s("..", exist_ok=True)

    def test_mkdir_s_should_create_a_dir_with_default_mode_when_non_existent_path(self):
        parent_dir = "non-existent-parent-dir"
        child_dir = os.path.join(parent_dir, "non-existent-child-dir")
        mkdir_s(child_dir)

        self.assertTrue(os.path.exists(parent_dir))
        self.assertTrue(os.path.exists(child_dir))

        self.assertEqual(os.stat(parent_dir).st_mode, os.st.S_IFDIR | 0o750)
        self.assertEqual(os.stat(child_dir).st_mode, os.st.S_IFDIR | 0o750)
        
        shutil.rmtree(parent_dir)

    def test_mkdir_s_should_raise_xxx_when_providing_existent_file(self):
        self.assertRaises(FileExistsError, mkdir_s, __file__)
        self.assertRaises(FileExistsError, mkdir_s, os.path.join(__file__, "random"))
 

