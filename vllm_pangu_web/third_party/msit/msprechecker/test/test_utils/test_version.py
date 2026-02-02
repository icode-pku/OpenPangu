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
from unittest.mock import patch

from msprechecker.utils.version import Version


class TestVersion(unittest.TestCase):
    """Version类单元测试"""

    def test_init_with_valid_version_string(self):
        """测试使用有效版本字符串初始化Version对象"""
        version_obj = Version("1.2.3")
        self.assertEqual(version_obj.major, "1")
        self.assertEqual(version_obj.minor, "2")
        self.assertEqual(version_obj.patch, "3")
        self.assertIsNone(version_obj.rc)
        self.assertIsNone(version_obj.beta)

    def test_init_with_rc_version(self):
        """测试使用RC版本字符串初始化Version对象"""
        version_obj = Version("1.2.rc3")
        self.assertEqual(version_obj.major, "1")
        self.assertEqual(version_obj.minor, "2")
        self.assertIsNone(version_obj.patch)
        self.assertEqual(version_obj.rc, "3")
        self.assertIsNone(version_obj.beta)

    def test_init_with_rc_beta_version(self):
        """测试使用RC+Beta版本字符串初始化Version对象"""
        version_obj = Version("1.2.rc3.b4")
        self.assertEqual(version_obj.major, "1")
        self.assertEqual(version_obj.minor, "2")
        self.assertIsNone(version_obj.patch)
        self.assertEqual(version_obj.rc, "3")
        self.assertEqual(version_obj.beta, "4")

    def test_init_with_invalid_version_string(self):
        """测试使用无效版本字符串初始化Version对象时抛出异常"""
        with self.assertRaises(ValueError):
            Version("invalid")

    @patch('msprechecker.utils.version.get_pkg_version')
    def test_init_with_package_name(self, mock_get_pkg_version):
        """测试使用包名初始化Version对象"""
        mock_get_pkg_version.return_value = "1.2.3"
        version_obj = Version("numpy")
        self.assertEqual(version_obj.major, "1")
        self.assertEqual(version_obj.minor, "2")
        self.assertEqual(version_obj.patch, "3")

    def test_repr_with_patch_version(self):
        """测试补丁版本号的字符串表示"""
        version_obj = Version("1.2.3")
        self.assertEqual(repr(version_obj), "1.2.3")

    def test_repr_with_rc_version(self):
        """测试RC版本号的字符串表示"""
        version_obj = Version("1.2.rc3")
        self.assertEqual(repr(version_obj), "1.2.rc3")

    def test_repr_with_rc_beta_version(self):
        """测试RC+Beta版本号的字符串表示"""
        version_obj = Version("1.2.rc3.b4")
        self.assertEqual(repr(version_obj), "1.2.rc3.b4")

    def test_eq_with_same_version(self):
        """测试相同版本号的相等比较"""
        v1 = Version("1.2.3")
        v2 = Version("1.2.3")
        self.assertEqual(v1, v2)

    def test_eq_with_different_version(self):
        """测试不同版本号的相等比较"""
        v1 = Version("1.2.3")
        v2 = Version("1.2.4")
        self.assertNotEqual(v1, v2)

    def test_lt_with_lower_version(self):
        """测试较低版本号的小于比较"""
        v1 = Version("1.2.3")
        v2 = Version("1.2.4")
        self.assertLess(v1, v2)

    def test_lt_with_higher_version(self):
        """测试较高版本号的小于比较"""
        v1 = Version("1.2.4")
        v2 = Version("1.2.3")
        self.assertGreater(v1, v2)

    def test_properties_access(self):
        """测试版本对象属性的正确访问"""
        version_obj = Version("1.2.rc3.b4")
        self.assertEqual(version_obj.major, "1")
        self.assertEqual(version_obj.minor, "2")
        self.assertIsNone(version_obj.patch)
        self.assertEqual(version_obj.rc, "3")
        self.assertEqual(version_obj.beta, "4")

    def test_cmp_tuple_with_patch(self):
        """测试补丁版本号的比较元组生成"""
        version_obj = Version("1.2.3")
        self.assertEqual(version_obj.cmp_tuple(), (1, 2, 3, float('inf'), float('inf')))

    def test_cmp_tuple_with_rc(self):
        """测试RC版本号的比较元组生成"""
        version_obj = Version("1.2.rc3")
        self.assertEqual(version_obj.cmp_tuple(), (1, 2, 0, 3, float('inf')))

    def test_cmp_tuple_with_rc_beta(self):
        """测试RC+Beta版本号的比较元组生成"""
        version_obj = Version("1.2.rc3.b4")
        self.assertEqual(version_obj.cmp_tuple(), (1, 2, 0, 3, 4))

    @patch('msprechecker.utils.version.get_pkg_version')
    def test_parse_version_str_with_package_name(self, mock_get_pkg_version):
        """测试使用包名解析版本字符串"""
        mock_get_pkg_version.return_value = "1.2.3"
        result = Version._parse_version_str("numpy")
        self.assertIsNotNone(result)
        self.assertEqual(result.group("major"), "1")
        self.assertEqual(result.group("minor"), "2")
        self.assertEqual(result.group("patch"), "3")

    @patch('msprechecker.utils.version.get_pkg_version')
    def test_parse_version_str_with_invalid_package(self, mock_get_pkg_version):
        """测试使用无效包名解析版本字符串"""
        mock_get_pkg_version.return_value = None
        with self.assertRaises(ValueError):
            Version._parse_version_str("invalid_package")

    def test_version_comparision_should_return_true(self):
        self.assertGreater(Version("8.2.0"), Version("8.2.rc1"))
        self.assertGreater(Version("8.2.rc1"), Version("8.1.rc2"))
        self.assertGreater(Version("8.1.rc2"), Version("8.1.rc2.b020"))
