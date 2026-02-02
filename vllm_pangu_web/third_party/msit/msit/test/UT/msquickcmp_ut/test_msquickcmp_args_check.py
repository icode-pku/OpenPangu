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
import argparse
from contextlib import ExitStack

from components.debug.compare.msquickcmp.common.args_check import (
    check_model_path_legality, check_weight_path_legality, check_om_path_legality, is_saved_model_valid,
    valid_json_file_or_dir, check_path_exit, check_output_path_legality, check_cann_path_legality,
    check_input_path_legality, check_dict_kind_string, check_device_range_valid, check_number_list,
    check_dym_range_string, check_fusion_cfg_path_legality,
    check_quant_json_path_legality, safe_string, str2bool
)


class ModelPathMock:
    """Mock class for model path validation testing"""

    def __init__(self,
                 is_dir=False,
                 is_file=True,
                 exists=None,
                 is_saved_model_valid_=True,
                 is_basically_legal=True,
                 is_legal_file_type=True,
                 is_legal_file_size=True,
                 file_stat_exception=None):
        self.is_dir = is_dir
        self.is_file = is_file
        self.exists = exists
        self.is_saved_model_valid = is_saved_model_valid_
        self.is_basically_legal = is_basically_legal
        self.is_legal_file_type = is_legal_file_type
        self.is_legal_file_size = is_legal_file_size
        self.file_stat_exception = file_stat_exception

        # 设置mock的FileStat类
        self.mock_file_stat = type('MockFileStat', (), {
            'is_basically_legal': lambda _, mode, *args, **kwargs: self.is_basically_legal,
            'is_legal_file_type': lambda _, allowed_types, *args, **kwargs: self.is_legal_file_type,
            'is_legal_file_size': lambda _, size_limit, *args, **kwargs: self.is_legal_file_size,
            'is_dir': self.is_dir
        })

    def setup_mocks(self):
        """Setup all required patches"""
        patches = [
            patch('os.path.isdir', return_value=self.is_dir),
            patch('os.path.isfile', return_value=self.is_file),
            patch('components.debug.compare.msquickcmp.common.args_check.is_saved_model_valid',
                  return_value=self.is_saved_model_valid)
        ]

        if self.exists is not None:
            patches.append(patch('os.path.exists', return_value=self.exists))

        # 如果设置了file_stat_exception，则FileStat会抛出异常
        if self.file_stat_exception:
            patches.append(
                patch('components.debug.compare.msquickcmp.common.args_check.FileStat',
                      side_effect=self.file_stat_exception)
            )
        else:
            patches.append(
                patch('components.debug.compare.msquickcmp.common.args_check.FileStat',
                      return_value=self.mock_file_stat())
            )
            patch('components.debug.compare.msquickcmp.common.args_check.is_legal_args_path_string',
                  return_value=True)

        return patches


class BastCheckTestCase(unittest.TestCase):
    @staticmethod
    def apply_patches(mock_env):
        """Helper method to apply all patches"""
        stack = ExitStack()
        for patch_item in mock_env.setup_mocks():
            stack.enter_context(patch_item)
        return stack


class TestCheckModelPathLegality(BastCheckTestCase):
    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_normal_file_success(self, mock_path_exists, mock_path_readability, mock_file_size,
                                 mock_path_owner_consistent, mock_group_writable, mock_others_writable):
        """测试普通文件的成功场景"""
        mock_env = ModelPathMock(is_dir=False)
        with self.apply_patches(mock_env):
            result = check_model_path_legality("test_model.onnx")
            self.assertEqual(result, "test_model.onnx")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_basic_legality_check_fails(self, mock_path_exists, mock_path_readability, mock_file_size,
                                        mock_path_owner_consistent):
        """测试基本合法性检查失败的场景"""
        mock_env = ModelPathMock(is_dir=False, is_basically_legal=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_model_path_legality("test_model.onnx")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_file_type_check_fails(self, mock_path_exists, mock_path_readability, mock_file_size,
                                   mock_path_owner_consistent):
        """测试文件类型检查失败的场景"""
        mock_env = ModelPathMock(is_dir=False, is_legal_file_type=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_model_path_legality("test_model.invalid")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_file_size_check_fails(self, mock_path_exists, mock_path_readability, mock_file_size,
                                   mock_path_owner_consistent):
        """测试文件大小检查失败的场景"""
        mock_env = ModelPathMock(is_dir=False, is_legal_file_size=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_model_path_legality("test_model.onnx")

    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_directory_valid_saved_model(self, mock_path_exists, mock_path_readability,
                                         mock_path_owner_consistent, mock_group_writable, mock_others_writable):
        """测试有效的保存模型目录"""
        mock_env = ModelPathMock(is_dir=True, is_saved_model_valid_=True)
        with self.apply_patches(mock_env):
            result = check_model_path_legality("valid_model_dir")
            self.assertEqual(result, "valid_model_dir")

    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_directory_invalid_saved_model(self, mock_path_exists, mock_path_readability,
                                           mock_path_owner_consistent, mock_group_writable, mock_others_writable):
        """测试无效的保存模型目录"""
        mock_env = ModelPathMock(is_dir=True, is_saved_model_valid_=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_model_path_legality("invalid_model_dir")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_file_stat_raises_exception(self, mock_path_exists, mock_path_readability, mock_file_size,
                                        mock_path_owner_consistent):
        """测试FileStat初始化异常的场景"""
        mock_env = ModelPathMock(is_dir=False, file_stat_exception=Exception("File stat error"))
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_model_path_legality("invalid_path")


class TestCheckOmPathLegality(BastCheckTestCase):
    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_normal_file_success(self, mock_path_exists, mock_path_readability, mock_file_size,
                                 mock_path_owner_consistent, mock_group_writable, mock_others_writable):
        """测试普通om文件的成功场景"""
        mock_env = ModelPathMock(is_dir=False)
        with self.apply_patches(mock_env):
            result = check_om_path_legality("test_model.om")
            self.assertEqual(result, "test_model.om")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_basic_legality_check_fails(self, mock_path_exists, mock_path_readability, mock_file_size,
                                        mock_path_owner_consistent):
        """测试基本合法性检查失败的场景"""
        mock_env = ModelPathMock(is_dir=False, is_basically_legal=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_om_path_legality("test_model.om")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_file_type_check_fails(self, mock_path_exists, mock_path_readability, mock_file_size,
                                   mock_path_owner_consistent):
        """测试文件类型检查失败的场景"""
        mock_env = ModelPathMock(is_dir=False, is_legal_file_type=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_om_path_legality("test_model.invalid")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_file_size_check_fails(self, mock_path_exists, mock_path_readability, mock_file_size,
                                   mock_path_owner_consistent):
        """测试文件大小检查失败的场景"""
        mock_env = ModelPathMock(is_dir=False, is_legal_file_size=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_om_path_legality("test_model.om")

    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_directory_valid_saved_model(self, mock_path_exists, mock_path_readability, mock_path_owner_consistent,
                                         mock_group_writable, mock_others_writable):
        """测试有效的保存模型目录"""
        mock_env = ModelPathMock(is_dir=True, is_saved_model_valid_=True)
        with self.apply_patches(mock_env):
            result = check_om_path_legality("valid_model_dir")
            self.assertEqual(result, "valid_model_dir")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_file_stat_raises_exception(self, mock_path_exists, mock_path_readability, mock_file_size,
                                        mock_path_owner_consistent):
        """测试FileStat初始化异常的场景"""
        mock_env = ModelPathMock(is_dir=False, file_stat_exception=Exception("File stat error"))
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_om_path_legality("invalid_path")


def test_is_saved_model_valid():
    """ test is_saved_model_valid """
    mock_env = ModelPathMock(is_dir=True, is_file=True)
    with BastCheckTestCase.apply_patches(mock_env):
        assert is_saved_model_valid("valid_path") is True


class TestIsSavedModelValid(BastCheckTestCase):
    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('os.path.join')
    def test_valid_saved_model(self, mock_join, mock_isfile, mock_isdir):
        """测试有效的SavedModel目录结构"""
        # 设置mock返回值
        def mock_isdir_func(path):
            return {'test_dir': True, 'test_dir/variables': True}.get(path, False)
        mock_isdir.side_effect = mock_isdir_func

        def mock_isfile_func(path):
            return path == 'test_dir/saved_model.pb'
        mock_isfile.side_effect = mock_isfile_func

        def mock_join_func(dir_path, file_name):
            return f"{dir_path}/{file_name}"
        mock_join.side_effect = mock_join_func

        # 验证结果
        self.assertTrue(is_saved_model_valid('test_dir'))

    @patch('os.path.isdir')
    def test_invalid_directory(self, mock_isdir):
        """测试输入路径不是目录的情况"""
        mock_isdir.return_value = False
        self.assertFalse(is_saved_model_valid('not_a_dir'))

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('os.path.join')
    def test_missing_saved_model_pb(self, mock_join, mock_isfile, mock_isdir):
        """测试缺少saved_model.pb文件的情况"""
        # 目录存在，但saved_model.pb不存在
        mock_isdir.return_value = True
        mock_isfile.return_value = False

        def mock_join_func(dir_path, file_name):
            return f"{dir_path}/{file_name}"
        mock_join.side_effect = mock_join_func

        self.assertFalse(is_saved_model_valid('test_dir'))

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('os.path.join')
    def test_missing_variables_dir(self, mock_join, mock_isfile, mock_isdir):
        """测试缺少variables目录的情况"""
        # 设置mock返回值
        def mock_isdir_func(path):
            return {'test_dir': True, 'test_dir/variables': False}.get(path, False)
        mock_isdir.side_effect = mock_isdir_func

        def mock_isfile_func(path):
            return path == 'test_dir/saved_model.pb'
        mock_isfile.side_effect = mock_isfile_func

        def mock_join_func(dir_path, file_name):
            return f"{dir_path}/{file_name}"
        mock_join.side_effect = mock_join_func

        self.assertFalse(is_saved_model_valid('test_dir'))

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('os.path.join')
    def test_path_join_calls(self, mock_join, mock_isfile, mock_isdir):
        """测试os.path.join的调用参数是否正确"""
        test_dir = 'test_dir'

        # 设置基本的mock返回值
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        def mock_join_func(dir_path, file_name):
            return f"{dir_path}/{file_name}"
        mock_join.side_effect = mock_join_func

        is_saved_model_valid(test_dir)

        # 验证os.path.join的调用
        expected_calls = [
            unittest.mock.call(test_dir, 'saved_model.pb'),
            unittest.mock.call(test_dir, 'variables')
        ]
        mock_join.assert_has_calls(expected_calls, any_order=False)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('os.path.join')
    def test_empty_directory_path(self, mock_join, mock_isfile, mock_isdir):
        """测试空目录路径"""
        mock_isdir.return_value = False
        self.assertFalse(is_saved_model_valid(''))

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('os.path.join')
    def test_special_characters_in_path(self, mock_join, mock_isfile, mock_isdir):
        """测试路径中包含特殊字符的情况"""
        test_dir = '/test/dir/with/@#$%^&*()'
        def mock_isdir_func(path):
            return {test_dir: True, f'{test_dir}/variables': True}.get(path, False)
        mock_isdir.side_effect = mock_isdir_func

        def mock_isfile_func(path):
            return path == f'{test_dir}/saved_model.pb'
        mock_isfile.side_effect = mock_isfile_func

        def mock_join_func(dir_path, file_name):
            return f"{dir_path}/{file_name}"
        mock_join.side_effect = mock_join_func

        self.assertTrue(is_saved_model_valid(test_dir))


class TestCheckWeightPathLegality(BastCheckTestCase):
    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_normal_file_success(self, mock_path_exists, mock_path_readability, mock_file_size,
                                 mock_path_owner_consistent, mock_group_writable, mock_others_writable):
        """测试普通caffemodel文件的成功场景"""
        mock_env = ModelPathMock(is_dir=False)
        with self.apply_patches(mock_env):
            result = check_weight_path_legality("test_model.caffemodel")
            self.assertEqual(result, "test_model.caffemodel")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_basic_legality_check_fails(self, mock_path_exists, mock_path_readability, mock_file_size,
                                        mock_path_owner_consistent):
        """测试基本合法性检查失败的场景"""
        mock_env = ModelPathMock(is_dir=False, is_basically_legal=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_weight_path_legality("test_model.caffemodel")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_file_type_check_fails(self, mock_path_exists, mock_path_readability, mock_file_size,
                                   mock_path_owner_consistent):
        """测试文件类型检查失败的场景"""
        mock_env = ModelPathMock(is_dir=False, is_legal_file_type=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_weight_path_legality("test_model.invalid")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_file_size_check_fails(self, mock_path_exists, mock_path_readability, mock_file_size,
                                   mock_path_owner_consistent):
        """测试文件大小检查失败的场景"""
        mock_env = ModelPathMock(is_dir=False, is_legal_file_size=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_weight_path_legality("test_model.caffemodel")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_file_stat_raises_exception(self, mock_path_exists, mock_path_readability, mock_file_size,
                                        mock_path_owner_consistent):
        """测试FileStat初始化异常的场景"""
        mock_env = ModelPathMock(is_dir=False, file_stat_exception=Exception("File stat error"))
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_weight_path_legality("invalid_path")


class TestCheckInputPathLegality(BastCheckTestCase):
    def test_empty_input(self):
        """测试空输入"""
        self.assertIsNotNone(check_input_path_legality(""))
        self.assertIsNone(check_input_path_legality(None))

    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_single_valid_path(self, mock_path_exists, mock_path_readability, mock_file_size,
                               mock_path_owner_consistent, mock_group_writable, mock_others_writable):
        """测试单个有效路径"""
        mock_env = ModelPathMock(is_basically_legal=True)
        with self.apply_patches(mock_env):
            result = check_input_path_legality("valid/path")
            self.assertEqual(result, "valid/path")

    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_multiple_valid_paths(self, mock_path_exists, mock_path_readability, mock_file_size,
                                  mock_path_owner_consistent, mock_group_writable, mock_others_writable):
        """测试多个有效路径"""
        mock_env = ModelPathMock(is_basically_legal=True)
        with self.apply_patches(mock_env):
            result = check_input_path_legality("path1,path2,path3")
            self.assertEqual(result, "path1,path2,path3")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_invalid_path(self, mock_path_exists, mock_path_readability, mock_file_size, mock_path_owner_consistent):
        """测试无效路径"""
        mock_env = ModelPathMock(is_basically_legal=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_input_path_legality("invalid/path")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_file_stat_exception(self, mock_path_exists, mock_path_readability, mock_file_size,
                                 mock_path_owner_consistent):
        """测试FileStat异常"""
        mock_env = ModelPathMock(file_stat_exception=Exception("File stat error"))
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_input_path_legality("error/path")


class TestCheckCannPathLegality(BastCheckTestCase):
    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    @patch("os.path.isdir", return_value=True)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    @patch('components.debug.compare.msquickcmp.common.args_check.is_legal_args_path_string', return_value=True)
    def test_valid_path(self, mock_is_legal, mock_path_exists, mock_path_readability, mock_path_owner_consistent, mock_path_isdir, mock_group_writable, mock_others_writable):
        """测试有效路径"""
        result = check_cann_path_legality("valid/path")
        self.assertEqual(result, "valid/path")

    @patch("os.path.isdir", return_value=True)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    @patch('components.debug.compare.msquickcmp.common.args_check.is_legal_args_path_string', return_value=False)
    def test_invalid_path(self, mock_is_legal, mock_path_exists, mock_path_readability, mock_path_owner_consistent, mock_path_isdir):
        """测试无效路径"""
        with self.assertRaises(argparse.ArgumentTypeError):
            check_cann_path_legality("invalid/path")


class TestCheckOutputPathLegality(BastCheckTestCase):
    def test_empty_input(self):
        """测试空输入"""
        self.assertIsNotNone(check_output_path_legality(""))
        self.assertIsNone(check_output_path_legality(None))

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_path_writability", return_value=None)
    def test_valid_path(self, mock_path_writability, mock_path_owner_consistent):
        """测试有效输出路径"""
        mock_env = ModelPathMock(is_basically_legal=True)
        with self.apply_patches(mock_env):
            result = check_output_path_legality("valid/path")
            self.assertEqual(result, "valid/path")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_path_writability", return_value=None)
    def test_invalid_path(self, mock_path_writability, mock_path_owner_consistent):
        """测试无效输出路径"""
        mock_env = ModelPathMock(is_basically_legal=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_output_path_legality("invalid/path")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_path_writability", return_value=None)
    def test_file_stat_exception(self, mock_path_writability, mock_path_owner_consistent):
        """测试FileStat异常"""
        mock_env = ModelPathMock(file_stat_exception=Exception("File stat error"))
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_output_path_legality("error/path")


class TestCheckPathExit(BastCheckTestCase):
    def test_existing_path(self):
        """测试存在的路径"""
        mock_env = ModelPathMock(exists=True)
        with self.apply_patches(mock_env):
            result = check_path_exit("existing/path")
            self.assertEqual(result, "existing/path")

    def test_non_existing_path(self):
        """测试不存在的路径"""
        mock_env = ModelPathMock(exists=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(ValueError):
                check_path_exit("non_existing/path")


class TestValidJsonFileOrDir(BastCheckTestCase):
    def test_empty_input(self):
        """测试空输入"""
        self.assertIsNotNone(valid_json_file_or_dir(""))
        self.assertIsNone(valid_json_file_or_dir(None))

    def test_valid_directory(self):
        """测试有效目录"""
        mock_env = ModelPathMock(is_dir=True, is_basically_legal=True)
        with self.apply_patches(mock_env):
            result = valid_json_file_or_dir("valid/dir")
            self.assertEqual(result, "valid/dir")

    def test_valid_json_file(self):
        """测试有效JSON文件"""
        mock_env = ModelPathMock(is_dir=False, is_basically_legal=True, is_legal_file_type=True,
                                 is_legal_file_size=True)
        with self.apply_patches(mock_env):
            result = valid_json_file_or_dir("valid.json")
            self.assertEqual(result, "valid.json")

    def test_invalid_file_type(self):
        """测试无效文件类型"""
        mock_env = ModelPathMock(is_dir=False, is_basically_legal=True, is_legal_file_type=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                valid_json_file_or_dir("invalid.txt")

    def test_file_too_large(self):
        """测试文件过大"""
        mock_env = ModelPathMock(is_dir=False, is_basically_legal=True, is_legal_file_type=True,
                                 is_legal_file_size=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                valid_json_file_or_dir("large.json")

    def test_file_stat_exception(self):
        """测试FileStat异常"""
        mock_env = ModelPathMock(file_stat_exception=Exception("File stat error"))
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                valid_json_file_or_dir("error.json")


class TestCheckDictKindString(BastCheckTestCase):
    def test_empty_input(self):
        """测试空输入"""
        self.assertIsNotNone(check_dict_kind_string(""))
        self.assertIsNone(check_dict_kind_string(None))

    def test_valid_input(self):
        """测试有效输入"""
        valid_inputs = [
            "input_name1:1,224,224,3",
            "input_name1:1,224,224,3;input_name2:3,300",
            "test.name:1,2,3",
            "test_name:1,2,3"
        ]
        for input_str in valid_inputs:
            self.assertEqual(check_dict_kind_string(input_str), input_str)

    def test_invalid_input(self):
        """测试无效输入"""
        invalid_inputs = [
            "input@name:1,2,3",
            "input name:1,2,3",
            "input+name:1,2,3",
            "input$name:1,2,3"
        ]
        for input_str in invalid_inputs:
            with self.assertRaises(argparse.ArgumentTypeError):
                check_dict_kind_string(input_str)


class TestCheckDeviceRangeValid(BastCheckTestCase):
    def test_valid_range(self):
        """测试有效范围的值"""
        valid_values = ["0", "100", "255"]
        for value in valid_values:
            self.assertEqual(check_device_range_valid(value), value)

    def test_invalid_range(self):
        """测试无效范围的值"""
        invalid_values = ["-1", "256", "1000"]
        for value in invalid_values:
            with self.assertRaises(argparse.ArgumentTypeError):
                check_device_range_valid(value)

    def test_invalid_input(self):
        """测试非数字输入"""
        invalid_inputs = ["abc", "12.34", ""]
        for value in invalid_inputs:
            with self.assertRaises(argparse.ArgumentTypeError):
                check_device_range_valid(value)


class TestCheckNumberList(BastCheckTestCase):
    def test_empty_input(self):
        """测试空输入"""
        self.assertIsNotNone(check_number_list(""))
        self.assertIsNone(check_number_list(None))

    def test_valid_number_list(self):
        """测试有效的数字列表"""
        valid_inputs = [
            "123",
            "123,456",
            "123,456,789"
        ]
        for input_str in valid_inputs:
            self.assertEqual(check_number_list(input_str), input_str)

    def test_invalid_number_list(self):
        """测试无效的数字列表"""
        invalid_inputs = [
            "abc",
            "123,abc",
            "123,456.789",
            "123,-456"
        ]
        for input_str in invalid_inputs:
            with self.assertRaises(argparse.ArgumentTypeError):
                check_number_list(input_str)


class TestCheckDymRangeString(BastCheckTestCase):
    def test_empty_input(self):
        """测试空输入"""
        self.assertIsNotNone(check_dym_range_string(""))
        self.assertIsNone(check_dym_range_string(None))

    def test_valid_dym_range(self):
        """测试有效的动态范围字符串"""
        valid_inputs = [
            "1,2,3",
            "test_name:1,2,3",
            "test.name:1~3",
            "test-name:1,2,3"
        ]
        for input_str in valid_inputs:
            self.assertEqual(check_dym_range_string(input_str), input_str)

    def test_invalid_dym_range(self):
        """测试无效的动态范围字符串"""
        invalid_inputs = [
            "test@name:1,2,3",
            "test name:1,2,3",
            "test+name:1,2,3"
        ]
        for input_str in invalid_inputs:
            with self.assertRaises(argparse.ArgumentTypeError):
                check_dym_range_string(input_str)


class TestCheckFusionCfgPathLegality(BastCheckTestCase):
    def test_empty_input(self):
        """测试空输入"""
        self.assertIsNotNone(check_fusion_cfg_path_legality(""))
        self.assertIsNone(check_fusion_cfg_path_legality(None))

    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_valid_cfg_file(self, mock_path_exists, mock_path_readability, mock_file_size, mock_path_owner_consistent, mock_group_writable, mock_others_writable):
        """测试有效的配置文件"""
        mock_env = ModelPathMock(is_basically_legal=True, is_legal_file_type=True, is_legal_file_size=True)
        with self.apply_patches(mock_env):
            result = check_fusion_cfg_path_legality("valid.cfg")
            self.assertEqual(result, "valid.cfg")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_invalid_file_type(self, mock_path_exists, mock_path_readability, mock_file_size,
                               mock_path_owner_consistent):
        """测试无效的文件类型"""
        mock_env = ModelPathMock(is_basically_legal=True, is_legal_file_type=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_fusion_cfg_path_legality("invalid.txt")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_file_too_large(self, mock_path_exists, mock_path_readability, mock_file_size, mock_path_owner_consistent):
        """测试文件过大"""
        mock_env = ModelPathMock(is_basically_legal=True, is_legal_file_type=True, is_legal_file_size=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_fusion_cfg_path_legality("large.cfg")


class TestCheckQuantJsonPathLegality(BastCheckTestCase):
    def test_empty_input(self):
        """测试空输入"""
        self.assertIsNotNone(check_quant_json_path_legality(""))
        self.assertIsNone(check_quant_json_path_legality(None))

    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_valid_json_file(self, mock_path_exists, mock_path_readability, mock_file_size, mock_path_owner_consistent, mock_group_writable, mock_others_writable):
        """测试有效的JSON文件"""
        mock_env = ModelPathMock(is_basically_legal=True, is_legal_file_type=True, is_legal_file_size=True)
        with self.apply_patches(mock_env):
            result = check_quant_json_path_legality("valid.json")
            self.assertEqual(result, "valid.json")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_invalid_file_type(self, mock_path_exists, mock_path_readability, mock_file_size,
                               mock_path_owner_consistent):
        """测试无效的文件类型"""
        mock_env = ModelPathMock(is_basically_legal=True, is_legal_file_type=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_quant_json_path_legality("invalid.txt")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_file_too_large(self, mock_path_exists, mock_path_readability, mock_file_size, mock_path_owner_consistent):
        """测试文件过大"""
        mock_env = ModelPathMock(is_basically_legal=True, is_legal_file_type=True, is_legal_file_size=False)
        with self.apply_patches(mock_env):
            with self.assertRaises(argparse.ArgumentTypeError):
                check_quant_json_path_legality("large.json")


class TestSafeString(BastCheckTestCase):
    def test_empty_input(self):
        """测试空输入"""
        self.assertIsNotNone(safe_string(""))
        self.assertIsNone(safe_string(None))

    def test_valid_string(self):
        """测试有效字符串"""
        valid_inputs = [
            "test_string",
            "test123",
            "test-string",
            "test.string",
            "test:string",
            "test/string"
        ]
        for input_str in valid_inputs:
            self.assertEqual(safe_string(input_str), input_str)

    def test_invalid_string(self):
        """测试无效字符串"""
        invalid_inputs = [
            "test@string",
            "test#string",
            "test$string",
            "test%string"
        ]
        for input_str in invalid_inputs:
            with self.assertRaises(ValueError):
                safe_string(input_str)


class TestStr2Bool(BastCheckTestCase):
    def test_boolean_input(self):
        """测试布尔值输入"""
        self.assertTrue(str2bool(True))
        self.assertFalse(str2bool(False))

    def test_valid_true_strings(self):
        """测试表示True的有效字符串"""
        true_strings = ["yes", "true", "t", "y", "1"]
        for input_str in true_strings:
            self.assertTrue(str2bool(input_str))
            self.assertTrue(str2bool(input_str.upper()))

    def test_valid_false_strings(self):
        """测试表示False的有效字符串"""
        false_strings = ["no", "false", "f", "n", "0"]
        for input_str in false_strings:
            self.assertFalse(str2bool(input_str))
            self.assertFalse(str2bool(input_str.upper()))

    def test_invalid_strings(self):
        """测试无效字符串"""
        invalid_strings = ["maybe", "2", "invalid"]
        for input_str in invalid_strings:
            with self.assertRaises(argparse.ArgumentTypeError):
                str2bool(input_str)
