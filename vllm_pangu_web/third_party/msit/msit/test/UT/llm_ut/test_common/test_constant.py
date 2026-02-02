# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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

from __future__ import print_function
import datetime
import os
import unittest
from unittest.mock import patch, MagicMock

from msit_llm.common.constant import get_visible_device, get_timestamp_sync, get_ait_dump_path


class TestDeviceRelatedFunctions(unittest.TestCase):
    @patch.dict(os.environ, {"ASCEND_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "1"})
    def test_get_visible_device(self):
        envionment_name = "ASCEND_VISIBLE_DEVICES"
        result = get_visible_device(envionment_name)
        self.assertEqual(result, 0)

        envionment_name = "CUDA_VISIBLE_DEVICES"
        result = get_visible_device(envionment_name)
        self.assertEqual(result, 1)

    @patch.dict(os.environ, {"ASCEND_VISIBLE_DEVICES": "abc"})
    def test_get_visible_device_when_value_is_invalid(self):
        envionment_name = "ASCEND_VISIBLE_DEVICES"
        with self.assertRaises(ValueError) as context:
            get_visible_device(envionment_name)

        assert "The value of the variable ASCEND_VISIBLE_DEVICES is not valid" in str(context.exception)


class TestAitDumpPathFunction(unittest.TestCase):
    @patch('os.environ', MagicMock())
    @patch('msit_llm.common.constant.get_timestamp_sync', return_value=1609459200)  # 模拟一个时间戳
    def test_get_ait_dump_path(self, mock_get_timestamp_sync):
        result = get_ait_dump_path()
        expected_path = "msit_dump_20210101_080000"
        self.assertEqual(result, expected_path)


class TestGetTimestampSync(unittest.TestCase):
    def setUp(self):
        """
        在每个测试方法执行前进行通用的模拟初始化操作，包括模拟相关函数的返回值等，
        并保存模拟对象以便在测试方法中进一步设置或验证。
        """
        # 模拟torch.distributed.is_initialized函数，保存模拟对象
        self.mock_is_initialized = patch('torch.distributed.is_initialized').start()
        # 模拟torch.distributed.init_process_group函数，保存模拟对象
        self.mock_init_process_group = patch('torch.distributed.init_process_group').start()
        # 模拟torch.distributed.destroy_process_group函数，保存模拟对象
        self.mock_destroy_process_group = patch('torch.distributed.destroy_process_group').start()
        # 模拟torch.distributed.all_reduce函数，保存模拟对象
        self.mock_all_reduce = patch('torch.distributed.all_reduce').start()
        # 模拟torch.tensor函数，保存模拟对象
        self.mock_tensor = patch('torch.tensor').start()
        # 模拟datetime.datetime函数，保存模拟对象
        self.mock_datetime = patch('datetime.datetime').start()

        # 设置datetime.datetime.now函数的通用返回值
        self.mock_datetime.now.return_value = datetime.datetime(2024, 12, 4, 12, 0, 0, tzinfo=datetime.timezone.utc)

        # 设置torch.tensor函数返回的MagicMock对象，方便后续在各测试方法中进一步设置属性或方法返回值
        self.mock_tensor.return_value = MagicMock()

    def tearDown(self):
        """
        在每个测试方法执行后停止所有启动的patch模拟，进行清理操作，确保测试环境的独立性。
        """
        patch.stopall()

    def test_single_process(self):
        """
        测试单进程场景下get_timestamp_sync函数的行为。
        """
        # 设置单进程相关环境变量
        os.environ["LOCAL_WORLD_SIZE"] = "1"

        # 设置torch.distributed.is_initialized返回值为False，适用于单进程场景
        self.mock_is_initialized.return_value = False

        # 调用函数
        result = get_timestamp_sync()

        # 检查init_process_group是否未被调用
        self.mock_init_process_group.assert_not_called()

        # 检查all_reduce是否未被调用
        self.mock_all_reduce.assert_not_called()

        # 检查destroy_process_group是否未被调用
        self.mock_destroy_process_group.assert_not_called()

        # 检查结果
        expected_timestamp = int(self.mock_datetime.now().strftime("%s"))
        self.assertEqual(result, expected_timestamp)

    def test_multi_process(self):
        """
        测试多进程场景下get_timestamp_sync函数的行为。
        """
        # 设置多进程相关环境变量
        os.environ["LOCAL_WORLD_SIZE"] = "2"
        os.environ["LOCAL_RANK"] = "0"

        # 设置torch.distributed.is_initialized返回值为False，适用于多进程初始化场景
        self.mock_is_initialized.return_value = False

        # 设置torch.tensor返回值的item方法返回值，用于多进程场景测试
        self.mock_tensor.return_value.item.return_value = 1700000000

        # 调用函数
        result = get_timestamp_sync()

        # 检查init_process_group是否被调用
        self.mock_init_process_group.assert_called_once()

        # 检查all_reduce是否被调用
        self.mock_all_reduce.assert_called_once()

        # 检查destroy_process_group是否被调用
        self.mock_destroy_process_group.assert_called_once()

        # 检查结果
        self.assertEqual(result, 1700000000)

    def test_distributed_initialized(self):
        """
        测试分布式环境已初始化场景下get_timestamp_sync函数的行为。
        """
        # 设置分布式环境已初始化相关环境变量
        os.environ["LOCAL_WORLD_SIZE"] = "2"
        os.environ["LOCAL_RANK"] = "1"

        # 设置torch.distributed.is_initialized返回值为True，适用于分布式已初始化场景
        self.mock_is_initialized.return_value = True

        # 设置torch.tensor返回值的item方法返回值，与其他多进程场景保持一致
        self.mock_tensor.return_value.item.return_value = 1700000000

        # 调用函数
        result = get_timestamp_sync()

        # 检查init_process_group是否未被调用
        self.mock_init_process_group.assert_not_called()

        # 检查all_reduce是否被调用
        self.mock_all_reduce.assert_called_once()

        # 检查destroy_process_group是否未被调用
        self.mock_destroy_process_group.assert_not_called()

        # 检查结果
        self.assertEqual(result, 1700000000)
