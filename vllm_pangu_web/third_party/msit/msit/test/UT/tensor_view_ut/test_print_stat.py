# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
from io import StringIO
import logging
import torch

from components.utils.log import logger
from components.tensor_view.ait_tensor_view.print_stat import print_stat


class TestPrintStat(unittest.TestCase):
    def setUp(self):
        # 创建一个StringIO对象来捕获输出
        self.captured_output = StringIO()
        # 配置日志记录器，将其输出重定向到StringIO对象
        logger.setLevel(logging.INFO)
        self.stream_handler = logging.StreamHandler(self.captured_output)
        formatter = logging.Formatter('%(message)s')
        self.stream_handler.setFormatter(formatter)
        logger.addHandler(self.stream_handler)

    def tearDown(self):
        # 从日志记录器中移除stream_handler，避免影响其他测试
        logger.removeHandler(self.stream_handler)

    def test_valid_dtype(self):
        # 创建一个测试张量（有效的数据类型）
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        print_stat(tensor)
        log_content = self.captured_output.getvalue()
        self.assertIn("min", log_content)
        self.assertIn("max", log_content)
        self.assertIn("mean", log_content)
        self.assertIn("std", log_content)
        self.assertIn("var", log_content)

    def test_invalid_dtype(self):
        # 创建一个测试张量（无效的数据类型）
        tensor = torch.tensor([1, 2, 3], dtype=torch.int)
        print_stat(tensor)
        log_content = self.captured_output.getvalue()
        self.assertIn("min", log_content)
        self.assertIn("max", log_content)
        self.assertIn("mean", log_content)
        self.assertIn("std", log_content)
        self.assertIn("var", log_content)

    def test_tensor_with_nan(self):
        # 创建一个包含NaN值的张量
        tensor = torch.tensor([1.0, float('nan'), 3.0], dtype=torch.float32)
        print_stat(tensor)
        log_content = self.captured_output.getvalue()
        self.assertIn("nan", log_content)

    def test_large_tensor(self):
        # 创建一个大张量
        tensor = torch.randn(10000)
        print_stat(tensor)
        log_content = self.captured_output.getvalue()
        self.assertIn("min", log_content)
        self.assertIn("max", log_content)
        self.assertIn("mean", log_content)
        self.assertIn("std", log_content)
        self.assertIn("var", log_content)


if __name__ == '__main__':
    unittest.main()