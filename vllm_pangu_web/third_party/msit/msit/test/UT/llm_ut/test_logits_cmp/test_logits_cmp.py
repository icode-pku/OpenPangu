# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd. All rights reserved.
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
import csv
import shutil
import unittest
import tempfile
from unittest.mock import patch, MagicMock

import torch

from msit_llm.logits_compare.logits_cmp import compute_ulp, parse_key_id, compare_tensor,\
                                               check_tensor, LogitsComparison


class TestParseKeyId(unittest.TestCase):
    # 测试Humaneval-X格式解析
    def test_humanevalx_python_case(self):
        filename = "test_python_123_456.pth"
        key, tid, err = parse_key_id(filename)
        self.assertEqual(key, "python/123")
        self.assertEqual(tid, 456)
        self.assertIsNone(err)

    # 测试无效格式处理
    def test_invalid_format(self):
        filename = "invalid_file.txt"
        key, tid, err = parse_key_id(filename)
        self.assertIsNone(key)
        self.assertIsNone(tid)
        self.assertIn("invalid_format_", err)

class TestTensorComparison(unittest.TestCase):
    """测试张量比较相关函数"""

    def setUp(self):
        # 创建测试张量
        self.golden = torch.randn(1, 4).squeeze(0)
        self.my_good = self.golden + 1e-6  # 微小差异
        self.my_bad_shape = torch.randn(1, 3).squeeze(0)
        self.my_nan = torch.tensor([1.0, float('nan')])

    def test_shape_check(self):
        passed, msg = check_tensor(self.golden, self.my_bad_shape)
        self.assertFalse(passed)
        self.assertIn("shape", msg)

    def test_nan_check(self):
        passed, msg = check_tensor(self.golden, self.my_nan)
        self.assertFalse(passed)
        self.assertIn("my_data", msg)

    def test_cosine_similarity(self):
        result = compare_tensor(self.golden, self.my_good, "fp32")
        self.assertGreater(result["cosine_similarity"], 0.99)

class TestLogitsComparison(unittest.TestCase):
    """测试 LogitsComparison 类"""
    
    def setUp(self):
        # 创建临时目录
        self.test_dir = tempfile.mkdtemp()
        self.golden_path = os.path.join(self.test_dir, "golden")
        self.my_path = os.path.join(self.test_dir, "my")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.golden_path, mode=0o750)
        os.makedirs(self.my_path, mode=0o750)
        
        # 生成测试文件
        torch.save(torch.randn(5), os.path.join(self.golden_path, "test_0_0.pth"))
        torch.save(torch.randn(5), os.path.join(self.my_path, "test_0_0.pth"))
        
        # 模拟参数
        self.args = MagicMock()
        self.args.golden_path = self.golden_path
        self.args.my_path = self.my_path
        self.args.cosine_similarity = 0.9
        self.args.kl_divergence = 0.1
        self.args.l1_norm = 0.1
        self.args.output_dir = self.output_dir
        self.args.dtype = "fp32"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('components.utils.util.safe_torch_load')
    @patch('msit_llm.logits_compare.logits_cmp.parse_key_id')
    def test_compare_logits_normal(self, mock_parse, mock_load):
        """ 测试正常文件解析和比较流程 """
        # 模拟解析结果
        mock_parse.return_value = ("python/123", 456, None)
        
        # 模拟加载张量
        dummy_tensor = torch.randn(5)
        mock_load.side_effect = [dummy_tensor, dummy_tensor]

        comparator = LogitsComparison(self.args)
        rows = comparator.compare_logits()

        # 验证结果行结构
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row[0], "test_0_0.pth")
        self.assertEqual(row[1], "python/123")
        self.assertEqual(row[2], 456)
        self.assertNotEqual(row[3], "NA")  # 应有具体数值
        self.assertEqual(row[9], None)

    def test_compare_logits_missing_files(self):
        """ 测试文件不匹配场景 """
        # 增加一个文件制造差异
        torch.save(torch.randn(5), os.path.join(self.golden_path, "test_0_1.pth"))

        comparator = LogitsComparison(self.args)
        rows = comparator.compare_logits()
        
        # 应包含两条记录：共同文件 + golden独有
        self.assertEqual(len(rows), 2)
        self.assertEqual("only_in_golden_path", rows[1][9])
        os.remove(os.path.join(self.golden_path, "test_0_1.pth"))

    @patch('msit_llm.logits_compare.logits_cmp.parse_key_id')
    def test_compare_logits_parse_error(self, mock_parse):
        """ 测试文件名解析错误处理 """
        mock_parse.return_value = (None, None, "invalid_format")
        
        comparator = LogitsComparison(self.args)
        rows = comparator.compare_logits()
        
        self.assertIn("parse_error:invalid_format", rows[0][-1])

    def test_compare_with_baseline_pass_all(self):
        """ 测试全部指标达标 """
        test_row = [None]*10  # 占位符
        test_row[3] = 0.95   # cosine_similarity > 0.9
        test_row[4] = 0.05    # kl_divergence < 0.1
        test_row[5] = 0.08    # l1_norm < 0.1
        test_row[6] = 0.5     # ulp_max_diff
        test_row[7] = 1.0     # ulp
        
        comparator = LogitsComparison(self.args)
        comparator.compare_with_baseline([test_row])
        
        self.assertEqual(test_row[8], "True")

    def test_compare_with_baseline_ulp_pass(self):
        """ 测试ULP条件通过 """
        test_row = [None]*10
        test_row[3] = 0.8     # 不满足cosine
        test_row[6] = 0.5     # ulp_max_diff <= ulp (0.5 <= 1.0)
        test_row[7] = 1.0
        
        comparator = LogitsComparison(self.args)
        comparator.compare_with_baseline([test_row])
        
        self.assertEqual(test_row[8], "True")

    def test_compare_with_baseline_fail(self):
        """ 测试完全失败的情况 """
        test_row = [None]*10
        test_row[3] = 0.8     # cosine不达标
        test_row[6] = 1.5     # ulp_max_diff > ulp
        test_row[7] = 1.0
        
        comparator = LogitsComparison(self.args)
        comparator.compare_with_baseline([test_row])
        
        self.assertEqual(test_row[8], "False")

    def test_save_result_normal(self):
        """ 测试正常保存流程 """
        # 构造测试数据
        test_rows = [
            ["file1.pth", "key1", 123, 0.95, 0.05, 0.08, 0.5, 1.0, "True", None],
            ["file2.pth", "key2", 456, "NA", "NA", "NA", "NA", "NA", "NA", "error"]
        ]

        comparator = LogitsComparison(self.args)
        comparator.save_result(test_rows)
        
        # 验证文件生成
        output_files = [f for f in os.listdir(self.output_dir) if f.startswith("logits_cmp_res")]
        expected_path = os.path.join(self.output_dir, output_files[0])
        self.assertTrue(os.path.exists(expected_path))
        
        # 验证CSV内容
        with open(expected_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            self.assertEqual(headers, ["file_name", "key", "token_id", "cosine_similarity",
                                      "kl_divergence", "l1_norm", "ulp_max_diff", "ulp",
                                      "passed", "cmp_fail_reason"])
            self.assertEqual(len(list(reader)), 2)

    # 测试完整流程
    @patch('components.utils.security_check.ms_makedirs')
    @patch('msit_llm.common.log.logger')
    def test_full_process(self, mock_logger, mock_makedirs):
        comparator = LogitsComparison(self.args)
        comparator.process_comparsion()
        
        # 验证输出文件
        output_files = [f for f in os.listdir(self.output_dir) if f.startswith("logits_cmp_res")]
        self.assertEqual(len(output_files), 1)
        
        # 验证CSV内容
        with open(os.path.join(self.output_dir, output_files[0])) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["cmp_fail_reason"], "")

if __name__ == '__main__':
    unittest.main()
