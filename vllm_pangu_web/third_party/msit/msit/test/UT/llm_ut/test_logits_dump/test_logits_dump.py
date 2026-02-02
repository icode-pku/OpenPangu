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
import subprocess
from unittest import TestCase
from unittest.mock import patch, MagicMock

import pandas as pd

from msit_llm.logits_dump.logits_dump import build_bad_case_list, build_humaneval_bad_case_list,\
                                             build_humanevalx_bad_case_list, build_others_bad_case_list,\
                                             check_gpu, check_npu, execute_command, del_env, LogitsDumper


class TestBuildBadCaseListFromKeyList(TestCase):
    # HumanEval-X 测试用例
    def test_humanevalx_valid(self):
        keys = ["CPP/123", "Go/456"]
        result = build_humanevalx_bad_case_list(keys)
        self.assertEqual(result, [[123], [456], [], [], []])

    def test_humanevalx_invalid_prefix(self):
        with self.assertRaises(ValueError):
            build_humanevalx_bad_case_list(["Invalid/123"])

    def test_humanevalx_invalid_format(self):
        with self.assertRaises(ValueError):
            build_humanevalx_bad_case_list(["Python/abc"])  # 非数字索引

    # HumanEval 测试用例
    def test_humaneval_valid(self):
        keys = ["HumanEval/789"]
        result = build_humaneval_bad_case_list(keys)
        self.assertEqual(result, [789])

    def test_humaneval_wrong_prefix(self):
        with self.assertRaises(ValueError):
            build_humaneval_bad_case_list(["WrongPrefix/123"])

    # 其他数据集测试用例
    def test_others_valid(self):
        keys = [0, 1, 2]
        result = build_others_bad_case_list(keys)
        self.assertEqual(result, keys)

    def test_others_invalid_type(self):
        with self.assertRaises(ValueError):
            build_others_bad_case_list(["123"])  # 字符串类型

    def test_others_negative_number(self):
        with self.assertRaises(ValueError):
            build_others_bad_case_list([-1])


class TestBuildBadCaseListFromCsv(TestCase):
    def setUp(self):
        self.temp_csv = "temp_test.csv"

    def tearDown(self):
        if os.path.exists(self.temp_csv):
            os.remove(self.temp_csv)

    def _create_csv(self, data):
        df = pd.DataFrame({"key": data})
        df.to_csv(self.temp_csv, index=False)

    # 测试不同数据集类型
    def test_humanevalx_flow(self):
        self._create_csv(["Java/123", "JavaScript/456"])
        result = build_bad_case_list(self.temp_csv)
        self.assertEqual(result, [[], [], [123], [456], []])

    def test_humaneval_flow(self):
        self._create_csv(["HumanEval/789"])
        result = build_bad_case_list(self.temp_csv)
        self.assertEqual(result, [789])

    def test_others_flow(self):
        self._create_csv([0, 1, 2])
        result = build_bad_case_list(self.temp_csv)
        self.assertEqual(result, [0, 1, 2])

    # 异常场景测试
    def test_empty_csv(self):
        self._create_csv([])
        with self.assertRaises(RuntimeError):
            build_bad_case_list(self.temp_csv)

    def test_invalid_csv_path(self):
        with self.assertRaises(ValueError):
            build_bad_case_list("invalid_path.txt")

    def test_invalid_key_data(self):
        self._create_csv(["python"])
        with self.assertRaises(TypeError):
            build_bad_case_list(self.temp_csv)

    def test_mixed_data_types(self):
        self._create_csv(["Python/123", 456])  # 混合类型
        with self.assertRaises(ValueError):
            build_bad_case_list(self.temp_csv)


class TestHardwareCheck(TestCase):
    @patch('torch.cuda.is_available')
    def test_check_gpu(self, mock_cuda):
        # 测试GPU可用场景
        mock_cuda.return_value = True
        self.assertTrue(check_gpu())
        
        # 测试GPU不可用场景
        mock_cuda.return_value = False
        self.assertFalse(check_gpu())

    def test_check_npu(self):
        # 测试torch npu未安装场景
        self.assertFalse(check_npu())

        MockTorchNpu = MagicMock()
        with patch.dict("sys.modules", {"torch_npu": MockTorchNpu}):
            # 测试NPU可用场景
            MockTorchNpu.npu.is_available = lambda: True
            self.assertTrue(check_npu())
            # 测试NPU不可用场景
            MockTorchNpu.npu.is_available = lambda: False
            self.assertFalse(check_npu())


class TestCommandExecution(TestCase):
    @patch('subprocess.run')
    def test_successful_execution(self, mock_run):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        execute_command(["valid_command"], info_need=False)
        mock_run.assert_called_once_with(
            ["valid_command"],
            shell=False,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

    @patch('subprocess.run')
    def test_failed_execution(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["invalid_command"],
            output="Mocked error output"
        )
        
        with self.assertRaises(RuntimeError):
            execute_command(["invalid_command"])


class TestDelEnvironment(TestCase):
    def test_del_env(self):
        os.environ['BAD_CASE_LOGITS_DUMP'] = "True"
        os.environ['LOGITS_DUMP_TOKEN_MAX_LENGTH'] = "10"
        os.environ['BAD_CASE_LIST'] = "[0,1,2]"

        self.assertEqual(os.getenv('BAD_CASE_LOGITS_DUMP', "False"), "True")
        self.assertEqual(os.getenv('LOGITS_DUMP_TOKEN_MAX_LENGTH', "0"), "10")
        self.assertEqual(os.getenv('BAD_CASE_LIST', "[]"), "[0,1,2]")

        del_env()

        self.assertNotIn('BAD_CASE_LOGITS_DUMP', os.environ)
        self.assertNotIn('LOGITS_DUMP_TOKEN_MAX_LENGTH', os.environ)
        self.assertNotIn('BAD_CASE_LIST', os.environ)


class TestLogitsDumper(TestCase):
    def setUp(self):
        class Args:
            pass
        args = Args()
        args.exec = "modeltest cmd"
        args.bad_case_result_csv = "cases.csv"
        args.token_range = 512
        self.dumper = LogitsDumper(args)

    @patch('msit_llm.logits_dump.logits_dump.build_bad_case_list')
    @patch('msit_llm.logits_dump.logits_dump.check_npu')
    @patch('msit_llm.logits_dump.logits_dump.check_gpu')
    @patch('msit_llm.logits_dump.logits_dump.execute_command')
    @patch('msit_llm.logits_dump.logits_dump.del_env')
    def test_dump_logits_no_device(self, mock_del_env, mock_execute_command, mock_check_gpu, 
                                   mock_check_npu, mock_build_bad_case_list):
        # 测试 dump_logits 方法在没有可用设备的情况下
        mock_check_npu.return_value = False
        mock_check_gpu.return_value = False
        mock_build_bad_case_list.return_value = []
        with self.assertRaises(RuntimeError) as cm:
            self.dumper.dump_logits()
        self.assertEqual(str(cm.exception), "NPU/GPU is not available")
        mock_del_env.assert_called()

    @patch('msit_llm.logits_dump.logits_dump.build_bad_case_list')
    @patch('msit_llm.logits_dump.logits_dump.check_npu')
    @patch('msit_llm.logits_dump.logits_dump.check_gpu')
    @patch('subprocess.run')
    @patch('msit_llm.logits_dump.logits_dump.del_env')
    def test_dump_logits_no_device(self, mock_del_env, mock_execute_command, mock_check_gpu, 
                                   mock_check_npu, mock_build_bad_case_list):
        # 测试 dump_logits 方法在NPU设备的情况下
        mock_check_npu.return_value = True
        mock_check_gpu.return_value = False
        mock_build_bad_case_list.return_value = []
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_execute_command.return_value = mock_process

        self.dumper.dump_logits()
        mock_del_env.assert_called()
