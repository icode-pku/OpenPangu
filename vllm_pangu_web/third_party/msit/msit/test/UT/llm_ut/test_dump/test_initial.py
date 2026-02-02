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

from unittest import mock
from unittest import mock, TestCase

from components.llm.msit_llm.dump.initial import is_use_cxx11, read_cpu_profiling_data, \
    split_cpu_profiling_data, clear_dump_task, run_pipeline


class TestIsUseCxx11(TestCase):
    @mock.patch('os.environ.get', return_value="/path/to/atb/home")
    @mock.patch('os.path.exists', side_effect=[True, True])
    @mock.patch('components.llm.msit_llm.dump.initial.run_pipeline', return_value=(0, "some output containing Probe and cxx11"))
    def test_is_use_cxx11_success(self, mock_subproc, mock_exists, mock_env):
        result = is_use_cxx11()
        self.assertTrue(result)

    @mock.patch('os.environ.get', return_value="/path/to/atb/home")
    @mock.patch('os.path.exists', side_effect=[True, False])
    def test_is_use_cxx11_lib_not_exists(self, mock_exists, mock_env):
        with self.assertRaises(OSError) as context:
            is_use_cxx11()
        self.assertIn("not exists", str(context.exception))

    @mock.patch('os.environ.get', return_value="")
    def test_is_use_cxx11_empty_atb_home_path(self, mock_env):
        with self.assertRaises(OSError) as context:
            is_use_cxx11()
        self.assertIn("empty or invalid", str(context.exception))

    @mock.patch('os.environ.get', return_value="/path/to/atb/home")
    @mock.patch('os.path.exists', side_effect=[False])
    def test_is_use_cxx11_invalid_atb_home_path(self, mock_exists, mock_env):
        with self.assertRaises(OSError) as context:
            is_use_cxx11()
        self.assertIn("empty or invalid", str(context.exception))

    @mock.patch('os.environ.get', return_value="/path/to/atb/home")
    @mock.patch('os.path.exists', side_effect=[True, True])
    @mock.patch('components.llm.msit_llm.dump.initial.run_pipeline', return_value=(1, ""))
    def test_is_use_cxx11_not_found(self, mock_subproc, mock_exists, mock_env):
        result = is_use_cxx11()
        self.assertFalse(result)


class TestReadCpuProfilingData(TestCase):
    def test_read_cpu_profiling_data(self):
        lines = [
            '[opname1]:stat1',
            '[opname2]:stat2',
            '[opname1]:stat3',
            'invalid line',  # This line should be ignored
            '[opname3]:stat4'
        ]
        data_map = {}
        read_cpu_profiling_data(lines, data_map)
        expected_data_map = {
            'opname1': ['stat1', 'stat3'],
            'opname2': ['stat2'],
            'opname3': ['stat4']
        }
        self.assertEqual(data_map, expected_data_map)

    def test_read_cpu_profiling_data_empty_input(self):
        lines = []
        data_map = {}
        read_cpu_profiling_data(lines, data_map)
        self.assertEqual(data_map, {})

    def test_read_cpu_profiling_data_existing_entries(self):
        lines = [
            '[opname1]:stat1',
            '[opname2]:stat2'
        ]
        existing_data_map = {
            'opname1': ['existing_stat']
        }
        read_cpu_profiling_data(lines, existing_data_map)
        expected_data_map = {
            'opname1': ['existing_stat', 'stat1'],
            'opname2': ['stat2']
        }

        self.assertEqual(existing_data_map, expected_data_map)

    def test_read_cpu_profiling_data_invalid_lines(self):
        lines = [
            'invalid line 1',
            'invalid line 2',
            '[opname1]:stat1'  # Only this line should be processed
        ]
        data_map = {}
        read_cpu_profiling_data(lines, data_map)
        expected_data_map = {
            'opname1': ['stat1']
        }
        self.assertEqual(data_map, expected_data_map)


class TestSplitCpuProfilingData(TestCase):
    def setUp(self):
        # 初始化一个包含不同统计信息的数据字典
        self.data = {
            'opname1': [
                'some info kernelExecuteTime:1234 more info',
                'other info runnerSetupTime:5678 more details'
            ],
            'opname2': [
                'info with no execute or setup time'
            ],
            'opname3': [
                'runnerSetupTime:9876 and some other info',
                'kernelExecuteTime:4321 with additional data'
            ]
        }

    def test_split_cpu_profiling_data_with_execute_and_setup(self):
        execute_data, setup_data = split_cpu_profiling_data(self.data, 'opname1')
        self.assertEqual(execute_data, 'some info kernelExecuteTime:1234 more info')
        self.assertEqual(setup_data, 'other info runnerSetupTime:5678 more details')

    def test_split_cpu_profiling_data_only_execute(self):
        execute_data, setup_data = split_cpu_profiling_data(self.data, 'opname3')
        self.assertEqual(execute_data, 'kernelExecuteTime:4321 with additional data')
        self.assertEqual(setup_data, 'runnerSetupTime:9876 and some other info')

    def test_split_cpu_profiling_data_no_execute_or_setup(self):
        execute_data, setup_data = split_cpu_profiling_data(self.data, 'opname2')
        self.assertEqual(execute_data, '')
        self.assertEqual(setup_data, '')


class TestClearDumpTask(TestCase):
    @mock.patch('components.llm.msit_llm.dump.initial.json_to_onnx')
    def test_clear_dump_task_onnx_model(self, mock_json_to_onnx):
        args = mock.Mock(type="onnx,model")
        clear_dump_task(args)
        mock_json_to_onnx.assert_called_once_with(args)

    @mock.patch('components.llm.msit_llm.dump.initial.json_to_onnx')
    def test_clear_dump_task_onnx_layer(self, mock_json_to_onnx):
        args = mock.Mock(type="onnx,layer")
        clear_dump_task(args)
        mock_json_to_onnx.assert_called_once_with(args)

    @mock.patch('components.llm.msit_llm.dump.initial.merge_cpu_profiling_data')
    @mock.patch('os.environ.get', return_value='')
    @mock.patch('os.path.exists', return_value=False)
    def test_clear_dump_task_cpu_profiling_no_atb_output_dir(self, mock_exists, mock_env_get, mock_merge):
        args = mock.Mock(type="cpu_profiling")
        clear_dump_task(args)
        mock_merge.assert_not_called()