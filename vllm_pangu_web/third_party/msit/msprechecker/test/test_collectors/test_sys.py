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
import subprocess
from unittest.mock import patch, mock_open, MagicMock

from msprechecker.collectors.sys import (
    LscpuCollector, VirtualMachineCollector, CPUHighPerformanceCollector,
    PsutilStrategy, ScalingGovernorStrategy, CpupowerStrategy,
    DmidecodeStrategy, LshwStrategy,
    KernelInfoCollector, MemoryInfoCollector, SysCollector
)
from msprechecker.utils import CollectErrorHandler


class TestLscpuCollector(unittest.TestCase):
    def setUp(self):
        self.collector = LscpuCollector()

    @patch('subprocess.check_output')
    def test_parse_output_with_primary_key_should_collect_successfully(self, mock_check_output):
        test_output = "Model name: I Love U"
        mock_check_output.return_value = test_output
        result = self.collector.collect()
        self.assertEqual(result.data['model_name'], "I Love U")

    @patch('subprocess.check_output')
    def test_parse_output_with_secondary_key_should_collect_successfully(self, mock_check_output):
        test_output = "BIOS Model name: I Love U"
        mock_check_output.return_value = test_output
        result = self.collector.collect()
        self.assertEqual(result.data['model_name'], "I Love U")

    @patch('subprocess.check_output')
    def test_parse_output_with_multiple_keys_should_collect_primary(self, mock_check_output):
        test_output = "Model name: I Love U\nBIOS Model name: He doesn't love me"
        mock_check_output.return_value = test_output
        result = self.collector.collect()
        self.assertEqual(result.data['model_name'], "I Love U")

    @patch('subprocess.check_output')
    def test_parse_output_with_chinese_key_should_collect_successfully(self, mock_check_output):
        test_output = "你别管\n型号名称: I Love U"
        mock_check_output.return_value = test_output
        result = self.collector.collect()
        self.assertEqual(result.data['model_name'], "I Love U")

    @patch('subprocess.check_output')
    def test_when_command_fails_should_return_empty(self, mock_check_output):
        mock_check_output.side_effect = subprocess.CalledProcessError(1, 'lscpu')
        result = self.collector.collect()
        self.assertEqual(result.data, {})
        self.assertEqual(len(result.error_handler.errors), 1)


class TestVirtualMachineCollector(unittest.TestCase):
    def setUp(self):
        self.collector = VirtualMachineCollector()

    @patch('msprechecker.collectors.sys.open_s', new_callable=mock_open, read_data="hypervisor")
    def test_when_read_hypervisor_should_collect_true(self, mock_file):
        result = self.collector.collect()
        self.assertEqual(result.data, {"virtual_machine": True})

    @patch('msprechecker.collectors.sys.open_s', new_callable=mock_open, read_data="其他的玩意儿")
    def test_when_read_other_should_collect_false(self, mock_file):
        result = self.collector.collect()
        self.assertEqual(result.data, {"virtual_machine": False})

    @patch('msprechecker.collectors.sys.open_s')
    def test_when_file_open_fails_should_add_error(self, mock_file):
        mock_file.side_effect = IOError("不准读")
        result = self.collector.collect()
        self.assertEqual(result.data, {})
        self.assertEqual(len(result.error_handler.errors), 1)


class TestCPUHighPerformanceCollector(unittest.TestCase):
    @patch('psutil.cpu_freq')
    def test_psutil_strategy_when_current_freq_eq_max_should_return_true(self, mock_cpu_freq):
        collector = CPUHighPerformanceCollector(strategies=[PsutilStrategy()])
        mock_cpu_freq.return_value = MagicMock(current=3000, max=3000)
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": True})

    @patch('psutil.cpu_freq')
    def test_psutil_strategy_when_current_freq_not_eq_max_should_return_false(self, mock_cpu_freq):
        collector = CPUHighPerformanceCollector(strategies=[PsutilStrategy()])
        mock_cpu_freq.return_value = MagicMock(current=2000, max=3000)
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": False})
    
    @patch('psutil.cpu_freq')
    def test_psutil_strategy_when_current_freq_not_exists_should_return_false(self, mock_cpu_freq):
        collector = CPUHighPerformanceCollector(strategies=[PsutilStrategy()])
        mock_cpu_freq.return_value = None
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": False})

    @patch('subprocess.check_output')
    def test_dmidecode_strategy_when_current_freq_eq_max_should_return_true(self, mock_check_output):
        mock_check_output.return_value = """
Processor Information
    Max Speed: 3000 MHz
    Current Speed: 3000 MHz
"""
        collector = CPUHighPerformanceCollector(strategies=[DmidecodeStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": True})

    @patch('subprocess.check_output')
    def test_dmidecode_strategy_when_current_freq_not_eq_max_should_return_false(self, mock_check_output):
        mock_check_output.return_value = """
Processor Information
    Max Speed: 3000 MHz
    Current Speed: 0 MHz
"""
        collector = CPUHighPerformanceCollector(strategies=[DmidecodeStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": False})
    
    @patch('subprocess.check_output')
    def test_dmidecode_strategy_when_run_failed_should_return_false(self, mock_check_output):
        mock_check_output.side_effect = RuntimeError("出错啦")
        collector = CPUHighPerformanceCollector(strategies=[DmidecodeStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": False})

    @patch('subprocess.check_output')
    def test_cpupower_strategy_when_current_freq_eq_max_should_return_true(self, mock_check_output):
        mock_check_output.return_value = """
hardware limits: 1.20 GHz - 3.00 GHz
current CPU frequency: 3.00 GHz
"""
        collector = CPUHighPerformanceCollector(strategies=[CpupowerStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": True})
    
    @patch('subprocess.check_output')
    def test_cpupower_strategy_when_current_freq_not_eq_max_should_return_false(self, mock_check_output):
        mock_check_output.return_value = """
hardware limits: 1.20 GHz - 3.00 GHz
current CPU frequency: 1.00 GHz
"""
        collector = CPUHighPerformanceCollector(strategies=[CpupowerStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": False})

    @patch('subprocess.check_output')
    def test_cpupower_strategy_when_run_failed_should_return_false(self, mock_check_output):
        mock_check_output.side_effect = RuntimeError("跑不动")
        collector = CPUHighPerformanceCollector(strategies=[CpupowerStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": False})

    @patch('subprocess.check_output')
    def test_cpupower_strategy_when_run_unexpected_should_return_false(self, mock_check_output):
        mock_check_output.return_value = """
hardware limits: 1.20 GHz - 3.00 GHz
"""
        collector = CPUHighPerformanceCollector(strategies=[CpupowerStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": False})

    @patch('subprocess.check_output')
    def test_lshw_strategy_when_current_freq_eq_max_should_return_true(self, mock_check_output):
        mock_check_output.return_value = """
size: 1
capacity: 1
"""
        collector = CPUHighPerformanceCollector(strategies=[LshwStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": True})
    
    @patch('subprocess.check_output')
    def test_lshw_strategy_when_current_freq_not_eq_max_should_return_false(self, mock_check_output):
        """only have capacity, no size found"""
        mock_check_output.return_value = """
capacity: 1
"""
        collector = CPUHighPerformanceCollector(strategies=[LshwStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": False})
    
    @patch('subprocess.check_output')
    def test_lshw_strategy_when_run_failed_should_return_false(self, mock_check_output):
        mock_check_output.side_effect = RuntimeError("不准跑")
        collector = CPUHighPerformanceCollector(strategies=[LshwStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": False})

    @patch('msprechecker.collectors.sys.open_s', new_callable=mock_open, read_data="performance")
    def test_scaling_governor_strategy_should_return_true_when_all_cpu_is_performance(self, mock_open_s):
        collector = CPUHighPerformanceCollector(strategies=[ScalingGovernorStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": True})
    
    @patch('msprechecker.collectors.sys.open_s', new_callable=mock_open, read_data="不知道")
    def test_scaling_governor_strategy_should_return_false_when_any_cpu_is_not_performance(self, mock_open_s):
        collector = CPUHighPerformanceCollector(strategies=[ScalingGovernorStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": False})
    
    @patch('os.cpu_count', return_value=None)
    def test_scaling_governor_strategy_should_return_false_when_no_cpu_found(self, mock_cpu_count):
        collector = CPUHighPerformanceCollector(strategies=[ScalingGovernorStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": False})
    
    @patch('msprechecker.collectors.sys.open_s', new_callable=mock_open)
    def test_scaling_governor_strategy_should_return_false_when_open_failed(self, mock_open_s):
        mock_open_s.side_effect = RuntimeError("你错了")
        collector = CPUHighPerformanceCollector(strategies=[ScalingGovernorStrategy()])
        result = collector.collect()
        self.assertEqual(result.data, {"high_performance": False})


class TestKernelInfoCollector(unittest.TestCase):
    def setUp(self):
        self.collector = KernelInfoCollector()

    @patch('platform.uname')
    @patch('msprechecker.collectors.sys.open_s', new_callable=mock_open, read_data="[always] madvise never")
    def test_collect_kernel_info(self, mock_file, mock_uname):
        mock_uname.return_value = MagicMock()
        mock_uname()._asdict.return_value = dict(
            system="Linux",
            node="test-node",
            release="5.4.0",
            version="#1 SMP",
            machine="x86_64",
            processor="x86_64"
        )

        result = self.collector.collect()
        expected_keys = ['system', 'node', 'release', 'version', 'machine', 'processor', 'transparent_hugepage']
        self.assertCountEqual(result.data.keys(), expected_keys)
        self.assertEqual(result.data['transparent_hugepage'], "always")

    @patch('platform.uname')
    @patch('msprechecker.collectors.sys.open_s')
    def test_when_hugepage_file_fails(self, mock_file, mock_uname):
        mock_file.side_effect = IOError("有问题")
        mock_uname()._asdict.return_value = dict(
            system="Linux",
            node="test-node",
            release="5.4.0",
            version="#1 SMP",
            machine="x86_64",
            processor="x86_64"
        )
        result = self.collector.collect()
        self.assertNotIn('transparent_hugepage', result.data)
        self.assertEqual(len(result.error_handler.errors), 1)


class TestMemoryInfoCollector(unittest.TestCase):
    def setUp(self):
        self.collector = MemoryInfoCollector()

    @patch('os.sysconf')
    @patch('msprechecker.collectors.sys.open_s', new_callable=mock_open, read_data="1")
    def test_collect_memory_info(self, mock_file, mock_sysconf):
        mock_sysconf.return_value = 4096
        result = self.collector.collect()
        self.assertEqual(result.data['page_size'], 4096)
        self.assertEqual(result.data['overcommit_memory'], "1")

    @patch('os.sysconf')
    @patch('msprechecker.collectors.sys.open_s')
    def test_when_sysconf_fails_when_os_sysconf_raises(self, mock_file, mock_sysconf):
        mock_sysconf.side_effect = ValueError
        mock_file.return_value.read.return_value = "1"
        result = self.collector.collect()
        self.assertNotIn('page_size', result.data)
        self.assertEqual(len(result.error_handler.errors), 1)

    @patch('os.sysconf')
    @patch('msprechecker.collectors.sys.open_s')
    def test_when_overcommit_file_fails_when_overcommit_memory_path_open_failed(self, mock_file, mock_sysconf):
        mock_sysconf.return_value = 4096
        mock_file.side_effect = IOError("Permission denied")
        result = self.collector.collect()
        self.assertNotIn('overcommit_memory', result.data)
        self.assertEqual(len(result.error_handler.errors), 1)


class TestSysCollector(unittest.TestCase):
    def test_collect_all_info(self):
        lscpu_mock = MagicMock()
        lscpu_mock.collect.return_value = MagicMock(
            data={"model_name": "Test CPU"},
            error_handler=CollectErrorHandler()
        )

        vm_mock = MagicMock()
        vm_mock.collect.return_value = MagicMock(
            data={"virtual_machine": "False"},
            error_handler=CollectErrorHandler()
        )

        cpu_mock = MagicMock()
        cpu_mock.collect.return_value = MagicMock(
            data={"high_performance": "True"},
            error_handler=CollectErrorHandler()
        )

        kernel_mock = MagicMock()
        kernel_mock.collect.return_value = MagicMock(
            data={"system": "Linux"},
            error_handler=CollectErrorHandler()
        )

        mem_mock = MagicMock()
        mem_mock.collect.return_value = MagicMock(
            data={"page_size": 4096},
            error_handler=CollectErrorHandler()
        )

        subcollectors = [lscpu_mock, vm_mock, cpu_mock, kernel_mock, mem_mock]
        collector = SysCollector(subcollectors=subcollectors)
        result = collector.collect()
        expected_keys = ['model_name', 'virtual_machine', 'high_performance', 'system', 'page_size']
        self.assertCountEqual(result.data.keys(), expected_keys)
