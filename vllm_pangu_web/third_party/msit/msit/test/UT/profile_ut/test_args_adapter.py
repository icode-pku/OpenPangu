# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
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
from components.profile.msit_prof.msprof.args_adapter import MsProfArgsAdapter

class TestMsProfArgsAdapter(unittest.TestCase):
    def test_msprofargsadapter_initialization(self):
        # 准备测试数据
        application = "test_app"
        output = "test_output"
        model_execution = True
        sys_hardware_mem = False
        sys_cpu_profiling = True
        sys_profiling = False
        sys_pid_profiling = True
        dvpp_profiling = False
        runtime_api = "test_runtime_api"
        task_time = 100
        aicpu = True

        # 创建 MsProfArgsAdapter 实例
        msprof_args_adapter = MsProfArgsAdapter(
            application, output, model_execution, sys_hardware_mem, sys_cpu_profiling, sys_profiling,
            sys_pid_profiling, dvpp_profiling, runtime_api, task_time, aicpu
        )

        # 验证属性是否正确赋值
        self.assertEqual(msprof_args_adapter.application, application)
        self.assertEqual(msprof_args_adapter.output, output)
        self.assertEqual(msprof_args_adapter.model_execution, model_execution)
        self.assertEqual(msprof_args_adapter.sys_hardware_mem, sys_hardware_mem)
        self.assertEqual(msprof_args_adapter.sys_cpu_profiling, sys_cpu_profiling)
        self.assertEqual(msprof_args_adapter.sys_profiling, sys_profiling)
        self.assertEqual(msprof_args_adapter.sys_pid_profiling, sys_pid_profiling)
        self.assertEqual(msprof_args_adapter.dvpp_profiling, dvpp_profiling)
        self.assertEqual(msprof_args_adapter.runtime_api, runtime_api)
        self.assertEqual(msprof_args_adapter.task_time, task_time)
        self.assertEqual(msprof_args_adapter.aicpu, aicpu)

if __name__ == '__main__':
    unittest.main()