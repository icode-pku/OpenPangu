# Copyright (c) 2023-2025 Huawei Technologies Co., Ltd.
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

class MsProfArgsAdapter():
    def __init__(self, application, output, model_execution, sys_hardware_mem, sys_cpu_profiling, sys_profiling,
                 sys_pid_profiling, dvpp_profiling, runtime_api, task_time, aicpu):
        self.application = application
        self.output = output
        self.model_execution = model_execution
        self.sys_hardware_mem = sys_hardware_mem
        self.sys_cpu_profiling = sys_cpu_profiling
        self.sys_profiling = sys_profiling
        self.sys_pid_profiling = sys_pid_profiling
        self.dvpp_profiling = dvpp_profiling
        self.runtime_api = runtime_api
        self.task_time = task_time
        self.aicpu = aicpu