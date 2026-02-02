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

import logging
from test.st.executor.exec_command import CommandExecutor


class ExecParse(CommandExecutor):
    def __init__(self):
        super().__init__()
        self.input_path = ""
        self.output_path = ""
        self.params = []

    def set_input_path(self, input_path):
        self.input_path = input_path

    def set_output_path(self, output_path):
        self.output_path = output_path

    def add_param(self, *param_str):
        self.params.extend(param_str)

    def ready_go(self):
        # 执行
        self.execute(["python", "-m", "ms_service_profiler.parse",
                      "--input-path", self.input_path,
                      "--output-path", self.output_path] + self.params)

        exit_code, _ = self.wait(timeout=600) # 等个10分钟，解析不完直接自杀
        logging.info(f"wait result: {exit_code}")
        return exit_code == 0