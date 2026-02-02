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


class ExecBenchmark(CommandExecutor):
    def __init__(self):
        super().__init__()
        self.model_name = "llama"
        self.model_path = "/model"
        self.dataset_path = "/dataset"
        self.server_port = 1025
        self.server_ip = "127.0.0.1"
        self.server_mannager_port = 1026
        self.server_mannager_ip = "127.0.0.1"

    def set_model_path(self, model_path, model_name="llama"):
        self.model_name = model_name
        self.model_path = model_path

    def set_dataset_path(self, dataset_path, dataset_type="gsm8k"):
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path

    def set_service_port(self, server_port, server_ip="127.0.0.1"):
        self.server_port = server_port
        self.server_ip = server_ip

    def set_service_mannager_port(self, server_mannager_port, server_mannager_ip="127.0.0.1"):
        self.server_mannager_port = server_mannager_port
        self.server_mannager_ip = server_mannager_ip

    def curl_test(self):
        # 执行
        self.execute(
            ["curl", f"http://{self.server_ip}:{self.server_port}",
             "-X", "POST",
             "-d", '{"inputs":"Please introduce yourself.","parameters":{"max_new_tokens":250, "temperature":0.3, '
                   '"top_p":0.3, "top_k":5, "do_sample":true, "repetition_penalty":1.05, "seed":128}}'
             ])

        exit_code, _ = self.wait()
        return exit_code == 0

    def ready_go(self, wait_for=None):
        # 执行
        self.execute(
            ["benchmark",
             "--DatasetPath", self.dataset_path,
             "--DatasetType", self.dataset_type,
             "--ModelName", self.model_name,
             "--ModelPath", self.model_path,
             "--TestType", "client",
             "--Http", f"http://{self.server_ip}:{self.server_port}",
             "--ManagementHttp", f"http://{self.server_mannager_ip}:{self.server_mannager_port}",
             "--Concurrency", 200,
             "--RequestRate", 10,
             "--MaxOutputLen", 128,
             "--Tokenizer", True],
            dict(MINDIE_LOG_TO_STDOUT="benchmark:1; client:1"))

        exit_code, _ = self.wait(target=wait_for)
        logging.info(f"wait result: {exit_code}")
        return exit_code == 0