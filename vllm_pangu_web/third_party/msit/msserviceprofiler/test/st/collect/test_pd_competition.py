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
import os
import uuid

from test.st.executor.exec_benchmark import ExecBenchmark
from test.st.executor.exec_mindie_server import ExecMindIEServer
from test.st.executor.exec_parse import ExecParse
from pytest_check import check


def test_example(devices, mindie_path, tmp_workspace):
    try:
        workspace_path = tmp_workspace
        model_path = '/model'
        dataset_path = '/dataset'

        # 启动服务
        mindie_server = ExecMindIEServer(workspace_path)
        mindie_server.set_device_id(*devices)
        mindie_server.set_mindie_path(mindie_path)
        mindie_server.set_model_path(model_path)
        mindie_server.set_prof_config(prof_dir=os.path.join(workspace_path, "prof_data"))
        mindie_server.set_prof_config(enable=1)
        assert mindie_server.ready_go()

        # curl 一条试试深浅
        benchmark = ExecBenchmark()
        benchmark.set_model_path(model_path)
        benchmark.set_dataset_path(dataset_path)
        assert benchmark.curl_test()

        mindie_server.set_prof_config(acl_task_time=0, enable=0)
        mindie_server.kill()

    finally:
        if mindie_server:
            mindie_server.kill()
        logging.info(f"workspace: {workspace_path}")