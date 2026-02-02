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

import os
import shutil
import json
from pathlib import Path
from unittest import TestCase

from st.utils import execute_cmd
from msserviceprofiler.msguard.security import open_s


def check_request_json_content(json_path):
    # 校验请求对应轮次数有没有正确生成
    try:
        # 读取JSON文件
        with open_s(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 检查JSON文件中是否有10个键值对
        if len(data) != 10:
            result = False
            message = f"JSON文件中键值对的数量不正确，实际数量为 {len(data)}，应为10个。"
        else:
            # 检查键 "0" 是否存在且值为50
            if data.get("0") != 50:
                result = False
                message = "键 '0' 的值不正确，应为50。"
            # 检查键 "10" 是否存在且值为50
            elif data.get("9") != 50:
                result = False
                message = "键 '9' 的值不正确，应为50。"
            else:
                result = True
        return result
    except Exception as e:
        return False


class TestTrainCmd(TestCase):
    ST_DATA_PATH = os.getenv("MS_SERVICE_PROFILER",
                             "/data/ms_service_profiler")
    INPUT_PATH = os.path.join(ST_DATA_PATH, "vllm_profiling_output")
    TRAIN_OUTPUT_PATH = os.path.join(ST_DATA_PATH, "train_vllm")
    COMMAND_SUCCESS = 0

    def setUp(self):
        os.makedirs(self.TRAIN_OUTPUT_PATH, mode=0o750, exist_ok=True)


    def tearDown(self):
        shutil.rmtree(self.TRAIN_OUTPUT_PATH)


    def test_compare_ms_service_profiler_data(self):
        train_cmd = [
            "msserviceprofiler", "train",
            "-i", self.INPUT_PATH,
            "-o", self.TRAIN_OUTPUT_PATH,
            "-t", "vllm"
        ]
        if execute_cmd(train_cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.TRAIN_OUTPUT_PATH):
            self.assertFalse(
                True, msg="enable optimizer train vllm task failed.")
            return

        self.assertTrue((Path(self.TRAIN_OUTPUT_PATH) / 'req_id_and_decode_num.json').exists())
        self.assertTrue((Path(self.TRAIN_OUTPUT_PATH) / 'model/xgb_model.ubj').exists())
        result = check_request_json_content((Path(self.TRAIN_OUTPUT_PATH) / 'req_id_and_decode_num.json'))
        self.assertTrue(result)

