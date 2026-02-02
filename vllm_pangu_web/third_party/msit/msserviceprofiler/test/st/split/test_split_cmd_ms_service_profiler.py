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
from unittest import TestCase
import pandas as pd
from st.utils import execute_cmd, check_split_csv_content
   

class TestAnalyzeCmd(TestCase):
    ST_DATA_PATH = os.getenv("MS_SERVICE_PROFILER",
                             "/data/ms_service_profiler")
    INPUT_PATH = os.path.join(ST_DATA_PATH, "input/split/MindIE_latest_PD_complete")
    PREFILL_INPUT_PATH = os.path.join(ST_DATA_PATH, "input/split/MindIE_latest_PD_split/p")
    DECODE_INPUT_PATH = os.path.join(ST_DATA_PATH, "input/split/MindIE_latest_PD_split/d")
    OUTPUT_PATH = os.path.join(ST_DATA_PATH, "output/split")
    REQUEST_PATH = os.path.join(OUTPUT_PATH, "request.csv")
    PREFILL_CSV = "prefill.csv"
    DECODE_CSV = "decode.csv"
    COMMAND_SUCCESS = 0
    SPLIT_PROFILER = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
                                    "msserviceprofiler/__main__.py")
    COMMON_BATCH_SIZE = "1"
    PREFILL_RID = "2728857197956474597"
    DECODE_RID = "2728857197956474597"

    def setUp(self):
        os.makedirs(self.OUTPUT_PATH, mode=0o750, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.OUTPUT_PATH)

    def check_split_task(self, task_name, output_path, csv_file_name):
        with self.subTest(f"Check {csv_file_name} content"):
            try:
                check_split_csv_content(output_path, csv_file_name)
            except Exception as e:
                self.fail(f"{task_name}: check {csv_file_name} wrong: {e}")

    def get_request_http_rid(self, input_path, output_path):
        cmd = [
            "python", self.SPLIT_PROFILER, "analyze",
            "--input-path", input_path,
            "--output-path", output_path
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS:
            self.fail("execute analyze failed")
        try:
            request_data = pd.read_csv(self.REQUEST_PATH)
            http_rid_row = request_data[request_data['execution_time(ms)'].notna()]
            if http_rid_row.empty:
                rid = request_data.iloc[0]["http_rid"]
            else:
                rid = http_rid_row.iloc[0]["http_rid"]
        except Exception as e:
            self.fail(f"get http_rid failed: {e}")

        return str(rid)

    def test_split_by_batch_size(self):
        # PD竞争 根据batch_size拆解 校验输出文件及内容
        cmd = [
            "python", self.SPLIT_PROFILER, "split",
            "--input-path", self.INPUT_PATH,
            "--output-path", self.OUTPUT_PATH,
            "--prefill-batch-size", self.COMMON_BATCH_SIZE,
            "--decode-batch-size", self.COMMON_BATCH_SIZE,
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(
                True, msg="enable split task by batch size failed.")
            
        self.check_split_task("test_split_by_batch_size", self.OUTPUT_PATH, self.PREFILL_CSV)
        self.check_split_task("test_split_by_batch_size", self.OUTPUT_PATH, self.DECODE_CSV)

    def test_split_by_rid(self):
        # PD竞争 根据rid拆解 校验输出文件及内容
        pd_complete_rid = self.get_request_http_rid(self.INPUT_PATH, self.OUTPUT_PATH)
        cmd = [
            "python", self.SPLIT_PROFILER, "split",
            "--input-path", self.INPUT_PATH,
            "--output-path", self.OUTPUT_PATH,
            "--prefill-rid", pd_complete_rid,
            "--decode-rid", pd_complete_rid,
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(
                True, msg="enable split task by rid failed.")
            
        self.check_split_task("test_split_by_rid", self.OUTPUT_PATH, self.PREFILL_CSV)
        self.check_split_task("test_split_by_rid", self.OUTPUT_PATH, self.DECODE_CSV)

    def test_split_data_in_p_node_by_batch_size(self):
        # PD分离 P 节点根据batch_size拆解 校验输出文件及内容
        cmd = ["python", self.SPLIT_PROFILER, "split",
               "--input-path", self.PREFILL_INPUT_PATH,
               "--output-path", self.OUTPUT_PATH,
               "--prefill-batch-size", self.COMMON_BATCH_SIZE]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(True, msg="enable split task by batch size in P Node failed.")

        self.check_split_task("test_split_data_in_p_node_by_batch_size", self.OUTPUT_PATH, self.PREFILL_CSV)

    def test_split_data_in_p_node_by_rid(self):
        # PD分离 P 节点根据rid拆解 校验输出文件及内容
        cmd = ["python", self.SPLIT_PROFILER, "split",
               "--input-path", self.PREFILL_INPUT_PATH,
               "--output-path", self.OUTPUT_PATH,
               "--prefill-rid", self.PREFILL_RID]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(True, msg="enable split task by rid in P Node failed.")

    def test_split_data_in_d_node_by_batch_size(self):
        # PD分离 P 节点根据batch_size拆解 校验输出文件及内容
        cmd = ["python", self.SPLIT_PROFILER, "split",
               "--input-path", self.DECODE_INPUT_PATH,
               "--output-path", self.OUTPUT_PATH,
               "--decode-batch-size", self.COMMON_BATCH_SIZE]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(True, msg="enable split task by batch size in D Node failed.")

        self.check_split_task("test_split_data_in_d_node_by_batch_size", self.OUTPUT_PATH, self.DECODE_CSV)

    def test_split_data_in_d_node_by_rid(self):
        # PD分离 D 节点根据rid拆解 校验输出文件及内容
        cmd = ["python", self.SPLIT_PROFILER, "split",
               "--input-path", self.DECODE_INPUT_PATH,
               "--output-path", self.OUTPUT_PATH,
               "--decode-rid", self.DECODE_RID]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(True, msg="enable split task by rid in D Node failed.")

        self.check_split_task("test_split_data_in_d_node_by_rid", self.OUTPUT_PATH, self.DECODE_CSV)