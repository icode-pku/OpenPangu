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


class TestSplitVllmCmd(TestCase):
    ST_DATA_PATH = os.getenv("MS_SERVICE_PROFILER",
                             "/data/ms_service_profiler")
    INPUT_PATH = os.path.join(ST_DATA_PATH, "input/split/vllm_latest")
    OUTPUT_PATH_ANALYZE_FOR_SPLIT = os.path.join(ST_DATA_PATH, "output/analyze_for_split")
    OUTPUT_PATH = os.path.join(ST_DATA_PATH, "output/split")
    OUTPUT_PATH_P = os.path.join(ST_DATA_PATH, "output/split_p")
    OUTPUT_PATH_D = os.path.join(ST_DATA_PATH, "output/split_d")
    OUTPUT_PATH_RID = os.path.join(ST_DATA_PATH, "output/split_rid")
    OUTPUT_PATH_RID_P = os.path.join(ST_DATA_PATH, "output/split_rid_p")
    OUTPUT_PATH_RID_D = os.path.join(ST_DATA_PATH, "output/split_rid_d")
    PREFILL_CSV = "prefill.csv"
    DECODE_CSV = "decode.csv"
    COMMAND_SUCCESS = 0
    ANALYZE_PROFILER = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
                                  "msserviceprofiler/__main__.py")
    SPLIT_PROFILER = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
                                  "msserviceprofiler/__main__.py")
    COMMON_BATCH_SIZE = '1'
    PREFILL_BATCH_SIZE = '2'
    DECODE_BATCH_SIZE = '8'
    PREFILL_NUMBER = '100'
    DECODE_NUMBER = '100'


    def setUp(self):
        os.makedirs(self.OUTPUT_PATH_ANALYZE_FOR_SPLIT, mode=0o750, exist_ok=True)
        os.makedirs(self.OUTPUT_PATH, mode=0o750, exist_ok=True)
        os.makedirs(self.OUTPUT_PATH_P, mode=0o750, exist_ok=True)
        os.makedirs(self.OUTPUT_PATH_D, mode=0o750, exist_ok=True)
        os.makedirs(self.OUTPUT_PATH_RID, mode=0o750, exist_ok=True)
        os.makedirs(self.OUTPUT_PATH_RID_P, mode=0o750, exist_ok=True)
        os.makedirs(self.OUTPUT_PATH_RID_D, mode=0o750, exist_ok=True)
        cmd = [
            "python", self.ANALYZE_PROFILER, "analyze",
            "--input-path", self.INPUT_PATH,
            "--output-path", self.OUTPUT_PATH_ANALYZE_FOR_SPLIT
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH_ANALYZE_FOR_SPLIT):
            self.assertFalse(
                True, msg="enable ms service profiler analyze task failed.")

        request_data = pd.read_csv(os.path.join(self.OUTPUT_PATH_ANALYZE_FOR_SPLIT, 'request.csv'))
        self.rid = request_data.loc[0, 'http_rid']



    def tearDown(self):
        shutil.rmtree(self.OUTPUT_PATH_ANALYZE_FOR_SPLIT)
        shutil.rmtree(self.OUTPUT_PATH)
        shutil.rmtree(self.OUTPUT_PATH_P)
        shutil.rmtree(self.OUTPUT_PATH_D)
        shutil.rmtree(self.OUTPUT_PATH_RID)
        shutil.rmtree(self.OUTPUT_PATH_RID_P)
        shutil.rmtree(self.OUTPUT_PATH_RID_D)


    def check_split_task(self, task_name, output_path, csv_file_name):
        with self.subTest(f"Check {csv_file_name} content"):
            try:
                check_split_csv_content(output_path, csv_file_name)
            except Exception as e:
                self.fail(f"{task_name}: 检查 {csv_file_name} 时发生异常: {e}")

    def test_split_vllm_prefill_decode(self):
        # PD竞争 根据batch_size拆解 校验输出文件及内容
        cmd = [
            "python", self.SPLIT_PROFILER, "split",
            "--input-path", self.INPUT_PATH,
            "--output-path", self.OUTPUT_PATH,
            "--prefill-batch-size", self.COMMON_BATCH_SIZE,
            "--prefill-number", self.PREFILL_NUMBER,
            "--decode-batch-size", self.COMMON_BATCH_SIZE,
            "--decode-number", self.DECODE_NUMBER
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH):
            self.assertFalse(
                True, msg="enable split vllm prefill and decode task failed.")

        self.check_split_task("test_split_vllm_prefill_decode", self.OUTPUT_PATH, self.PREFILL_CSV)
        self.check_split_task("test_split_vllm_decode_prefill", self.OUTPUT_PATH, self.DECODE_CSV)

    def test_split_vllm_prefill(self):
        # PD竞争 根据batch_size拆解 校验输出文件及内容
        cmd = [
            "python", self.SPLIT_PROFILER, "split",
            "--input-path", self.INPUT_PATH,
            "--output-path", self.OUTPUT_PATH_P,
            "--prefill-batch-size", self.PREFILL_BATCH_SIZE,
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH_P):
            self.assertFalse(
                True, msg="enable split vllm prefill task failed.")

        self.check_split_task("test_split_vllm_prefill", self.OUTPUT_PATH_P, self.PREFILL_CSV)

    def test_split_vllm_decode(self):
        # PD竞争 根据batch_size拆解 校验输出文件及内容
        cmd = [
            "python", self.SPLIT_PROFILER, "split",
            "--input-path", self.INPUT_PATH,
            "--output-path", self.OUTPUT_PATH_D,
            "--decode-batch-size", self.DECODE_BATCH_SIZE,
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH_D):
            self.assertFalse(
                True, msg="enable split vllm decode task failed.")

        self.check_split_task("test_split_vllm_decode", self.OUTPUT_PATH_D, self.DECODE_CSV)

    def test_split_vllm_prefill_decode_rid(self):
        # PD竞争 根据batch_size拆解 校验输出文件及内容
        cmd = [
            "python", self.SPLIT_PROFILER, "split",
            "--input-path", self.INPUT_PATH,
            "--output-path", self.OUTPUT_PATH_RID,
            "--prefill-rid", self.rid,
            "--decode-rid", self.rid
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH_RID):
            self.assertFalse(
                True, msg="enable split vllm prefill and decode rid task failed.")

        self.check_split_task("test_split_vllm_prefill_decode_rid", self.OUTPUT_PATH_RID, self.PREFILL_CSV)
        self.check_split_task("test_split_vllm_decode_prefill_rid", self.OUTPUT_PATH_RID, self.DECODE_CSV)

    def test_split_vllm_prefill_rid(self):
        # PD竞争 根据batch_size拆解 校验输出文件及内容
        cmd = [
            "python", self.SPLIT_PROFILER, "split",
            "--input-path", self.INPUT_PATH,
            "--output-path", self.OUTPUT_PATH_RID_P,
            "--prefill-rid", self.rid,
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH_RID_P):
            self.assertFalse(
                True, msg="enable split vllm prefill rid task failed.")

        self.check_split_task("test_split_vllm_prefill_rid", self.OUTPUT_PATH_RID_P, self.PREFILL_CSV)

    def test_split_vllm_decode_rid(self):
        # PD竞争 根据batch_size拆解 校验输出文件及内容
        cmd = [
            "python", self.SPLIT_PROFILER, "split",
            "--input-path", self.INPUT_PATH,
            "--output-path", self.OUTPUT_PATH_RID_D,
            "--decode-rid", self.rid,
        ]
        if execute_cmd(cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.OUTPUT_PATH_RID_D):
            self.assertFalse(
                True, msg="enable split vllm decode rid task failed.")

        self.check_split_task("test_split_vllm_decode_rid", self.OUTPUT_PATH_RID_D, self.DECODE_CSV)