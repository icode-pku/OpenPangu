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
from pathlib import Path
from unittest import TestCase

from st.utils import execute_cmd


class TestCompareCmd(TestCase):
    ST_DATA_PATH = os.getenv("MS_SERVICE_PROFILER",
                             "/data/ms_service_profiler")
    INPUT_PATH = os.path.join(ST_DATA_PATH, "input/analyze/1225-196-10Req")
    ANALYZE_OUTPUT_PATH = os.path.join(ST_DATA_PATH, "output/analyze")
    COMPARE_OUTPUT_PATH = os.path.join(ST_DATA_PATH, "output/compare")
    COMMAND_SUCCESS = 0
    ANALYZE_SCRIPT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
                                    "msserviceprofiler/__main__.py")
    COMPARE_SCRIPT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
                                    "msserviceprofiler/__main__.py")

    def setUp(self):
        os.makedirs(self.ANALYZE_OUTPUT_PATH, mode=0o750, exist_ok=True)
        os.makedirs(self.COMPARE_OUTPUT_PATH, mode=0o750, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.ANALYZE_OUTPUT_PATH)
        shutil.rmtree(self.COMPARE_OUTPUT_PATH)

    def test_compare_ms_service_profiler_data(self):
        # 校验msserviceprofiler打点采集数据解析功能是否正常解析，校验输出文件及内容
        analyze_cmd = [
            "python", self.ANALYZE_SCRIPT, "analyze",
            "--input-path", self.INPUT_PATH,
            "--output-path", self.ANALYZE_OUTPUT_PATH
        ]
        if execute_cmd(analyze_cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.ANALYZE_OUTPUT_PATH):
            self.assertFalse(
                True, msg="enable ms service profiler analyze task failed.")
            return
        
        compare_cmd = [
            "python", self.COMPARE_SCRIPT, "compare",
            self.ANALYZE_OUTPUT_PATH,
            self.ANALYZE_OUTPUT_PATH,
            "--output-path", self.COMPARE_OUTPUT_PATH
        ]
        if execute_cmd(compare_cmd) != self.COMMAND_SUCCESS or not os.path.exists(self.COMPARE_OUTPUT_PATH):
            self.assertFalse(
                True, msg="enable ms service profiler compare task failed.")
            return
        self.assertTrue((Path(self.COMPARE_OUTPUT_PATH) / 'compare_result.xlsx').exists())
        self.assertTrue((Path(self.COMPARE_OUTPUT_PATH) / 'compare_result.db').exists())
        self.assertTrue((Path(self.COMPARE_OUTPUT_PATH) / 'compare_visualization.json').exists())
