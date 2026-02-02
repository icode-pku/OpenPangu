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
import glob
from unittest import TestCase
from msserviceprofiler.msguard.security import open_s


def check_csv_no_empty_start(csv_file):
    with open_s(csv_file, 'r', encoding='utf-8') as f:
        next(f)  # 跳过标题行（第一行）

        for _, line in enumerate(f, start=2):  # 从第2行开始检查（数据行）
            stripped_line = line.strip()
            
            # 检查是否以逗号开头（即第一个单元格是否为空）
            if stripped_line.startswith(','):
                return False

        return True


def get_modelevalstate_path():
    # optimizer命令当前只能在该目录下运行
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    target_path = os.path.abspath(os.path.join(dir_path, "../../../msserviceprofiler/modelevalstate"))
    return target_path


class TestTrainCmd(TestCase):
    COMMAND_SUCCESS = 0
    MODELEVALSTATE_DIR = get_modelevalstate_path()
    RESULT_DIR = os.path.join(MODELEVALSTATE_DIR, "result")

    def setUp(self):
        if os.path.exists(self.RESULT_DIR):
            shutil.rmtree(self.RESULT_DIR)

    def tearDown(self):
        if os.path.exists(self.RESULT_DIR):
            shutil.rmtree(self.RESULT_DIR)

    def test_compare_ms_service_profiler_data(self):
        train_cmd = "cd " + self.MODELEVALSTATE_DIR + " && msserviceprofiler optimizer"

        self.assertEqual(os.system(train_cmd), self.COMMAND_SUCCESS)
        self.assertTrue(os.path.exists(self.RESULT_DIR))

        pattern = os.path.join(self.RESULT_DIR, "store", "data_storage_*.csv")
        matched_files = glob.glob(pattern)

        self.assertEqual(len(matched_files), 1)

        csv_file = matched_files[0]  # 取第一个匹配的文件
        result = check_csv_no_empty_start(csv_file)  # 调用检查函数
        self.assertTrue(result)
