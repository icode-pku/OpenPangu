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

import unittest
from unittest.mock import patch
import pandas as pd
from msserviceprofiler.ms_service_profiler_ext.split_processor.base_processor import (
    BaseFrameworkProcessor
)


class TestBatchSizeRecommend(unittest.TestCase):
    def setUp(self):
        self.framework_df = pd.DataFrame({
            "name": ["batch_start", "batch_end", "batch_start", "batch_end"],
            "batch_type": ["type1", "type1", "type2", "type2"],
            "batch_size": [32, 64, 1, 128]
        })
        self.instance = BaseFrameworkProcessor()  # 替换为实际的类实例

    @patch('msserviceprofiler.ms_service_profiler_ext.split_processor.base_processor.logger')
    def test_no_matching_rows(self, mock_logger):
        result = self.instance._get_batch_size_recommend(self.framework_df, "nonexistent_type")
        self.assertEqual(result, [-1])
        mock_logger.warning.assert_not_called()

    def test_non_empty_batch_size(self):
        result = self.instance._get_batch_size_recommend(self.framework_df, "type1")
        self.assertEqual(result, [32])

    @patch('msserviceprofiler.ms_service_profiler_ext.split_processor.base_processor.logger')
    def test_preprocess_framework_df_key_error(self, mock_logger):
        # 测试KeyError的情况
        df = pd.DataFrame({
            "name": ["Framework1", "Framework2"],
            "during_time": ["1.0", "2.0"],
            'pid': [1, 1]
        })
        missing_columns = ['tid', 'start_time', 'end_time', 'rid', 'start_datetime', 
                            'end_datetime', 'batch_type', 'batch_size', 'rid_list', 'token_id_list']
        result = self.instance.preprocess_framework_df(df)
        mock_logger.warning.assert_called_once_with(f"Field '{missing_columns} not in index' not found in datasource.")
        self.assertTrue(result.empty)