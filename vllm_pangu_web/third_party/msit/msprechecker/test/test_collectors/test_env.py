# -*- coding: utf-8 -*-
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
import unittest
from unittest.mock import patch
from msprechecker.collectors import EnvCollector


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.normal_collector = EnvCollector()
        self.filtered_collector = EnvCollector(filter_env=True)

    def test_normal_collect_should_return_all_os_environ(self):
        test_data = {"abc": "123", "ASCEND": "123"}
        with patch.dict(os.environ, test_data, clear=True):
            collect_result = self.normal_collector.collect()
            self.assertEqual(collect_result.data, test_data)

    def test_filtered_collect_should_return_ascend_related(self):
        test_data = {"abc": "123", "ASCEND": "123"}
        with patch.dict(os.environ, test_data, clear=True):
            collect_result = self.filtered_collector.collect()
            self.assertEqual(collect_result.data, {"ASCEND": "123"})
    
    @patch('os.environ')
    def test_when_occur_collect_error_should_added_in_error_handler(self, mock_environ):
        mock_environ.items.side_effect = RuntimeError
        collect_result = self.normal_collector.collect()
        self.assertEqual(len(collect_result.error_handler.errors), 1)
