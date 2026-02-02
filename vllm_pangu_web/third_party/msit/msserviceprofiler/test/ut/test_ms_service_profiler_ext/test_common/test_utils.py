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
import unittest
from unittest.mock import patch
import pandas as pd

from msserviceprofiler.ms_service_profiler_ext.common.utils import confirmation_interaction


class TestUtilsFuctions(unittest.TestCase):
    @patch('builtins.input', return_value='yes')
    def test_confirmation_interaction_true(self, mock_input):
        prompt = "(y/n): "
        result = confirmation_interaction(prompt)
        self.assertTrue(result)
    
    @patch('builtins.input', side_effect=Exception)
    def test_confirmation_interaction_exception(self, mock_input):
        prompt = "(y/n): "
        result = confirmation_interaction(prompt)
        self.assertFalse(result)
    