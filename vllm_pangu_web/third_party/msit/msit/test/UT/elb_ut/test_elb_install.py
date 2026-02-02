# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
from unittest import mock
from unittest.mock import patch

from components.expert_load_balancing.elb.__install__ import ExpertLoadBalanceInstall


class TestExpertLoadBalanceInstallCheck(unittest.TestCase):
    @patch('pkg_resources.working_set')
    def test_check_ok(self, mock_working_set):
        mock_pkg = unittest.mock.MagicMock()
        mock_pkg.key = "ortools"
        mock_working_set.__iter__.return_value = [mock_pkg]
        
        result = ExpertLoadBalanceInstall.check()
        self.assertEqual(result, "OK")

    @patch('pkg_resources.working_set')
    def test_check_missing_ortools(self, mock_working_set):
        mock_working_set.__iter__.return_value = []
        
        result = ExpertLoadBalanceInstall.check()
        self.assertEqual(result, "[error] ortools not installed. Please install ortools packages.")

    @patch('pkg_resources.working_set')
    def test_check_multiple_missing_packages(self, mock_working_set):
        mock_pkg1 = unittest.mock.MagicMock()
        mock_pkg1.key = "other_package"
        mock_working_set.__iter__.return_value = [mock_pkg1]
        
        result = ExpertLoadBalanceInstall.check()
        self.assertIn("ortools not installed", result)
        self.assertEqual(len(result.split("\n")), 1)  # 当前只有ortools一个检查项