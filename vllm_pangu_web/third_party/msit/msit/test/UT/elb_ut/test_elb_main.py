

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
import sys
import unittest
from unittest.mock import patch, MagicMock
from argparse import ArgumentParser
from components.expert_load_balancing.elb.__main__ import ExpertLoadBalanceCommmand, check_input_path_legality



class TestExpertLoadBalanceCommandHandle(unittest.TestCase):

    def setUp(self):
        help_info = "Large Language Model(llm) Debugger Tools."
        self.command = ExpertLoadBalanceCommmand("expert-load-balancing", help_info)
        self.args = MagicMock()
        self.original_environ = dict(os.environ)
        self.original_path = list(sys.path)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_environ)
        sys.path = self.original_path

    @patch('os.path.exists')
    @patch('os.stat')
    @patch('os.getuid')
    @patch('pwd.getpwuid')
    @patch('components.expert_load_balancing.elb.__main__.logger')
    def test_handle_success(self, mock_logger, mock_getpwuid, mock_getuid, mock_stat, mock_exists):
        os.environ["ASCEND_TOOLKIT_HOME"] = "/valid/path"
        mock_exists.side_effect = lambda x: True
        
        mock_stat.return_value.st_uid = 1000
        mock_stat.return_value.st_mode = 0o750  # 安全权限
        mock_getuid.return_value = 1000
        mock_getpwuid.return_value.pw_name = "testuser"

        with patch.dict('sys.modules', {'elb.eplb_runner': MagicMock()}):
            self.command.handle(self.args)
        
        mock_logger.info.assert_any_call("===================load balancing algorithm start====================")
        mock_logger.info.assert_any_call("===================load balancing algorithm end====================")


    @patch('os.path.exists')
    @patch('components.expert_load_balancing.elb.__main__.logger')
    def test_handle_cann_path_not_exist(self, mock_logger, mock_exists):
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            self.command.handle(self.args)
        
        mock_logger.error.assert_called_with("CANN toolkit path does not exist. Please check your environment variables.")

    @patch('os.path.exists')
    @patch('os.stat')
    @patch('os.getuid')
    @patch('pwd.getpwuid')
    @patch('components.expert_load_balancing.elb.__main__.logger')
    def test_handle_unsafe_permissions(self, mock_logger, mock_getpwuid, mock_getuid, mock_stat, mock_exists):
        os.environ["ASCEND_TOOLKIT_HOME"] = "/valid/path"
        mock_exists.return_value = True
    
        mock_stat.return_value.st_uid = 1000
        mock_stat.return_value.st_mode = 0o777  # 不安全权限
        mock_getuid.return_value = 1000
        mock_getpwuid.return_value.pw_name = "testuser"
        with patch.dict('sys.modules', {'elb.eplb_runner': MagicMock()}):
            self.command.handle(self.args)

        mock_logger.warning.assert_called_with(
            "Algorithm path has unsafe permissions. Recommended permissions are <= 750."
        )
    
    @patch('os.path.exists')
    @patch('os.stat')
    @patch('os.getuid')
    @patch('pwd.getpwuid')
    @patch('components.expert_load_balancing.elb.__main__.logger')
    def test_handle_different_owner(self, mock_logger, mock_getpwuid, mock_getuid, mock_stat, mock_exists):
        os.environ["ASCEND_TOOLKIT_HOME"] = "/valid/path"
        
        mock_exists.return_value = True
        mock_stat.return_value.st_uid = 1001  # 不同用户
        mock_getuid.return_value = 1000
        mock_getpwuid.side_effect = [
            MagicMock(pw_name="testuser"),  # 当前用户
            MagicMock(pw_name="otheruser")  # 路径属主
        ]

        with patch.dict('sys.modules', {'elb.eplb_runner': MagicMock()}):
            self.command.handle(self.args)
    
        mock_logger.warning.assert_called_with(
            "Algorithm path is owned by another userinstead of the current user. This may cause permission issues."
        )

    @patch('os.path.exists')
    @patch('os.stat')
    @patch('os.getuid')
    @patch('pwd.getpwuid')
    @patch('components.expert_load_balancing.elb.__main__.logger')
    def test_handle_import_failure(self, mock_logger, mock_getpwuid, mock_getuid, mock_stat, mock_exists):
        os.environ["ASCEND_TOOLKIT_HOME"] = "/valid/path"
        mock_exists.return_value = True
    
        mock_stat.return_value.st_uid = 1000
        mock_stat.return_value.st_mode = 0o750
        mock_getuid.return_value = 1000
        mock_getpwuid.return_value.pw_name = "testuser"
        
        with self.assertRaises(Exception) as context:
            self.command.handle(self.args)
        
        self.assertEqual(str(context.exception), "Failed to import load_balancing module")



class TestExpertLoadBalanceCommmandAddArguments(unittest.TestCase):
    def setUp(self):
        help_info = "Large Language Model(llm) Debugger Tools."
        self.command = ExpertLoadBalanceCommmand("expert-load-balancing", help_info)
        self.parser = ArgumentParser()
        self.mock_parser = MagicMock()

    def test_add_arguments_all_required(self):
        """测试添加所有必选参数"""
        with patch.object(self.mock_parser, 'add_argument') as mock_add:
            self.command.add_arguments(self.mock_parser)

            mock_add.assert_any_call(
                '--info-csv-path', '-icp',
                dest="expert_popularity_csv_load_path",
                required=True,
                type=check_input_path_legality,
                help="Data input directory. Contains  CSV fileswhich might have been generated during prefill or decoder."
            )
            mock_add.assert_any_call(
                '--device-type', '-dt',
                dest="device_type",
                type=str,
                required=True,
                choices=['a2', 'a3'],
                help="device type. a2 代表适用于Atlas 800I A2推理服务器, a3 代表适用于Atlas 800I A3推理服务器。"
            )