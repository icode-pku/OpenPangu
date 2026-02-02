# -*- coding: utf-8 -*-
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

import unittest
from unittest.mock import patch, MagicMock
from auto_optimizer.ait_main import (
    ListCommand,
    EvaluateCommand,
    OptimizeCommand
)


class TestSurgeonCommands(unittest.TestCase):

    def test_list_command_handle_given_no_args_when_called_then_list_knowledges_called(self):
        with patch('auto_optimizer.ait_main.list_knowledges') as mock_list_knowledges:
            cmd = ListCommand("list", "List available Knowledges")
            cmd.handle(None)
            mock_list_knowledges.assert_called_once()

    def test_evaluate_command_handle_given_valid_path_when_called_then_cli_eva_called(self):
        args = MagicMock()
        args.path = 'valid_path'
        args.knowledges = 'knowledge1,knowledge2'
        args.recursive = False
        args.verbose = False
        args.processes = 1

        with patch('auto_optimizer.ait_main.check_input_path', return_value=True), \
             patch('auto_optimizer.ait_main.pathlib.Path'), \
             patch('auto_optimizer.ait_main.GraphOptimizer'), \
             patch('auto_optimizer.ait_main.cli_eva') as mock_cli_eva:
            cmd = EvaluateCommand("evaluate", "Evaluate model matching specified knowledges", alias_name="eva")
            cmd.handle(args)
            mock_cli_eva.assert_called_once()

    def test_evaluate_command_handle_given_invalid_path_when_called_then_cli_eva_not_called(self):
        args = MagicMock()
        args.path = 'invalid_path'
        args.knowledges = 'knowledge1,knowledge2'
        args.recursive = False
        args.verbose = False
        args.processes = 1

        with patch('auto_optimizer.ait_main.check_input_path', return_value=False), \
             patch('auto_optimizer.ait_main.cli_eva') as mock_cli_eva:
            cmd = EvaluateCommand("evaluate", "Evaluate model matching specified knowledges", alias_name="eva")
            cmd.handle(args)
            mock_cli_eva.assert_not_called()

    def test_optimize_command_handle_given_invalid_paths_when_called_then_optimize_onnx_not_called(self):
        args = MagicMock()
        args.input_model = 'invalid_input_path'
        args.output_model = 'invalid_output_path'
        args.knowledges = 'knowledge1,knowledge2'
        args.infer_test = False
        args.big_kernel = False
        args.attention_start_node = ''
        args.attention_end_node = ''
        args.device = 0
        args.loop = 100
        args.threshold = 0
        args.input_shape = None
        args.input_shape_range = None
        args.dynamic_shape = None
        args.output_size = None

        with patch('auto_optimizer.ait_main.check_input_path', return_value=False), \
             patch('auto_optimizer.ait_main.check_output_model_path', return_value=False), \
             patch('auto_optimizer.ait_main.optimize_onnx') as mock_optimize_onnx:
            cmd = OptimizeCommand("optimize", "Optimize model with specified knowledges", alias_name="opt")
            cmd.handle(args)
            mock_optimize_onnx.assert_not_called()


if __name__ == '__main__':
    unittest.main()
