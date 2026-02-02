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

import pathlib
import unittest
from unittest.mock import MagicMock, patch
from auto_optimizer.common.click_utils import (
    safe_string,
    check_input_path,
    check_output_model_path,
    is_graph_input_static,
    list_knowledges,
    convert_to_graph_optimizer,
    optimize_onnx,
    FormatMsg,
    evaluate_onnx,
    validate_opt_converter,
)
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_optimizer.optimizer import (
    GraphOptimizer,
    InferTestConfig,
)



class TestSafeString(unittest.TestCase):
    def test_safe_string_given_valid_string_when_checked_then_return_string(self):
        self.assertEqual(safe_string("valid_string"), "valid_string")

    def test_safe_string_given_empty_string_when_checked_then_return_empty_string(self):
        self.assertEqual(safe_string(""), "")

    def test_safe_string_given_invalid_string_when_checked_then_raise_value_error(self):
        with self.assertRaises(ValueError):
            safe_string("invalid_string!")


class TestCheckInputPath(unittest.TestCase):
    @patch('os.access')
    @patch('components.debug.common.logger.error')
    def test_check_input_path_given_existing_readable_path_when_checked_then_return_true(self, mock_logger, 
                                                                                         mock_access):
        mock_access.side_effect = [True, True]
        self.assertTrue(check_input_path("existing_path"))
        mock_logger.assert_not_called()

    @patch('os.access')
    @patch('components.debug.common.logger.error')
    def test_check_input_path_given_non_existing_path_when_checked_then_return_false(self, mock_logger, mock_access):
        mock_access.side_effect = [False, True]
        self.assertFalse(check_input_path("non_existing_path"))
        mock_logger.assert_called_once_with("Input path %r is not exist.", "non_existing_path")

    @patch('os.access')
    @patch('components.debug.common.logger.error')
    def test_check_input_path_given_existing_non_readable_path_when_checked_then_return_false(self, mock_logger, 
                                                                                              mock_access):
        mock_access.side_effect = [True, False]
        self.assertFalse(check_input_path("existing_non_readable_path"))
        mock_logger.assert_called_once_with("Input path %r is not readable.", "existing_non_readable_path")


class TestCheckOutputModelPath(unittest.TestCase):
    @patch('os.path.isdir')
    @patch('os.path.exists')
    @patch('components.debug.common.logger.error')
    def test_check_output_model_path_given_valid_path_when_checked_then_return_true(self, mock_logger, mock_exists, 
                                                                                    mock_isdir):
        mock_isdir.return_value = False
        mock_exists.return_value = True
        self.assertTrue(check_output_model_path("valid_path"))
        mock_logger.assert_not_called()

    @patch('os.path.isdir')
    @patch('os.path.exists')
    @patch('components.debug.common.logger.error')
    def test_check_output_model_path_given_directory_path_when_checked_then_return_false(self, mock_logger, 
                                                                                         mock_exists, mock_isdir):
        mock_isdir.return_value = True
        self.assertFalse(check_output_model_path("directory_path"))
        mock_logger.assert_called_once_with("Output path %r is a directory.", "directory_path")

    @patch('os.path.isdir')
    @patch('os.path.exists')
    @patch('components.debug.common.logger.error')
    def test_check_output_model_path_given_non_existing_path_when_checked_then_return_false(self, mock_logger, 
                                                                                            mock_exists, mock_isdir):
        mock_isdir.return_value = False
        mock_exists.return_value = False
        self.assertFalse(check_output_model_path("non_existing_path"))
        mock_logger.assert_called_once_with("Output path %r is not exist.", "non_existing_path")


class TestIsGraphInputStatic(unittest.TestCase):
    def test_is_graph_input_static_given_static_graph_when_checked_then_return_true(self):
        graph = MagicMock(spec=BaseGraph)
        graph.inputs = [MagicMock(shape=[1, 2, 3])]
        self.assertTrue(is_graph_input_static(graph))

    def test_is_graph_input_static_given_dynamic_graph_when_checked_then_return_false(self):
        graph = MagicMock(spec=BaseGraph)
        graph.inputs = [MagicMock(shape=['a', 2, 3])]
        self.assertFalse(is_graph_input_static(graph))

    def test_is_graph_input_static_given_graph_with_negative_dim_when_checked_then_return_false(self):
        graph = MagicMock(spec=BaseGraph)
        graph.inputs = [MagicMock(shape=[1, -2, 3])]
        self.assertFalse(is_graph_input_static(graph))


class TestListKnowledges(unittest.TestCase):
    @patch('auto_optimizer.KnowledgeFactory.get_knowledge_pool')
    @patch('components.debug.common.logger.info')
    def test_list_knowledges_given_knowledges_when_listed_then_log_info(self, mock_logger, mock_get_knowledge_pool):
        mock_get_knowledge_pool.return_value = ['Knowledge1', 'Knowledge2']
        list_knowledges()
        mock_logger.assert_any_call('Available knowledges:')
        mock_logger.assert_any_call('   0 Knowledge1')
        mock_logger.assert_any_call('   1 Knowledge2')


class TestOptimizeOnnx(unittest.TestCase):
    @patch('auto_optimizer.graph_refactor.onnx.graph.OnnxGraph.parse')
    @patch('components.debug.common.logger.warning')
    def test_optimize_onnx_given_invalid_model_when_optimized_then_return_empty_list(self, mock_logger, mock_parse):
        mock_parse.side_effect = Exception("Parse error")
        optimizer = MagicMock(spec=GraphOptimizer)
        input_model = pathlib.Path("invalid_model.onnx")
        output_model = pathlib.Path("output_model.onnx")
        self.assertEqual(optimize_onnx(optimizer, input_model, output_model, False, InferTestConfig()), [])
        mock_logger.assert_any_call('%s model parse failed.', 'invalid_model.onnx')

    @patch('auto_optimizer.graph_refactor.onnx.graph.OnnxGraph.parse')
    @patch('components.debug.common.logger.warning')
    def test_optimize_onnx_given_valid_model_when_optimized_then_return_applied_knowledges(self, mock_logger, 
                                                                                           mock_parse):
        optimizer = MagicMock(spec=GraphOptimizer)
        optimizer.apply_knowledges.return_value = (MagicMock(), ['Knowledge1', 'Knowledge2'])
        input_model = pathlib.Path("valid_model.onnx")
        output_model = pathlib.Path("output_model.onnx")
        self.assertEqual(optimize_onnx(optimizer, input_model, output_model, False, InferTestConfig()), 
                         ['Knowledge1', 'Knowledge2'])
        mock_logger.assert_not_called()


class TestEvaluateOnnx(unittest.TestCase):
    @patch('auto_optimizer.graph_refactor.onnx.graph.OnnxGraph.parse')
    @patch('components.debug.common.logger.warning')
    def test_evaluate_onnx_given_invalid_model_when_evaluated_then_return_empty_list(self, mock_logger, mock_parse):
        mock_parse.side_effect = Exception("Parse error")
        optimizer = MagicMock(spec=GraphOptimizer)
        model = pathlib.Path("invalid_model.onnx")
        self.assertEqual(evaluate_onnx(model, optimizer, False), [])
        mock_logger.assert_any_call('%s match failed.', 'invalid_model.onnx')

    @patch('auto_optimizer.graph_refactor.onnx.graph.OnnxGraph.parse')
    @patch('components.debug.common.logger.warning')
    def test_evaluate_onnx_given_valid_model_when_evaluated_then_return_applied_knowledges(self, mock_logger, 
                                                                                           mock_parse):
        optimizer = MagicMock(spec=GraphOptimizer)
        optimizer.apply_knowledges.return_value = (MagicMock(), ['Knowledge1', 'Knowledge2'])
        model = pathlib.Path("valid_model.onnx")
        self.assertEqual(evaluate_onnx(model, optimizer, False), ['Knowledge1', 'Knowledge2'])
        mock_logger.assert_not_called()


class TestFormatMsg(unittest.TestCase):
    @patch('components.debug.common.logger.error')
    def test_format_msg_show_given_file_when_called_then_log_error(self, mock_logger):
        format_msg = FormatMsg()
        format_msg.format_message = MagicMock(return_value="Error message")
        format_msg.show("test_file.txt")
        mock_logger.assert_called_once_with("Error message")


class TestConvertToGraphOptimizer(unittest.TestCase):
    def test_convert_to_graph_optimizer_given_valid_value_when_converted_then_return_graph_optimizer(self):
        optimizer = convert_to_graph_optimizer(MagicMock(), MagicMock(), "Knowledge1,Knowledge2")
        self.assertIsInstance(optimizer, GraphOptimizer)


class TestValidateOptConverter(unittest.TestCase):
    def test_validate_opt_converter_given_valid_value_when_validated_then_return_lowercase_value(self):
        self.assertEqual(validate_opt_converter(MagicMock(), MagicMock(), "ATC"), "atc")


if __name__ == '__main__':
    unittest.main()
