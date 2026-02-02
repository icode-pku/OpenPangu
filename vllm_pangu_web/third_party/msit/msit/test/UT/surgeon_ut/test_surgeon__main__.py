import unittest
from unittest import mock
from unittest.mock import patch, MagicMock, call
import pathlib
import sys
from click.testing import CliRunner
from auto_optimizer import __main__ as main_mod
import tempfile
import os


class TestMainCLI(unittest.TestCase):
    def setUp(self):
        # Patch sys.argv so click doesn't try to parse unittest arguments
        self.sys_argv_backup = sys.argv
        sys.argv = ["auto_optimizer"]
        self.runner = CliRunner()

    @patch("auto_optimizer.__main__.list_knowledges")
    def test_command_list(self, mock_list_knowledges):
        result = self.runner.invoke(main_mod.cli, ["list"])
        self.assertEqual(result.exit_code, 0)
        mock_list_knowledges.assert_called_once()

    @patch("auto_optimizer.__main__.cli_eva")
    def test_command_evaluate(self, mock_cli_eva):
        # 创建一个临时文件作为存在的路径
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmpfile:
            result = self.runner.invoke(
                main_mod.cli,
                [
                    "evaluate",
                    tmpfile.name,
                    "-k",
                    "some_knowledge",
                    "-p",
                    "2",
                    "-v",
                    "-r",
                ],
            )
            self.assertEqual(result.exit_code, 0)
            mock_cli_eva.assert_called()

    @patch("auto_optimizer.__main__.optimize_onnx")
    @patch("auto_optimizer.__main__.logger")
    def test_command_optimize_success(self, mock_logger, mock_optimize_onnx):
        mock_optimize_onnx.return_value = ["KnowledgeA", "KnowledgeB"]
        with tempfile.NamedTemporaryFile(
            suffix=".onnx"
        ) as input_file, tempfile.NamedTemporaryFile(suffix=".onnx") as output_file:
            args = [
                "optimize",
                input_file.name,
                output_file.name,
                "-k",
                "some_knowledge",
                "-d",
                "0",
                "-l",
                "10",
                "--threshold",
                "0.1",
                "--soc",
                "Ascend310P3",
            ]
            result = self.runner.invoke(main_mod.cli, args)
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(
                any(
                    "Success" in call.args[0]
                    for call in mock_logger.info.call_args_list
                )
            )

    @patch("auto_optimizer.__main__.optimize_onnx")
    @patch("auto_optimizer.__main__.logger")
    def test_command_optimize_no_knowledges(self, mock_logger, mock_optimize_onnx):
        mock_optimize_onnx.return_value = []
        with tempfile.NamedTemporaryFile(
            suffix=".onnx"
        ) as input_file, tempfile.NamedTemporaryFile(suffix=".onnx") as output_file:
            args = [
                "optimize",
                input_file.name,
                output_file.name,
                "-k",
                "some_knowledge",
            ]
            result = self.runner.invoke(main_mod.cli, args)
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(
                any(
                    "Unable to optimize" in call.args[0]
                    for call in mock_logger.info.call_args_list
                )
            )

    @patch("auto_optimizer.__main__.logger")
    def test_command_optimize_input_output_same(self, mock_logger):
        with tempfile.NamedTemporaryFile(suffix=".onnx") as same_file:
            args = ["optimize", same_file.name, same_file.name, "-k", "some_knowledge"]
            result = self.runner.invoke(main_mod.cli, args)
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(
                any(
                    "refuse to overwrite" in call.args[0]
                    for call in mock_logger.warning.call_args_list
                )
            )

    @patch("auto_optimizer.__main__.OnnxGraph")
    @patch("auto_optimizer.__main__.check_output_model_path")
    def test_command_extract_success(self, mock_check_path, mock_onnxgraph):
        mock_check_path.return_value = True
        mock_graph = MagicMock()
        mock_onnxgraph.parse.return_value = mock_graph
        with tempfile.NamedTemporaryFile(
            suffix=".onnx"
        ) as input_file, tempfile.NamedTemporaryFile(suffix=".onnx") as output_file:
            args = [
                "extract",
                input_file.name,
                output_file.name,
                "start1,start2",
                "end1,end2",
            ]
            result = self.runner.invoke(main_mod.cli, args)
            self.assertEqual(result.exit_code, 0)
            mock_graph.extract_subgraph.assert_called_once()

    @patch("auto_optimizer.__main__.OnnxGraph")
    @patch("auto_optimizer.__main__.check_output_model_path")
    @patch("auto_optimizer.__main__.logger")
    def test_command_extract_value_error(
        self, mock_logger, mock_check_path, mock_onnxgraph
    ):
        mock_check_path.return_value = True
        mock_graph = MagicMock()
        mock_graph.extract_subgraph.side_effect = ValueError("test error")
        mock_onnxgraph.parse.return_value = mock_graph
        with tempfile.NamedTemporaryFile(
            suffix=".onnx"
        ) as input_file, tempfile.NamedTemporaryFile(suffix=".onnx") as output_file:
            args = ["extract", input_file.name, output_file.name, "start", "end"]
            result = self.runner.invoke(main_mod.cli, args)
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(
                any(
                    "test error" in str(call.args[0])
                    for call in mock_logger.error.call_args_list
                )
            )

    @patch("auto_optimizer.__main__.check_output_model_path")
    @patch("auto_optimizer.__main__.logger")
    def test_command_extract_input_output_same(self, mock_logger, mock_check_path):
        with tempfile.NamedTemporaryFile(suffix=".onnx") as same_file:
            args = ["extract", same_file.name, same_file.name, "start", "end"]
            result = self.runner.invoke(main_mod.cli, args)
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(
                any(
                    "refuse to overwrite" in call.args[0]
                    for call in mock_logger.warning.call_args_list
                )
            )

    @patch("auto_optimizer.__main__.check_output_model_path")
    def test_command_extract_invalid_output_path(self, mock_check_path):
        mock_check_path.return_value = False
        with tempfile.NamedTemporaryFile(
            suffix=".onnx"
        ) as input_file, tempfile.NamedTemporaryFile(suffix=".onnx") as output_file:
            args = ["extract", input_file.name, output_file.name, "start", "end"]
            result = self.runner.invoke(main_mod.cli, args)
            self.assertEqual(result.exit_code, 0)

    def tearDown(self):
        sys.argv = self.sys_argv_backup


if __name__ == "__main__":
    unittest.main()
