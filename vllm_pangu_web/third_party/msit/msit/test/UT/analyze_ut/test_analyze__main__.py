import unittest
from unittest.mock import patch, MagicMock
import argparse
from model_evaluation.common.enum import Framework
from model_evaluation.bean import ConvertConfig

# msit/components/analyze/model_evaluation/test___main__.py


from model_evaluation.__main__ import (
    check_model_path_legality,
    check_weight_path_legality,
    check_soc_string,
    check_output_path_legality,
    parse_input_param,
    AnalyzeCommand,
    get_cmd_instance,
)

class TestCheckModelPathLegality(unittest.TestCase):
    @patch("model_evaluation.__main__.FileStat")
    def test_valid(self, mock_filestat):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = True
        mock_stat.is_legal_file_type.return_value = True
        mock_stat.is_legal_file_size.return_value = True
        mock_filestat.return_value = mock_stat
        # 需要mock所有FileStat方法
        self.assertEqual(check_model_path_legality("somefile.onnx"), "somefile.onnx")

    @patch("model_evaluation.__main__.FileStat")
    def test_invalid_permission(self, mock_filestat):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = False
        mock_stat.is_legal_file_type.return_value = True
        mock_stat.is_legal_file_size.return_value = True
        mock_filestat.return_value = mock_stat
        with self.assertRaises(argparse.ArgumentTypeError):
            check_model_path_legality("file.onnx")

    @patch("model_evaluation.__main__.FileStat")
    def test_invalid_type(self, mock_filestat):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = True
        mock_stat.is_legal_file_type.return_value = False
        mock_stat.is_legal_file_size.return_value = True
        mock_filestat.return_value = mock_stat
        with self.assertRaises(argparse.ArgumentTypeError):
            check_model_path_legality("file.txt")

    @patch("model_evaluation.__main__.FileStat")
    def test_invalid_size(self, mock_filestat):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = True
        mock_stat.is_legal_file_type.return_value = True
        mock_stat.is_legal_file_size.return_value = False
        mock_filestat.return_value = mock_stat
        with self.assertRaises(argparse.ArgumentTypeError):
            check_model_path_legality("file.onnx")

    @patch("model_evaluation.__main__.FileStat", side_effect=Exception("fail"))
    def test_filestat_exception(self, mock_filestat):
        with self.assertRaises(argparse.ArgumentTypeError):
            check_model_path_legality("file.onnx")

class TestCheckWeightPathLegality(unittest.TestCase):
    @patch("model_evaluation.__main__.FileStat")
    def test_valid(self, mock_filestat):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = True
        mock_stat.is_legal_file_type.return_value = True
        mock_stat.is_legal_file_size.return_value = True
        mock_filestat.return_value = mock_stat
        self.assertEqual(check_weight_path_legality("somefile.caffemodel"), "somefile.caffemodel")

    @patch("model_evaluation.__main__.FileStat")
    def test_invalid_permission(self, mock_filestat):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = False
        mock_filestat.return_value = mock_stat
        with self.assertRaises(argparse.ArgumentTypeError):
            check_weight_path_legality("file.caffemodel")

    @patch("model_evaluation.__main__.FileStat")
    def test_invalid_type(self, mock_filestat):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = True
        mock_stat.is_legal_file_type.return_value = False
        mock_filestat.return_value = mock_stat
        with self.assertRaises(argparse.ArgumentTypeError):
            check_weight_path_legality("file.txt")

    @patch("model_evaluation.__main__.FileStat")
    def test_invalid_size(self, mock_filestat):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = True
        mock_stat.is_legal_file_type.return_value = True
        mock_stat.is_legal_file_size.return_value = False
        mock_filestat.return_value = mock_stat
        with self.assertRaises(argparse.ArgumentTypeError):
            check_weight_path_legality("file.caffemodel")

    @patch("model_evaluation.__main__.FileStat", side_effect=Exception("fail"))
    def test_filestat_exception(self, mock_filestat):
        with self.assertRaises(argparse.ArgumentTypeError):
            check_weight_path_legality("file.caffemodel")

class TestCheckSocString(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(check_soc_string("Ascend310"), "Ascend310")
        self.assertEqual(check_soc_string("soc_123-abc"), "soc_123-abc")

    def test_invalid(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            check_soc_string("soc@123")

class TestCheckOutputPathLegality(unittest.TestCase):
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    @patch("model_evaluation.__main__.FileStat")
    def test_valid(self, mock_filestat, mock_path_exists, mock_path_writability, mock_path_owner_consistent):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = True
        mock_filestat.return_value = mock_stat
        self.assertEqual(check_output_path_legality("outdir"), "outdir")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    @patch("model_evaluation.__main__.FileStat")
    def test_invalid_permission(self, mock_filestat, mock_path_exists, mock_path_writability, mock_path_owner_consistent):
        mock_stat = MagicMock()
        mock_stat.is_basically_legal.return_value = False
        mock_filestat.return_value = mock_stat
        with self.assertRaises(argparse.ArgumentTypeError):
            check_output_path_legality("outdir")

    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("os.access", return_value=True)
    @patch("os.path.exists", return_value=True)
    @patch("model_evaluation.__main__.FileStat", side_effect=Exception("fail"))
    def test_filestat_exception(self, mock_filestat, mock_path_exists, mock_path_writability, mock_path_owner_consistent):
        with self.assertRaises(argparse.ArgumentTypeError):
            check_output_path_legality("outdir")

class TestParseInputParam(unittest.TestCase):
    @patch("model_evaluation.common.utils.get_framework")
    @patch("model_evaluation.common.utils.get_soc_type")
    def test_valid(self, mock_get_soc_type, mock_get_framework):
        mock_get_framework.return_value = Framework.ONNX
        mock_get_soc_type.return_value = "Ascend310"
        cfg = parse_input_param("model.onnx", None, "w", None)
        self.assertIsInstance(cfg, ConvertConfig)
        self.assertEqual(cfg.framework, Framework.ONNX)
        self.assertEqual(cfg.soc_type, "Ascend310")
        self.assertEqual(cfg.weight, "w")

    def test_invalid_framework(self):
        with self.assertRaises(ValueError):
            parse_input_param("model.onnx", "abc", "w", "soc")

    @patch("model_evaluation.common.utils.get_framework")
    def test_unknown_framework(self, mock_get_framework):
        mock_get_framework.return_value = Framework.UNKNOWN
        with self.assertRaises(ValueError):
            parse_input_param("model.abc", None, "w", "soc")

class TestAnalyzeCommand(unittest.TestCase):
    @patch("model_evaluation.__main__.os.path.isfile", return_value=False)
    @patch("model_evaluation.__main__.logger")
    def test_handle_model_not_file(self, mock_logger, mock_isfile):
        cmd = AnalyzeCommand("analyze", "help")
        args = argparse.Namespace(
            golden_model="not_exist.onnx",
            framework=None,
            weight="",
            soc_version="",
            output="out"
        )
        cmd.handle(args)
        mock_logger.error.assert_called_with('input model is not file.')

    @patch("model_evaluation.__main__.FileStat")
    @patch("model_evaluation.__main__.os.path.isfile", return_value=True)
    @patch("model_evaluation.__main__.logger")
    @patch("model_evaluation.__main__.parse_input_param", side_effect=ValueError("fail"))
    def test_handle_parse_input_param_fail(self, mock_parse, mock_logger, mock_isfile, mock_filestat):
        cmd = AnalyzeCommand("analyze", "help")
        args = argparse.Namespace(
            golden_model="model.onnx",
            framework=None,
            weight="",
            soc_version="",
            output="out"
        )
        cmd.handle(args)
        mock_logger.error.assert_called_with('fail')

    @patch("model_evaluation.__main__.FileStat")
    @patch("model_evaluation.__main__.os.path.isfile", return_value=True)
    @patch("model_evaluation.__main__.logger")
    @patch("model_evaluation.__main__.parse_input_param")
    @patch("model_evaluation.__main__.Analyze", return_value=None)
    def test_handle_analyze_none(self, mock_analyze, mock_parse, mock_logger, mock_isfile, mock_filestat):
        cmd = AnalyzeCommand("analyze", "help")
        args = argparse.Namespace(
            golden_model="model.onnx",
            framework=None,
            weight="",
            soc_version="",
            output="out"
        )
        cmd.handle(args)
        mock_logger.error.assert_called_with("the object of 'Analyze' create failed.")

    @patch("model_evaluation.__main__.FileStat")
    @patch("model_evaluation.__main__.os.path.isfile", return_value=True)
    @patch("model_evaluation.__main__.logger")
    @patch("model_evaluation.__main__.parse_input_param")
    @patch("model_evaluation.__main__.Analyze")
    def test_handle_success(self, mock_analyze, mock_parse, mock_logger, mock_isfile, mock_filestat):
        mock_analyze_instance = MagicMock()
        mock_analyze.return_value = mock_analyze_instance
        cmd = AnalyzeCommand("analyze", "help")
        args = argparse.Namespace(
            golden_model="model.onnx",
            framework=None,
            weight="",
            soc_version="",
            output="out"
        )
        cmd.handle(args)
        mock_analyze_instance.analyze_model.assert_called_once()
        mock_logger.info.assert_called_with('analyze model finished.')

class TestGetCmdInstance(unittest.TestCase):
    def test_instance(self):
        cmd = get_cmd_instance()
        self.assertIsInstance(cmd, AnalyzeCommand)
        self.assertEqual(cmd.name, "analyze")
        self.assertIn("Analyze tool", cmd.help_info)