import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import sys


class TestDumpConfigInitSuccess(unittest.TestCase):

    @patch.dict("os.environ", {"ASCEND_TOOLKIT_HOME": "/mock/ascend"})
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.json.dump")
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.ms_open", new_callable=mock_open)
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.os.path.abspath")
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.os.path.dirname")
    @patch("os.path.isdir", return_value=True)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("components.utils.file_utils.check_path_readability", return_value=None)
    @patch("components.utils.file_utils.check_path_exists", return_value=None)
    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    def test_dump_config_success(
        self, mock_group_writable, mock_others_writable, mock_path_exists, mock_path_readability, mock_file_size, mock_path_owner_consistent, mock_path_isdir,
            mock_dirname, mock_abspath, mock_ms_open, mock_json_dump
    ):
        from components.debug.compare.msquickcmp.dump.mietorch.dump_config import DumpConfig
        mock_abspath.return_value = "/mock/path/to/file.py"
        mock_dirname.return_value = "/mock/path/to"
        config = DumpConfig(dump_path="/tmp", mode="all", op_switch="on", api_list="a,b")
        expected_config = {
            "dump": {
                "dump_path": "/tmp",
                "dump_mode": "all",
                "dump_op_switch": "on",
                "dump_list": [{"model_name": "Graph", "layer": ["a", "b"]}]
            }
        }
        self.assertEqual(config.config, expected_config)
        mock_json_dump.assert_called_once()
        mock_ms_open.assert_called_once()


class TestDumpConfigFileNotFound(unittest.TestCase):
    @patch.dict("os.environ", {"ASCEND_TOOLKIT_HOME": "/mock/ascend"})
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.ms_open", side_effect=FileNotFoundError)
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.os.path.abspath")
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.os.path.dirname")
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.utils.logger")
    @patch("os.path.isdir", return_value=True)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("components.utils.file_utils.check_path_readability", return_value=None)
    @patch("components.utils.file_utils.check_path_exists", return_value=None)
    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    def test_file_not_found_error(
        self, mock_group_writable, mock_others_writable, mock_path_exists, mock_path_readability, mock_file_size, mock_path_owner_consistent, mock_path_isdir,
            mock_logger, mock_dirname, mock_abspath, mock_ms_open
    ):
        from components.debug.compare.msquickcmp.dump.mietorch.dump_config import DumpConfig
        mock_abspath.return_value = "/mock/file.py"
        mock_dirname.return_value = "/mock"
        with self.assertRaises(FileNotFoundError):
            DumpConfig()
        mock_logger.error.assert_called_with("File not found.")


class TestDumpConfigJsonDecodeError(unittest.TestCase):
    @patch.dict("os.environ", {"ASCEND_TOOLKIT_HOME": "/mock/ascend"})
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.os.path.abspath")
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.os.path.dirname")
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.ms_open")
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.utils.logger")
    @patch("components.debug.compare.msquickcmp.dump.mietorch.dump_config.json.dump", side_effect=json.JSONDecodeError("Expecting value", "", 0))
    @patch("os.path.isdir", return_value=True)
    @patch("components.utils.file_utils.check_path_owner_consistent", return_value=None)
    @patch("components.utils.file_utils.check_file_size", return_value=None)
    @patch("components.utils.file_utils.check_path_readability", return_value=None)
    @patch("components.utils.file_utils.check_path_exists", return_value=None)
    @patch("components.utils.file_utils.check_others_writable", return_value=None)
    @patch("components.utils.file_utils.check_group_writable", return_value=None)
    def test_json_decode_error(
        self, mock_group_writable, mock_others_writable, mock_path_exists, mock_path_readability, mock_file_size, mock_path_owner_consistent, mock_path_isdir,
            mock_json_dump, mock_logger, mock_ms_open, mock_dirname, mock_abspath
    ):
        from components.debug.compare.msquickcmp.dump.mietorch.dump_config import DumpConfig
        mock_abspath.return_value = "/mock/file.py"
        mock_dirname.return_value = "/mock"
        with self.assertRaises(json.JSONDecodeError):
            DumpConfig()
        self.assertTrue(mock_logger.error.called)
        self.assertIn("JSON decode error", mock_logger.error.call_args[0][0])
