import unittest
from unittest.mock import patch, MagicMock

from components.debug.compare.msquickcmp.common.convert import convert_bin_dump_data_to_npy, convert_bin_file_to_npy


class TestConvertFunctions(unittest.TestCase):

    @patch("components.debug.compare.msquickcmp.common.convert.utils.logger")
    @patch("components.debug.compare.msquickcmp.common.convert.utils.execute_command")
    @patch("components.debug.compare.msquickcmp.common.convert.ms_makedirs")
    @patch("components.debug.compare.msquickcmp.common.convert.os.path.exists")
    @patch("components.debug.compare.msquickcmp.common.convert.os.path.normpath")
    @patch("components.debug.compare.msquickcmp.common.convert.os.path.join")
    @patch("components.debug.compare.msquickcmp.common.convert.os.path.relpath")
    @patch("components.debug.compare.msquickcmp.common.convert.os.path.commonprefix")
    @patch("components.debug.compare.msquickcmp.common.convert.sys.executable", new="/usr/bin/python3")
    def test_convert_bin_dump_data_to_npy(
        self,
        mock_commonprefix,
        mock_relpath,
        mock_join,
        mock_normpath,
        mock_exists,
        mock_makedirs,
        mock_execute,
        mock_logger
    ):
        # Setup mock values
        mock_commonprefix.return_value = "/dump"
        mock_relpath.return_value = "20240601/otherpath"
        mock_join.side_effect = lambda *args: "/".join(args)
        mock_normpath.side_effect = lambda x: x
        mock_exists.return_value = False

        # Inputs
        dump_path = "/dump/20240601/otherpath"
        net_output_path = "/dump/20240601/output"
        cann_path = "/opt/cann"

        result = convert_bin_dump_data_to_npy(dump_path, net_output_path, cann_path)

        expected_convert_dir = "/dump/20240601/otherpath_bin2npy"
        expected_cmd = [
            "python3",
            "/opt/cann/toolkit/tools/operator_cmp/compare/msaccucmp.py",
            "convert",
            "-d",
            dump_path,
            "-out",
            expected_convert_dir,
        ]

        mock_commonprefix.assert_called_once_with([dump_path, net_output_path])
        mock_relpath.assert_called_once()
        mock_exists.assert_called_once_with(expected_convert_dir)
        mock_makedirs.assert_called_once_with(expected_convert_dir)
        mock_execute.assert_called_once_with(expected_cmd, False)
        mock_logger.info.assert_called_once()
        self.assertEqual(result, expected_convert_dir)

    @patch("components.debug.compare.msquickcmp.common.convert.utils.logger")
    @patch("components.debug.compare.msquickcmp.common.convert.utils.execute_command")
    @patch("components.debug.compare.msquickcmp.common.convert.os.path.join")
    @patch("components.debug.compare.msquickcmp.common.convert.sys.executable", new="/usr/bin/python3")
    def test_convert_bin_file_to_npy(
        self,
        mock_join,
        mock_execute,
        mock_logger
    ):
        # Setup
        bin_file_path = "/some/file.bin"
        npy_dir_path = "/some/output"
        cann_path = "/opt/cann"
        mock_join.side_effect = lambda *args: "/".join(args)

        convert_bin_file_to_npy(bin_file_path, npy_dir_path, cann_path)

        expected_cmd = [
            "python3",
            "/opt/cann/toolkit/tools/operator_cmp/compare/msaccucmp.py",
            "convert",
            "-d",
            bin_file_path,
            "-out",
            npy_dir_path,
        ]

        mock_logger.info.assert_called_once()
        mock_execute.assert_called_once_with(expected_cmd)
