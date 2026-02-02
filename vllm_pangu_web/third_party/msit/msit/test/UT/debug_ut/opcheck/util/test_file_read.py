import pytest
from unittest.mock import patch, Mock
from msit_opcheck.util.file_read import (
    execute_convert_npy_command,
    convert_ge_dump_file_to_npy,
    get_ascbackend_ascgraph
)


class TestFileRead:

    @patch('subprocess.run')
    def test_execute_convert_npy_command_given_valid_command_when_run_then_returns_stdout(
        self, mock_run
    ):
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "success"
        mock_run.return_value = mock_result
        command = ["dummy"]
        assert execute_convert_npy_command(command) == "success"

    @patch('subprocess.run')
    def test_execute_convert_npy_command_given_error_returncode_when_run_then_returns_error_message(
        self, mock_run
    ):
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "error"
        mock_run.return_value = mock_result
        command = ["dummy"]
        assert "Error while converting bin data to npy: error" in execute_convert_npy_command(command)

    @patch("msit_opcheck.util.file_read.check_input_path_legality")
    @patch('msit_opcheck.util.file_read.execute_convert_npy_command')
    def test_convert_ge_dump_file_to_npy_given_valid_command_when_executed_then_returns_success(
        self, mock_execute, mock_check
    ):
        mock_check.side_effect = lambda x: x
        mock_execute.return_value = "success"
        convert_ge_dump_file_to_npy("input", "output")

    @patch("msit_opcheck.util.file_read.check_input_path_legality")
    @patch('msit_opcheck.util.file_read.execute_convert_npy_command')
    def test_convert_ge_dump_file_to_npy_given_error_message_when_executed_then_raises_runtime_error(
        self, mock_execute, mock_check
    ):
        mock_check.side_effect = lambda x: x
        mock_execute.return_value = "Error while converting bin data to npy: dummy"
        with pytest.raises(RuntimeError):
            convert_ge_dump_file_to_npy("input", "output")

    @patch('os.listdir')
    def test_get_ascbackend_ascgraph_given_valid_dir_with_multiple_files_when_listed_then_returns_filtered_list(
        self, mock_listdir
    ):
        mock_listdir.return_value = ["file1.py", "autofuse_fused1.py", "file2.py"]
        assert get_ascbackend_ascgraph("dummy") == ["file1.py", "file2.py"]

    @patch('os.listdir')
    def test_get_ascbackend_ascgraph_given_dir_with_no_py_files_when_listed_then_returns_empty_list(
        self, mock_listdir
    ):
        mock_listdir.return_value = ["file1.txt", "file2.dat"]
        assert get_ascbackend_ascgraph("dummy") == []