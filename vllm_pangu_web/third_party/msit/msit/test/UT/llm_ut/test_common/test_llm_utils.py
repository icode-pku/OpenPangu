import os
import stat
import unittest
from unittest.mock import patch
from argparse import ArgumentTypeError

import pytest

from msit_llm.common.utils import (
    check_positive_integer,
    check_dump_time_integer,
    safe_string,
    check_number_list,
    check_ids_string,
    check_exec_script_file,
    check_input_args,
    check_exec_cmd,
    check_output_path_legality,
    check_input_path_legality,
    check_data_file_size,
    str2bool,
    load_file_to_read_common_check,
    check_device_integer_range_valid,
    check_device_range_valid,
    check_cosine_similarity, 
    check_kl_divergence,
    check_l1_norm,
    safe_int_env
)


@pytest.fixture(scope='function')
def temp_large_file(tmp_path_factory):
    file_path = str(tmp_path_factory.mktemp("data") / "data_file.txt")
   # 文件权限为 640 (-rw-r-----)
    file_permissions = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP

    with os.fdopen(os.open(file_path, os.O_CREAT | os.O_WRONLY, file_permissions), 'wb') as f:
        f.write(b'hello')
    yield file_path
    if os.path.exists(file_path):
        os.remove(file_path)


@pytest.mark.parametrize("value, expected", [(1, 1), (0, 0), (2, 2)])
def test_check_positive_integer_valid(value, expected):
    assert check_positive_integer(value) == expected


@pytest.mark.parametrize("value", [-1, 3])
def test_check_positive_integer_invalid(value):
    with pytest.raises(ArgumentTypeError):
        check_positive_integer(value)


@pytest.mark.parametrize("value, expected", [(1, 1), (0, 0), (2, 2), (3, 3)])
def test_check_dump_time_integer_valid(value, expected):
    assert check_dump_time_integer(value) == expected


@pytest.mark.parametrize("value", [-1, 4])
def test_check_dump_time_integer_invalid(value):
    with pytest.raises(ArgumentTypeError):
        check_dump_time_integer(value)


@pytest.mark.parametrize("value", ["ValidString123", "", None])
def test_safe_string_valid(value):
    assert safe_string(value) == value


@pytest.mark.parametrize("value", ["Invalid|String"])
def test_safe_string_invalid(value):
    with pytest.raises(ValueError):
        safe_string(value)


@pytest.mark.parametrize("value, expected", [("1,2,3", "1,2,3"), ("", ""), (None, None)])
def test_check_number_list_valid(value, expected):
    assert check_number_list(value) == expected


@pytest.mark.parametrize("value", ["1,2,invalid", "string", "1,2,3,invalid"])
def test_check_number_list_invalid(value):
    with pytest.raises(ArgumentTypeError):
        check_number_list(value)


@pytest.mark.parametrize("value, expected", [("1_2,3_4", "1_2,3_4"), ("", ""), (None, None)])
def test_check_ids_string_valid(value, expected):
    assert check_ids_string(value) == expected


@pytest.mark.parametrize("value", ["invalid_ids", "-1,-1", "_1_0,1_0"])
def test_check_ids_string_invalid(value):
    with pytest.raises(ArgumentTypeError):
        check_ids_string(value)


def test_check_exec_script_file_existing_file():
    with pytest.raises(ArgumentTypeError):
        check_exec_script_file("non_existing_script.sh")


@pytest.mark.parametrize("args", [["arg1", "|", "arg3"]])
def test_check_input_args(args):
    with pytest.raises(ArgumentTypeError):
        check_input_args(args)


@pytest.mark.parametrize("command", ["python3 aa.sh", "invalid command"])
def test_check_exec_cmd(command):
    with pytest.raises(ArgumentTypeError):
        check_exec_cmd(command)


def test_check_output_path_legality_existing_path():
    with pytest.raises(ArgumentTypeError):
        check_output_path_legality("invalid_&&_file|path")


def test_check_input_path_legality_existing_paths():
    with pytest.raises(ArgumentTypeError):
        check_input_path_legality("non_existing_input_path1,non_existing_input_path2")


# Test cases for check_data_file_size function
def test_check_data_file_size_existing_legal_file(temp_large_file):
    assert check_data_file_size(temp_large_file) == True


def test_check_data_file_size_non_existing_file():
    non_existing_file = "non_existing_file.txt"
    assert not os.path.exists(non_existing_file)
    with pytest.raises(Exception):
        check_data_file_size(non_existing_file)


@pytest.mark.parametrize("value, expected", [
    ("True", True), ("true", True), ("T", True), ("t", True), ("1", True),
    ("False", False), ("false", False), ("F", False), ("f", False), ("0", False),
])
def test_str2bool_valid(value, expected):
    assert str2bool(value) == expected


@pytest.mark.parametrize("value", ["invalid", "2", ""])
def test_str2bool_invalid(value):
    with pytest.raises(ArgumentTypeError):
        str2bool(value)

def test_check_device_integer_range_valid():
    valid_id = 5
    check_device_integer_range_valid(valid_id)
    invalid_id = -1
    with pytest.raises(ArgumentTypeError):
        check_device_integer_range_valid(invalid_id)

def test_check_device_range_valid():
    valid_single = "5"
    result = check_device_range_valid(valid_single)
    assert result == 5

    valid_multi = "1,2,3"
    result = check_device_range_valid(valid_multi)
    assert result == [1, 2, 3]

    invalid_char = "a"
    with pytest.raises(ArgumentTypeError):
        check_device_range_valid(invalid_char)

def test_check_cosine_similarity():
    valid_cosine = "0.5"
    result = check_cosine_similarity(valid_cosine)
    assert result == 0.5

    invalid_cosine = "2"
    with pytest.raises(ArgumentTypeError):
        check_cosine_similarity(invalid_cosine)

    non_num_cosine = "abc"
    with pytest.raises(ArgumentTypeError):
        check_cosine_similarity(non_num_cosine)

def test_check_kl_divergence():
    valid_kl = "0.5"
    result = check_kl_divergence(valid_kl)
    assert result == 0.5

    invalid_kl = "-0.5"
    with pytest.raises(ArgumentTypeError):
        check_kl_divergence(invalid_kl)

    non_num_kl = "abc"
    with pytest.raises(ArgumentTypeError):
        check_kl_divergence(non_num_kl)

def test_check_l1_norm():
    valid_l1 = "0.5"
    result = check_l1_norm(valid_l1)
    assert result == 0.5

    invalid_l1 = "-2"
    with pytest.raises(ArgumentTypeError):
        check_l1_norm(invalid_l1)

    non_num_l1 = "abc"
    with pytest.raises(ArgumentTypeError):
        check_l1_norm(non_num_l1)


class TestCommon(unittest.TestCase):

    def test_load_file_to_read_common_check_invalid_char(self):
        with self.assertLogs('msit_logger', 'ERROR') as cm:
            self.assertRaises(ValueError, load_file_to_read_common_check, "\n\r")
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r'Invalid character')
            
    def test_load_file_to_read_common_check_invalid_exts_input(self):
        with self.assertLogs('msit_logger', 'ERROR') as cm:
            self.assertRaises(TypeError, load_file_to_read_common_check, "abc.abc", exts='abc')
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r"Expected 'exts' to be")

    def test_load_file_to_read_common_check_invalid_exts_value(self):
        with self.assertLogs('msit_logger', 'ERROR') as cm:
            self.assertRaises(ValueError, load_file_to_read_common_check, "abc.abc", 
                              exts=['a', 'b', 'c'])
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r"Expected extenstion to be one")

    def test_load_file_to_read_common_check_file_name_too_long(self):
        with self.assertLogs('msit_logger', 'ERROR') as cm:
            self.assertRaises(OSError, load_file_to_read_common_check, "s" * 256)
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r'File name too long')

    def test_load_file_to_read_common_check_file_not_exist(self):
        with self.assertLogs('msit_logger', 'ERROR') as cm:
            self.assertRaises(FileNotFoundError, load_file_to_read_common_check, "abcde12345")
            logger_output = cm.output
            self.assertEqual(len(logger_output), 1)
            self.assertRegex(logger_output[0], r'No such file or directory')

    def test_load_file_to_read_common_check_not_reg_file(self):
        temp_dir = 'perm_dir'
        os.makedirs(temp_dir, 0, exist_ok=True)

        try:
            self.assertRaises(ValueError, load_file_to_read_common_check, temp_dir)
        finally:
            os.rmdir(temp_dir)

    def test_load_file_to_read_common_check_file_too_large(self):
        with patch('os.path.getsize', return_value=300 * 1024 * 1024 * 1024):
            with self.assertLogs('msit_logger', 'ERROR') as cm:
                self.assertRaises(ValueError, load_file_to_read_common_check, __file__)
                logger_output = cm.output
                self.assertEqual(len(logger_output), 1)
                self.assertRegex(logger_output[0], r'File too large')

    def test_load_file_to_read_common_check_file_other_writeable(self):
        file_stat = list(os.stat(__file__))
        file_stat[0] |= os.st.S_IWOTH
        file_stat = os.stat_result(file_stat)
        with patch('os.stat', return_value=file_stat):
            with self.assertLogs('msit_logger', 'ERROR') as cm:
                self.assertRaises(PermissionError, load_file_to_read_common_check, __file__)
                logger_output = cm.output
                self.assertEqual(len(logger_output), 1)
                self.assertRegex(logger_output[0], r'Vulnerable path')

    def test_load_file_to_read_common_check_file_uid_not_matched(self):
        file_stat = list(os.stat(__file__))
        with patch('os.geteuid', return_value=1001):
            file_stat[4] = 1002
            file_stat = os.stat_result(file_stat)

            with patch('os.stat', return_value=file_stat):
                self.assertRaises(PermissionError, load_file_to_read_common_check, __file__)     

    @patch("os.getenv")
    def test_valid_integer(self, mock_getenv):
        mock_getenv.return_value = "42"
        self.assertEqual(safe_int_env("RANK", 10), 42)
    
    @patch("os.getenv")
    def test_invalid_integer(self, mock_getenv):
        mock_getenv.return_value = "\ninvalid\n"
        self.assertEqual(safe_int_env("LOCAL_RANK", 10), 10)
    
    @patch("os.getenv")
    def test_empty_string(self, mock_getenv):
        mock_getenv.return_value = ""
        self.assertEqual(safe_int_env("RANK", 10), 10)
    
    @patch("os.getenv")
    def test_whitespace_string(self, mock_getenv):
        mock_getenv.return_value = "   "
        self.assertEqual(safe_int_env("RANK", 10), 10)
    
    @patch("os.getenv")
    def test_negative_integer(self, mock_getenv):
        mock_getenv.return_value = "-5"
        self.assertEqual(safe_int_env("RANK", 10), -5)
    
    @patch("os.getenv") 
    def test_float_value(self, mock_getenv):
        mock_getenv.return_value = "3.14"
        self.assertEqual(safe_int_env("RANK", 10), 10)
