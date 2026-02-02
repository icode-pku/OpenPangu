# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import stat
import unittest
import argparse
import tempfile

from unittest.mock import patch, Mock

import pytest

from components.utils.check import PathChecker
from components.utils.security_check import (
    is_belong_to_user_or_group,
    is_endswith_extensions,
    get_valid_path,
    check_write_directory,
    get_valid_read_path,
    type_to_str,
    check_type,
    check_number,
    check_int,
    check_element_type,
    check_character,
    check_dict_character,
    find_existing_path,
    is_enough_disk_space_left,
    ms_makedirs,
    check_positive_integer,
    check_output_path_legality,
    check_input_opsummary_legality,
    valid_ops_map_file
)
from components.utils.file_open_check import FileStat


MAX_READ_FILE_SIZE_4G = 4294967296  # 4G, 4 * 1024 * 1024 * 1024
MAX_READ_FILE_SIZE_32G = 34359738368  # 32G, 32 * 1024 * 1024 * 1024
MIN_DUMP_DISK_SPACE = 2147483648  # 2G, 2 * 1024 * 1024 * 1024
READ_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH
WRITE_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH


class TestMakedirs(unittest.TestCase):

    def setUp(self) -> None:
        self.dp = tempfile.TemporaryDirectory()
        self.dp_invalid = tempfile.TemporaryDirectory()
        os.chmod(self.dp_invalid.name, mode=0o777)

    def test_makedirs_valid(self) -> None:
        target_dir = os.path.join(self.dp.name, "d1")
        ms_makedirs(target_dir)
        assert os.path.exists(target_dir)

        target_dir = os.path.join(self.dp.name, "d2/d3/d4")
        ms_makedirs(target_dir)
        assert os.path.exists(target_dir)

    @patch('components.utils.log.logger.warning')
    def test_makedirs_invalid(self, mock_logger) -> None:
        target_dir = os.path.join(self.dp_invalid.name, "d1")
        if not PathChecker().is_safe_parent_dir().check(os.path.join(self.dp_invalid.name, "d1")):
            ms_makedirs(target_dir)
            mock_logger.assert_called_once_with(f"Output parent directory path {target_dir} is not safe.")

    def tearDown(self) -> None:
        self.dp.cleanup()


def test_is_belong_to_user_or_group_given_path_when_valid_then_pass():
    mock_stat = Mock(st_uid=1000, st_gid=1001)
    with patch('os.getuid', return_value=1000), \
         patch('os.getgroups', return_value=[1001]):
        result = is_belong_to_user_or_group(mock_stat)
        assert result is True


def test_is_belong_to_user_or_group_given_path_when_invalid_then_fail():
    mock_stat = Mock(st_uid=2000, st_gid=2001)
    with patch('os.getuid', return_value=1000), \
         patch('os.getgroups', return_value=[1001]):
        result = is_belong_to_user_or_group(mock_stat)
        assert result is False


def test_is_endswith_extensions_given_list_of_extensions_when_matches_then_true():
    path = "file.txt"
    extensions = ["txt", "md"]
    result = is_endswith_extensions(path, extensions)
    assert result is True


def test_is_endswith_extensions_given_single_extension_when_not_match_then_false():
    path = "file.txt"
    extensions = "md"
    result = is_endswith_extensions(path, extensions)
    assert result is False


# Additional tests would follow the same pattern for each function in security_check.py

def test_get_valid_path_given_empty_path_when_called_then_raises_value_error():
    with pytest.raises(ValueError):
        get_valid_path("")


def test_get_valid_path_given_path_with_special_char_when_called_then_raises_value_error():
    with pytest.raises(ValueError):
        get_valid_path("/path/with*special?char")


def test_get_valid_path_given_soft_link_when_called_then_raises_value_error():
    with patch('os.path.islink', return_value=True), \
         patch('os.path.abspath', return_value='/path/to/symlink'):
        with pytest.raises(ValueError, match="cannot be soft link"):
            get_valid_path("/path/to/symlink")


def test_get_valid_path_given_long_filename_when_called_then_raises_value_error():
    long_filename = "a" * 257
    with pytest.raises(ValueError):
        get_valid_path(f"/path/{long_filename}")


def test_get_valid_path_given_long_path_when_called_then_raises_value_error():
    long_path = "/".join(["a" * 1000] * 5)
    with pytest.raises(ValueError):
        get_valid_path(long_path)


# The following tests are simplified examples. More tests should be added to reach the required coverage.

def test_check_write_directory_given_nonexistent_directory_when_called_then_raises_value_error():
    with pytest.raises(ValueError):
        check_write_directory("/nonexistent/directory")


def test_type_to_str_given_tuple_of_types_when_called_then_returns_string():
    types = (int, float)
    result = type_to_str(types)
    assert result == "int or float"


def test_check_type_given_valid_type_when_called_then_passes():
    try:
        check_type(42, int)
        check_type([1, 2, 3], list)
    except TypeError:
        pytest.fail("check_type() raised TypeError unexpectedly!")


def test_check_type_given_invalid_type_when_called_then_raises_type_error():
    with pytest.raises(TypeError):
        check_type("string", int)


def test_check_number_given_within_range_when_called_then_passes():
    try:
        check_number(42, min_value=0, max_value=100)
    except ValueError:
        pytest.fail("check_number() raised ValueError unexpectedly!")


def test_check_number_given_outside_range_when_called_then_raises_value_error():
    with pytest.raises(ValueError):
        check_number(101, min_value=0, max_value=100)


def test_check_int_given_float_when_called_then_raises_type_error():
    with pytest.raises(TypeError):
        check_int(42.5)


def test_check_element_type_given_invalid_element_when_called_then_raises_value_error():
    with pytest.raises(ValueError):
        check_element_type([1, "two", 3], element_type=int)


def test_check_character_given_invalid_string_when_called_then_raises_value_error():
    with pytest.raises(ValueError):
        check_character("invalid*string?")


def test_check_dict_character_given_invalid_key_when_called_then_raises_value_error():
    with pytest.raises(ValueError):
        check_dict_character({"invalid*key": "value"})


def test_find_existing_path_given_nonexistent_path_when_depth_exceeded_then_raises_recursion_error():
    with pytest.raises(RecursionError):
        find_existing_path("/nonexistent/path", depth=0)


def test_is_enough_disk_space_left_given_insufficient_space_when_called_then_returns_false():
    dump_path = "/path/to/check"
    with patch('shutil.disk_usage', return_value=Mock(free=MIN_DUMP_DISK_SPACE - 1)):
        result = is_enough_disk_space_left(dump_path)
        assert result is False

def test_check_dict_character_given_long_key_when_called_then_raises_value_error():
    long_key = "k" * 513
    with pytest.raises(ValueError, match=r"Length of dict key exceeds limitation 512\."):
        check_dict_character({long_key: "value"})


def test_check_dict_character_given_deep_nested_dict_when_called_then_raises_value_error():
    # Create a deeply nested dictionary to trigger the recursion depth error.
    deep_dict = {}
    current_level = deep_dict
    for _ in range(101):  # This should exceed max_depth=100
        next_level = {"next": {}}
        current_level["next"] = next_level
        current_level = next_level

    with pytest.raises(ValueError, match=r"Recursion depth of dict exceeds limitation\."):
        check_dict_character(deep_dict)


def test_check_dict_character_given_non_dict_values_when_called_then_passes():
    try:
        check_dict_character({
            "string": "a string",
            "integer": 42,
            "float": 3.14,
            "list": ["item"],
            "tuple": ("item",),
        })
    except ValueError as e:
        pytest.fail(f"check_dict_character() raised ValueError unexpectedly: {e}")


def test_check_dict_character_given_invalid_characters_in_keys_when_called_then_raises_value_error():
    invalid_key = "invalid*key"
    with pytest.raises(ValueError, match=r"dict key contains invalid characters\."):
        check_dict_character({invalid_key: "value"})


def test_get_valid_read_path_given_nonexistent_file_when_called_then_raises_value_error():
    with patch('components.utils.security_check.get_valid_path', return_value="/nonexistent/file"), \
         patch('os.path.isfile', return_value=False):
        with pytest.raises(ValueError, match=r"The path .+ doesn't exist or not a file\."):
            get_valid_read_path("/nonexistent/file")


def test_get_valid_read_path_given_nonexistent_directory_when_called_then_raises_value_error():
    with patch('components.utils.security_check.get_valid_path', return_value="/nonexistent/dir"), \
         patch('os.path.isdir', return_value=False):
        with pytest.raises(ValueError, match=r"The path .+ doesn't exist or not a directory\."):
            get_valid_read_path("/nonexistent/dir", is_dir=True)


def test_get_valid_read_path_given_directory_not_belonging_to_user_or_group_when_called_then_raises_value_error():
    mock_stat = Mock(st_uid=os.getuid() + 1, st_gid=9999)  # 不属于当前用户的UID和GID
    with patch('components.utils.security_check.get_valid_path', return_value="/path/to/dir"), \
         patch('os.path.isdir', return_value=True), \
         patch('os.stat', return_value=mock_stat), \
         patch('sys.platform', 'linux'):
        with pytest.raises(ValueError, match=r"The file .+ doesn't belong to the current user or group\."):
            get_valid_read_path("/path/to/dir", check_user_stat=True, is_dir=True)


def test_get_valid_read_path_given_file_with_incorrect_permissions_when_called_then_raises_value_error():
    mock_stat = Mock(st_uid=os.getuid(), st_mode=stat.S_IRUSR | READ_FILE_NOT_PERMITTED_STAT)
    with patch('components.utils.security_check.get_valid_path', return_value="/path/to/file"), \
         patch('os.path.isfile', return_value=True), \
         patch('os.stat', side_effect=[mock_stat, mock_stat]), \
         patch('sys.platform', 'linux'):
        with pytest.raises(ValueError, match=r"The file .+ is group writable, or is others writable\."):
            get_valid_read_path("/path/to/file", check_user_stat=True)


def test_get_valid_read_path_given_file_without_read_permission_when_called_then_raises_value_error():
    mock_stat = Mock(st_uid=os.getuid(), st_mode=0o200)  # 没有读取权限
    with patch('components.utils.security_check.get_valid_path', return_value="/path/to/file"), \
         patch('os.path.isfile', return_value=True), \
         patch('os.stat', return_value=mock_stat), \
         patch('os.access', return_value=False):
        with pytest.raises(ValueError, match=r"Current user doesn't have read permission to the file .+\." ):
            get_valid_read_path("/path/to/file")


def test_get_valid_read_path_given_large_file_when_called_then_raises_value_error():
    mock_stat = Mock(st_uid=os.getuid(), st_gid=os.getgid(), st_mode=0o600, st_size=MAX_READ_FILE_SIZE_4G + 1)
    with patch('components.utils.security_check.get_valid_path', return_value="/path/to/largefile"), \
         patch('os.path.isfile', return_value=True), \
         patch('os.stat', return_value=mock_stat), \
         patch('os.access', return_value=True):
        with pytest.raises(ValueError, match=r"The file .+ exceeds size limitation of \d+\." ):
            get_valid_read_path("/path/to/largefile", size_max=MAX_READ_FILE_SIZE_4G)


def test_get_valid_read_path_given_existing_file_with_correct_permissions_when_called_then_passes():
    mock_stat = Mock(st_uid=os.getuid(), st_gid=os.getgid(), st_mode=0o600, st_size=1024)  # 确保 st_size 是整数
    with patch('components.utils.security_check.get_valid_path', return_value="/path/to/existing/file"), \
         patch('os.path.isfile', return_value=True), \
         patch('os.stat', return_value=mock_stat), \
         patch('os.access', return_value=True):
        try:
            get_valid_read_path("/path/to/existing/file")
        except ValueError as e:
            pytest.fail(f"get_valid_read_path() raised an unexpected ValueError: {e}")


def test_get_valid_read_path_given_existing_directory_with_correct_permissions_when_called_then_passes():
    mock_stat = Mock(st_uid=os.getuid(), st_mode=0o700)
    with patch('components.utils.security_check.get_valid_path', return_value="/path/to/existing/dir"), \
         patch('os.path.isdir', return_value=True), \
         patch('os.stat', return_value=mock_stat), \
         patch('os.access', return_value=True):
        try:
            get_valid_read_path("/path/to/existing/dir", is_dir=True)
        except ValueError as e:
            pytest.fail(f"get_valid_read_path() raised an unexpected ValueError: {e}")


class TestCheckPositiveInteger(unittest.TestCase):
    def test_valid_values(self):
        test_cases = [
            ("0", 0),     
            ("1", 1),     
            ("1000000", 1e6), 
            (123, 123),     
            ("  456  ", 456) 
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input=input_val, expected=expected):
                result = check_positive_integer(input_val)
                self.assertEqual(result, expected)
        
    def test_invalid_ranges(self):
        test_cases = [
            "1000001",
            "-5",       
        ]
        
        for value in test_cases:
            with self.subTest(value=value), \
                 self.assertRaises(ValueError) as cm:
                check_positive_integer(value)
                
            self.assertIn(f"{value} is an invalid positive int value", str(cm.exception))

    def test_invalid_types(self):
        test_cases = [
            "abc",     
            "123abc",  
            None,      
            [],         
            {"key": 1}  
        ]
        
        for value in test_cases:
            with self.subTest(value=value), \
                 self.assertRaises((ValueError, TypeError)):
                check_positive_integer(value)

    def test_extremely_large_values(self):
        with self.assertRaises(ValueError):
            check_positive_integer(1e100)
            
        with self.assertRaises(ValueError):
            check_positive_integer("12345678901234567890")


def test_check_output_path_legality_when_path_is_valid_then_pass():
    legal_path = "/path/to/legal/output"
    assert check_output_path_legality(legal_path) == legal_path


def test_check_output_path_illegal_when_path_is_invalid_then_raise_error():
    with pytest.raises(argparse.ArgumentTypeError):
        check_output_path_legality("/ille@gal/path/output")


def test_check_input_opsummary_legality_given_valid_path_when_all_checks_pass_then_return_path():
    with patch.object(FileStat, "is_basically_legal") as mock_legal, \
         patch.object(FileStat, "is_legal_file_type") as mock_file_type:
        mock_legal.return_value = True
        mock_file_type.return_value = True
        file_path = 'valid.csv'
        result = check_input_opsummary_legality(file_path)
        assert result == file_path


def test_check_input_opsummary_legality_given_non_csv_when_checked_then_raise_error():
    with patch.object(FileStat, "is_basically_legal") as mock_legal, \
         patch.object(FileStat, "is_legal_file_type") as mock_file_type, \
         pytest.raises(argparse.ArgumentTypeError, match="Op_summary file muse be 'csv'") as exc_info:
        mock_legal.return_value = True
        mock_file_type.return_value = False
        check_input_opsummary_legality('file.txt')


@patch("components.utils.file_open_check.FileStat")
def test_valid_ops_map_file_given_valid_file_when_all_checks_pass_then_return_path(mock_filestat):
    with patch.object(FileStat, "is_basically_legal") as mock_legal, \
         patch.object(FileStat, "is_legal_file_type") as mock_file_type, \
         patch.object(FileStat, "is_dir") as mock_is_dir, \
         pytest.raises(argparse.ArgumentTypeError, match="Not found ge_proto_xx_Build.txt in ./tmp") as exc_info:
        mock_legal.return_value = True
        mock_file_type.return_value = True
        mock_is_dir.return_value = False
        valid_ops_map_file('./tmp')