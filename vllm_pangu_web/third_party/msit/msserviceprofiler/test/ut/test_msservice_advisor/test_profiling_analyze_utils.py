# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from collections import namedtuple
import tempfile
import shutil
import pytest

from msserviceprofiler.msservice_advisor.profiling_analyze import utils


# Helper functions for file/directory testing
def create_temp_file(size_bytes, content=None):
    """Create a temporary file with specified size"""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    if content:
        temp_file.write(content.encode())
    else:
        temp_file.write(b"0" * size_bytes)
    temp_file.close()
    return temp_file.name


# Test str_ignore_case
def test_str_ignore_case_given_normal_string_when_lowercase_then_processed():
    assert utils.str_ignore_case("Hello") == "hello"
    assert utils.str_ignore_case("HELLO_WORLD") == "helloworld"
    assert utils.str_ignore_case("Mix-Ed_Case") == "mixedcase"


def test_str_ignore_case_given_empty_string_when_processed_then_empty():
    assert utils.str_ignore_case("") == ""


def test_str_ignore_case_given_non_string_when_processed_then_error():
    with pytest.raises(AttributeError):
        utils.str_ignore_case(None)
    with pytest.raises(AttributeError):
        utils.str_ignore_case(123)


def test_str_ignore_case_given_unicode_string_when_processed_then_lowercase():
    assert utils.str_ignore_case("Héllö_Wörld") == "héllöwörld"


# Test set_log_level
def test_set_log_level_given_valid_level_when_set_then_level_changed():
    original_level = utils.logger.level
    utils.set_log_level("debug")
    assert utils.logger.level == logging.DEBUG
    utils.set_log_level("INFO")
    assert utils.logger.level == logging.INFO


def test_set_log_level_given_case_variations_when_set_then_works():
    utils.set_log_level("Debug")
    assert utils.logger.level == logging.DEBUG
    utils.set_log_level("ERROR")
    assert utils.logger.level == logging.ERROR


# Test set_logger
def test_set_logger_given_new_logger_when_configured_then_has_handler():
    test_logger = utils.logging.getLogger("test_logger")
    utils.set_logger(test_logger)
    assert len(test_logger.handlers) == 1
    assert test_logger.propagate is False


def test_set_logger_given_existing_handler_when_configured_then_no_duplicate():
    test_logger = logging.getLogger("test_logger2")
    test_logger.addHandler(logging.StreamHandler())
    utils.set_logger(test_logger)
    assert len(test_logger.handlers) == 1  # Original handler replaced?


# Test get_directory_size
def test_get_directory_size_given_empty_dir_when_calculated_then_zero():
    temp_dir = tempfile.mkdtemp()
    try:
        size = utils.get_directory_size(temp_dir)
        assert size == 0.0
    finally:
        os.rmdir(temp_dir)


def test_get_directory_size_given_dir_with_files_when_calculated_then_correct():
    """Create a temporary directory with specified number of files. 3 files of 1KB each"""
    temp_dir = tempfile.mkdtemp()
    for i in range(3):
        with open(os.path.join(temp_dir, f"file_{i}.txt"), "w") as f:
            f.write("0" * 1024)

    try:
        size = utils.get_directory_size(temp_dir)
        expected = (3 * 1024) / utils.BYTES_TO_GB
        assert pytest.approx(size) == expected
    finally:
        shutil.rmtree(temp_dir)


def test_get_directory_size_given_symlinks_when_calculated_then_ignored():
    temp_dir = tempfile.mkdtemp()
    temp_file = create_temp_file(1024)
    try:
        os.symlink(temp_file, os.path.join(temp_dir, "symlink"))
        size = utils.get_directory_size(temp_dir)
        assert size == 0.0  # Symlinks should be ignored
    finally:
        os.unlink(temp_file)
        shutil.rmtree(temp_dir)
