import os
import sys

import pytest
import tempfile
from unittest.mock import patch, mock_open, MagicMock
from clang import cindex

from msit_llm.common.log import logger
from msit_llm.transform.float_atb_to_quant_atb.transform_quant import TransformQuant, transform_quant


# Mocking the necessary modules and functions
@pytest.fixture
def mock_clang():
    with patch('clang.cindex') as mock_cindex:
        yield mock_cindex

@pytest.fixture
def mock_ms_open():
    with patch('msit_llm.transform.float_atb_to_quant_atb.transform_quant.ms_open', new_callable=mock_open, read_data="test_data") as mock_file:
        yield mock_file

@pytest.fixture
def mock_load_file_to_read_common_check():
    with patch('msit_llm.transform.float_atb_to_quant_atb.transform_quant.load_file_to_read_common_check') as mock_load:
        yield mock_load

@pytest.fixture
def mock_check_libclang_so():
    with patch('msit_llm.transform.float_atb_to_quant_atb.transform_quant.check_libclang_so') as mock_check:
        yield mock_check

@pytest.fixture
def mock_glob():
    with patch('glob.glob') as mock_glob:
        yield mock_glob
        
@pytest.fixture
def mock_path_exists():
    with patch("os.path.exists") as mock_exists:
        yield mock_exists

@pytest.fixture
def mock_transform_quant():
    with patch("msit_llm.transform.float_atb_to_quant_atb.transform_quant.TransformQuant") as mock_transform:
        yield mock_transform

# Test cases
def test_transform_quant_given_valid_cpp_and_hpp_files_when_transforming_then_success(mock_clang, mock_ms_open, mock_load_file_to_read_common_check, mock_check_libclang_so, mock_glob):
    # 创建一个临时文件
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as temp_cpp:
        temp_cpp_path = temp_cpp.name
        temp_cpp.write(b"enum MyEnum { A, B, C };")

    mock_glob.return_value = [temp_cpp_path]
    mock_load_file_to_read_common_check.return_value = temp_cpp_path
    mock_clang.Index.create.return_value.parse.return_value.cursor.get_children.return_value = [MagicMock(kind=mock_clang.CursorKind.ENUM_DECL)]
    mock_clang.CursorKind.ENUM_DECL = mock_clang.CursorKind.ENUM_DECL
    mock_clang.CursorKind.FUNCTION_DECL = mock_clang.CursorKind.FUNCTION_DECL
    mock_clang.CursorKind.STRUCT_DECL = mock_clang.CursorKind.STRUCT_DECL
    mock_clang.CursorKind.VAR_DECL = mock_clang.CursorKind.VAR_DECL

    try:
        transform_quant(temp_cpp_path)
    finally:
        # 删除临时文件
        os.remove(temp_cpp_path)

def test_transform_quant_given_cpp_file_only_when_transforming_then_success(mock_clang, mock_ms_open, mock_load_file_to_read_common_check, mock_check_libclang_so, mock_glob):
    # 创建一个临时文件
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as temp_cpp:
        temp_cpp_path = temp_cpp.name
        temp_cpp.write(b"enum MyEnum { A, B, C };")

    mock_glob.return_value = [temp_cpp_path]
    mock_load_file_to_read_common_check.return_value = temp_cpp_path
    mock_clang.Index.create.return_value.parse.return_value.cursor.get_children.return_value = [MagicMock(kind=mock_clang.CursorKind.ENUM_DECL)]
    mock_clang.CursorKind.ENUM_DECL = mock_clang.CursorKind.ENUM_DECL
    mock_clang.CursorKind.FUNCTION_DECL = mock_clang.CursorKind.FUNCTION_DECL
    mock_clang.CursorKind.STRUCT_DECL = mock_clang.CursorKind.STRUCT_DECL
    mock_clang.CursorKind.VAR_DECL = mock_clang.CursorKind.VAR_DECL

    try:
        transform_quant(temp_cpp_path)
    finally:
        # 删除临时文件
        os.remove(temp_cpp_path)

def test_transform_quant_given_enable_sparse_when_transforming_then_success(mock_clang, mock_ms_open, mock_load_file_to_read_common_check, mock_check_libclang_so, mock_glob):
    # 创建一个临时文件
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as temp_cpp:
        temp_cpp_path = temp_cpp.name
        temp_cpp.write(b"enum MyEnum { A, B, C };")

    mock_glob.return_value = [temp_cpp_path]
    mock_load_file_to_read_common_check.return_value = temp_cpp_path
    mock_clang.Index.create.return_value.parse.return_value.cursor.get_children.return_value = [MagicMock(kind=mock_clang.CursorKind.ENUM_DECL)]
    mock_clang.CursorKind.ENUM_DECL = mock_clang.CursorKind.ENUM_DECL
    mock_clang.CursorKind.FUNCTION_DECL = mock_clang.CursorKind.FUNCTION_DECL
    mock_clang.CursorKind.STRUCT_DECL = mock_clang.CursorKind.STRUCT_DECL
    mock_clang.CursorKind.VAR_DECL = mock_clang.CursorKind.VAR_DECL

    try:
        transform_quant(temp_cpp_path, enable_sparse=True)
    finally:
        # 删除临时文件
        os.remove(temp_cpp_path)

def test_transform_quant_multi_cpp_file_success(
    mock_clang,
    mock_check_libclang_so,
    mock_glob,
    mock_path_exists,
    mock_transform_quant,
):
    dummy_instance = MagicMock(
        side_effect=[
            ("cpp_file_1.cpp", "hpp_file_1.h"),
            ("cpp_file_2.cpp", "hpp_file_2.h"),
            ("cpp_file_3.cpp", "hpp_file_3.h"),
        ]
    )
    # 当调用 TransformQuant(...) 时返回 dummy_instance
    mock_transform_quant.return_value = dummy_instance
    mock_glob.return_value = ["layer1.cpp", "layer2.cpp", "layer3.cpp"]
    mock_clang.Index.create.return_value.parse.return_value.cursor.get_children.return_value = [
        MagicMock(kind=mock_clang.CursorKind.ENUM_DECL)
    ]
    mock_clang.CursorKind.ENUM_DECL = mock_clang.CursorKind.ENUM_DECL
    mock_clang.CursorKind.FUNCTION_DECL = mock_clang.CursorKind.FUNCTION_DECL
    mock_clang.CursorKind.STRUCT_DECL = mock_clang.CursorKind.STRUCT_DECL
    mock_clang.CursorKind.VAR_DECL = mock_clang.CursorKind.VAR_DECL
    mock_path_exists.return_value = True

    transform_quant("models/qwen2_7b", enable_sparse=True)

def test_transform_quant_given_non_existent_cpp_file_when_transforming_then_failure(mock_check_libclang_so):
    with pytest.raises(FileNotFoundError):
        transform_quant("non_existent.cpp")