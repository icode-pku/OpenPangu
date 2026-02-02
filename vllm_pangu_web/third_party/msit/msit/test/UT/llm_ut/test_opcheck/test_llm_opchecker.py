import sys
import unittest 
import os
import stat
from unittest.mock import patch, MagicMock
import pytest
from pathlib import Path


@pytest.fixture(scope="function")
def import_opchecker_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.opchecker import _is_atb_only_saved_before, OpChecker
    from msit_llm.common.log import logger
    functions = {
        "_is_atb_only_saved_before": _is_atb_only_saved_before,
        "OpChecker": OpChecker,
        "logger": logger
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


@pytest.fixture()
def mock_logger(import_opchecker_module):
    logger = import_opchecker_module['logger']
    with patch.object(logger, 'error') as mock_error:
        yield mock_error


@pytest.mark.parametrize(
    "dump_scenario, expected_result",
    [
        (['before'], True),
        (['after'], False),
        (['before', 'after'], False)
    ]
)
def test_is_atb_only_saved_before(mock_logger, tmp_path, dump_scenario, expected_result, import_opchecker_module):
    _is_atb_only_saved_before = import_opchecker_module['_is_atb_only_saved_before']
    input_path = tmp_path / 'token_id'
    input_path.mkdir()
    layer_dir = input_path / '0_Decoder_layer'
    layer_dir.mkdir()
    for folder in dump_scenario:
        (layer_dir / folder).mkdir()
    res = _is_atb_only_saved_before(str(input_path))

    assert res is expected_result
    mock_logger.assert_not_called()


def test_is_atb_only_saved_before_false_no_folders(mock_logger, tmp_path, import_opchecker_module):
    _is_atb_only_saved_before = import_opchecker_module['_is_atb_only_saved_before']
    input_path = tmp_path / 'token_id'
    input_path.mkdir()
    res = _is_atb_only_saved_before(str(input_path))

    assert res is False
    mock_logger.assert_called_once()


def third_party_init_env_path(tmp_path_factory):
    # 预置so加载环境变量
    dir_path = tmp_path_factory.mktemp("test_ait_opcheck_lib_path")
    file_path = os.path.join(dir_path, "libopchecker.so")
    os.environ['AIT_OPCHECK_LIB_PATH'] = file_path
    return file_path


def test_third_party_init_load_not_exist_file(tmp_path_factory, import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    third_party_init_env_path(tmp_path_factory)

    res = OpChecker().third_party_init()

    del os.environ['AIT_OPCHECK_LIB_PATH']
    assert res is False


def test_third_party_init_load_other_writable_file(tmp_path_factory, import_opchecker_module):
    file_path = third_party_init_env_path(tmp_path_factory)
    OpChecker = import_opchecker_module["OpChecker"]
    # 创建他人可写文件
    file_permissions = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    with open(file_path, 'w') as f:
        os.chmod(file_path, file_permissions)

    res = OpChecker().third_party_init()

    del os.environ['AIT_OPCHECK_LIB_PATH']
    if os.path.exists(file_path):
        os.remove(file_path)
    assert res is False


@patch('os.path.exists', return_value=True)
def test_get_base_path_valid(mock_exists, import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    op_checker = OpChecker()
    result_path, pid = op_checker.get_base_path('/msit_dump/tensors/operation_12345/path')
    assert result_path == '/msit_dump/tensors/operation_12345/path'
    assert pid == '12345'


@patch('os.path.exists', return_value=False)
def test_get_base_path_invalid(mock_exists, import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    op_checker = OpChecker()
    result_path, pid = op_checker.get_base_path('/invalid/path')
    assert result_path is None
    assert pid is None


@patch('os.path.exists', return_value=False)
@patch('msit_llm.common.log.logger.error')
def test_check_input_legality_invalid(mock_logger_error, mock_exists, import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    op_checker = OpChecker()
    input_path, base_path, pid, ret = op_checker.check_input_legality('/invalid/input/path')
    assert ret is False
    assert input_path == '/invalid/input/path'
    assert base_path is None
    assert pid is None
    mock_logger_error.assert_called_once()


def test_check_id_range_match(import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    op_checker = OpChecker()
    op_checker.operation_ids = '123,456'
    op_checker.check_ids_string = ['123', '456']
    assert op_checker.check_id_range('123_0') is True


def test_check_id_range_no_match(import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    op_checker = OpChecker()
    op_checker.operation_ids = '123,456'
    op_checker.check_ids_string = ['123', '456']
    assert op_checker.check_id_range('789_0') is False


def test_check_name_match(import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    op_checker = OpChecker()
    op_checker.operation_name = 'operation,test'
    op_checker.check_patterns = ['operation', 'test']
    assert op_checker.check_name('Operation') is True


def test_check_name_no_match(import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    op_checker = OpChecker()
    op_checker.operation_name = 'operation,test'
    op_checker.check_patterns = ['operation', 'test']
    assert op_checker.check_name('invalid') is False


def test_is_exec_node_match(import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    op_checker = OpChecker()
    op_checker.operation_ids = '123'
    op_checker.check_ids_string = ['123']
    op_checker.operation_name = 'operation'
    op_checker.check_patterns = ['operation']
    case_info = {'op_id': '123_0', 'op_name': 'Operation'}
    assert op_checker.is_exec_node(case_info) is True


def test_is_exec_node_no_match(import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    op_checker = OpChecker()
    op_checker.operation_ids = '123'
    op_checker.check_ids_string = ['123']
    op_checker.operation_name = 'operation'
    op_checker.check_patterns = ['operation']
    case_info = {'op_id': '456_0', 'op_name': 'Invalid'}
    assert op_checker.is_exec_node(case_info) is False


def test_check_id_range(import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    op_checker = OpChecker()
    op_checker.operation_ids = '0'
    op_checker.check_ids_string = ['0']

    op_id = '0_1_2'
    assert op_checker.check_id_range(op_id) is True

    op_id = '1_2_3'
    assert op_checker.check_id_range(op_id) is False

    op_checker.operation_ids = ''
    assert op_checker.check_id_range(None) is False


def test_check_name(import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    op_checker = OpChecker()
    op_checker.operation_name = 'attention'
    op_checker.check_patterns = ['attention']

    op_name = 'SelfAttentionOperation'
    assert op_checker.check_name(op_name) is True

    op_name = 'ConvOperation'
    assert op_checker.check_name(op_name) is False

    op_checker.operation_name = None
    assert op_checker.check_name(None) is False


def test_is_exec_node(import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    op_checker = OpChecker()
    case_info = {'op_id': '0_1_2', 'op_name': 'op_name'}

    # Both operation_ids and operation_name are unspecified
    assert op_checker.is_exec_node(case_info) is True

    op_checker.operation_ids = '0'
    op_checker.check_ids_string = ['0']
    op_checker.operation_name = 'op_name'
    op_checker.check_patterns = ['name']

    assert op_checker.is_exec_node(case_info) is True


def test_add_case_to_cases(import_opchecker_module):
    OpChecker = import_opchecker_module["OpChecker"]
    op_checker = OpChecker()
    case_info = {'op_name': 'KvCacheOperation', 'op_id': '0_1_2'}
    op_checker.add_case_to_cases(case_info)
    assert '0_1_2' in op_checker.cases_info
    assert op_checker.cases_info['0_1_2']['inplace_idx'] == [2]

    case_info = {'op_name': 'ReshapeAndCacheOperation', 'op_id': '0_1_2'}
    op_checker.add_case_to_cases(case_info)
    assert op_checker.cases_info['0_1_2']['inplace_idx'] == [2, 3]

    case_info = {'op_name': 'SelfAttentionOperation', 'op_id': '0_1_2'}
    op_checker.add_case_to_cases(case_info)
    assert '0_1_2' in op_checker.cases_info

    case_info = {'op_name': 'UnknownOperation', 'op_id': '0_1_2'}
    op_checker.add_case_to_cases(case_info)
    assert '0_1_2' in op_checker.cases_info
