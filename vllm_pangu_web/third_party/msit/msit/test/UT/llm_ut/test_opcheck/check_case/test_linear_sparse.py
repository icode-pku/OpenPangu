import sys
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import pytest
import torch

from mock_operation_test import MockOperationTest

@pytest.fixture(scope="function")
def import_linear_sparse_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case import OpcheckLinearSparseOperation
    from msit_llm.common.log import logger
    OpcheckLinearSparseOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckLinearSparseOperation": OpcheckLinearSparseOperation,
        "logger": logger
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


# Mock the logger to capture log messages
@pytest.fixture
def mock_logger(import_linear_sparse_module):
    logger = import_linear_sparse_module["logger"]
    with patch.object(logger, 'info') as mock_info, patch.object(logger, 'debug') as mock_debug:
        yield mock_info, mock_debug



@dataclass
class InvalidTestParams:
    transposeA: bool
    transposeB: bool
    in_tensors: list
    expected_error: type


@pytest.mark.parametrize("params", [
    InvalidTestParams(False, True, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
    InvalidTestParams(True, False, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
    InvalidTestParams(False, False, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
])
def test_golden_calc_when_invalid_input(mock_logger, params, import_linear_sparse_module):
    OpcheckLinearSparseOperation = import_linear_sparse_module["OpcheckLinearSparseOperation"]
    op = OpcheckLinearSparseOperation()
    op.op_param = {"transposeA": params.transposeA, "transposeB": params.transposeB}
    with pytest.raises(params.expected_error):
        op.golden_calc(params.in_tensors)
