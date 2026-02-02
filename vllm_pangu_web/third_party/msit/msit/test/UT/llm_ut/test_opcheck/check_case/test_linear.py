import sys
from unittest.mock import patch, MagicMock

import pytest
import torch


from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_linear_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case import OpcheckLinearOperation
    OpcheckLinearOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckLinearOperation": OpcheckLinearOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


@pytest.mark.parametrize("op_param, in_tensors, expected_error", [
    ({'transposeA': False, 'transposeB': True, 'hasBias': True}, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
    ({'transposeA': True, 'transposeB': False, 'hasBias': True}, [torch.randn(2, 3), torch.randn(2, 3)], RuntimeError),
])
def test_golden_calc_when_invalid_input(op_param, in_tensors, expected_error, import_linear_module):
    OpcheckLinearOperation = import_linear_module['OpcheckLinearOperation']
    op = OpcheckLinearOperation()
    op.op_param = op_param

    with pytest.raises(expected_error):
        op.golden_calc(in_tensors)


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'transposeA': False, 'transposeB': True, 'hasBias': False}, True, True),
    ({'transposeA': True, 'transposeB': False, 'hasBias': False}, True, True),
    ({'transposeA': False, 'transposeB': True, 'hasBias': True}, True, True),
    ({'transposeA': True, 'transposeB': False, 'hasBias': True}, True, True),
    ({'transposeA': False, 'transposeB': True}, False, False),
    ({'transposeA': True, 'transposeB': False}, False, False),
    ({}, False, False),
])
def test_test_when_valid_input(op_param, validate_param_return, expected_execute_call, import_linear_module):
    OpcheckLinearOperation = import_linear_module['OpcheckLinearOperation']
    op = OpcheckLinearOperation()
    op.op_param = op_param

    with patch.object(op, 'validate_param', return_value=validate_param_return):
        with patch.object(op, 'execute') as mock_execute:
            op.test()

    if expected_execute_call:
        mock_execute.assert_called_once()
    else:
        mock_execute.assert_not_called()
