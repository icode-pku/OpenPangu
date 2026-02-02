import sys
from unittest.mock import patch, MagicMock

import pytest
import torch
    
from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_concat_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case import OpcheckConcatOperation
    OpcheckConcatOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckConcatOperation": OpcheckConcatOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_op_param_in_tensors_when_valid_input_then_correct_shape(import_concat_module):
    test_cases = [
        ({'concatDim': 0}, [torch.randn(2, 3), torch.randn(3, 3)], (5, 3)),
        ({'concatDim': 1}, [torch.randn(2, 3), torch.randn(2, 4)], (2, 7)),
        ({'concatDim': -1}, [torch.randn(2, 3), torch.randn(2, 4)], (2, 7)),
        ({'concatDim': -2}, [torch.randn(2, 3), torch.randn(3, 3)], (5, 3))
    ]
    for op_param, in_tensors, expected_shape in test_cases:
        OpcheckConcatOperation = import_concat_module['OpcheckConcatOperation']
        op = OpcheckConcatOperation()
        op.op_param = op_param

        result = op.golden_calc(in_tensors)

        assert result[0].shape == expected_shape


def test_golden_calc_given_op_param_in_tensors_when_invalid_input_then_raise_error(import_concat_module):
    test_cases = [
        ({'concatDim': 0}, [torch.randn(2, 3), torch.randn(2, 4)], RuntimeError),
        ({'concatDim': 1}, [torch.randn(2, 3), torch.randn(3, 3)], RuntimeError),
    ]
    for op_param, in_tensors, expected_error in test_cases:
        OpcheckConcatOperation = import_concat_module['OpcheckConcatOperation']
        op = OpcheckConcatOperation()
        op.op_param = op_param

        with pytest.raises(expected_error):
            op.golden_calc(in_tensors)


def test_test_given_op_param_when_valid_input_then_execute_successfully(import_concat_module):
    test_cases = [
        ({'concatDim': 0}, True, True),
        ({'concatDim': 1}, True, True),
        ({'concatDim': -1}, True, True),
        ({'concatDim': -2}, True, True),
        ({}, False, False),
        ({'concatDim': 0}, False, False)
    ]
    for op_param, validate_param_return, expected_execute_call in test_cases:
        OpcheckConcatOperation = import_concat_module['OpcheckConcatOperation']
        op = OpcheckConcatOperation()
        op.op_param = op_param

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test()

        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()
