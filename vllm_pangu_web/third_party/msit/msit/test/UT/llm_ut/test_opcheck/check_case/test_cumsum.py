import sys
from unittest.mock import patch, MagicMock

import pytest
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_cumsum_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case import OpcheckCumsumOperation
    OpcheckCumsumOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckCumsumOperation": OpcheckCumsumOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_op_param_in_tensors_when_valid_input_then_correct_result(import_cumsum_module):
    test_cases = [
        ({'axes': [0]}, [torch.tensor([1.0, 2.0, 3.0])], [torch.tensor([1.0, 3.0, 6.0])]),
        ({'axes': [1]}, [torch.tensor([[1.0, 2.0], [3.0, 4.0]])], [torch.tensor([[1.0, 3.0], [3.0, 7.0]])]),
        ({'axes': [0]}, [torch.tensor([[1.0, 2.0], [3.0, 4.0]])], [torch.tensor([[1.0, 2.0], [4.0, 6.0]])]),
    ]
    for op_param, in_tensors, expected_result in test_cases:
        OpcheckCumsumOperation = import_cumsum_module['OpcheckCumsumOperation']
        op = OpcheckCumsumOperation()
        op.op_param = op_param

        result = op.golden_calc(in_tensors)

        assert torch.equal(result[0], expected_result[0])


def test_golden_calc_given_op_param_in_tensors_when_invalid_input_then_raise_error(import_cumsum_module):
    test_cases = [
        ({'axes': [2]}, [torch.tensor([[1.0, 2.0], [3.0, 4.0]])], IndexError),
    ]
    for op_param, in_tensors, expected_error in test_cases:
        OpcheckCumsumOperation = import_cumsum_module['OpcheckCumsumOperation']
        op = OpcheckCumsumOperation()
        op.op_param = op_param

        if expected_error:
            with pytest.raises(expected_error):
                op.golden_calc(in_tensors)
        else:
            result = op.golden_calc(in_tensors)
            assert torch.equal(result[0], in_tensors[0])


def test_test_given_op_param_when_valid_input_then_execute_successfully(import_cumsum_module):
    test_cases = [
        ({'axes': [0]}, True, True),
        ({'axes': [1]}, True, True),
        ({'axes': [0]}, False, False),
        ({'axes': [1]}, False, False),
        ({}, False, False),
    ]
    for op_param, validate_param_return, expected_execute_call in test_cases:
        OpcheckCumsumOperation = import_cumsum_module['OpcheckCumsumOperation']
        op = OpcheckCumsumOperation()
        op.op_param = op_param

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test()

        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()
