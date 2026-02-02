import sys
from unittest.mock import patch, MagicMock

import pytest
import torch

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_fastsoftmax_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case import OpcheckFastSoftMaxOperation
    OpcheckFastSoftMaxOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckFastSoftMaxOperation": OpcheckFastSoftMaxOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_op_param_in_tensors_when_invalid_input_then_raise_error(import_fastsoftmax_module):
    test_cases = [
        {
            "op_param": {'qSeqLen': [2, 2], 'headNum': 1},
            "in_tensors": [torch.tensor([[1.0, 2.0]])],
            "expected_error": RuntimeError
        },
        {
            "op_param": {'qSeqLen': [3, 3], 'headNum': 1},
            "in_tensors": [torch.tensor([[1.0, 2.0, 3.0]])],
            "expected_error": RuntimeError
        },
        {
            "op_param": {'qSeqLen': [2, 2], 'headNum': 2},
            "in_tensors": [torch.tensor([[1.0, 2.0], [3.0, 4.0]])],
            "expected_error": RuntimeError
        }
    ]

    for case in test_cases:
        OpcheckFastSoftMaxOperation = import_fastsoftmax_module['OpcheckFastSoftMaxOperation']
        op = OpcheckFastSoftMaxOperation()
        op.op_param = case["op_param"]

        if case["expected_error"]:
            with pytest.raises(case["expected_error"]):
                op.golden_calc(case["in_tensors"])
        else:
            result = op.golden_calc(case["in_tensors"])
            assert torch.allclose(result[0], case["in_tensors"][0], atol=1e-4), \
                "Unexpected result for valid input"


def test_test_fastsoftmax_given_op_param_when_valid_input_then_execute_successfully(import_fastsoftmax_module):
    test_cases = [
        ({'qSeqLen': [2, 2], 'headNum': 1}, True, True),
        ({'qSeqLen': [3, 3], 'headNum': 1}, True, True),
        ({'qSeqLen': [2, 2], 'headNum': 2}, True, True),
        ({'qSeqLen': [2, 2]}, False, False),
        ({'headNum': 1}, False, False),
        ({}, False, False),
    ]
    
    for op_param, validate_param_return, expected_execute_call in test_cases:
        OpcheckFastSoftMaxOperation = import_fastsoftmax_module['OpcheckFastSoftMaxOperation']
        op = OpcheckFastSoftMaxOperation()
        op.op_param = op_param

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test_fastsoftmax()

        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()