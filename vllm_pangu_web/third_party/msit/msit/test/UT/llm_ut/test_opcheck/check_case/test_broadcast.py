import sys
from unittest.mock import patch, MagicMock

import torch
import pytest

from mock_operation_test import MockOperationTest
    
    
@pytest.fixture(scope="function")
def import_broadcast_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case import OpcheckBroadcastOperation
    OpcheckBroadcastOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckBroadcastOperation": OpcheckBroadcastOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_op_param_in_tensors_when_valid_input_then_correct_result(import_broadcast_module):
    test_cases = [
        ({'rankRoot': 0}, [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([1.0, 2.0])]),
        ({'rankRoot': 1}, [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([3.0, 4.0])]),
        ({'rankRoot': 0}, [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])], [torch.tensor([1.0])]),
        ({'rankRoot': 2}, [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])], [torch.tensor([3.0])])
    ]
    for op_param, in_tensors, expected_result in test_cases:
        OpcheckBroadcastOperation = import_broadcast_module["OpcheckBroadcastOperation"]
        op = OpcheckBroadcastOperation()
        op.op_param = op_param

        result = op.golden_calc(in_tensors)

        assert torch.equal(result[0], expected_result[0])


def test_test_broadcast_given_op_param_when_valid_input_then_execute_successfully(import_broadcast_module):
    test_cases = [
         ({'rankRoot': 0}, True, True),
        ({'rankRoot': 1}, True, True),
        ({}, False, False),
        ({'rankRoot': 0}, False, False)
    ]
    for op_param, validate_param_return, expected_execute_call in test_cases:
        OpcheckBroadcastOperation = import_broadcast_module["OpcheckBroadcastOperation"]
        op = OpcheckBroadcastOperation()
        op.op_param = op_param

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test_broadcast()

        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()
