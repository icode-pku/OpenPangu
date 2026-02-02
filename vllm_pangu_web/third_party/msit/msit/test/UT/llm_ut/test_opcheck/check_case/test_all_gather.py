import sys
from unittest.mock import patch, MagicMock

import torch
import pytest

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_allgather_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case import OpcheckAllGatherOperation
    OpcheckAllGatherOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckAllGatherOperation": OpcheckAllGatherOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_in_tensors_when_valid_input_then_correct_shape(import_allgather_module):
    test_cases = [
        ([torch.randn(4, 8), torch.randn(4, 8)], (2, 4, 8)),
        ([torch.randn(4, 8), torch.randn(4, 9)], None),  # Mismatched shapes
        ([], None),  # Empty list
    ]
    for in_tensors, expected_shape in test_cases:
        OpcheckAllGatherOperation = import_allgather_module['OpcheckAllGatherOperation']
        op = OpcheckAllGatherOperation()

        def get_new_in_tensors():
            return in_tensors

        op.get_new_in_tensors = get_new_in_tensors

        if expected_shape is None:
            with pytest.raises(RuntimeError):
                op.golden_calc(None)
        else:
            result = op.golden_calc(None)

            assert result[0].shape == expected_shape


def test_test_all_gather_given_pid_op_param_when_valid_input_then_execute_successfully(import_allgather_module):

    test_cases = [
        (1, {'rank': 0, 'rankRoot': 0, 'rankSize': 2}, True, True),
        (None, {'rank': 0, 'rankRoot': 0, 'rankSize': 2}, True, False),
        (1, {'rank': 0, 'rankRoot': 0, 'rankSize': 2}, False, False),
        (1, {'rankRoot': 0, 'rankSize': 2}, False, False),
        (1, {'rank': 0, 'rankSize': 2}, False, False),
        (1, {'rank': 0, 'rankRoot': 0}, False, False)
    ]
    for pid, op_param, validate_param_return, expected_execute_call in test_cases:
        OpcheckAllGatherOperation = import_allgather_module['OpcheckAllGatherOperation']
        op = OpcheckAllGatherOperation()
        op.pid = pid
        op.op_param = op_param


        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test_all_gather()


        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()
