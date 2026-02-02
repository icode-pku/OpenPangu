import sys
from unittest.mock import patch, MagicMock

import torch
import pytest

from mock_operation_test import MockOperationTest


@pytest.fixture(scope="function")
def import_as_strided_module():
    backup = {}
    for mod in ['torch_npu']:
        if mod in sys.modules:
            backup[mod] = sys.modules[mod]
    mock_torch_npu = MagicMock()
    sys.modules['torch_npu'] = mock_torch_npu
    from msit_llm.opcheck.check_case import OpcheckAsStridedOperation
    OpcheckAsStridedOperation.__bases__ = (MockOperationTest,)
    functions = {
        "OpcheckAsStridedOperation": OpcheckAsStridedOperation
    }
    yield functions
    
    for mod, module_obj in backup.items():
        sys.modules[mod] = module_obj
    for mod in ['torch_npu']:
        if mod not in backup and mod in sys.modules:
            del sys.modules[mod]


def test_golden_calc_given_op_param_in_tensors_when_valid_input_then_correct_shape(import_as_strided_module):
    test_cases = [
        ({'size': [4, 8, 16], 'stride': [128, 16, 1], 'offset': [0]}, [torch.randn(4, 8, 16)], (4, 8, 16), None),
        ({'size': [4, 8, 16], 'stride': [128, 16, 1], 'offset': [0]}, [torch.randn(4, 8, 15)], (4, 8, 16), RuntimeError)
    ]
    for op_param, in_tensors, expected_shape, expected_error in test_cases:
        OpcheckAsStridedOperation = import_as_strided_module['OpcheckAsStridedOperation']
        op = OpcheckAsStridedOperation()
        op.op_param = op_param

        if expected_error:
            with pytest.raises(expected_error):
                op.golden_calc(in_tensors)
        else:
            result = op.golden_calc(in_tensors)
            assert result[0].shape == expected_shape


def test_test_given_op_param_when_valid_input_then_execute_successfully(import_as_strided_module):
    test_cases = [
        ({'size': [4, 8, 16], 'stride': [128, 16, 1], 'offset': [0]}, True, True),
        ({'size': [4, 8, 16], 'stride': [128, 16, 1], 'offset': [0]}, False, False),
        ({'stride': [128, 16, 1], 'offset': [0]}, False, False),
        ({'size': [4, 8, 16], 'offset': [0]}, False, False),
        ({'size': [4, 8, 16], 'stride': [128, 16, 1]}, False, False)
    ]
    for op_param, validate_param_return, expected_execute_call in test_cases:
        OpcheckAsStridedOperation = import_as_strided_module['OpcheckAsStridedOperation']
        op = OpcheckAsStridedOperation()
        op.op_param = op_param

        with patch.object(op, 'validate_param', return_value=validate_param_return):
            with patch.object(op, 'execute') as mock_execute:
                op.test()

        if expected_execute_call:
            mock_execute.assert_called_once()
        else:
            mock_execute.assert_not_called()
